struct FittedState
  is_fitted::Bool
  is_transformed::Bool
  rho::Float64
end

struct SpatialVertexModel{V <: AbstractVertexModel, P <: AbstractPenalty, C} <: AbstractSpatialModel
  triobs::Vector{TriangleObs}
  vertex::Vector{V}
  penalty::P
  caches::C
  state::FittedState
end

function SpatialVertexModel(triobs, vertex, penalty, caches)
  state = FittedState(false, false, 0.0)
  return SpatialVertexModel(triobs, vertex, penalty, caches, state)
end

function create_vertex_set(
  ::Type{VT},
  tri::Triangulation,
  triobs::Vector{TriangleObs},
  nvar::Int;
  kwargs...
) where VT <: AbstractVertexModel
  vertex = VT[]
  
  # recreate mapping from triangulation labels to our labels
  indices = each_solid_vertex(tri) |> collect |> sort
  id2vertex = Dict{Int,Int}(id => j for (j, id) in enumerate(indices))
  
  # This assumes triangles in tobs are labeled using OUR scheme, not the one in the triangulation.
  wmin, wmax = Inf, -Inf
  for (j, id) in enumerate(indices)
    # count the total number of samples incident with vertex j
    nobs, part, triangles = 0, UnitRange{Int}[], Int[]
    for k in eachindex(triobs)
      if has_vertex(triobs[k], j)
        n = length(triobs[k].y)
        push!(part, (1+nobs):(nobs+n))
        push!(triangles, k)
        nobs += n
      end
    end

    # determine the number of incident vertices
    neighbors = [id2vertex[u] for u in DelaunayTriangulation.iterated_neighbourhood(tri, id, 1)]
    nneighbors = length(neighbors)

    # initialize weights
    weights = ones(nneighbors)
    for (k, other) in enumerate(DelaunayTriangulation.iterated_neighbourhood(tri, id, 1))
      weights[k] = DelaunayTriangulation.dist(get_point(tri, id), get_point(tri, other))
      wmin = min(wmin, weights[k])
      wmax = max(wmax, weights[k])
    end

    push!(vertex,
      create_vertex(VT, j, part, triangles, neighbors, weights, nobs, nvar; kwargs...)
    )
  end
  
  for v in vertex
    v.weights .= v.weights
  end

  return vertex
end

function create_model(::Type{VT}, y::Vector{T}, X::Matrix{T}, S::Matrix{T}, tri::Triangulation;
  penalty::P = L2Squared(),
  nchunks::Int = Threads.nthreads(),
  kwargs...
  ) where {VT <: AbstractVertexModel, T <: Real, P <: AbstractPenalty}
  #
  !isconcretetype(VT) && @warn "Detected non-concrete type for vertices. Performance may be degraded." VT
  nvar = size(X, 2)
  triobs = create_triobs_set(y, X, S, tri; nchunks = nchunks)
  vertex = create_vertex_set(VT, tri, triobs, nvar; kwargs...)
  caches = build_caches(triobs, vertex, penalty, nvar, nchunks)
  return SpatialVertexModel(triobs, vertex, penalty, caches)
end
#
# INITIALIZATION HEURISTICS
#
function initialize!(model::SpatialVertexModel)
  for v in eachvertex(model)
    if !isempty(v.triangles)
      model.vertex[v.index] = initialize_coefficients!(v, model.triobs)
      model.vertex[v.index] = Accessors.@set v.dispersion = 1.0
    end
  end
end
#
# ITERATION
#
eachobsindex(triobs::TriangleObs) = eachindex(triobs.y)

eachtriobs(triobs::Vector{TriangleObs}) = TriangleObsIterator(triobs)
eachtriobs(triobs::Vector{TriangleObs}, v::AbstractVertexModel) = TriangleObsIterator(triobs, v.triangles)
eachtriobs(model::SpatialVertexModel) = eachtriobs(model.triobs)
eachtriobs(model::SpatialVertexModel, v::AbstractVertexModel) = eachtriobs(model.triobs, v.triangles)

eachvertex(model::SpatialVertexModel) = model.vertex
#
# CACHES
#
function init_cache_logf(triobs)
  return Dict(triidx => zeros(3, length(t.y)) for (triidx, t) in TriangleObsIterator(triobs))
end

function update_cache_logf!(cache, v::AbstractVertexModel, triobs::Vector{TriangleObs})
  j, β, η, μ = v.index, v.beta, v.eta, v.mu
  family, link = v.family, v.link
  φ = v.dispersion
  for (k, idxrange) in zip(v.triangles, v.part)
    T = triobs[k]
    triidx, y, X = T.triangle, T.y, T.X
    t = get_triangle_vertices(T, j)
    logf = view(cache[triidx], t, :)
    mul!(view(η, idxrange), X, β) # η = Xβ
    @inbounds for (i, idx) in zip(eachindex(logf, y), idxrange)
      μ[idx] = meanfun(link, η[idx])
      logfᵢ = GLMUtilities.logpdf(y[i], μ[idx], φ, η[idx], family, link)
      logf[i] = logfᵢ
    end
  end
end

function init_cache_gamma(vertex, nvar)
  return [zeros(nvar, length(v.neighbors)) for v in vertex]
end

function update_cache_gamma!(cache, v::V, vertex::Vector{V}) where V <: AbstractVertexModel
  β = v.beta
  Γ = cache[v.index]
  for (i, k) in zip(axes(Γ, 2), v.neighbors)
    βₖ = vertex[k].beta
    @views begin
      @. Γ[:, i] = 1//2 * (β + βₖ)
    end
  end
end
#
# LOSS
#
function (f::SpatialVertexModel)(rho; kwargs...)
  loss = eval_loss(f.triobs, f.vertex, f.caches; kwargs...)
  penalty1 = eval_penalty(f.penalty, f.vertex)
  return loss + rho*penalty1
end

function eval_loss(triobs::Vector{TriangleObs}, ::Vector{<:AbstractVertexModel}, caches; nchunks::Int = Threads.nthreads(), kwargs...)
  cache = caches.logf
  logl = @tasks for T in triobs
    @set begin
      ntasks = nchunks
      reducer = +
      outputtype = Float64
    end
    triidx = T.triangle # can't use eachtriobs() here due to type instability
    F = cache[triidx]
    local c = zero(Float64)
    for i in eachobsindex(T)
      alpha = @get_triple T.A i
      logf = @get_triple F i
      tmp = weighted_logsumexp(alpha, logf)
      c += tmp
    end
    c
  end
  return -logl
end

function eval_negloglik(y, X, S, model, mesh;
  nchunks::Int = Threads.nthreads(),
  )
  #
  helper = create_triobs_helper(mesh)
  vertex = helper.loc
  id2loc = helper.id2loc
  logl = @tasks for idxset in chunks(eachindex(y); n = nchunks)
    @set begin
      chunking = false
      reducer = +
      outputtype = Float64
    end
    local c = zero(Float64)
    local alpha = zeros(Float64, 3)
    local logf = zeros(Float64, 3)
    local V = zeros(Float64, 2, 3)
    @views for i in idxset
      s = S[:, i]
      x = X[i, :]
      # Assemble matrix V containing vertices of enclosing triangle
      j, k, l = find_triangle(mesh, s, concavity_protection = true)
      j, k, l = sort((id2loc[j], id2loc[k], id2loc[l]))
      for (col, t) in enumerate((j, k, l)), row in 1:2
        V[row, col] = vertex[t][row]
      end
      # Solve for barycentric coordinates
      barycentric!(alpha, s, V)
      # Evaluate log-likelihood terms in triangle
      for (m, t) in enumerate((j, k, l))
        v = model.vertex[t]
        beta, family, link = v.beta, v.family, v.link
        if needs_dispersion(family)
          phi = model.state.is_transformed ? 1 / v.dispersion : v.dispersion
        else
          phi = one(v.dispersion)
        end
        eta = dot(x, beta)
        mu = meanfun(link, eta)
        logf[m] = GLMUtilities.logpdf(y[i], mu, phi, eta, family, link)
      end
      c += weighted_logsumexp(alpha, logf)
    end
    return c
  end
  return -logl
end
#
# SURROGATE
#
struct CoefficientSurrogate{V <: AbstractVertexModel, P <: AbstractPenalty, C, NT}
  index::Int
  model::SpatialVertexModel{V,P,C}
  opt::NT
end

function (g::CoefficientSurrogate)(beta)
  global ξ
  model = g.model
  caches = model.caches
  j = g.index
  v = model.vertex[j]
  loss = eval_loss_surrogate(beta, v, model.triobs, caches, ξ[])
  penalty1 = eval_penalty_surrogate(model.penalty, beta, v.weights, caches.gamma[j], v.beta)
  return loss + g.opt.rho*penalty1
end
#
# ESTIMATION
#
function update_coefficients!(model, opt, nchunks)
  workspace = ChannelLike(model.caches.workspace)
  workitr = ChannelLike(eachvertex(model))
  @safe_blas begin
    tforeach(1:nchunks; chunking = false) do _
      map(workspace) do wrk
        for v in workitr
          local g = CoefficientSurrogate(v.index, model, opt)
          if isempty(v.triangles)
            # Case: Incident observation sets are empty
            # Update the coefficients using the penalty term
            model.vertex[v.index] = update_empty_case!(model.penalty, v, v.weights, model.caches)
          else
            # Case: There is at least one non-empty observation set
            model.vertex[v.index] = mm_update_coef!(model.penalty, g, v, model.triobs, wrk, model.caches)
          end
        end
      end
    end
  end nchunks=nchunks

  # Apply all updates
  for v in eachvertex(model)
    @. v.beta = v.beta_new
  end

  return nothing
end

function update_dispersion!(model, opt, nchunks)
  add_tuple(x, y) = x .+ y
  workspace = model.caches.workspace
  workitr = ChannelLike(eachvertex(model))
  a, b = @tasks for i in 1:nchunks
    @set begin
      chunking = false
      reducer = add_tuple
      init = (0.0, 0.0)
      outputtype = Tuple{Float64,Float64}
    end
    local wrk = workspace[i]
    local a = 0.0
    local b = 0.0
    for v in workitr
      aj, bj = local_deviance(v.family, v, model.triobs, wrk, model.caches)
      a += aj
      b += bj
    end
    (a, b) # return value for reduction
  end

  phi = a / b

  for v in eachvertex(model)
    model.vertex[v.index] = (Accessors.@set v.dispersion = phi)
  end

  return nothing
end
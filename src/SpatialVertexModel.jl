struct SpatialVertexModel{V <: AbstractVertexModel, P <: AbstractPenalty, C} <: AbstractSpatialModel
  triobs::Vector{TriangleObs}
  vertex::Vector{V}
  penalty::P
  caches::C
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
  wmax = 0.0
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
    sort!(neighbors)
    nneighbors = length(neighbors)

    # initialize weights
    weights = ones(nneighbors)
    for (k, other) in enumerate(DelaunayTriangulation.iterated_neighbourhood(tri, id, 1))
      weights[k] = 1 / DelaunayTriangulation.dist(get_point(tri, id), get_point(tri, other))
      wmax = max(wmax, weights[k])
    end

    push!(vertex,
      create_vertex(VT, j, part, triangles, neighbors, weights, nobs, nvar; kwargs...)
    )
  end
  
  for v in vertex
    v.weights .= v.weights / wmax
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
  φ = v.extra_params[:dispersion]
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
  return Dict(v.index => zeros(nvar, length(v.neighbors)) for v in vertex)
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
  penalty = eval_penalty(f.penalty, f.vertex)
  prior_loss = eval_prior_loss(f.vertex; kwargs...)
  return loss + rho*penalty + prior_loss
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
#
# SURROGATE
#
struct VertexSurrogate{V <: AbstractVertexModel, P <: AbstractPenalty, C}
  index::Int
  model::SpatialVertexModel{V,P,C}
  rho::Float64
end

function (g::VertexSurrogate)(beta; kwargs...)
  j = g.index
  v = g.model.vertex[j]
  loss = eval_loss_surrogate(beta, v, g.model.triobs, g.model.caches)
  penalty = eval_penalty_surrogate(g.model.penalty, beta, v.weights, g.model.caches.gamma[j], v.beta)
  log_prior = eval_log_prior_vertex(beta, v; kwargs...)
  return loss + g.rho*penalty - log_prior
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
          local g = VertexSurrogate(v.index, model, opt.rho)
          if isempty(v.triangles)
            # Case: Incident observation sets are empty
            # Update the coefficients using the penalty term
            update_empty_case!(model.penalty, v, v.weights, model.caches)
          else
            # Case: There is at least one non-empty observation set
            mm_update_coef!(model.penalty, g, v, model.triobs, wrk, model.caches, opt)
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

function update_dispersion!(model, opt, nchunks, use_prior)
  workspace = ChannelLike(model.caches.workspace)
  workitr = ChannelLike(eachvertex(model))
  @safe_blas begin
    tforeach(1:nchunks; chunking = false) do _
      map(workspace) do wrk
        for v in workitr
          local g = VertexSurrogate(v.index, model, opt.rho)
          mm_update_disp!(v.family, g, v, model.triobs, wrk, model.caches, use_prior)
        end
      end
    end
  end nchunks=nchunks

  return nothing
end
struct VertexGLM{D <: Distribution, L <: Link} <: AbstractVertexModel
  family::D                   # distribution
  link::L                     # link function, g(μ) = η
  index::Int                  # vertex index, needed for consitency with triangulation
  part::Vector{UnitRange{Int}}# partition of indices into incident triangles
  d::Vector{Float64}          # IRLS weights, determined by GLM + mixture coefficients
  beta::Vector{Float64}       # coefficients, β
  eta::Vector{Float64}        # linear predictor, η = xᵀβ
  mu::Vector{Float64}         # mean parameter vector, μ = g⁻¹(η)
  workres::Vector{Float64}    # working residuals, (y - μ) g'(μ)
  neighbors::Vector{Int}      # index set representing neighboring vertices in penalty
  triangles::Vector{Int}      # index set representing incident triangles
  weights::Vector{Float64}    # penalty weights, w in ∑ w P(B)
  beta_new::Vector{Float64}   # proposed update
end

function create_vertex_set(::Type{VertexGLM}, tri::Triangulation, triobs::Vector{TriangleObs}, nvars::Int;
  family::D = Normal(),
  link::L = canonicallink(family),
  ) where {D <: UnivariateDistribution, L <: Link}
  vertex = VertexGLM{D,L}[]
  
  # recreate mapping from triangulation labels to our labels
  id2vertex = Dict{Int,Int}()
  for (j, id) in enumerate(each_solid_vertex(tri))
    id2vertex[id] = j
  end
  
  # This assumes triangles in tobs are labeled using OUR scheme, not the one in the triangulation.
  wmax = 0.0
  for (j, id) in enumerate(each_solid_vertex(tri))
    # count the total number of samples incident with vertex j
    nsamples, part, triangles = 0, UnitRange{Int}[], Int[]
    for k in eachindex(triobs)
      if has_vertex(triobs[k], j)
        n = length(triobs[k].y)
        push!(part, (1+nsamples):(nsamples+n))
        push!(triangles, k)
        nsamples += n
      end
    end

    # determine the number of incident vertices
    neighbors = [id2vertex[u] for u in DelaunayTriangulation.iterated_neighbourhood(tri, id, 1)]
    sort!(neighbors)
    nneighbors = length(neighbors)

    # allocate!
    d = zeros(nsamples)
    beta = zeros(nvars)
    eta = zeros(nsamples)
    mu = zeros(nsamples)
    workres = zeros(nsamples)
    weights = ones(nneighbors)

    for (k, other) in enumerate(DelaunayTriangulation.iterated_neighbourhood(tri, id, 1))
      weights[k] = 1 / DelaunayTriangulation.dist(get_point(tri, id), get_point(tri, other))
      wmax = max(wmax, weights[k])
    end

    push!(vertex, VertexGLM(family, link, j, part,
        d, beta, eta, mu, workres,
        neighbors, triangles, weights, similar(beta)
      )
    )
  end
  
  for v in vertex
    v.weights .= v.weights / wmax
  end

  return vertex
end
#
# CACHES
#
function build_caches(triobs::Vector{TriangleObs}, vertex::Vector{V}, ::AbstractPenalty, nvars, nchunks) where V <: VertexGLM
  return (;
    # Store log-likelihood values for each triangle
    logf = Dict(triidx => zeros(3, length(t.y)) for (triidx, t) in TriangleObsIterator(triobs)),
    # matrix of local average coefficients in MM algorithm, Γ
    gamma = Dict(v.index => zeros(nvars, length(v.neighbors)) for v in vertex),
    workspace = [(zeros(nvars), zeros(nvars, nvars), zeros(nvars)) for _ in 1:nchunks]
  )
end

function update_caches!(model::SpatialVertexModel{V}; nchunks::Int = Threads.nthreads()) where V <: VertexGLM
  workitr = ChannelLike(eachvertex(model))
  @safe_blas begin
    @localize model tforeach(1:nchunks; chunking = false) do _
      for v in workitr
        update_cache_logf!(model.caches.logf, v, model.triobs)
        update_cache_gamma!(model.caches.gamma, v, model.vertex)
      end
    end
  end nchunks=nchunks
end

function update_caches_serial!(model::SpatialVertexModel{V}) where V <: VertexGLM
  for v in eachvertex(model)
    update_cache_logf!(model.caches.logf, v, model.triobs)
    update_cache_gamma!(model.caches.gamma, v, model.vertex)
  end
end

function update_cache_logf!(cache, v::VertexGLM, triobs::Vector{TriangleObs})
  j, β, η, μ = v.index, v.beta, v.eta, v.mu
  family, link = v.family, v.link

  for (k, idxrange) in zip(v.triangles, v.part)
    T = triobs[k]
    triidx, y, X = T.triangle, T.y, T.X
    t = get_triangle_vertices(T, j)
    logf = view(cache[triidx], t, :)
    mul!(view(η, idxrange), X, β) # η = Xβ
    @inbounds for (i, idx) in zip(eachindex(logf, y), idxrange)
      μ[idx] = meanfun(link, η[idx])
      logfᵢ = GLMUtilities.log_likelihood(y[i], μ[idx], η[idx], family, link)
      logf[i] = logfᵢ
    end
  end
end

function update_cache_gamma!(cache, v::V, vertex::Vector{V}) where V <: VertexGLM
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
function eval_loss(triobs::Vector{TriangleObs}, ::Vector{V}, caches; nchunks::Int = Threads.nthreads()) where V <: VertexGLM
  cache = caches.logf
  logl = @localize cache @tasks for T in triobs
    @set begin
      ntasks = nchunks
      reducer = +
      outputtype = Float64
    end
    triidx = T.triangle # can't use eachtriobs() here due to type instability
    local c = zero(Float64)
    for i in eachobsindex(T)
      a = view(T.A, :, i)
      logf = view(cache[triidx], :, i)
      tmp = stable_logsumexp(a, logf)
      c += tmp
    end
    c
  end
  return -logl
end
#
# SURROGATE
#
function eval_loss_surrogate(beta, v::V, triobs::Vector{TriangleObs}, caches) where V <: VertexGLM
  cache = caches.logf
  logl = zero(Float64)
  for (triidx, T) in eachtriobs(triobs, v)
    t = get_triangle_vertices(T, v.index)
    for i in eachobsindex(T)
      x = view(T.X, i, :)
      a = view(T.A, :, i)
      logf = view(cache[triidx], :, i)

      # Evaluate log-likelihood term at β
      eta = dot(x, beta)
      mu = meanfun(v.link, eta)
      logfⱼ = GLMUtilities.log_likelihood(T.y[i], mu, eta, v.family, v.link)
      # Evaluate terms dependent on the anchor point βₙ
      zweight = stable_eval_mm_weight(t, a, logf)

      tmp = zweight * logfⱼ + zweight * (log(a[t]) + log(inv(zweight)))
      logl += tmp
    end
  end
  return -logl
end
#
# UPDATES
#
function mm_update!(penalty::AbstractPenalty, g::VertexSurrogate, v::V, triobs::Vector{TriangleObs}, workspace, caches, backtrack) where V <: VertexGLM
  # Setup local variables to match notation
  family = v.family
  link = v.link
  Γ = caches.gamma[v.index]
  d = v.d
  r = v.workres
  η = v.eta
  μ = v.mu
  w = v.weights
  ∇L, ∇²L, Δ = workspace
  
  fill!(∇L, 0); fill!(∇²L, 0)
  if isempty(v.triangles)
    # Case: Incident observation sets are empty
    update_empty_case!(penalty, v, w, Γ)
  else
    itr = zip(eachtriobs(triobs, v), v.part)
    for ((triidx, T), idxrange) in itr
      y, X, A, t = T.y, T.X, T.A, get_triangle_vertices(T, v.index)
      for (i, idx) in zip(eachobsindex(T), idxrange)
        x = view(X, i, :)
        a = view(A, :, i)
        logf = view(caches.logf[triidx], :, i)
        zweight = stable_eval_mm_weight(t, a, logf)
        dμdη = meanderiv(v.link, η[idx]) 
        r[idx] = (y[i] - μ[idx]) / dμdη
        d[idx] = zweight * stable_irls_weight(family, link, η[idx], μ[idx], dμdη)

        # Evaluate gradient + Hessian
        BLAS.axpy!(-2*d[idx]*r[idx], x, ∇L)
        BLAS.syr!('U', 2*d[idx], x, ∇²L)
      end
    end
    accumulate_penalty_derivs!(penalty, ∇L, ∇²L, v, Γ, g.rho)
    H = Symmetric(∇²L, :U)
    ldiv!(Δ, cholesky!(H), ∇L)
    linesearch!(g, v.beta_new, v.beta, Δ, backtrack)
  end
end

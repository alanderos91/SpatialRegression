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
  extra_params::Dict{Symbol,Float64}
end

function create_vertex_set(::Type{VertexGLM}, tri::Triangulation, triobs::Vector{TriangleObs}, nvars::Int;
  family::D = Normal(),
  link::L = canonicallink(family),
  ) where {D <: UnivariateDistribution, L <: Link}
  vertex = VertexGLM{D,L}[]
  
  # recreate mapping from triangulation labels to our labels
  indices = each_solid_vertex(tri) |> collect |> sort
  id2vertex = Dict{Int,Int}(id => j for (j, id) in enumerate(indices))
  
  # This assumes triangles in tobs are labeled using OUR scheme, not the one in the triangulation.
  wmax = 0.0
  for (j, id) in enumerate(indices)
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
        neighbors, triangles, weights, similar(beta),
        Dict(:dispersion => 1.0),
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
      alpha = view(T.A, :, i)
      logf = view(cache[triidx], :, i)
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
function eval_loss_surrogate(beta, v::V, triobs::Vector{TriangleObs}, caches) where V <: VertexGLM
  cache = caches.logf
  logl = zero(Float64)
  phi = v.extra_params[:dispersion]
  j, family, link = v.index, v.family, v.link
  for (triidx, T) in eachtriobs(triobs, v)
    t = get_triangle_vertices(T, j)
    for i in eachobsindex(T)
      x = view(T.X, i, :)
      alpha = view(T.A, :, i)
      logf = view(cache[triidx], :, i)

      # Evaluate log-likelihood term at β
      eta = dot(x, beta)
      mu = meanfun(link, eta)
      logfⱼ = GLMUtilities.logpdf(T.y[i], mu, phi, eta, family, link)

      # Evaluate terms dependent on the anchor point βₙ
      zweight = stable_convex_weight(t, alpha, logf)
      tmp = zweight < eps() ? zero(zweight) : zweight * logfⱼ + zweight * (log(alpha[t]) + log(inv(zweight)))
      @assert !isnan(tmp) && !isinf(tmp) "bad weighted log-likelihood @ vertex $(v.index)\n\tη: $(eta)\n\tμ: $(mu)\n\tlogf: $(logfⱼ)\n\tz: $(zweight)"
      logl += tmp
    end
  end
  return -logl
end
#
# UPDATES
#
function initialize_coefficients!(v::VertexGLM, triobs, vertex)
  X = similar(v.beta, length(v.d), length(v.beta))
  y = similar(v.beta, length(v.d))
  for (k, idxrange) in zip(v.triangles, v.part)
    T = triobs[k]
    X[idxrange, :] .= T.X
    y[idxrange] .= T.y
  end
  fitted = glm(X, y, v.family, v.link)
  v.beta .= coef(fitted)
  if needs_dispersion(v.family)
    v.extra_params[:dispersion] = 1
  end
  return nothing
end

function mm_update_coef!(penalty::AbstractPenalty, g::VertexSurrogate, v::V, triobs::Vector{TriangleObs}, workspace, caches, backtrack) where V <: VertexGLM
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
    # Update the coefficients using the penalty term
    update_empty_case!(penalty, v, w, Γ)

    # Dispersion parameter cannot be determined, so we keep it equal to 1.
  else
    # Case: There is at least one non-empty observation set
    itr = zip(eachtriobs(triobs, v), v.part)
    phi = v.extra_params[:dispersion]

    # Update the coefficients using IRWLS
    for ((triidx, T), idxrange) in itr
      y, X, A, t = T.y, T.X, T.A, get_triangle_vertices(T, v.index)
      for (i, idx) in zip(eachobsindex(T), idxrange)
        x = view(X, i, :)
        alpha = view(A, :, i)
        logf = view(caches.logf[triidx], :, i)
        zweight = stable_convex_weight(t, alpha, logf)
        dμdη = meanderiv(v.link, η[idx]) 
        r[idx] = (y[i] - μ[idx]) / dμdη
        d[idx] = zweight * stable_irls_weight(family, link, η[idx], μ[idx], dμdη)
        if iszero(zweight)
            d[idx] = one(zweight)
            r[idx] = zero(zweight)
        end
        # @assert zweight > 0 "MM Update w/ Zero Weight @ Vertex $(v.index)\n\tObs. Index: $(idx)\n\talpha = $(alpha)\n\tlogf = $(logf)\n\tResidual: $(r[idx])\n\tIRLS Weight: $(stable_irls_weight(family, link, η[idx], μ[idx], dμdη))"

        # Evaluate gradient + Hessian
        BLAS.axpy!(-d[idx]*r[idx] * dispfun(family, phi), x, ∇L)
        BLAS.syr!('U', d[idx] * dispfun(family, phi), x, ∇²L)
      end
    end
    accumulate_penalty_derivs!(penalty, ∇L, ∇²L, v, Γ, g.rho)
    H = Symmetric(∇²L, :U)
    ldiv!(Δ, cholesky!(H), ∇L)
    linesearch!(g, v.beta_new, v.beta, Δ, backtrack)
  end
end

function mm_update_disp!(::Distribution, g::VertexSurrogate, v::V, triobs::Vector{TriangleObs}, workspace, caches) where V <: VertexGLM
  error("MM updates not implemented for the $(v.family) case.")
end

function mm_update_disp!(::Normal, g::VertexSurrogate, v::V, triobs::Vector{TriangleObs}, workspace, caches) where V <: VertexGLM
  # Setup local variables to match notation
  family = v.family
  link = v.link
  d = v.d
  r = v.workres
  η = v.eta
  μ = v.mu

  itr = zip(eachtriobs(triobs, v), v.part)
  g_prev = g(v.beta)

  # Fix β, update φ
  N = 0
  num = den = zero(eltype(v.beta))
  for ((triidx, T), idxrange) in itr
    y, X, A, t = T.y, T.X, T.A, get_triangle_vertices(T, v.index)
    for (i, idx) in zip(eachobsindex(T), idxrange)
      x = view(X, i, :)
      alpha = view(A, :, i)
      logf = view(caches.logf[triidx], :, i)
      zweight = stable_convex_weight(t, alpha, logf)
      η[idx] = dot(x, v.beta)
      μ[idx] = meanfun(link, η[idx])
      num += zweight * abs2(y[i] - μ[idx])
      den += zweight
      N += 1
      # v.index == 408 && @show zweight, abs2(y[i] - μ[idx])
    end
  end
  phi = den / num
  v.extra_params[:dispersion] = isinf(phi) ? one(phi) : phi 

  g_curr = g(v.beta)
  @assert abs(g_curr - g_prev) <= 10 * (1 + abs(g_prev)) "\n\tVertex: $(v.index)\n\tNobs: $(N)\n\tPrevious: $(g_prev)\n\t Current: $(g_curr)\n\tDen/Num: $(den) / $(num)\n\tφ: $(phi)\n\tβ: $(v.beta)"
  # @assert g_curr <= g_prev "\n\tVertex: $(v.index)\n\tNobs: $(N)\n\tPrevious: $(g_prev)\n\t Current: $(g_curr)\n\tDen/Num: $(den) / $(num)\n\tφ: $(phi)\n\tβ: $(v.beta)"

end

function fitmodel(::Type{V}, yfull, Xfull, Sfull, tri;
    maxiter::Int = 100,
    backtrack::Int = 5,
    tol::Real = 1e-3,
    rho::Real = 1.0,
    nchunks::Int = Threads.nthreads(),
    kwargs...
    # intercept = all(isequal(1), view(Xfull, :, 1)),
  ) where V <: VertexGLM
  # Initialize
  nvars = size(Xfull, 2)
  model = f = create_model(V, yfull, Xfull, Sfull, tri; nchunks, kwargs...)
  initialize_coefficients!(model.triobs, model.vertex)
  update_caches!(model; nchunks)
  nlogl = f(rho; nchunks)
  nlogl_prev = zero(nlogl)
  iter = 0
  while iter < maxiter && abs(nlogl - nlogl_prev) > (1 + abs(nlogl_prev)) * tol
    iter += 1

    # Visit each vertex once to update local regression coefficients
    workspace = ChannelLike(model.caches.workspace)
    workitr = ChannelLike(eachvertex(model))
    @safe_blas begin
      @localize iter model rho backtrack workspace workitr tforeach(1:nchunks; chunking = false) do _
        map(workspace) do wrk
          for v in workitr
            g = VertexSurrogate(v.index, model, rho)
            if isempty(v.triangles)
              update_empty_case!(model.penalty, v, v.weights, model.caches)
            else
              mm_update_coef!(model.penalty, g, v, model.triobs, wrk, model.caches, backtrack)
            end
          end
        end
      end
    end nchunks=nchunks

    # Apply all updates
    for v in eachvertex(model)
      @. v.beta = v.beta_new
    end

    # Evaluate log-likelihood
    update_caches!(model; nchunks)
    nlogl_prev = nlogl
    nlogl = f(rho; nchunks)
    @show iter, nlogl, nlogl_prev - nlogl
    @assert nlogl < nlogl_prev

    if needs_dispersion(first(model.vertex).family) # TODO
      abs(nlogl - nlogl_prev) <= (1 + abs(nlogl_prev)) * tol && break

      # Visit each vertex once to update local dispersion parameters
      workspace = ChannelLike(model.caches.workspace)
      workitr = ChannelLike(eachvertex(model))
      @safe_blas begin
        @localize iter model rho workspace workitr tforeach(1:nchunks; chunking = false) do _
          map(workspace) do wrk
            for v in workitr
              if !isempty(v.triangles)
                g = VertexSurrogate(v.index, model, rho)
                mm_update_disp!(v.family, g, v, model.triobs, wrk, model.caches)
              end
            end
          end
        end
      end nchunks=nchunks

      # Evaluate log-likelihood
      update_caches!(model; nchunks)
      nlogl_prev = nlogl
      nlogl = f(rho; nchunks)
      @show iter, nlogl, nlogl_prev - nlogl
      @assert nlogl < nlogl_prev
    end
  end

  # Apply all updates
  for v in eachvertex(model)
    if needs_dispersion(v.family)
      phi = v.extra_params[:dispersion]
      v.extra_params[:dispersion] = 1 / phi
    end
  end

  return iter, model, nlogl
end

function assemble_mixture(x, s, m, tri, id2vertex, points)
    j, k, l = find_triangle(tri, s; concavity_protection = true) |> sort
    p, q, r = points[j], points[k], points[l]
    j, k, l = id2vertex[j], id2vertex[k], id2vertex[l]
    vertex = (m.vertex[j], m.vertex[k], m.vertex[l])
    alpha = barycentric(s, [p[1] q[1] r[1]; p[2] q[2] r[2]])
    return MixtureModel(
      [create_component(v.family, v.link, dot(x, v.beta), v.extra_params[:dispersion]) for v in vertex],
      alpha
    )
end

function assemble_mixture(family, link, x, s, B, Φ, tri, id2vertex, points)
    j, k, l = find_triangle(tri, s; concavity_protection = true) |> sort
    p, q, r = points[j], points[k], points[l]
    vertex_index = id2vertex[j], id2vertex[k], id2vertex[l]
    alpha = barycentric(s, [p[1] q[1] r[1]; p[2] q[2] r[2]])
    return @views MixtureModel(
      [create_component(family, link, dot(x, B[:,idx]), Φ[idx]) for idx in vertex_index],
      alpha
    )
end

function predict(X, S, m::SpatialVertexModel, tri; kind = :mean)
  indices = each_solid_vertex(tri) |> collect |> sort
  id2vertex = Dict{Int,Int}(id => j for (j, id) in enumerate(indices))
  points = get_points(tri)
  yhat = zeros(size(X, 1))
  @views for i in axes(X, 1)
    s = S[:, i]
    x = X[i, :]
    h = assemble_mixture(x, s, m, tri, id2vertex, points)
    
    yhat[i] = if kind == :mean
      mean(h)
    elseif kind == :mode
      _search_mode_(h)
    end
  end
  return yhat
end

function _search_mode_(h)
  # Initialize the predicted value using the weighted mean.
  y = mean(h)

  # Use an MM algorithm to search for a mode
  mu = [mean(d) for d in components(h)]
  logf = Distributions.logpdf.(components(h), y)
  logh = Distributions.logpdf(h, y)
  logh_prev = logh*10
  while abs(logh - logh_prev) > 1e-6 * (1 + abs(logh_prev))
    logh_prev = logh
    y = __mm_update_mode__(component_type(h), Distributions.probs(h), mu, logf)
    logf .= Distributions.logpdf.(components(h), y)
    logh = Distributions.logpdf(h, y)
  end

  return y
end

function __mm_update_mode__(::Type{Normal{T}}, alpha, mu, logf) where T <: Real
  num = den = zero(first(mu))
  for k in 1:3
    weight = stable_convex_weight(k, alpha, logf)
    num += weight * mu[k]
    den += weight # should be 1?
  end
  @assert den ≈ one(den)
  return num / den
end

# function __mm_update_mode__(::Union{Binomial,Bernoulli}, alpha, mu, logf)
#   a = b = zero(first(mu))
#   for k in 1:3
#     weight = stable_eval_mm_weight(k, alpha, logf)
#     a += weight * log(mu[k])
#     b += weight * log1p(-mu[k])
#   end
#   return a > b ? zero
# end

# function __mm_update_mode__(::Poisson, alpha, mu, logf)

# end

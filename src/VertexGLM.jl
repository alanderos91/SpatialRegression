struct VertexGLM{D <: UnivariateDistribution, L <: Link} <: AbstractVertexModel
  index::Int                  # vertex index, needed for consitency with triangulation
  part::Vector{UnitRange{Int}}# partition of indices into incident triangles
  triangles::Vector{Int}      # index set representing incident triangles
  neighbors::Vector{Int}      # index set representing neighboring vertices in penalty
  weights::Vector{Float64}    # penalty weights, w in ∑ w P(B)

  nobs::Int                   # number of incident observations
  nvar::Int                   # number of variables + intercept 
  family::D                   # distribution
  link::L                     # link function, g(μ) = η
  d::Vector{Float64}          # IRLS weights, determined by GLM + mixture coefficients
  beta::Vector{Float64}       # coefficients, β
  beta_new::Vector{Float64}   # proposed update
  eta::Vector{Float64}        # linear predictor, η = xᵀβ
  mu::Vector{Float64}         # mean parameter vector, μ = g⁻¹(η)
  workres::Vector{Float64}    # working residuals, (y - μ) g'(μ)
  extra_params::Dict{Symbol,Float64}
end

infer_vertex_type(::Type{VertexGLM}, ::D, ::L; kwargs...) where {D <: UnivariateDistribution, L <: Link} = VertexGLM{D,L}

function create_vertex(::Type{VertexGLM{D,L}}, index, part, triangles, neighbors, weights, nobs, nvar; family::D=Normal(), link::L=IdentityLink(), kwargs...) where {D <: UnivariateDistribution, L <: Link}
  d = zeros(nobs)
  beta = zeros(nvar)
  beta_new = similar(beta)
  eta = zeros(nobs)
  mu = zeros(nobs)
  workres = zeros(nobs)
  extra_params = Dict(:dispersion => 1.0)
  return VertexGLM(
    index, part, triangles, neighbors, weights,
    nobs, nvar, family, link, d, beta, beta_new,
    eta, mu, workres, extra_params
  )
end
#
# CACHES
#
function build_caches(triobs::Vector{TriangleObs}, vertex::Vector{V}, ::AbstractPenalty, nvar, nchunks) where V <: VertexGLM
  return (;
    # Store log-likelihood values for each triangle
    logf = init_cache_logf(triobs),
    # matrix of local average coefficients in MM algorithm, Γ
    gamma = init_cache_gamma(vertex, nvar),
    # ∇L, ∇²L, Δ
    workspace = [(zeros(nvar), zeros(nvar, nvar), zeros(nvar)) for _ in 1:nchunks]
  )
end

function update_caches!(model::SpatialVertexModel{<:VertexGLM}; nchunks::Int = Threads.nthreads())
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

function update_caches_serial!(model::SpatialVertexModel{<:VertexGLM})
  for v in eachvertex(model)
    update_cache_logf!(model.caches.logf, v, model.triobs)
    update_cache_gamma!(model.caches.gamma, v, model.vertex)
  end
end
#
# SURROGATE
#
function eval_loss_surrogate(beta::Vector, v::VertexGLM, triobs::Vector{TriangleObs}, caches)
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
      @assert !isnan(tmp) && !isinf(tmp) "bad weighted log-likelihood @ vertex $(v.index)\n\tη: $(eta)\n\tμ: $(mu)\n\tφ: $(phi)\n\tlogf: $(logfⱼ)\n\tz: $(zweight)"
      logl += tmp
    end
  end
  return -logl
end

function eval_loss_surrogate(phi::Real, v::VertexGLM, triobs::Vector{TriangleObs}, caches)
  cache = caches.logf
  logl = zero(Float64)
  beta = v.beta
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

      # Evaluate terms dependent on the anchor point φₙ
      zweight = stable_convex_weight(t, alpha, logf)
      tmp = zweight < eps() ? zero(zweight) : zweight * logfⱼ + zweight * (log(alpha[t]) + log(inv(zweight)))
      @assert !isnan(tmp) && !isinf(tmp) "bad weighted log-likelihood @ vertex $(v.index)\n\tNobs: $(v.nobs)\n\tη: $(eta)\n\tμ: $(mu)\n\tφ: $(phi)\n\tlogf: $(logfⱼ)\n\tz: $(zweight)"
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

function mm_update_coef!(penalty::AbstractPenalty, g::VertexSurrogate, v::VertexGLM, triobs::Vector{TriangleObs}, workspace, caches, opt)
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
  backtrack = opt.backtrack

  fill!(∇L, 0); fill!(∇²L, 0)
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

function mm_update_disp!(::UnivariateDistribution, g::VertexSurrogate, v::VertexGLM, triobs::Vector{TriangleObs}, workspace, caches)
  error("MM updates not implemented for the $(v.family) case.")
end

function mm_update_disp!(::Normal, g::VertexSurrogate, v::VertexGLM, triobs::Vector{TriangleObs}, workspace, caches)
  # Setup local variables to match notation
  η = v.eta
  μ = v.mu

  phi_old = v.extra_params[:dispersion]
  g_prev = g(phi_old)
  itr = zip(eachtriobs(triobs, v), v.part)

  # Fix β, update φ
  num = den = zero(eltype(v.beta))
  for ((triidx, T), idxrange) in itr
    y, X, A, t = T.y, T.X, T.A, get_triangle_vertices(T, v.index)
    for (i, idx) in zip(eachobsindex(T), idxrange)
      x = view(X, i, :)
      alpha = view(A, :, i)
      logf = view(caches.logf[triidx], :, i)
      zweight = stable_convex_weight(t, alpha, logf)
      η[idx] = dot(x, v.beta)
      μ[idx] = meanfun(v.link, η[idx])
      num += zweight * deviance(v.family, y[i], μ[idx])
      den += zweight
      # v.index == 1 && @show zweight, deviance(v.family, y[i], μ[idx])
    end
  end
  phi = den / num

  g_curr = g(phi)
  @assert g_curr < g_prev || abs(g_curr - g_prev) < sqrt(eps()) * (1 + abs(g_prev)) "\n\tVertex: $(v.index)\n\tNobs: $(v.nobs)\n\tPrevious: $(g_prev)\n\t Current: $(g_curr)\n\tDen/Num: $(den) / $(num)\n\tφ₀: $(phi_old)\n\tφ: $(phi)\n\tβ: $(v.beta)"

  v.extra_params[:dispersion] = isinf(phi) ? one(phi) : phi
end

function fitmodel(::Type{VertexGLM}, yfull, Xfull, Sfull, tri;
    family::D = Normal(),
    link::L = IdentityLink(),
    maxiter::Int = 100,
    backtrack::Int = 5,
    tol::Real = 1e-3,
    rho::Real = 1.0,
    nchunks::Int = Threads.nthreads(),
    kwargs...
    # intercept = all(isequal(1), view(Xfull, :, 1)),
  ) where {D <: UnivariateDistribution, L <: Link}
  # Initialize
  VT = infer_vertex_type(VertexGLM, family, link; kwargs...)
  model = f = create_model(VT, yfull, Xfull, Sfull, tri; nchunks, family, link, kwargs...)
  initialize_coefficients!(model.triobs, model.vertex)
  update_caches!(model; nchunks)
  nlogl = f(rho; nchunks)
  nlogl_prev = zero(nlogl)
  iter = 0
  opt = (; rho = rho,  backtrack = backtrack)
  while iter < maxiter && abs(nlogl - nlogl_prev) > (1 + abs(nlogl_prev)) * tol
    iter += 1

    # Visit each vertex once to update local regression coefficients
    update_coefficients!(model, opt, nchunks)

    # Evaluate log-likelihood
    update_caches!(model; nchunks)
    nlogl_prev = nlogl
    nlogl = f(rho; nchunks)
    @show iter, nlogl, nlogl_prev - nlogl
    @assert nlogl < nlogl_prev || abs(nlogl - nlogl_prev) < sqrt(eps()) * (1 + abs(nlogl_prev))

    if needs_dispersion(first(model.vertex).family) # TODO
      abs(nlogl - nlogl_prev) <= (1 + abs(nlogl_prev)) * tol && break

      # Visit each vertex once to update local dispersion parameters
      update_dispersion!(model, opt, nchunks)

      # Evaluate log-likelihood
      update_caches!(model; nchunks)
      nlogl_prev = nlogl
      nlogl = f(rho; nchunks)
      @show iter, nlogl, nlogl_prev - nlogl
      @assert nlogl < nlogl_prev || abs(nlogl - nlogl_prev) < sqrt(eps()) * (1 + abs(nlogl_prev))
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

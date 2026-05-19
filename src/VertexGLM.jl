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

function create_vertex(::Type{VertexGLM{D,L}}, index, part, triangles, neighbors, weights, nobs, nvar;
  family::D = Normal(),
  link::L   = IdentityLink(),
  nu::Real  = 2, 
  kwargs...) where {D <: UnivariateDistribution, L <: Link}
  d = zeros(nobs)
  beta = zeros(nvar)
  beta_new = similar(beta)
  eta = zeros(nobs)
  mu = zeros(nobs)
  workres = zeros(nobs)
  extra_params = Dict(
    :dispersion => 1.0,
    :global_dispersion => 1.0,
    :nu => float(nu),
  )
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
    tforeach(1:nchunks; chunking = false) do _
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

eval_log_prior_vertex(::Vector, v::VertexGLM; kwargs...) = eval_log_prior_vertex(v.extra_params[:dispersion], v; kwargs...)

function eval_log_prior_vertex(phi::Real, v::VertexGLM; use_prior::Bool = false)
  if use_prior
    # assumes conjugancy with Gaussian-like distribution
    nu = v.extra_params[:nu]
    phi0 = v.extra_params[:global_dispersion]
    Distributions.logpdf(Distributions.Gamma(nu/2+1, 2*phi0/nu), phi)
  else
    # convex log penalty on ratios; flipped signed!
    nu = v.extra_params[:nu]
    A = v.extra_params[:local_dispersion]
    nu * 1//2 * log(phi / A)
  end
end

function eval_prior_loss(vertex::Vector{V}; nchunks::Int = Threads.nthreads(), kwargs...) where V <: VertexGLM
  if needs_dispersion(first(vertex).family)
    logl = @tasks for v in vertex
      @set begin
        ntasks = nchunks
        reducer = +
        outputtype = Float64
      end
      local phi = v.extra_params[:dispersion]
      eval_log_prior_vertex(phi, v; kwargs...)
    end
  else
    logl = 0.0
  end
  return -logl
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
    F = cache[triidx]
    for i in eachobsindex(T)
      x = view(T.X, i, :)
      alpha = @get_triple T.A i
      logf = @get_triple F i

      # Evaluate log-likelihood term at β
      eta = v.nvar > 1 ? dot(x, beta) : x[1]*beta[1]
      mu = meanfun(link, eta)
      logfⱼ = GLMUtilities.logpdf(T.y[i], mu, phi, eta, family, link)

      # Evaluate terms dependent on the anchor point βₙ
      zweight = stable_convex_weight(t, alpha, logf)
      tmp = zweight < eps() ? zero(zweight) : zweight * logfⱼ + zweight * (log(alpha[t]) + log(inv(zweight)))
      # @assert !isnan(tmp) && !isinf(tmp) "bad weighted log-likelihood @ vertex $(v.index)\n\tη: $(eta)\n\tμ: $(mu)\n\tφ: $(phi)\n\tlogf: $(logfⱼ)\n\tz: $(zweight)"
      logl += tmp
    end
  end
  return -logl
end

function eval_loss_surrogate(phi::Real, v::VertexGLM, triobs::Vector{TriangleObs}, caches)
  cache = caches.logf
  logl = zero(Float64)
  j, family, link = v.index, v.family, v.link
  itr = zip(eachtriobs(triobs, v), v.part)
  for ((triidx, T), idxrange) in itr
    t = get_triangle_vertices(T, j)
    F = cache[triidx]
    for (i, idx) in zip(eachobsindex(T), idxrange)
      alpha = @get_triple T.A i
      logf = @get_triple F i

      # Evaluate log-likelihood term at ϕ
      eta = v.eta[idx]
      mu = v.mu[idx]
      logfⱼ = GLMUtilities.logpdf(T.y[i], mu, phi, eta, family, link)

      # Evaluate terms dependent on the anchor point φₙ
      zweight = stable_convex_weight(t, alpha, logf)
      tmp = zweight < eps() ? zero(zweight) : zweight * logfⱼ + zweight * (log(alpha[t]) + log(inv(zweight)))
      # @assert !isnan(tmp) && !isinf(tmp) "bad weighted log-likelihood @ vertex $(v.index)\n\tNobs: $(v.nobs)\n\tη: $(eta)\n\tμ: $(mu)\n\tφ: $(phi)\n\tlogf: $(logfⱼ)\n\tz: $(zweight)"
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
    F = caches.logf[triidx]
    for (i, idx) in zip(eachobsindex(T), idxrange)
      x = view(X, i, :)
      alpha = @get_triple A i
      logf = @get_triple F i
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
      if v.nvar > 1
        BLAS.axpy!(-d[idx]*r[idx] * dispfun(family, phi), x, ∇L)
        BLAS.syr!('U', d[idx] * dispfun(family, phi), x, ∇²L)
      else
        ∇L[1] = ∇L[1] - d[idx]*r[idx]*x[1]
        ∇²L[1] = ∇²L[1] + d[idx] * dispfun(family, phi) * x[1]*x[1]
      end
    end
  end
  accumulate_penalty_derivs!(penalty, ∇L, ∇²L, v, Γ, g.rho)
  if v.nvar > 1
    H = Symmetric(∇²L, :U)
    ldiv!(Δ, cholesky!(H), ∇L)
  else
    Δ[1] = ∇L[1] / ∇²L[1]
  end
  # linesearch!(g, v.beta_new, v.beta, Δ, backtrack)
  @. v.beta_new = v.beta - Δ
end

function mm_update_disp!(::UnivariateDistribution, g::VertexSurrogate, v::VertexGLM, triobs::Vector{TriangleObs}, workspace, caches, user_prior)
  error("MM updates not implemented for the $(v.family) case.")
end

function mm_update_disp!(::Normal, g::VertexSurrogate, v::VertexGLM, triobs::Vector{TriangleObs}, workspace, caches, use_prior)
  # Setup local variables to match notation
  μ = v.mu

  itr = zip(eachtriobs(triobs, v), v.part)

  # Fix β, update φ
  num = den = zero(eltype(v.beta))
  for ((triidx, T), idxrange) in itr
    y, A, t = T.y, T.A, get_triangle_vertices(T, v.index)
    F = caches.logf[triidx]
    for (i, idx) in zip(eachobsindex(T), idxrange)
      alpha = @get_triple A i
      logf = @get_triple F i
      zweight = stable_convex_weight(t, alpha, logf)
      num += zweight * deviance(v.family, y[i], μ[idx])
      den += zweight
      # v.index == 1 && @show zweight, deviance(v.family, y[i], μ[idx])
    end
  end

  if use_prior
    phi0 = v.extra_params[:global_dispersion]
    nu = v.extra_params[:nu]
    if !isempty(v.triangles)
      phi = (den + nu) / (num + nu / phi0)
    else
      phi = phi0
    end
  else
    nu = v.extra_params[:nu]
    A = iszero(den) ? zero(den) : num / den
    B = zero(den)
    C = nu
    wsum = sum(v.weights)
    for (idx, k) in enumerate(v.neighbors)
      u = g.model.vertex[k]
      B_k = u.extra_params[:local_dispersion]
      B += v.weights[idx]/wsum * inv(B_k)
    end
    invphi = den / (den + C) * A + C / (den + C) * B
    phi = inv(invphi)
  end

  # g_curr = g(phi)
  # @assert g_curr < g_prev || abs(g_curr - g_prev) < sqrt(eps()) * (1 + abs(g_prev)) "\n\tVertex: $(v.index)\n\tNobs: $(v.nobs)\n\tPrevious: $(g_prev)\n\t Current: $(g_curr)\n\tDen/Num: $(den) / $(num)\n\tφ₀: $(phi_old)\n\tφ: $(phi)\n\tβ: $(v.beta)"

  v.extra_params[:dispersion] = isinf(phi) ? one(phi) : phi
end

function update_pooling!(model, use_prior)
  if use_prior
    J, phi0 = length(model.vertex), 0.0
    for v in eachvertex(model)
      nu = v.extra_params[:nu]
      phi0 += 1/J * nu/(nu+2) * v.extra_params[:dispersion]
    end
    for v in eachvertex(model)
      v.extra_params[:global_dispersion] = phi0
    end
  else
    for v in eachvertex(model)
      wavg = zero(Float64)
      wsum = sum(v.weights)
      for (idx, k) in enumerate(v.neighbors)
        u = model.vertex[k]
        phi_k = u.extra_params[:dispersion]
        wavg += v.weights[idx]/wsum * phi_k
      end
      v.extra_params[:local_dispersion] = wavg
    end
  end
end

function fitmodel(::Type{VertexGLM}, yfull, Xfull, Sfull, tri;
    family::D = Normal(),
    link::L = IdentityLink(),
    maxiter::Int = 100,
    backtrack::Int = 5,
    tol::Real = 1e-3,
    rho::Real = 1.0,
    nchunks::Int = Threads.nthreads(),
    use_prior::Bool = false,
    kwargs...
    # intercept = all(isequal(1), view(Xfull, :, 1)),
  ) where {D <: UnivariateDistribution, L <: Link}
  # Initialize
  VT = infer_vertex_type(VertexGLM, family, link; kwargs...)
  model = f = create_model(VT, yfull, Xfull, Sfull, tri; nchunks, family, link, kwargs...)
  update_phi = needs_dispersion(family)
  initialize_coefficients!(model.triobs, model.vertex)
  update_caches!(model; nchunks)
  update_phi && update_pooling!(model, use_prior)
  nlogl = f(rho; nchunks, use_prior)
  nlogl_prev = zero(nlogl)
  iter = 0
  opt = (; rho = rho,  backtrack = backtrack)
  while iter < maxiter && abs(nlogl - nlogl_prev) > (1 + abs(nlogl_prev)) * tol
    iter += 1

    # Visit each vertex once to update local regression coefficients
    update_coefficients!(model, opt, nchunks)
    update_phi && update_caches!(model; nchunks)

    # Visit each vertex once to update auxiliary parameters
    update_phi && update_dispersion!(model, opt, nchunks, use_prior)

    # Synchronize algorithm state and evaluate current model
    update_caches!(model; nchunks)
    update_phi && update_pooling!(model, use_prior)
    nlogl_prev = nlogl
    nlogl = f(rho; nchunks, use_prior)
    @show iter, nlogl, nlogl_prev - nlogl
    @assert nlogl < nlogl_prev || abs(nlogl - nlogl_prev) < sqrt(eps()) * (1 + abs(nlogl_prev))
  end

  # Apply all updates
  if update_phi
    for v in eachvertex(model)
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

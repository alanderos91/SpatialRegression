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
  dispersion::Float64         # may be transformed while fitting
end

# convenience accesors to hide implementation details
get_family(v::VertexGLM) = v.family
get_link(v::VertexGLM) = v.link
needs_dispersion(v::VertexGLM) = needs_dispersion(get_family(v))

get_family(model::SpatialVertexModel{V}) where V <: VertexGLM = get_family(first(model.vertex))
get_link(model::SpatialVertexModel{V}) where V <: VertexGLM = get_link(first(model.vertex))
needs_dispersion(model::SpatialVertexModel{V}) where V <: VertexGLM = needs_dispersion(first(model.vertex))

infer_vertex_type(::Type{VertexGLM}, ::D, ::L) where {D <: UnivariateDistribution, L <: Link} = VertexGLM{D,L}

function transform_parameters!(model::SpatialVertexModel{V}) where V <: VertexGLM
  if needs_dispersion(model)
    for v in eachvertex(model)
      phi = v.dispersion
      model.vertex[v.index] = Accessors.@set v.dispersion = 1 / phi
    end
  end
end

function create_vertex(::Type{VertexGLM{D,L}}, index, part, triangles, neighbors, weights, nobs, nvar;
  family::D = Normal(),
  link::L   = IdentityLink(),
  ) where {D <: UnivariateDistribution, L <: Link}
  #
  d = zeros(nobs)
  beta = zeros(nvar)
  beta_new = similar(beta)
  eta = zeros(nobs)
  mu = zeros(nobs)
  workres = zeros(nobs)
  dispersion = 1.0
  return VertexGLM(
    index, part, triangles, neighbors, weights,
    nobs, nvar, family, link, d, beta, beta_new,
    eta, mu, workres, dispersion
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
    # local average dispersion parameters in MM algorithm, avgϕ
    avgphi = init_cache_avgphi(vertex),
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
        needs_dispersion(v.family) && update_cache_avgphi!(model.caches.avgphi, v, model.vertex)
      end
    end
  end nchunks=nchunks
end

function update_caches_serial!(model::SpatialVertexModel{<:VertexGLM})
  for v in eachvertex(model)
    update_cache_logf!(model.caches.logf, v, model.triobs)
    update_cache_gamma!(model.caches.gamma, v, model.vertex)
    needs_dispersion(v.family) && update_cache_avgphi!(model.caches.avgphi, v, model.vertex)
  end
end
#
# FAMILY-SPECIFIC PARAMETER PENALTIES
#
eval_dispersion_penalty(::UnivariateDistribution, vertex) = zero(Float64)

function eval_dispersion_penalty(::Normal, vertex)
  penalty = zero(Float64)
  for v in vertex
    phi_j = v.dispersion
    penalty += -log(phi_j) + 1//2*abs2(phi_j)
    for (i, k) in enumerate(v.neighbors)
      u = vertex[k]
      phi_k = u.dispersion
      penalty += 1//4 * v.weights[i] * abs2(phi_j - phi_k)
    end
  end
  return 1//2*penalty
end
#
# SURROGATE
#
function eval_loss_surrogate(beta::Vector, v::VertexGLM, triobs::Vector{TriangleObs}, caches, xi)
  cache = caches.logf
  logl = zero(Float64)
  phi = v.dispersion
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
      logl += tmp
    end
  end
  return -logl
end

function eval_loss_surrogate(phi::Real, v::VertexGLM, triobs::Vector{TriangleObs}, caches, xi)
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
      mu = meanfun(link, v.mu[idx])
      logfⱼ = GLMUtilities.logpdf(T.y[i], mu, phi, eta, family, link)

      # Evaluate terms dependent on the anchor point φₙ
      zweight = stable_convex_weight(t, alpha, logf)
      tmp = zweight < eps() ? zero(zweight) : zweight * logfⱼ + zweight * (log(alpha[t]) + log(inv(zweight)))
      logl += tmp
    end
  end
  return -logl
end

function eval_dispersion_surrogate(::Normal, phi_j, weights, avgphi)
  penalty = -log(phi_j) + 1//2*abs2(phi_j)
  @inbounds for (w_jk, avgphi_jk) in zip(weights, avgphi)
    penalty += w_jk * abs2(phi_j - avgphi_jk)
  end
  return 1//2*penalty
end
#
# UPDATES
#
function initialize_coefficients!(v::VertexGLM, triobs)
  X = similar(v.beta, length(v.d), length(v.beta))
  y = similar(v.beta, length(v.d))
  for (k, idxrange) in zip(v.triangles, v.part)
    T = triobs[k]
    X[idxrange, :] .= T.X
    y[idxrange] .= T.y
  end
  fitted = glm(X, y, v.family, v.link)
  v.beta .= coef(fitted)
  return v
end

function mm_update_coef!(penalty::AbstractPenalty, g::CoefficientSurrogate, v::VertexGLM, triobs::Vector{TriangleObs}, workspace, caches)
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
  itr = zip(eachtriobs(triobs, v), v.part)
  phi = v.dispersion

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

      # Evaluate gradient + Hessian
      if v.nvar > 1
        BLAS.axpy!(-d[idx]*r[idx] * dispfun(family, phi), x, ∇L)
        BLAS.syr!('U', d[idx] * dispfun(family, phi), x, ∇²L)
      else
        ∇L[1] = ∇L[1] - d[idx]*r[idx] * dispfun(family, phi) * x[1]
        ∇²L[1] = ∇²L[1] + d[idx] * dispfun(family, phi) * x[1]*x[1]
      end
    end
  end
  accumulate_penalty_derivs!(penalty, ∇L, ∇²L, v, Γ, g.opt.rho)
  if v.nvar > 1
    H = Symmetric(∇²L, :U)
    ldiv!(Δ, cholesky!(H), ∇L)
  else
    Δ[1] = ∇L[1] / ∇²L[1]
  end
  linesearch!(g, v.beta_new, v.beta, Δ, g.opt.backtrack)
  return v
end

function mm_update_disp!(::UnivariateDistribution, g::DispersionSurrogate, v::VertexGLM, triobs::Vector{TriangleObs}, workspace, caches)
  error("MM updates not implemented for the $(v.family) case.")
end

function mm_update_disp!(::Normal, g::DispersionSurrogate, v::VertexGLM, triobs::Vector{TriangleObs}, workspace, caches)
  # Setup local variables to match notation
  μ = v.mu
  nu = g.opt.nu
  avgphi = caches.avgphi[v.index]

  itr = zip(eachtriobs(triobs, v), v.part)

  # Fix β, update φ
  rss = n_j = zero(eltype(v.beta))
  for ((triidx, T), idxrange) in itr
    y, A, t = T.y, T.A, get_triangle_vertices(T, v.index)
    F = caches.logf[triidx]
    for (i, idx) in zip(eachobsindex(T), idxrange)
      alpha = @get_triple A i
      logf = @get_triple F i
      zweight = stable_convex_weight(t, alpha, logf)
      rss += zweight * deviance(v.family, y[i], μ[idx])
      n_j += zweight
    end
  end

  a = 1//2*nu * (1 + 2*sum(v.weights))
  b = nu * dot(avgphi, v.weights) - 1//2*rss
  c = 1//2 * (n_j + nu)
  phi1 = (b + sqrt(b*b + 4*a*c)) / (2*a)
  phi2 = (b - sqrt(b*b + 4*a*c)) / (2*a)
  if phi2 > 0
    phi = g(phi1) < g(phi2) ? phi1 : phi2
  else
    phi = phi1
  end
  v = Accessors.@set v.dispersion = phi
  return v
end

function fitmodel(::Type{VertexGLM}, yfull, Xfull, Sfull, tri;
    family::D = Normal(),
    link::L = IdentityLink(),
    maxiter::Int = 100,
    backtrack::Int = 5,
    tol::Real = 1e-3,
    rho::Real = 1.0,
    nu::Real = 1.0,
    nchunks::Int = Threads.nthreads(),
    verbose::Bool = false,
    kwargs...
    # intercept = all(isequal(1), view(Xfull, :, 1)),
  ) where {D <: UnivariateDistribution, L <: Link}
  # Initialize
  VT = infer_vertex_type(VertexGLM, family, link)
  model = create_model(VT, yfull, Xfull, Sfull, tri; nchunks, family, link, kwargs...)
  initialize!(model)

  opt = (; rho, nu, xi = zero(rho), backtrack, verbose, nchunks)
  _fitmodel_loop_(model, maxiter, tol, opt)
end

function _fitmodel_loop_(model::SpatialVertexModel{V}, maxiter, tol, opt) where V <: VertexGLM
  # unpacking + convenient definitions
  rho, nu, verbose, nchunks = opt.rho, opt.nu, opt.verbose, opt.nchunks
  family = get_family(model)
  f = model

  update_phi = needs_dispersion(family)
  update_caches!(model; nchunks)

  # Main loop
  nlogl = f(rho, nu; nchunks)
  nlogl_prev = zero(nlogl)
  iter = 0

  while iter < maxiter && abs(nlogl - nlogl_prev) > (1 + abs(nlogl_prev)) * tol
    iter += 1

    # Visit each vertex once to update local regression coefficients
    update_coefficients!(model, opt, nchunks)

    # Visit each vertex once to update auxiliary parameters
    update_phi && update_caches!(model; nchunks)
    update_phi && update_dispersion!(model, opt, nchunks)

    # Synchronize algorithm state and evaluate current model
    update_caches!(model; nchunks)

    # Check convergence
    nlogl_prev = nlogl
    nlogl = f(rho, nu; nchunks)
    verbose && @show iter, nlogl, nlogl_prev - nlogl
    @assert is_approx_decrease(nlogl, nlogl_prev, sqrt(eps()))
  end

  # Transform parameters as needed
  transform_parameters!(model)

  return iter, model, nlogl
end

function assemble_mixture(x, s, m, tri, id2vertex, points)
    j, k, l = find_triangle(tri, s; concavity_protection = true) |> sort
    p, q, r = points[j], points[k], points[l]
    j, k, l = id2vertex[j], id2vertex[k], id2vertex[l]
    vertex = (m.vertex[j], m.vertex[k], m.vertex[l])
    alpha = barycentric(s, [p[1] q[1] r[1]; p[2] q[2] r[2]])
    return MixtureModel(
      [create_component(v.family, v.link, dot(x, v.beta), v.dispersion) for v in vertex],
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

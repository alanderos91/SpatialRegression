#
# HELPER FUNCTIONS
#
prox_pinball(r, tau, mu) = begin
  if r >= mu*tau
    r - mu*tau
  elseif -mu*(1-tau) < r < mu*tau
    zero(r)
  else # r <= -mu*(1-tau)
    r + mu*(1-tau)
  end
end
#
# FAMILY-SPECIFIC PARAMETER PENALTIES
#
function eval_dispersion_penalty(::ALD, vertex)
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
  return penalty
end
#
# SURROGATE
#
function eval_loss_surrogate(beta::Vector, v::VertexGLM{<:ALD}, triobs::Vector{TriangleObs}, caches, xi)
  cache = caches.logf
  logl = zero(Float64)
  phi = v.dispersion
  j, family, link = v.index, v.family, v.link
  tau = family.τ
  for (triidx, T) in eachtriobs(triobs, v)
    t = get_triangle_vertices(T, j)
    F = cache[triidx]
    for i in eachobsindex(T)
      x = view(T.X, i, :)
      alpha = @get_triple T.A i
      logf = @get_triple F i

      # Evaluate log-likelihood term at β
      mu = v.nvar > 1 ? dot(x, beta) : x[1]*beta[1]
      r = T.y[i] - mu
      u = prox_pinball(r, tau, xi)
      loss = -phi*(pinball(u, tau) + 1//2*inv(xi)*abs2(r - u))

      # Evaluate terms dependent on the anchor point βₙ
      zweight = stable_convex_weight(t, alpha, logf)
      tmp = zweight < eps() ? zero(zweight) : zweight * loss + zweight * (log(alpha[t]) + log(inv(zweight)))
      logl += tmp
    end
  end
  return -logl
end

function eval_dispersion_surrogate(::ALD, phi_j, weights, avgphi)
  penalty = -log(phi_j) + 1//2*abs2(phi_j)
  @inbounds for (w_jk, avgphi_jk) in zip(weights, avgphi)
    penalty += w_jk * abs2(phi_j - avgphi_jk)
  end
  return penalty
end
#
# UPDATES
#
function mm_update_coef!(penalty::AbstractPenalty, g::CoefficientSurrogate, v::VertexGLM{<:ALD}, triobs::Vector{TriangleObs}, workspace, caches)
  # Setup local variables to match notation
  Γ = caches.gamma[v.index]
  d = v.d
  μ = v.mu
  τ = v.family.τ
  rho = g.opt.rho
  xi = g.opt.xi
  LHS, RHS, Δ = workspace

  fill!(LHS, 0); fill!(RHS, 0)
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
      r = (y[i] - μ[idx])
      dweight = zweight * phi * xi
      z = prox_pinball(r, τ, dweight)
      u = r - z
      d[idx] = inv(dweight)
      if iszero(zweight)
        d[idx] = one(zweight)
        u = zero(zweight)
      end

      # Evaluate gradient + Hessian
      if v.nvar > 1
        BLAS.axpy!(d[idx] * u, x, RHS)
        BLAS.syr!('U', d[idx], x, LHS)
      else
        RHS[1] = RHS[1] - d[idx] * u * x[1]
        LHS[1] = LHS[1] + d[idx] * x[1] * x[1]
      end
    end
  end
  accumulate_penalty_terms!(penalty, RHS, LHS, v, Γ, rho)
  if v.nvar > 1
    H = Symmetric(LHS, :U)
    ldiv!(Δ, cholesky!(H), RHS)
  else
    Δ[1] = RHS[1] / LHS[1]
  end
  linesearch!(g, v.beta_new, v.beta, Δ, g.opt.backtrack)
  return v
end

function mm_update_disp!(::ALD, g::DispersionSurrogate, v::VertexGLM, triobs::Vector{TriangleObs}, workspace, caches)
  # Setup local variables to match notation
  mu = v.mu
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
      rss += zweight * deviance(v.family, y[i], mu[idx])
      n_j += zweight
    end
  end

  a = nu * (1 + 2*sum(v.weights))
  b = 2 * nu * dot(avgphi, v.weights) - rss
  c = n_j + nu
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

function fitqreg(::Type{VertexGLM}, yfull, Xfull, Sfull, tri;
    maxiter::Int = 100,
    backtrack::Int = 5,
    tol::Real = 1e-3,
    rho::Real = 1.0,
    nu::Real = 1.0,
    quantile::Real = 0.5,
    smooth_max::Real = 10.0,
    smooth_min::Real = 0.1,
    smooth_itr::Real = 20,
    nchunks::Int = Threads.nthreads(),
    verbose::Bool = false,
    kwargs...
    # intercept = all(isequal(1), view(Xfull, :, 1)),
  )
  # Initialize
  tau = quantile
  family, link = ALD(tau), IdentityLink() # need to propagate tau!
  VT = infer_vertex_type(VertexGLM, family, link)
  model = create_model(VT, yfull, Xfull, Sfull, tri; nchunks, family, link, kwargs...)

  opt = (;
    rho, nu, xi = zero(rho), backtrack, verbose, nchunks,
    smooth_max, smooth_min, smooth_itr # unique to QREG
  )
  _fitqreg_loop_(model, maxiter, tol, opt)
end

function _fitqreg_loop_(model::SpatialVertexModel{V}, maxiter, tol, opt) where V <: VertexGLM
  # unpacking + convenient definitions
  rho, nu, verbose, nchunks = opt.rho, opt.nu, opt.verbose, opt.nchunks
  smooth_max, smooth_min, smooth_itr = opt.smooth_max, opt.smooth_min, opt.smooth_itr
  f = model

  update_caches!(model; nchunks)

  # Main loop
  nlogl = f(rho, nu; nchunks)
  nlogl_prev = zero(nlogl)
  iter = 0
  xi = smooth_max
  opt = (; opt..., xi = typeof(opt.xi)(xi))

  while iter < maxiter && abs(nlogl - nlogl_prev) > (1 + abs(nlogl_prev)) * tol
    iter += 1

    # Visit each vertex once to update local regression coefficients
    update_coefficients!(model, opt, nchunks)
    
    # Visit each vertex once to update auxiliary parameters
    update_caches!(model; nchunks)
    update_dispersion!(model, opt, nchunks)

    # Synchronize algorithm state and evaluate current model
    xi = iter > smooth_itr ? smooth_min : smooth_max * (smooth_min / smooth_max) ^ (iter / smooth_itr)
    opt = (; opt..., xi = typeof(opt.xi)(xi))
    update_caches!(model; nchunks)

    # Check convergence
    nlogl_prev = nlogl
    nlogl = f(rho, nu; nchunks)
    verbose && @show iter, nlogl, nlogl_prev - nlogl
    @assert is_approx_decrease(nlogl, nlogl_prev, sqrt(eps()))
  end

  # Apply all updates
  transform_parameters!(model)

  return iter, model, nlogl
end

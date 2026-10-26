#
# HELPER FUNCTIONS
#
function prox_pinball(r, tau, mu)
  if r >= mu*tau
    r - mu*tau
  elseif -mu*(1-tau) < r < mu*tau
    zero(r)
  else # r <= -mu*(1-tau)
    r + mu*(1-tau)
  end
end

function moreau_pinball(r, tau, mu)
  if r >= mu*tau
    tau*r - 1//2*mu*abs2(tau)
  elseif r <= -mu * (1-tau)
    -(1-tau)*r - 1//2*mu*abs2(1-tau)
  else
    1//2 * inv(mu) * abs2(r)
  end
end
#
# MOREAU REGULARIZATION
#
function update_cache_logf!(cache, v::VertexGLM{<:ALD}, triobs::Vector{TriangleObs})
  global ξmin
  j, β, η, μ = v.index, v.beta, v.eta, v.mu
  family, link = v.family, v.link
  φ = v.dispersion
  τ = family.τ
  for (k, idxrange) in zip(v.triangles, v.part)
    T = triobs[k]
    triidx, y, X = T.triangle, T.y, T.X
    t = get_triangle_vertices(T, j)
    logf = view(cache[triidx], t, :)
    mul!(view(η, idxrange), X, β) # η = Xβ
    @inbounds for (i, idx) in zip(eachindex(logf, y), idxrange)
      μ[idx] = meanfun(link, η[idx])
      # REPLACE PINBALL LOSS WITH MOREAU ENVELOPE
      logfᵢ = log(τ * (1 - τ)) + log(φ) - moreau_pinball(y[i] - μ[idx], τ, ξmin[])*φ
      logf[i] = logfᵢ
    end
  end
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
      mut = v.nvar > 1 ? dot(x, v.beta) : x[1]*v.beta[1]
      rt = T.y[i] - mut
      ut = prox_pinball(rt, tau, xi)
      logfⱼ = -phi*(pinball(ut, tau) + 1//2*inv(xi)*abs2(r - ut))
      logfⱼ += log(tau * (1 - tau)) + log(phi)

      # Evaluate terms dependent on the anchor point βₙ
      zweight = stable_convex_weight(t, alpha, logf)
      tmp = zweight < eps() ? zero(zweight) : zweight * logfⱼ + zweight * (log(alpha[t]) + log(inv(zweight)))
      logl += tmp
    end
  end
  return -logl
end
#
# UPDATES
#
function mm_update_coef!(penalty::AbstractPenalty, g::CoefficientSurrogate, v::VertexGLM{<:ALD}, triobs::Vector{TriangleObs}, workspace, caches)
  global ξ
  # Setup local variables to match notation
  family = v.family
  link = v.link
  Γ = caches.gamma[v.index]
  d = v.d
  r = v.workres
  η = v.eta
  μ = v.mu
  τ = v.family.τ
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
      # Evaluate prox of pinball function
      u = prox_pinball(y[i] - μ[idx], τ, ξ[])
      # Setup IRWLS structure
      dμdη = meanderiv(link, η[idx]) # = 1.0
      r[idx] = (y[i] - u - μ[idx]) / dμdη
      d[idx] = zweight * dμdη / ξ[]
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

function local_deviance(::ALD, v::VertexGLM, triobs::Vector{TriangleObs}, workspace, caches)
  # Setup local variables to match notation
  mu = v.mu

  itr = zip(eachtriobs(triobs, v), v.part)

  # Fix β, update φ
  n_j = rss = zero(eltype(v.beta))
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

  return n_j, rss
end

function fitqreg(::Type{VertexGLM}, yfull, Xfull, Sfull, tri;
    maxiter::Int = 100,
    backtrack::Int = 5,
    tol::Real = 1e-3,
    rho::Real = 1.0,
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
  initialize!(model)

  opt = (;
    rho, backtrack, verbose, nchunks,
    smooth_max, smooth_min, smooth_itr # unique to QREG
  )
  _fitqreg_loop_(model, maxiter, tol, opt, true)
end

function _fitqreg_loop_(model::SpatialVertexModel{V}, maxiter, tol, opt, needs_transform) where V <: VertexGLM
  # unpacking + convenient definitions
  global ξ, ξmin
  rho, verbose, nchunks = opt.rho, opt.verbose, opt.nchunks
  smooth_max, smooth_min, smooth_itr = opt.smooth_max, opt.smooth_min, opt.smooth_itr
  f = model

  update_caches!(model; nchunks)

  # Main loop
  nlogl = f(rho; nchunks)
  nlogl_prev = zero(nlogl)
  iter = 0
  ξ[] = smooth_max
  ξmin[] = smooth_min

  while iter < maxiter && abs(nlogl - nlogl_prev) > (1 + abs(nlogl_prev)) * tol
    iter += 1

    # Visit each vertex once to update local regression coefficients
    update_coefficients!(model, opt, nchunks)
    
    # Visit each vertex once to update auxiliary parameters
    update_caches!(model; nchunks)
    update_dispersion!(model, opt, nchunks)

    # Synchronize algorithm state and evaluate current model
    ξ[] = iter > smooth_itr ? smooth_min : smooth_max * (smooth_min / smooth_max) ^ (iter / smooth_itr)
    update_caches!(model; nchunks)

    # Check convergence
    nlogl_prev = nlogl
    nlogl = f(rho; nchunks)
    verbose && @show iter, nlogl, nlogl_prev - nlogl
    @assert is_approx_decrease(nlogl, nlogl_prev, sqrt(eps()))
  end

  # Report negative log-likelihood without penalty
  nlogl = eval_loss(model.triobs, model.vertex, model.caches; nchunks)

  # Transform parameters as needed
  needs_transform && transform_parameters!(model)

  # Reflect updated state in model
  model = Accessors.@set model.state =
    FittedState(true, needs_transform, rho)

  return iter, model, nlogl
end
#
# CROSS VALIDATION
#
function cv(::Type{VertexGLM{AsymmetricLaplace}}, yfull, Xfull, Sfull, tri;
    train_prop::Real = 0.8,
    nfolds::Int = 5,
    grid_rho::AbstractVector=[1e-6, 1e-3, 1.0],
    maxiter::Int = 100,
    backtrack::Int = 5,
    tol::Real = 1e-3,
    nchunks::Int = Threads.nthreads(),
    verbose::Bool = false,
    penalty::P = L2Squared(),
    quantile::Real = 0.5,
    smooth_max::Real = 10.0,
    smooth_min::Real = 0.1,
    smooth_itr::Real = 20,
    kwargs...
    # intercept = all(isequal(1), view(Xfull, :, 1)),
  ) where {P <: AbstractPenalty}
  # Setup QREG-specific information
  global ξ, ξmin
  tau = quantile
  family, link = ALD(tau), IdentityLink() # propagate tau

  # Extract vertex data so we can quickly build up TriangleObs sets
  helper = create_triobs_helper(tri)
  tri_vertex_coords = DelaunayTriangulation.get_points(tri)

  # Shuffle data, then split data into CV subset and test subset.
  n = length(yfull)
  idx_shuffle = shuffleobs(1:n)
  aa, bb = splitobs(n; at = train_prop)
  idx_cv, idx_tes = getobs(idx_shuffle, aa), getobs(idx_shuffle, bb)

  # Buffers to help us materialize data.
  # We do our best to minimize the memory footprint.
  tmp_tra, tmp_val = kfolds(length(idx_cv), nfolds)
  max_size_tra = maximum(length, tmp_tra)
  max_size_val = maximum(length, tmp_val)
  max_size_tes = length(idx_tes)
  n_buf = max(max_size_tra, max_size_val, max_size_tes)
  y_tmp = similar(yfull, n_buf)
  y_buf = similar(yfull, n_buf)
  X_buf = similar(Xfull, (n_buf, size(Xfull, 2)))
  S_buf = similar(Sfull, (size(Sfull, 1), n_buf))
  dst = (y_buf, X_buf, S_buf)
  src = (yfull, Xfull, Sfull)

  # Setup before any CV or model fitting takes place
  itr = grid_rho
  nvals = length(itr)
  nvar = size(Xfull, 2)
  VT = infer_vertex_type(VertexGLM, family, link)
  PT = create_component(family, link, 0.0, 1.0) |> typeof
  buffers = (Vector{PT}(undef, 3), zeros(3))

  score_tra = zeros(nvals, nfolds)
  score_val = zeros(nvals, nfolds)
  score_tes = zeros(nvals, nfolds)

  niter_tra = zeros(Int, nvals, nfolds)
  nlogl_tra = zeros(nvals, nfolds)

  times_fit = zeros(nvals, nfolds)
  times_tra = zeros(nvals, nfolds)
  times_val = zeros(nvals, nfolds)
  times_tes = zeros(nvals, nfolds)

  result1 = (score_tra, score_val, score_tes)
  result2 = (times_tra, times_val, times_tes)

  for (k, (idx_tra, idx_val)) in enumerate(kfolds(idx_cv; k=nfolds))
    # iterator specifying each of the three subsets
    sub = (idx_tra, idx_val, idx_tes)

    # retrieve training data
    y_tra, X_tra, S_tra = materialize_data!(dst, src, idx_tra)

    # Initialize model with training data
    triobs = _create_triobs_set_with_helper_(helper, y_tra, X_tra, S_tra, tri, nchunks)
    vertex = create_vertex_set(VT, tri, triobs, nvar; family, link, kwargs...)
    caches = build_caches(triobs, vertex, penalty, nvar, nchunks)
    model = SpatialVertexModel(triobs, vertex, penalty, caches)

    # Cross validate on penalty strength, rho
    for (i, rho) in enumerate(itr)
      verbose && println("Fold $(k), rho = $(round(rho, sigdigits=4))")
      # fit model with target hyperparameter + other fixed parameters
      opt = (; rho, backtrack, verbose=false, nchunks,
        smooth_max, smooth_min, smooth_itr # unique to QREG
      )
      fitstats = @timed _fitqreg_loop_(model, maxiter, tol, opt, false)

      # record model fit information
      niter_tra[i,k] = fitstats.value[1]
      nlogl_tra[i,k] = fitstats.value[3]
      times_fit[i,k] = fitstats.time

      # score model on different subsets
      for (idx, arr1, arr2) in zip(sub, result1, result2)
        y, X, S = materialize_data!(dst, src, idx)
        predstats = @timed eval_negloglik(y, X, S, model, tri; nchunks)
        arr1[i,k] = predstats.value
        arr2[i,k] = predstats.time
      end
    end
  end

  # assemble results
  cv_scores = mean(score_val, dims=2) |> vec
  idx = argmin(cv_scores)

  best_rho = grid_rho[idx]
  results = (;
    # cross validation settings
    idx_cv, idx_tes, nfolds, train_prop,
    # cross validation results
    score_tra, score_val, score_tes,
    times_fit, niter_tra, nlogl_tra,
    times_tra, times_val, times_tes,
    cv_scores,
    # optimal hyperparameter(s)
    grid_rho,
    best_score = cv_scores,
    best_rho,
    best_rho_index = idx,
  )

  return results
end
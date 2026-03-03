function eval_surrogate(tobs, vmod, j, penalty_type::AbstractPenalty, caches)
  v = vmod[j]
  eval_surrogate(v.beta, v.index, v.gamma, v.weights, v.rho, vmod, tobs, penalty_type, caches)
end

function eval_surrogate(beta, j, gamma, weights, rho, vmod, tobs, penalty_type::AbstractPenalty, caches)
  logl, vⱼ = zero(Float64), vmod[j]

  # Log-Likelihood
  for triidx in vⱼ.triangles
    T = tobs[triidx]
    A, y, X = T.A, T.y, T.X
    t, v = get_triangle_vertices(T, j, vmod)
    cache = caches.logf[triidx]

    for i in eachindex(T.y)
      x = view(X, i, :)
      a = view(A, :, i)
      logf = view(cache, :, i)

      # Evaluate log-likelihood term at β
      eta = dot(x, beta)
      mu = meanfun(vⱼ.link, eta)
      logfⱼ = GLMUtilities.log_likelihood(y[i], mu, eta, vⱼ.family, vⱼ.link)

      # Evaluate terms dependent on the anchor point βₙ
      zweight = stable_eval_mm_weight(t, a, logf)

      tmp = zweight * logfⱼ + zweight * (log(a[t]) + log(inv(zweight)))
      logl += tmp
    end
  end

  # Penalty
  penalty = eval_penalty_surrogate(penalty_type, rho, beta, weights, gamma, vⱼ.beta)

  return logl + penalty
end

function update_empty_case!(::L2Squared, v, weights, gamma)
  mul!(v.beta_new, gamma, weights)
  sumw = sum(weights)
  @. v.beta_new = v.beta_new / sumw
  return nothing
end

function update_empty_case!(p::L1Approx, v, weights, gamma)
  epsilon = p.epsilon
  beta = v.beta
  for k in eachindex(beta)
    num, den = zero(Float64), zero(Float64)
    for (idx, γ) in enumerate(eachcol(gamma))
      eta = beta[k] - γ[k]
      q = l1apx(eta, epsilon/2)
      num += γ[k] * weights[idx] / q
      den += weights[idx] / q
    end
    v.beta_new[k] = num / den
  end
  return nothing
end

function eval_gamma!(vmod)
  for v in vmod
    β = v.beta
    Γ = v.gamma
    for (idx, k) in enumerate(v.neighbors)
      if v.index == k continue end
      βₖ = vmod[k].beta
      @views begin
        @. Γ[:, idx] = 1//2 * (β + βₖ)
      end
    end
  end
end

function mm_update!(penalty, vⱼ, vmod, tobs, workspace, caches)
  # Setup local variables to match notation
  β = vⱼ.beta
  Γ = vⱼ.gamma
  d = vⱼ.d
  r = vⱼ.workres
  η = vⱼ.eta
  μ = vⱼ.mu
  w = vⱼ.weights
  ∇L, ∇²L, search_direction = workspace
  
  idx = 1; fill!(∇L, 0); fill!(∇²L, 0)
  if isempty(vⱼ.triangles)
    # Case: Incident observation sets are empty
    update_empty_case!(penalty, vⱼ, w, Γ)
  else
    # Case: Incident observation sets are not empty
    for triidx in vⱼ.triangles
      T = tobs[triidx]
      A, y, X = T.A, T.y, T.X
      t, v = get_triangle_vertices(T, vⱼ.index, vmod)
      cache = caches.logf[triidx]

      # Compute weights and working residuals
      for i in eachindex(y)
        x = view(X, i, :)
        a = view(A, :, i)
        logf = view(cache, :, i)

        zweight = stable_eval_mm_weight(t, a, logf)
        # η[idx] = dot(x, β)
        μ[idx] = meanfun(vⱼ.link, η[idx])
        dμdη   = meanderiv(vⱼ.link, η[idx]) 
        r[idx] = (y[i] - μ[idx]) / dμdη
        d[idx] = zweight * stable_irls_weight(vⱼ.family, vⱼ.link, η[idx], μ[idx], dμdη)
        idx += 1
      end

      # Evaluate gradient + Hessian
      idxrange = (idx-length(y)):(idx-1)
      dd = view(d, idxrange)
      rr = view(r, idxrange)
      @inbounds for k in axes(X, 1)
        xx = view(X, k, :)
        BLAS.axpy!(dd[k] * rr[k], xx, ∇L)
        BLAS.syr!('U', dd[k], xx, ∇²L)
      end
    end

    accumulate_penalty_derivs!(penalty, ∇L, ∇²L, vⱼ.rho, β, w, Γ)
    H = Symmetric(∇²L, :U)
    ldiv!(search_direction, cholesky!(H), ∇L)
  end
end

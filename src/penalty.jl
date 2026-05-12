struct L2Squared <: AbstractPenalty end
struct L1Approx <: AbstractPenalty epsilon::Float64 end

function eval_penalty(::L2Squared, vmod)
  penalty = zero(Float64)
  for v in vmod
    for (idx, k) in enumerate(v.neighbors)
      u = vmod[k]
      diff = zero(eltype(v.beta))
      @inbounds @simd for j in eachindex(v.beta)
        diff += abs2(v.beta[j] - u.beta[j])
      end
      penalty += 1//4 * v.weights[idx] * diff
    end
  end
  return penalty
end

function l1apx(x, ϵ)
  return sqrt(x*x + ϵ*ϵ)
end

function eval_penalty(p::L1Approx, vmod)
  epsilon = p.epsilon
  penalty = zero(Float64)
  for v in vmod
    for (idx, k) in enumerate(v.neighbors)
      u = vmod[k]
      diff = zero(eltype(v.beta))
      @inbounds @simd for j in eachindex(v.beta)
        diff += l1apx(v.beta[j] - u.beta[j], epsilon)
      end
      penalty += 1//2 * v.weights[idx] * diff
    end
  end
  return penalty
end

function eval_penalty_surrogate(penalty::AbstractPenalty, ::Real, weights, gamma, betan)
  eval_penalty_surrogate(penalty, betan, weights, gamma, betan)
end

function eval_penalty_surrogate(::L2Squared, beta::Vector, weights, gamma, betan)
  penalty = zero(Float64)
  for (idx, γ) in enumerate(eachcol(gamma))
    diff = zero(eltype(beta))
    @inbounds @simd for k in eachindex(beta)
      diff += abs2(beta[k] - γ[k])
    end
    penalty += weights[idx] * diff
  end
  return penalty
end

function eval_penalty_surrogate(p::L1Approx, beta::Vector, weights, gamma, betan)
  epsilon = p.epsilon
  penalty = zero(Float64)
  for (idx, γ) in enumerate(eachcol(gamma))
    diff = zero(eltype(beta))
    @inbounds @simd for k in eachindex(beta)
      eta = betan[k] - γ[k]
      q = l1apx(eta, epsilon/2)
      diff += 1 / (2*q) * ((beta[k] - γ[k])^2 - eta^2) + q
    end
    penalty += weights[idx] * diff
  end
  return penalty
end

# TODO: Check SIGNS here
function accumulate_penalty_derivs!(::L2Squared, grad, hess, v, gamma, rho)
  beta, weights = v.beta, v.weights
  wsum = sum(weights)
  wrho = 2 * rho
  nvar = length(beta)
  if nvar > 1
    BLAS.gemv!('N', -wrho, gamma, weights, true, grad)
    BLAS.axpy!(wrho*wsum, beta, grad)
    @inbounds @simd for k in axes(hess, 2)
      hess[k,k] += wrho*wsum
    end
  else
    grad[1] = grad[1] - wrho*dot(gamma, weights) + wrho*wsum*beta[1]
    hess[1] = hess[1] + wrho*wsum
  end
end

# TODO: Check SIGNS here
function accumulate_penalty_derivs!(p::L1Approx, grad, hess, v, gamma, rho)
  beta, weights = v.beta, v.weights
  epsilon = p.epsilon
  for k in eachindex(beta)
    c, d = zero(Float64), zero(Float64)
    for (idx, γ) in enumerate(eachcol(gamma))
      eta = beta[k] - γ[k]
      q = l1apx(eta, epsilon/2)
      c += weights[idx] / q
      d += eta * weights[idx] / q
    end
    grad[k] += rho*d
    hess[k,k] += rho*c
  end
end

function accumulate_penalty_terms!(::L2Squared, RHS, LHS, v, gamma, rho)
  beta, weights = v.beta, v.weights
  wsum = sum(weights)
  wrho = 2 * rho
  nvar = length(beta)
  if nvar > 1
    BLAS.gemv!('N', wrho, gamma, weights, true, RHS)
    @inbounds @simd for k in axes(LHS, 2)
      LHS[k,k] += wrho*wsum
    end
  else
    RHS[1] = RHS[1] + wrho*dot(gamma, weights)
    LHS[1] = LHS[1] + wrho*wsum
  end
end

function accumulate_penalty_terms!(p::L1Approx, RHS, LHS, v, gamma, rho)
  beta, weights = v.beta, v.weights
  epsilon = p.epsilon
  for k in eachindex(beta)
    c, d = zero(Float64), zero(Float64)
    for (idx, γ) in enumerate(eachcol(gamma))
      eta = beta[k] - γ[k]
      q = l1apx(eta, epsilon/2)
      c += weights[idx] / q
      d += γ[k] * weights[idx] / q
    end
    RHS[k] += rho*d
    LHS[k,k] += rho*c
  end
end

function update_empty_case!(::L2Squared, v, weights, caches)
  gamma = caches.gamma[v.index]
  mul!(v.beta_new, gamma, weights)
  sumw = sum(weights)
  @. v.beta_new = v.beta_new / sumw
  return nothing
end

function update_empty_case!(p::L1Approx, v, weights, caches)
  gamma = caches.gamma[v.index]
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

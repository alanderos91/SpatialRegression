abstract type AbstractPenalty end

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
      penalty -= v.rho/4 * v.weights[idx] * diff
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
      penalty -= v.rho/2 * v.weights[idx] * diff
    end
  end
  return penalty
end


function eval_penalty_surrogate(::L2Squared, rho, beta, weights, gamma, betan)
  penalty = zero(Float64)
  for (idx, γ) in enumerate(eachcol(gamma))
    diff = zero(eltype(beta))
    @inbounds @simd for k in eachindex(beta)
      diff += abs2(beta[k] - γ[k])
    end
    penalty -= rho * weights[idx] * diff
  end
  return penalty
end

function eval_penalty_surrogate(p::L1Approx, rho, beta, weights, gamma, betan)
  epsilon = p.epsilon
  penalty = zero(Float64)
  for (idx, γ) in enumerate(eachcol(gamma))
    diff = zero(eltype(beta))
    @inbounds @simd for k in eachindex(beta)
      eta = betan[k] - γ[k]
      q = l1apx(eta, epsilon/2)
      diff += 1 / (2*q) * ((beta[k] - γ[k])^2 - eta^2) + q
    end
    penalty -= rho * weights[idx] * diff
  end
  return penalty
end

function accumulate_penalty_derivs!(::L2Squared, grad, hess, rho, beta, weights, gamma)
  wsum = sum(weights)
  wrho = 2 * rho
  BLAS.gemm!('N', 'N', wrho, gamma, weights, true, grad)
  BLAS.axpy!(-wrho*wsum, beta, grad)
  # grad .= grad - wrho*(sum(w)*beta - gamma*weights)
  @inbounds @simd for k in axes(hess, 2)
    hess[k,k] += wrho*wsum
  end
  # hess .= hess + wrho*wsum*I
end

function accumulate_penalty_derivs!(p::L1Approx, grad, hess, rho, beta, weights, gamma)
  epsilon = p.epsilon
  for k in eachindex(beta)
    c, d = zero(Float64), zero(Float64)
    for (idx, γ) in enumerate(eachcol(gamma))
      eta = beta[k] - γ[k]
      q = l1apx(eta, epsilon/2)
      c += weights[idx] / q
      d += eta * weights[idx] / q
    end
    grad[k] -= rho*d
    hess[k,k] += rho*c
  end
end

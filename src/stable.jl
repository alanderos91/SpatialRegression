function stable_logsumexp(a, f)
  s, c = stable_explogsum(a, f)
  if isinf(c)
    s = argmax(f)
    return log(a[s]) + f[s] 
  else
    return log(a[s]) + f[s] + log(c)
  end
end

function stable_explogsum(a, f)
  s = argmin(f)
  c = zero(eltype(f))
  for k in eachindex(f)
    c += a[k] / a[s] * exp(f[k] - f[s])
  end
  return s, c
end

stable_glmvar(family::Distribution, μ, η) = GLM.glmvar(family, μ)

function stable_glmvar(family::Binomial, μ, η)
  if iszero(μ) || isone(μ)
    expabs = exp(-abs(η))
    return expabs / (1 + 2*expabs + expabs^2) 
  else
    return GLM.glmvar(family, μ)
  end
end

"""
    stable_eval_mm_weight(t, a, v, y, x)

Evaluate the MM weight in a log-likelihood surrogate, `a[t]*f[t] / (a[1]*f[1]+a[2]*f[2]+a[3]*f[3])`, using the exp-log-sum trick.
Here `a` and `v` are vectors of length 3 containing convex weights and `VertexModel` objects, respectively.
The index `t` indicates which vertex of `(v[1], v[2], v[3])` is the argument of the surrogate.

Returns the MM weight.
"""
function stable_eval_mm_weight(t::Integer, a::AbstractVector{T}, logf) where T <: Real
  s, c = stable_explogsum(a, logf)
  if isinf(c)
    zweight = one(c)        
  else
    zweight = a[t] / a[s] * exp(logf[t] - logf[s]) / c
  end
  return zweight
end

function stable_eval_mm_weight(t::Integer, a, v::NTuple{3,T}, y, x) where T <: VertexModel
  logf = eval_loglikelihoods(v, y, x)
  return stable_eval_mm_weight(t, a, logf)
end

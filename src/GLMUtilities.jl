module GLMUtilities
# Helper functions for working with GLMs.
# Source: Hua's GLM implementation + GLM.jl; favors LogExpFunctions.jl when possible.

using LogExpFunctions
using SpecialFunctions
using Distributions: 
  Distribution, UnivariateDistribution, MixtureModel,
  Normal, Bernoulli, Binomial, Poisson,
  components, component_type
using GLM:
  Link,
  IdentityLink, LogitLink, LogLink,
  canonicallink, glm, coef

export Distribution, UnivariateDistribution, MixtureModel,
  Normal, Bernoulli, Binomial, Poisson,
  components, component_type,
  Link,
  IdentityLink, LogitLink, LogLink,
  canonicallink
#
# LOG-LIKELIHOOD: Use mean parameter only
#
logpdf(y, mu, phi, ::Normal, n = one(y)) = -1//2 * (abs2(y - mu) * phi + log(2*π/phi))
logpdf(y, mu, phi, ::Bernoulli, n = one(y)) = isone(y) ? log(mu) : log(1-mu)
logpdf(y, mu, phi, ::Binomial, n = one(y)) = begin
  if isone(n)
    logpdf(y, mu, phi, Bernoulli(), n)
  else
    y * log(mu) + (n-y) * log(1-mu) + loggamma(n+1) - loggamma(y+1) - loggamma(n-y+1)
  end
end
logpdf(y, mu, phi, ::Poisson, n = one(y)) = y * log(mu) - mu - loggamma(y+1)
#
# LOG-LIKELIHOOD: Use linear predictor to avoid -Inf/NaN
#
logpdf(y, mu, phi, eta, family::Distribution, link::Link, n = one(y)) = logpdf(y, mu, phi, family, n)
logpdf(y, mu, phi, eta, ::Bernoulli, ::LogitLink, n = one(y)) = begin
  if abs(eta) ≥ 32
    isone(y) ? loglogistic(eta) : log1mlogistic(eta)
  else
    logpdf(y, mu, phi, Bernoulli(), n)
  end
end
logpdf(y, mu, phi, eta, ::Binomial, ::LogitLink, n = one(y)) = begin
  if isone(n)
    logpdf(y, mu, phi, eta, Bernoulli(), LogitLink(), n)
  else
    y * loglogistic(eta) + (n-y) * log1mlogistic(eta) + loggamma(n+1) - loggamma(y+1) - loggamma(n-y+1)
  end
end

export logpdf
#
# DEVIANCE
#
deviance(::Normal, y, mu) = abs2(y - mu)
deviance(::Bernoulli, y, mu) = -2 * (isone(y) ? log(mu) : log1p(-mu))
deviance(::Binomial, y, mu) = begin
  if y <= 1 # 0 <= y <= 1
    deviance(Bernoulli(), y, mu)
  else y > 1
    2 * (y * (log(y) - log(mu)) + (1-y) * (log1p(-y) - log1p(-mu)))
  end
end
deviance(::Poisson, y, mu) = 2 * (xlogy(y, y / mu) - (y - mu))

export deviance
#
# DISPERSION
#
needs_dispersion(::Distribution) = false
needs_dispersion(::Normal) = true

dispfun(d::Distribution, phi) = begin
  needs_dispersion(d) ? _dispfun(d, phi) : one(phi)
end
_dispfun(::Normal, phi) = phi

export needs_dispersion, dispfun
#
# (INVERSE) LINK FUNCTIONS
#
meanfun(::IdentityLink, eta) = eta
meanfun(::LogitLink, eta) = logistic(eta)
meanfun(::LogLink, eta) = exp(eta)

linpred(::IdentityLink, mu) = mu
linpred(::LogitLink, mu) = logit(mu)
linpred(::LogLink, mu) = log(mu)

export meanfun, linpred
#
# MEAN FUNCTION DERIVATIVE
#
meanderiv(::IdentityLink, eta) = one(eta)
meanderiv(::LogitLink, eta) = begin
  CUTOFF = abs(log10(eps(zero(eta))))
  eta = clamp(eta, -CUTOFF, CUTOFF)
  return exp(loglogistic(eta) + log1mlogistic(eta))
end
meanderiv(::LogLink, eta) = exp(eta)

export meanderiv
#
# VARIANCE FUNCTIONS
#
varfun(::Normal, mu) = one(mu)
varfun(::Union{Bernoulli,Binomial}, mu) = mu * (1-mu)
varfun(::Poisson) = mu

export varfun
#
# CANCELLATION TRICKS
#
# based on GLM.cancancel()
# https://github.com/JuliaStats/GLM.jl/blob/1c62ab087ccb6df8247d25883d4d997b00d115d9/src/glmfit.jl#L92-L105
cancancel(family, link) = false
cancancel(::Union{Bernoulli,Binomial}, ::LogitLink) = true
cancancel(::Normal, ::IdentityLink) = true
cancancel(::Poisson, ::LogLink) = true

export cancancel
#
# INITIALIZATION
#
mustart(::Normal, y) = y
mustart(::Bernoulli, y) = (y + oftype(y, 1/2)) / 2
mustart(::Binomial, y) = mustart(Bernoulli(), y)
mustart(::Poisson, y) = begin
  fy = float(y)
  fy + oftype(fy, 1/10)
end

export mustart
#
# MIXTURES
#
function create_component(::Normal, link, η, φ)
  σ = sqrt(φ)
  μ = meanfun(link, η)
  return Normal(μ, σ)
end

function create_component(::Binomial, link, η, φ)
  return Bernoulli(meanfun(link, η))
end

function create_component(::Poisson, link, η, φ)
  return Poisson(meanfun(link, η))
end

export create_component
#
# NUMERICAL STABILITY
#
function stable_irls_weight(family, link, η, μ, dμdη)
  return cancancel(family, link) ? dμdη : abs2(dμdη) / varfun(family, μ)
end

function weighted_sumexp(alpha, logf, m = argmax(logf))
  logf_m = logf[m]
  s = zero(eltype(logf))
  for (a_k, logf_k) in zip(alpha, logf)
    s += a_k * exp(logf_k - logf_m)
  end
  return s, logf_m
end

function weighted_logsumexp(alpha, logf, m = argmax(logf))
  s, logf_m = weighted_sumexp(alpha, logf, m)
  return log(s) + logf_m
end

function stable_convex_weight(k::Integer, alpha, logf)
  s, logf_m = weighted_sumexp(alpha, logf)
  return alpha[k]*exp(logf[k] - logf_m) / s
end

export stable_irls_weight, weighted_sumexp, weighted_logsumexp, stable_convex_weight

end # module

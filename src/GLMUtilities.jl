module GLMUtilities
# Helper functions for working with GLMs.
# Source: Hua's GLM implementation + GLM.jl; favors LogExpFunctions.jl when possible.

using LogExpFunctions
using SpecialFunctions
using Distributions: 
  Distribution, UnivariateDistribution,
  Normal, Bernoulli, Binomial, Poisson
using GLM:
  Link,
  IdentityLink, LogitLink, LogLink,
  canonicallink, glm, coef

export Distribution, UnivariateDistribution,
  Normal, Bernoulli, Binomial, Poisson,
  Link,
  IdentityLink, LogitLink, LogLink,
  canonicallink
#
# LOG-LIKELIHOOD: Use mean parameter only
#
log_likelihood(y, mu, ::Normal, n = one(y)) = -abs2(y - mu)
log_likelihood(y, mu, ::Bernoulli, n = one(y)) = isone(y) ? log(mu) : log(1-mu)
log_likelihood(y, mu, ::Binomial, n = one(y)) = begin
  if isone(n)
    log_likelihood(y, mu, Bernoulli(), n)
  else
    y * log(mu) + (n-y) * log(1-mu) + loggamma(n+1) - loggamma(y+1) - loggamma(n-y+1)
  end
end
log_likelihood(y, mu, ::Poisson, n = one(y)) = y * log(mu) - mu - loggamma(y+1)
#
# LOG-LIKELIHOOD: Use linear predictor to avoid -Inf/NaN
#
log_likelihood(y, mu, eta, family::Distribution, link::Link, n = one(y)) = log_likelihood(y, mu, family, n)
log_likelihood(y, mu, eta, ::Bernoulli, ::LogitLink, n = one(y)) = begin
  if abs(eta) ≥ 32
    isone(y) ? loglogistic(eta) : log1mlogistic(eta)
  else
    log_likelihood(y, mu, Bernoulli(), n)
  end
end
log_likelihood(y, mu, eta, ::Binomial, ::LogitLink, n = one(y)) = begin
  if isone(n)
    log_likelihood(y, mu, eta, Bernoulli(), LogitLink(), n)
  else
    y * loglogistic(eta) + (n-y) * log1mlogistic(eta) + loggamma(n+1) - loggamma(y+1) - loggamma(n-y+1)
  end
end

export log_likelihood
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
  CUTOFF = abs(log(eps(typeof(eta))))
  eta = clamp(eta, -CUTOFF, CUTOFF)
  return inv(2 * cosh(eta/2))^2
end
meanderiv(::LogLink, eta) = exp(eta)

export meanderiv
#
# VARIANCE FUNCTIONS
#
varfun(::Normal, mu) = one(mu)
varfun(::Union{Bernoulli,Binomial}, mu) = mu * (1+mu)
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
  fy + oftype(fy, 1/2)
end

export mustart

end # module

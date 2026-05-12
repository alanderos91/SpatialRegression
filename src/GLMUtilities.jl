module GLMUtilities
# Helper functions for working with GLMs.
# Source: Hua's GLM implementation + GLM.jl; favors LogExpFunctions.jl when possible.

using LogExpFunctions
using SpecialFunctions
using Distributions: 
  Distribution,
  ContinuousUnivariateDistribution,
  UnivariateDistribution,
  MixtureModel,
  Normal, Bernoulli, Binomial, Poisson,
  components, component_type,
  @check_args, @distr_support
using Random: AbstractRNG
using GLM:
  Link,
  IdentityLink, LogitLink, LogLink,
  canonicallink, glm, coef
import Base: eltype, rand
import Distributions
import Distributions: params, partype, location, scale, pdf, logpdf, cdf, logcdf
import Statistics: mean, median, quantile, std, var
import StatsBase: kurtosis, skewness, entropy, mode, params

export Distribution, UnivariateDistribution, MixtureModel,
  Normal, Bernoulli, Binomial, Poisson,
  components, component_type,
  Link,
  IdentityLink, LogitLink, LogLink,
  canonicallink,
  AsymmetricLaplace, ALD, pinball
#
# Asymmetric Laplace Distribution
#
struct AsymmetricLaplace{T<:Real} <: ContinuousUnivariateDistribution
  μ::T
  σ::T
  τ::T
  AsymmetricLaplace{T}(μ::T, σ::T, τ::T) where {T<:Real} = new{T}(μ, σ, τ)
end

function AsymmetricLaplace(μ::T, σ::T, τ::T; check_args::Bool = true) where {T<:Real}
  @check_args AsymmetricLaplace (σ, σ >= zero(σ)) (τ, 0 < τ < 1)
  return AsymmetricLaplace{T}(μ, σ, τ)
end

#### Outer Constructors
AsymmetricLaplace(μ::Real, σ::Real, τ::Real; check_args::Bool = true) = AsymmetricLaplace(promote(μ, σ, τ)...; check_args = check_args)
AsymmetricLaplace(τ::Real = 0.5) = AsymmetricLaplace(zero(τ), one(τ), τ; check_args = false)

const ALD = AsymmetricLaplace

#### Conversions
convert(::Type{AsymmetricLaplace{T}}, μ::S, σ::S, τ::S) where {T <: Real, S <: Real} = AsymmetricLaplace(T(μ), T(σ), T(τ))
Base.convert(::Type{AsymmetricLaplace{T}}, d::AsymmetricLaplace) where {T<:Real} = AsymmetricLaplace{T}(T(d.μ), T(d.σ), T(d.τ))
Base.convert(::Type{AsymmetricLaplace{T}}, d::AsymmetricLaplace{T}) where {T<:Real} = d

@distr_support ALD -Inf Inf

#### Parameters
params(d::AsymmetricLaplace) = (d.μ, d.σ, d.τ)
@inline partype(::AsymmetricLaplace{T}) where {T<:Real} = T

location(d::AsymmetricLaplace) = d.μ
scale(d::AsymmetricLaplace) = d.σ

Base.eltype(::Type{AsymmetricLaplace{T}}) where {T} = T

#### Statistics
function quantile(d::AsymmetricLaplace, p::Real)
  μ, σ, τ = params(d)
  if p <= τ
    μ + σ / (1-τ) * log(p/τ)
  else
    μ - σ / τ * log((1-p)/(1-τ))
  end
end

mean(d::AsymmetricLaplace) = d.μ + (1 - 2*d.τ) * inv(d.τ * (1 - d.τ)) * d.σ
median(d::AsymmetricLaplace) = quantile(d, 0.5)
mode(d::AsymmetricLaplace) = d.μ

var(d::AsymmetricLaplace) = inv(d.τ^2 * (1-d.τ)^2) * (1 - 2*d.τ + 2*d.τ^2) * d.σ
std(d::AsymmetricLaplace) = sqrt(var(d))
skewness(d::AsymmetricLaplace) = 2 * (1-2*d.τ) * (1-d.τ+d.τ^2) * inv(1-2*d.τ+2*d.τ^2)^(3/2)
kurtosis(d::AsymmetricLaplace) = 6 * (d.τ^4 + (1-d.τ)^4) * inv(1-2*d.τ+2*d.τ^2)^2

entropy(d::AsymmetricLaplace) = 1 + log(d.σ) - log(d.τ*(1-d.τ))

#### Evaluation
function Distributions.pdf(d::AsymmetricLaplace, x::Real)
  μ, σ, τ = params(d)
  c = τ*(1-τ)/σ
  r = (x-μ)/σ
  return c * exp(-(τ - 1//2) * r - 1//2 * abs(r))
end

function Distributions.logpdf(d::AsymmetricLaplace, x::Real)
  μ, σ, τ = params(d)
  r = (x-μ)/σ
  return log(τ) + log(1-τ) - log(σ) - (τ - 1//2)*r - 1//2 * abs(r)
end

function Distributions.cdf(d::AsymmetricLaplace, x::Real)
  μ, σ, τ = params(d)
  r = (x-μ)/σ
  if x < μ
    τ * exp((1-τ)*r)
  else
    1 - (1-τ) * exp(-τ*r)
  end
end

function Distributions.logcdf(d::AsymmetricLaplace, x::Real)
  μ, σ, τ = params(d)
  r = (x-μ)/σ
  if x < μ
    log(τ) + (1-τ)*r
  else
    log1mexp(log1p(-τ) - τ*r)
  end
end

function Distributions.mgf(d::AsymmetricLaplace, t::Real)
  μ, σ, τ, _t = promote(params(d)..., t)
  if (τ-1)/σ < _t < τ/σ
    τ*(1-τ) * exp(μ*_t) * inv((τ-σ*_t) * (σ*_t+1-τ))
  else
    typemax(_t)
  end
end

function Distributions.cf(d::AsymmetricLaplace, t::Real)
  μ, σ, τ, _t = promote(params(d)..., t)
  τ*(1-τ) * exp(im*μ*_t) * inv((τ-im*σ*_t) * (im*σ*_t+1-τ))
end

#### Sampling
# use the quantile fallback defined in univariates.jl

pinball(r, tau) = (tau - 1//2)*r + 1//2*abs(r)
#
# LOG-LIKELIHOOD: Use mean parameter only
#
function logpdf(y, mu, phi, ::Normal, n = one(y))
  if isfinite(phi)
    -1//2 * (abs2(y - mu) * phi + log(2*π/phi))
  else
    y == mu ? typemax(phi) : typemin(phi)
  end
end
logpdf(y, mu, phi, ::Bernoulli, n = one(y)) = isone(y) ? log(mu) : log(1-mu)
logpdf(y, mu, phi, ::Binomial, n = one(y)) = begin
  if isone(n)
    logpdf(y, mu, phi, Bernoulli(), n)
  else
    y * log(mu) + (n-y) * log(1-mu) + loggamma(n+1) - loggamma(y+1) - loggamma(n-y+1)
  end
end
logpdf(y, mu, phi, ::Poisson, n = one(y)) = y * log(mu) - mu - loggamma(y+1)
function logpdf(y, mu, phi, d::ALD, n = one(y))
  if isfinite(phi)
    τ = d.τ
    log(τ * (1 - τ)) + log(phi) - pinball(y - mu, τ) * phi
  else
    y == mu ? typemax(phi) : typemin(phi)
  end
end
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
deviance(d::ALD, y, mu) = pinball(y - mu, d.τ)

export deviance
#
# DISPERSION
#
needs_dispersion(::Distribution) = false
needs_dispersion(::Normal) = true
needs_dispersion(::ALD) = true

dispfun(d::Distribution, phi) = begin
  needs_dispersion(d) ? _dispfun(d, phi) : one(phi)
end
_dispfun(::Normal, phi) = phi
_dispfun(::ALD, phi) = phi

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
varfun(::Poisson, mu) = mu
varfun(::ALD, mu) = one(mu)

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
mustart(::ALD, y) = y

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

function create_component(d::ALD, link, η, φ)
  return ALD(η, φ, d.τ)
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

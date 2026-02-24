module SpatialRegression

using LinearAlgebra
using StaticArrays
using Distributions
using ForwardDiff
using DelaunayTriangulation
using OhMyThreads

import Distributions: loglikelihood

import GLM
using GLM: GlmResp, Link, CauchitLink, CloglogLink,
  IdentityLink, InverseLink, InverseSquareLink,
  LogitLink, LogLink, NegativeBinomialLink,
  PowerLink, ProbitLink, SqrtLink,
  canonicallink

#
# COORDINATE CONVERSIONS
#
"""
Convert Cartesian coordinates `s` to barycentric coordinates with respect
to triangle vertices in `V`. Vertices are assumed to stored in columns.
"""
barycentric(s::AbstractVector, V) = barycentric!(zeros(eltype(s), 3), s, V)

function barycentric!(out, s::AbstractVector{T}, V::AbstractMatrix{T}) where T <: AbstractFloat
  @assert length(s) == 2
  A = SMatrix{3,3,T,9}( # remember: column major
    V[1,1], V[2,1], 1, # column 1
    V[1,2], V[2,2], 1, # column 2
    V[1,3], V[2,3], 1  # column 3
  )
  b = SVector{3,T}(s[1], s[2], 1)
  out .= A \ b
  return out
end

"""
Converts Cartesian coordinates in `S` (stored in columns) to barycentric coordinates, using triangle vertices in `V`.
"""
function barycentric(S::AbstractMatrix, V)
  out = zeros(eltype(S), 3, size(S, 2))
  for (o, s) in zip(eachcol(out), eachcol(S))
    barycentric!(o, s, V)
  end
  return out
end

"""
Convert barycentric coordinates `a` with respect to triangle vertices `V` to Cartesian coordinates.
Assumes vertices are stored in columns.
"""
cartesian(a::AbstractVector, V) = cartesian!(zeros(eltype(a), 2), a, V)
cartesian!(out, a::AbstractVector{T}, V::AbstractMatrix{T}) where T <: AbstractFloat = mul!(out, V, a)

"""
Converts barycentric coordinates in `A` (stored in columns) to Cartesian coordinates, using triangle vertices in `V.`
"""
function cartesian(A::AbstractMatrix, V)
  out = zeros(eltype(A), 2, size(A, 2))
  for (o, a) in zip(eachcol(out), eachcol(A))
    cartesian!(o, a, V)
  end
  return out
end

struct TriangleObs
  triangle::Tuple{Int,Int,Int} # indices in ascending order
  idx::Vector{Int}    # set of indices into original data (y, X)
  y::Vector{Float64}  # response local to triangle
  X::Matrix{Float64}  # covariates local to triangle
  A::Matrix{Float64}  # mixture weights as barycentric coordinates
  V::Matrix{Float64}  # vertices of triangle, stored in columns
end

# check if triangle has vertex with index j as one of its vertices
has_vertex(T::TriangleObs, j::Int) = j in T.triangle

function is_inside_mesh(Δ::Triangulation, s)
  triangle = find_triangle(Δ, s, concavity_protection = true)
 return all(>(0), triangle)
end

is_inside_mesh(Δ::Triangulation) = Base.Fix1(is_inside_mesh, Δ)

function create_triobs_sets(y, X, S, tri; nchunks::Int = Threads.nthreads())
  # Get points + mapping to solid vertices.
  # The mapping is needed to ensure vertex label = index into some array
  vertex = get_points(tri)
  id2vertex = Dict{Int,Int}()

  # Matrix of vertices, stored along columns
  V = Matrix{Float64}(undef, 2, length(vertex))
  for (j, id) in enumerate(each_solid_vertex(tri))
    V[:, j] .= vertex[id]
    id2vertex[id] = j
  end

  # Assign each observation to a triangle in parallel
  TriType = Tuple{Int,Int,Int}
  IdxType = Vector{Int}
  itr = OhMyThreads.ChannelLike(axes(S, 2))
  tmp = OhMyThreads.@localize tri id2vertex itr OhMyThreads.tmap(Dict{TriType,IdxType}, 1:nchunks; chunking = false) do _
    local dict = Dict{TriType,IdxType}()
    map(itr) do i
      if iszero(i)
        triidx = (0, 0, 0)
      else
        s = view(S, :, i)
        j, k, l = find_triangle(tri, s, concavity_protection = true)
        triidx = sort((id2vertex[j], id2vertex[k], id2vertex[l]))
      end
      if !haskey(dict, triidx)
        dict[triidx] = Int[]
      end
      push!(dict[triidx], i)
      return nothing
    end
    return dict
  end
  
  # Aggregate results across tasks
  dict = Dict{TriType,IdxType}()
  mergewith!(union!, dict, tmp...)
  n_active = length(keys(dict))

  # Create TriangleObs for each 'active' triangle
  triobs = Vector{TriangleObs}(undef, n_active)
  for (i, (triidx, idx)) in enumerate(dict)
    sort!(idx) # mitigate random access patterns as much as possible
    v = V[:, [triidx[1], triidx[2], triidx[3]]]
    A = barycentric(view(S, :, idx), v)
    triobs[i] = TriangleObs(triidx, idx, y[idx], X[idx, :], A, v)
  end

  return triobs
end

struct VertexModel{D <: Distribution, L <: Link}
  family::D                 # distribution
  link::L                   # link function, g(μ) = η
  index::Int                # vertex index, needed for consitency with triangulation
  d::Vector{Float64}        # IRLS weights, determined by GLM + mixture coefficients
  beta::Vector{Float64}     # coefficients, β
  gamma::Matrix{Float64}    # matrix of local average coefficients in MM algorithm, Γ
  eta::Vector{Float64}      # linear predictor, η = xᵀβ
  mu::Vector{Float64}       # mean parameter vector, μ = g⁻¹(η)
  workres::Vector{Float64}  # working residuals, (y - μ) g'(μ)
  neighbors::Vector{Int}    # index set representing neighboring vertices in penalty
  triangles::Vector{Int}    # index set representing incident triangles
  weights::Vector{Float64}  # penalty weights, w in ∑ w P(B)
  rho::Float64              # penalty coefficient
  beta_new::Vector{Float64} # proposed update
end

function create_vertexmodel_set(family::D, link::L, tri::Triangulation, tobs::Vector{TriangleObs}, nvars; rho::Real = 1.0) where {D,L}
  mv = VertexModel{D,L}[]
  
  # recreate mapping from triangulation labels to our labels
  id2vertex = Dict{Int,Int}()
  for (j, id) in enumerate(each_solid_vertex(tri))
    id2vertex[id] = j
  end
  
  # This assumes triangles in tobs are labeled using OUR scheme, not the one in the triangulation.
  wmax = 0.0
  for (j, id) in enumerate(each_solid_vertex(tri))
    # count the total number of samples incident with vertex j
    nsamples, triangles = 0, Int[]
    for k in eachindex(tobs)
      if has_vertex(tobs[k], j)
        nsamples += length(tobs[k].y)
        push!(triangles, k)
      end
    end

    # determine the number of incident vertices
    neighbors = [id2vertex[u] for u in DelaunayTriangulation.iterated_neighbourhood(tri, id, 1)]
    sort!(neighbors)
    nneighbors = length(neighbors)

    # allocate!
    d = zeros(nsamples)
    beta = zeros(nvars)
    gamma = zeros(nvars, nneighbors)
    eta = zeros(nsamples)
    mu = zeros(nsamples)
    workres = zeros(nsamples)
    weights = ones(nneighbors)

    for (k, other) in enumerate(DelaunayTriangulation.iterated_neighbourhood(tri, id, 1))
      weights[k] = 1 / DelaunayTriangulation.dist(get_point(tri, id), get_point(tri, other))
      wmax = max(wmax, weights[k])
    end

    push!(mv, VertexModel(family, link, j,
        d, beta, gamma, eta, mu, workres,
        neighbors, triangles, weights, rho, similar(beta)
      )
    )
  end
  
  # sort!(mv, by = v -> v.index)
  for v in mv
    v.weights .= v.weights / wmax
  end

  return mv
end

function loglik_obs(v::VertexModel, y, x)
  loglik_obs(v.beta, y, x, v.family, v.link)
end

function loglik_obs(beta, y, x, family, link)
  η = dot(x, beta)
  μ, _ = GLM.inverselink(link, η)
  logl = GLM.loglik_obs(family, y, μ, 1, 1)
  return logl
end

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

function eval_loglikelihoods(v, y, x)
  return (  # TODO: Make this use BLAS; i.e. η₁, η₂, η₃ = Bᵀx with Bᵀ 3 × p
    loglik_obs(v[1], y, x),
    loglik_obs(v[2], y, x),
    loglik_obs(v[3], y, x)
  )
end

abstract type AbstractPenalty end

struct L2Squared <: AbstractPenalty end
struct L1Approx <: AbstractPenalty epsilon::Float64 end

function eval_loglikelihood(tobs, vmod, penalty_type::AbstractPenalty)
  logl = zero(Float64)
  for T in tobs
    A, y, X = T.A, T.y, T.X
    _, v = get_triangle_vertices(T, T.triangle[1], vmod)
    for i in eachindex(y)
      x = view(X, i, :)
      a = view(A, :, i)

      # evaluate log-likelihood at each vertex
      logf = eval_loglikelihoods(v, y[i], x)
      
      # shift log(∑ⱼ αⱼ exp(fⱼ)) by the most negative log-likelihood
      logl += stable_logsumexp(a, logf)
    end
  end

  penalty = eval_penalty(penalty_type, vmod)

  return logl + penalty
end

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

"""
    get_triangle_vertices(T::TriangleObs, index, vmod)

Match `index` to one of `(j, k, l)` and retrieve the vertices `(vⱼ, vₖ, vₗ)`.

Returns `pos` as one of `1`, `2`, or `3` along with a tuple of vertices.
"""
function get_triangle_vertices(T::TriangleObs, index, vmod)
  j, k, l = T.triangle
  if index == j
    pos = 1
  elseif index == k
    pos = 2
  elseif index == l
    pos = 3
  else
    error("The index $(index) is not in the triangle $(T.triangle).")
  end
  return pos, (vmod[j], vmod[k], vmod[l])
end

"""
    stable_eval_mm_weight(t, a, v, y, x)

Evaluate the MM weight in a log-likelihood surrogate, `a[t]*f[t] / (a[1]*f[1]+a[2]*f[2]+a[3]*f[3])`, using the exp-log-sum trick.
Here `a` and `v` are vectors of length 3 containing convex weights and `VertexModel` objects, respectively.
The index `t` indicates which vertex of `(v[1], v[2], v[3])` is the argument of the surrogate.

Returns the MM weight.
"""
function stable_eval_mm_weight(t::Integer, a::AbstractVector{T}, logf::NTuple{3,T}) where T <: Real
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

function eval_surrogate(tobs, vmod, j, penalty_type::AbstractPenalty)
  v = vmod[j]
  eval_surrogate(v.beta, v.index, v.gamma, v.weights, v.rho, vmod, tobs, penalty_type)
end

function eval_surrogate(beta, j, gamma, weights, rho, vmod, tobs, penalty_type::AbstractPenalty)
  logl, vⱼ = zero(Float64), vmod[j]

  # Log-Likelihood
  for triidx in vⱼ.triangles
    T = tobs[triidx]
    A, y, X = T.A, T.y, T.X
    t, v = get_triangle_vertices(T, j, vmod)
    
    for i in eachindex(T.y)
      x = view(X, i, :)
      a = view(A, :, i)

      # Evaluate log-likelihood term at β
      logfⱼ = loglik_obs(beta, y[i], x, vⱼ.family, vⱼ.link)

      # Evaluate terms dependent on the anchor point βₙ
      zweight = stable_eval_mm_weight(t, a, v, y[i], x)

      logl += zweight * logfⱼ + zweight * (log(a[t]) + log(inv(zweight)))
    end
  end

  # Penalty
  penalty = eval_penalty_surrogate(penalty_type, rho, beta, weights, gamma, vⱼ.beta)

  return logl + penalty
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

function initialize_coefficients!(tobs, vmod)
  for v in vmod
    initialize_coefficients!(v, tobs, vmod)
  end
end

function initialize_coefficients!(v::VertexModel, tobs, vmod)
  p = length(v.beta)
  for triidx in v.triangles
    T = tobs[triidx]
    j, k, l = T.triangle
    
    # Solve linear system associated with triangle
    η = GLM.linkfun.(v.link, T.y)
    X = [Diagonal(T.A[1,:])*T.X Diagonal(T.A[2,:])*T.X Diagonal(T.A[3,:])*T.X]
    β = [vmod[j].beta; vmod[k].beta; vmod[l].beta]
    β .= X \ η

    # Average solutions over incident triangles for each vertex
    for (r, u) in enumerate((vmod[j], vmod[k], vmod[l]))
      u.beta .+= 1/length(u.triangles) * β[p*(r-1)+1 : p*r]
    end
  end
  return nothing
end

function mm_update!(penalty, vⱼ, vmod, tobs, workspace)
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

      # Compute weights and working residuals
      for i in eachindex(y)
        x = view(X, i, :)
        a = view(A, :, i)

        zweight = stable_eval_mm_weight(t, a, v, y[i], x)
        η[idx] = dot(x, β)
        μ[idx], dμdη = GLM.inverselink(vⱼ.link, η[idx])
        r[idx] = (y[i] - μ[idx]) / dμdη
        d[idx] = zweight * dμdη^2 / stable_glmvar(vⱼ.family, μ[idx], η[idx])
        # if isnan(d[idx]) || isinf(d[idx]) || isnan(r[idx]) || isinf(r[idx])
        #   error("Encountered underflow/overflow. Check arrays!")
        # end
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
    # gg = ForwardDiff.gradient(b -> eval_surrogate(b, vⱼ.index, Γ, w, rho, vmod, tobs, penalty), β)
    # hh = ForwardDiff.hessian(b -> -eval_surrogate(b, vⱼ.index, Γ, w, rho, vmod, tobs, penalty), β)
    # @show norm(gg - ∇L)
    # @show norm(Symmetric(hh, :U) - Symmetric(∇²L, :U))
    ldiv!(search_direction, cholesky!(Symmetric(∇²L, :U)), ∇L)
    # ldiv!(search_direction, cholesky!(Symmetric(hh, :U)), ∇L)
    # @. vⱼ.beta_new = β + search_direction
  end
end

function linesearch!(βₙ₊₁, βₙ, Δ, objective, backtrack, index, gamma, weights, rho, vmod, tobs, penalty)
  # Backtracking line search
  t = 1.0
  for step in 0:backtrack
    @. βₙ₊₁ = βₙ + t * Δ
    objective_new = eval_surrogate(βₙ₊₁, index, gamma, weights, rho, vmod, tobs, penalty)
    if objective_new >= objective
      break
    elseif step < backtrack
      t /= 2
    else
      @show t, objective_new, objective
      error("Backtracking failed at iteration $(iter) for vertex $(vⱼ.index) after $(step) attempts!")
      @. βₙ₊₁ = βₙ
      break
    end
  end
  return t, step
end

function fitmodel(yfull, Xfull, Sfull, tri;
    maxiter::Int = 100,
    backtrack::Int = 5,
    tol::Real = 1e-3,
    family::UnivariateDistribution = Normal(),
    link::Link = GLM.canonicallink(family),
    rho::Real = 1.0,
    penalty::PT = L2Squared(),
    nchunks::Int = Threads.nthreads(),
    # intercept = all(isequal(1), view(Xfull, :, 1)),
  ) where PT <: AbstractPenalty
  # Initialize
  nvars = size(Xfull, 2)
  tobs = create_triobs_sets(yfull, Xfull, Sfull, tri; nchunks = 4*nchunks)
  vmod = create_vertexmodel_set(family, link, tri, tobs, nvars; rho = rho)
  # initialize_coefficients!(tobs, vmod)
  logl = eval_loglikelihood(tobs, vmod, penalty)
  logl_prev = zero(logl)

  BLAS_THREADS = BLAS.get_num_threads()
  cache = [(zeros(nvars), zeros(nvars, nvars), zeros(nvars)) for _ in 1:nchunks]

  iter = 0
  while iter < maxiter && abs(logl - logl_prev) > (1 + abs(logl_prev)) * tol
    iter += 1

    # Update local averaged estimates, γₙⱼₖ
    eval_gamma!(vmod)

    # Visit each vertex once to update local parameters
    workspace = OhMyThreads.ChannelLike(cache)
    workitr = OhMyThreads.ChannelLike(vmod)
    try
      BLAS.set_num_threads(max(1, div(BLAS_THREADS, nchunks)))
      OhMyThreads.@localize iter tobs vmod nvars OhMyThreads.tforeach(1:nchunks; chunking = false) do _
        map(workspace) do (∇L, ∇²L, search_direction)

          map(workitr) do vⱼ
            objective = eval_surrogate(vⱼ.beta, vⱼ.index, vⱼ.gamma, vⱼ.weights, vⱼ.rho, vmod, tobs, penalty)
            isinf(objective) && display(vⱼ.beta)
            mm_update!(penalty, vⱼ, vmod, tobs, (∇L, ∇²L, search_direction))
            linesearch!(
              vⱼ.beta_new,
              vⱼ.beta,
              search_direction,
              objective,
              backtrack,
              vⱼ.index, vⱼ.gamma, vⱼ.weights, vⱼ.rho, vmod, tobs, penalty
            )
          end
        end
      end
    finally
      BLAS.set_num_threads(BLAS_THREADS)
    end

    # Apply all updates
    for vⱼ in vmod
      @. vⱼ.beta = vⱼ.beta_new
    end

    # Evaluate log-likelihood
    logl_prev = logl
    logl = eval_loglikelihood(tobs, vmod, penalty)
    # if logl < logl_prev
    #   rel = abs(logl - logl_prev) / (1 + abs(logl_prev))
    #   @show iter, logl, logl_prev, rel
    #   @warn "Detected increase in log-likelihood, likely due to instability in evaluating it at current estimate."
    #   break
    # end
    # rel = abs(logl - logl_prev) / (1 + abs(logl_prev))
    # @show iter, logl, logl_prev, rel
  end
  return iter, tobs, vmod
end

function predict(X, S, vmod, tri)
  id2vertex = Dict{Int,Int}(id => j for (j, id) in enumerate(each_solid_vertex(tri)))
  points = get_points(tri)
  yhat = zeros(size(X, 1))
  @views for i in axes(X, 1)
    s = S[:, i]
    x = X[i, :]
    j, k, l = find_triangle(tri, s; concavity_protection = true) |> sort
    p, q, r = points[j], points[k], points[l]
    j, k, l = id2vertex[j], id2vertex[k], id2vertex[l]
    v, u, w = vmod[j], vmod[k], vmod[l]
    a, b, c = barycentric(s, [p[1] q[1] r[1]; p[2] q[2] r[2]])
    yhat[i] = 
      a*GLM.linkinv(v.link, dot(x, v.beta)) +
      b*GLM.linkinv(u.link, dot(x, u.beta)) +
      c*GLM.linkinv(w.link, dot(x, w.beta))
  end
  return yhat
end

export TriangleObs, create_triobs_sets,
  VertexModel, create_vertexmodel_set

end

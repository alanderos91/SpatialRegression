module SpatialRegression

using LinearAlgebra
using StaticArrays
using Distributions
using ForwardDiff
using DelaunayTriangulation

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

function create_triobs_sets(y, X, S, tri)
  stats = statistics(tri)
  vertex = tri.points

  # Matrix of vertices, stored along columns
  V = Matrix{Float64}(undef, 2, stats.num_solid_vertices)
  for j in each_solid_vertex(tri)
    V[:, j] .= vertex[j]
  end

  # Retrieve each triangle, sorting indices in ascending order
  dict = Dict{Tuple{Int,Int,Int},Vector{Int}}()
  for (i, s) in enumerate(eachcol(S))
    tri_ind = sort(find_triangle(tri, s))
    if !haskey(dict, tri_ind)
      dict[tri_ind] = Int[]
    end
    index_set = dict[tri_ind]
    push!(index_set, i)
  end

  triobs = TriangleObs[]
  k = 0
  for (tri_ind, idx) in dict
    k += 1
    Vₖ = V[:, [tri_ind...]]
    A = barycentric(view(S, :, idx), Vₖ)
    push!(triobs, TriangleObs(tri_ind, idx, y[idx], X[idx, :], A, Vₖ))
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
  weights::Vector{Float64}  # penalty weights, w in ∑ w P(B)
  rho::Float64              # penalty coefficient
  beta_new::Vector{Float64} # proposed update
end

function create_vertexmodel_set(family::D, link::L, tri::Triangulation, tobs::Vector{TriangleObs}, nvars; rho::Real = 1.0) where {D,L}
  mv = VertexModel{D,L}[]
  for j in each_solid_vertex(tri)
    # count the total number of samples incident with vertex j
    nsamples = 0
    for k in eachindex(tobs)
      if has_vertex(tobs[k], j)
        nsamples += length(tobs[k].y)
      end
    end

    # determine the number of incident vertices
    neighbors = DelaunayTriangulation.iterated_neighbourhood(tri, j, 1) |>
      collect |> sort!
    nneighbors = length(neighbors)

    # allocate!
    d = zeros(nsamples)
    beta = zeros(nvars)
    gamma = zeros(nvars, nneighbors)
    eta = zeros(nsamples)
    mu = zeros(nsamples)
    workres = zeros(nsamples)
    weights = ones(nneighbors)

    push!(mv, VertexModel(family, link, j,
        d, beta, gamma, eta, mu, workres,
        neighbors, weights, rho, similar(beta)
      )
    )
  end
  return sort!(mv, by = v -> v.index)
end

function loglik_obs(v::VertexModel, y, x)
  η = dot(x, v.beta)
  μ, _ = GLM.inverselink(v.link, η)
  logl = GLM.loglik_obs(v.family, y, μ, 1, 1)
  return logl
end

function eval_loglikelihood(tobs, vmod)
  logl = zero(Float64)
  # Log-Likelihood
  for T in tobs
    j, k, l = T.triangle
    A, y, X = T.A, T.y, T.X
    v1, v2, v3 = vmod[j], vmod[k], vmod[l]
    @views for i in eachindex(y)
      f1 = loglik_obs(v1, y[i], X[i, :]) |> exp
      f2 = loglik_obs(v2, y[i], X[i, :]) |> exp
      f3 = loglik_obs(v3, y[i], X[i, :]) |> exp
      f = A[1,i]*f1 + A[2,i]*f2 + A[3,i]*f3
      logl += ifelse(iszero(f), zero(f), log(f))
    end
  end
  # Penalty
  for v in vmod
    for (idx, k) in enumerate(v.neighbors)
      u = vmod[k]
      logl -= v.rho/2 * v.weights[idx] * sum(abs2, v.beta - u.beta)
    end
  end
  return logl
end

function eval_surrogate(v::VertexModel, vmod, tobs)
  logl = zero(Float64)
  # Log-Likelihood
  for T in tobs
    if !has_vertex(T, v.index) continue end
    j, k, l = T.triangle; pos = findfirst(==(v.index), T.triangle)
    v1, v2, v3 = vmod[j], vmod[k], vmod[l]
    A, y, X = T.A, T.y, T.X
    @views for i in eachindex(y)
      f1 = loglik_obs(v1, y[i], X[i, :])
      f2 = loglik_obs(v2, y[i], X[i, :])
      f3 = loglik_obs(v3, y[i], X[i, :])
      z1 = A[1,i] * ifelse(f1 > -600, exp(f1), one(f1))
      z2 = A[2,i] * ifelse(f2 > -600, exp(f2), one(f2))
      z3 = A[3,i] * ifelse(f3 > -600, exp(f3), one(f3))
      zsum = z1 + z2 + z3
      z, f = ifelse(v.index == j, (z1, f1), ifelse(v.index == k, (z2, f2), (z3, f3)))
      logl += z / zsum * f + z / zsum * (log(A[pos,i]) + log(zsum) - log(z))
      if isnan(logl)
        @show i, z1, z2, z3, zsum, f
        error()
      end
    end
  end
  # Penalty
  penalty = zero(logl)
  for (idx, γ) in enumerate(eachcol(v.gamma))
    penalty -= v.rho * v.weights[idx] * sum(abs2, v.beta - γ)
  end
  return logl + penalty
end

function G(beta, index, gamma, weights, rho, vmod, tobs)
  logl = zero(Float64)
  # Log-Likelihood
  for T in tobs
    if !has_vertex(T, index) continue end
    # j, k, l = T.triangle
    pos1 = findfirst(==(index), T.triangle)
    if pos1 == 1
      j, k, l = T.triangle
      pos2, pos3 = 2, 3
    elseif pos1 == 2
      k, j, l = T.triangle
      pos2, pos3 = 1, 3
    else # pos1 == 3
      k, l, j = T.triangle
      pos2, pos3 = 1, 2
    end
    v1, v2, v3 = vmod[j], vmod[k], vmod[l]
    family, link = v1.family, v1.link
    beta1, beta2, beta3 = v1.beta, v2.beta, v3.beta
    A, y, X = T.A, T.y, T.X
    @views for i in eachindex(T.y)
      x = X[i,:]
      η1, η2, η3 = dot(x, beta1), dot(x, beta2), dot(x, beta3)
      μ1, _, _ = GLM.inverselink(link, η1)  
      μ2, _, _ = GLM.inverselink(link, η2)
      μ3, _, _ = GLM.inverselink(link, η3)
      f1 = GLM.loglik_obs(family, y[i], μ1, 1, 1)
      f2 = GLM.loglik_obs(family, y[i], μ2, 1, 1)
      f3 = GLM.loglik_obs(family, y[i], μ3, 1, 1)

      z1 = A[pos1,i] * ifelse(f1 > -600, exp(f1), one(f1))
      z2 = A[pos2,i] * ifelse(f2 > -600, exp(f2), one(f2))
      z3 = A[pos3,i] * ifelse(f3 > -600, exp(f3), one(f3))

      zsum = z1 + z2 + z3

      η = dot(x, beta)
      μ, _, _ = GLM.inverselink(link, η)
      f = GLM.loglik_obs(family, y[i], μ, 1, 1)

      logl += z1 / zsum * f + z1 / zsum * (log(A[pos1,i]) + log(zsum) - log(z1))
    end
  end
  # Penalty
  penalty = zero(logl)
  for (idx, γ) in enumerate(eachcol(gamma))
    penalty -= rho * weights[idx] * sum(abs2, beta - γ)
  end
  return logl + penalty
end

function fitmodel(yfull, Xfull, Sfull, tri;
    maxiter::Int = 100,
    backtrack::Int = 5,
    tol = 1e-3,
    family = Normal(),
    link = GLM.canonicallink(family),
    rho = 1.0,
    intercept = all(isequal(1), view(Xfull, :, 1)),
  )
  # Initialize
  nvars = size(Xfull, 2)
  tobs = create_triobs_sets(yfull, Xfull, Sfull, tri)
  vmod = create_vertexmodel_set(family, link, tri, tobs, nvars; rho = rho)
  logl = eval_loglikelihood(tobs, vmod)
  logl_prev = zero(logl)
  ∇L = zeros(nvars)
  ∇²L = zeros(nvars, nvars)
  search_direction = zeros(nvars)

  iter = 0
  while iter < maxiter && abs(logl - logl_prev) > (1 + abs(logl_prev)) * tol
    iter += 1

    # Visit each vertex once to update local parameters
    for v in vmod
      # Setup local variables to match notation
      β = v.beta
      Γ = v.gamma
      d = v.d
      D = Diagonal(d)
      r = v.workres
      η = v.eta
      μ = v.mu
      w = v.weights

      # Update local averaged estimates, γₙⱼₖ
      for (idx, k) in enumerate(v.neighbors)
        if v.index == k continue end
        βₖ = vmod[k].beta
        @views begin
          @. Γ[:, idx] = 1//2 * (β + βₖ)
        end
      end
      objective = eval_surrogate(v, vmod, tobs)
      isinf(objective) && display(v.beta)
      
      fill!(∇L, 0); fill!(∇²L, 0)
      idx = 1
      for T in tobs
        if !has_vertex(T, v.index) continue end
        j, k, l = T.triangle
        A, y, X = T.A, T.y, T.X
        v1, v2, v3 = vmod[j], vmod[k], vmod[l]
        
        # Compute weights and working residuals
        for i in eachindex(y)
          # exp(f) may be close to 0; replace with A[]
          f1 = @views loglik_obs(v1, y[i], X[i, :])
          f2 = @views loglik_obs(v2, y[i], X[i, :])
          f3 = @views loglik_obs(v3, y[i], X[i, :])

          z1 = A[1,i] * ifelse(f1 > -745, exp(f1), one(f1))
          z2 = A[2,i] * ifelse(f2 > -745, exp(f2), one(f2))
          z3 = A[3,i] * ifelse(f3 > -745, exp(f3), one(f3))

          z = ifelse(v.index == j, z1, ifelse(v.index == k, z2, z3))
          zsum = z1 + z2 + z3

          η[idx] = @views dot(X[i, :], β)
          μ[idx], dμdη = GLM.inverselink(v.link, η[idx])
          r[idx] = (y[i] - μ[idx]) / dμdη
          d[idx] = z/zsum * dμdη^2 / GLM.glmvar(v.family, μ[idx])
          idx += 1
        end

        # Evaluate gradient + Hessian
        idxrange = (idx-length(y)):(idx-1)
        Dj = Diagonal(view(d, idxrange))
        rj = view(r, idxrange)
        ∇L .+= X' * Dj * rj
        ∇²L .+= X' * Dj * X
      end
      idxpen = (1+intercept):nvars
      ∇L[idxpen] .= ∇L[idxpen] - 2*v.rho*(sum(w)*β[idxpen] - Γ[idxpen,:]*w)
      ∇²L[idxpen,idxpen] .= ∇²L[idxpen,idxpen] + 2*v.rho*sum(w)*I
      # ∇G = ForwardDiff.gradient(beta -> G(beta, v.index, Γ, w, v.rho, vmod, tobs), β)
      # ∇²G = -ForwardDiff.hessian(beta -> G(beta, v.index, Γ, w, v.rho, vmod, tobs), β)

      # @show findall(isnan, D.diag)
      # @show v.index
      # display(∇²L)
      # Compute the search direction
      search_direction .= Symmetric(∇²L) \ ∇L

      # Backtracking line search
      t = 1.0
      v.beta_new .= β
      for step in 0:backtrack
        # if iter == 1 break end
        @. v.beta_new = β + t * search_direction
        # @assert β === v.beta
        # objective_new = eval_surrogate(v, vmod, tobs)
        objective_new = G(v.beta_new, v.index, Γ, w, v.rho, vmod, tobs)
        # @show objective_new, objective
        if objective_new >= objective
          break
        elseif step < backtrack
          t /= 2
        else
          @show t, objective_new, objective
          error("Backtracking failed at iteration $(iter) for vertex $(v.index) after $(step) attempts!")
        end
      end
      # β .= v.beta_new
      # @. v.beta_new = β + t * search_direction
      # println("Success for vertex $(v.index)")
    end

    # Apply all updates
    for v in vmod
      @. v.beta = v.beta_new
    end

    # Evaluate log-likelihood
    logl_prev = logl
    logl = eval_loglikelihood(tobs, vmod)
    rel = abs(logl - logl_prev) / (1 + abs(logl_prev))
    @show iter, logl, logl_prev, rel
  end

  return iter, tobs, vmod
end

export TriangleObs, create_triobs_sets,
  VertexModel, create_vertexmodel_set

end

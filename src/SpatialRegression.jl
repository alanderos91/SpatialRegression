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

exp_clamp(logf) = clamp()
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

  # Retrieve each triangle, sorting indices in ascending order
  dict = Dict{Tuple{Int,Int,Int},Vector{Int}}()
  for (i, s) in enumerate(eachcol(S))
    j, k, l = find_triangle(tri, s)
    tri_ind = sort((id2vertex[j], id2vertex[k], id2vertex[l]))
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
  η = dot(x, v.beta)
  μ, _ = GLM.inverselink(v.link, η)
  logl = GLM.loglik_obs(v.family, y, μ, 1, 1)
  return logl
end

function stable_logsumexp(a, f)
  s, c = stable_explogsum(a, f)
  return log(a[s]) + f[s] + log(c)
end

function stable_explogsum(a, f)
  s = argmin(f)
  c = zero(eltype(f))
  for k in eachindex(f)
    c += a[k] / a[s] * exp(f[k] - f[s])
  end
  return s, c
end

function eval_loglikelihood(tobs, vmod)
  logl = zero(Float64)
  # Log-Likelihood
  for T in tobs
    j, k, l = T.triangle
    A, y, X = T.A, T.y, T.X
    v1, v2, v3 = vmod[j], vmod[k], vmod[l]
    @views for i in eachindex(y)
      # evaluate log-likelihood at each vertex
      a1, f1 = A[1,i], loglik_obs(v1, y[i], X[i, :])
      a2, f2 = A[2,i], loglik_obs(v2, y[i], X[i, :])
      a3, f3 = A[3,i], loglik_obs(v3, y[i], X[i, :])
      
      # shift log(∑ⱼ αⱼ exp(fⱼ)) by the most negative log-likelihood
      f = (f1, f2, f3)
      a = (a1, a2, a3)
      logl += stable_logsumexp(a, f)
    end
  end
  # Penalty
  for v in vmod
    for (idx, k) in enumerate(v.neighbors)
      u = vmod[k]
      logl -= v.rho/4 * v.weights[idx] * sum(abs2, v.beta - u.beta)
    end
  end
  return logl
end

function eval_surrogate(tobs, vmod, j)
  v = vmod[j]
  eval_surrogate(v.beta, v.index, v.gamma, v.weights, v.rho, vmod, tobs)
end

function eval_surrogate(beta, index, gamma, weights, rho, vmod, tobs)
  logl, v = zero(Float64), vmod[index]
  # Log-Likelihood
  for triidx in v.triangles
    T = tobs[triidx]
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
      
      a1, f1 = A[pos1, i], GLM.loglik_obs(family, y[i], μ1, 1, 1)
      a2, f2 = A[pos2, i], GLM.loglik_obs(family, y[i], μ2, 1, 1)
      a3, f3 = A[pos3, i], GLM.loglik_obs(family, y[i], μ3, 1, 1)
      as = (a1, a2, a3)
      fs = (f1, f2, f3)
      s, c = stable_explogsum(as, fs)
      zweight = as[1] / as[s] * exp(fs[1] - fs[s]) / c

      η = dot(x, beta)
      μ, _, _ = GLM.inverselink(link, η)
      f = GLM.loglik_obs(family, y[i], μ, 1, 1)

      logl += zweight * f + zweight * (log(a1) + log(inv(zweight)))
    end
  end
  # Penalty
  penalty = zero(logl)
  for (idx, γ) in enumerate(eachcol(gamma))
    penalty -= rho * weights[idx] * sum(abs2, beta - γ)
  end
  return logl + penalty
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

function fitmodel(yfull, Xfull, Sfull, tri;
    maxiter::Int = 100,
    backtrack::Int = 5,
    tol = 1e-3,
    family = Normal(),
    link = GLM.canonicallink(family),
    rho = 1.0,
    # intercept = all(isequal(1), view(Xfull, :, 1)),
  )
  # Initialize
  nvars = size(Xfull, 2)
  tobs = create_triobs_sets(yfull, Xfull, Sfull, tri)
  vmod = create_vertexmodel_set(family, link, tri, tobs, nvars; rho = rho)
  # initialize_coefficients!(tobs, vmod)
  logl = eval_loglikelihood(tobs, vmod)
  logl_prev = zero(logl)
  ∇L = zeros(nvars)
  ∇²L = zeros(nvars, nvars)
  search_direction = zeros(nvars)

  iter = 0
  while iter < maxiter && abs(logl - logl_prev) > (1 + abs(logl_prev)) * tol
    iter += 1

    # Update local averaged estimates, γₙⱼₖ
    eval_gamma!(vmod)

    # Visit each vertex once to update local parameters
    for v in vmod
      # Setup local variables to match notation
      β = v.beta
      Γ = v.gamma
      d = v.d
      r = v.workres
      η = v.eta
      μ = v.mu
      w = v.weights

      objective = eval_surrogate(v.beta, v.index, Γ, w, v.rho, vmod, tobs)
      isinf(objective) && display(v.beta)
      
      fill!(∇L, 0); fill!(∇²L, 0)
      idx = 1
      for triidx in v.triangles
        T = tobs[triidx]
        j, k, l = T.triangle
        A, y, X = T.A, T.y, T.X
        v1, v2, v3 = vmod[j], vmod[k], vmod[l]
        
        # Compute weights and working residuals
        for i in eachindex(y)
          # exp(f) may be close to 0; replace with A[]
          a1, f1 = A[1, i], @views loglik_obs(v1, y[i], X[i, :])
          a2, f2 = A[2, i], @views loglik_obs(v2, y[i], X[i, :])
          a3, f3 = A[3, i], @views loglik_obs(v3, y[i], X[i, :])
          as = (a1, a2, a3)
          fs = (f1, f2, f3)

          t = findfirst(==(v.index), T.triangle)
          s, c = stable_explogsum(as, fs)
          zweight = as[t] / as[s] * exp(fs[t] - fs[s]) / c

          η[idx] = @views dot(X[i, :], β)
          μ[idx], dμdη = GLM.inverselink(v.link, η[idx])
          r[idx] = (y[i] - μ[idx]) / dμdη
          d[idx] = zweight * dμdη^2 / GLM.glmvar(v.family, μ[idx])
          idx += 1
        end

        # Evaluate gradient + Hessian
        idxrange = (idx-length(y)):(idx-1)
        Dj = Diagonal(view(d, idxrange))
        rj = view(r, idxrange)
        ∇L .+= X' * Dj * rj
        ∇²L .+= X' * Dj * X
      end
      # idxpen = (1+intercept):nvars
      # ∇L[idxpen] .= ∇L[idxpen] - 2*v.rho*(sum(w)*β[idxpen] - Γ[idxpen,:]*w)
      # ∇²L[idxpen,idxpen] .= ∇²L[idxpen,idxpen] + 2*v.rho*sum(w)*I
      ∇L .= ∇L - 2*v.rho*(sum(w)*β - Γ*w)
      ∇²L .= ∇²L + 2*v.rho*sum(w)*I
      # ∇G = ForwardDiff.gradient(beta -> G(beta, v.index, Γ, w, v.rho, vmod, tobs), β)
      # ∇²G = -ForwardDiff.hessian(beta -> G(beta, v.index, Γ, w, v.rho, vmod, tobs), β)

      # @show findall(isnan, D.diag)
      # @show v.index
      # display(∇²L)
      # Compute the search direction
      search_direction .= Symmetric(∇²L) \ ∇L
      # @show norm(∇L), norm(search_direction)

      # Backtracking line search
      t = 1.0
      v.beta_new .= β
      for step in 0:backtrack
        # if iter == 1 break end
        @. v.beta_new = β + t * search_direction
        # @assert β === v.beta
        # objective_new = eval_surrogate(v, vmod, tobs)
        objective_new = eval_surrogate(v.beta_new, v.index, Γ, w, v.rho, vmod, tobs)
        # @show objective_new, objective
        if objective_new >= objective
          break
        elseif step < backtrack
          t /= 2
        else
          @show t, objective_new, objective
          error("Backtracking failed at iteration $(iter) for vertex $(v.index) after $(step) attempts!")
          v.beta_new .= β
          break
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
    # rel = abs(logl - logl_prev) / (1 + abs(logl_prev))
    # @show iter, logl, logl_prev, rel
  end
  @show norm(∇L)
  return iter, tobs, vmod
end

function predict(X, S, vmod, tri)
  id2vertex = Dict{Int,Int}(id => j for (j, id) in enumerate(each_solid_vertex(tri)))
  points = get_points(tri)
  yhat = zeros(size(X, 1))
  @views for i in axes(X, 1)
    s = S[:, i]
    x = X[i, :]
    j, k, l = find_triangle(tri, s) |> sort
    p, q, r = points[j], points[k], points[l]
    j, k, l = id2vertex[j], id2vertex[k], id2vertex[l]
    v, u, w = vmod[j], vmod[k], vmod[l]
    a, b, c = barycentric(s, [p q r])
    yhat[i] = 
      GLM.linkinv(v.link, a*dot(x, v.beta)) +
      GLM.linkinv(u.link, b*dot(x, u.beta)) +
      GLM.linkinv(w.link, c*dot(x, w.beta))
  end
  return yhat
end

export TriangleObs, create_triobs_sets,
  VertexModel, create_vertexmodel_set

end

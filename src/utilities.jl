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
#
# TRIANGULATION
#
# check if triangle has vertex with index j as one of its vertices
has_vertex(T::TriangleObs, j::Int) = j in T.triangle

function is_inside_mesh(Δ::Triangulation, s)
  triangle = find_triangle(Δ, s, concavity_protection = true)
 return all(>(0), triangle)
end

is_inside_mesh(Δ::Triangulation) = Base.Fix1(is_inside_mesh, Δ)
#
# LINESEARCH
#
function linesearch!(βₙ₊₁, βₙ, Δ, iter, objective, backtrack, index, gamma, weights, rho, vmod, tobs, penalty, caches)
  # Backtracking line search
  t = 1.0
  for step in 0:backtrack
    @. βₙ₊₁ = βₙ + t * Δ
    objective_new = eval_surrogate(βₙ₊₁, index, gamma, weights, rho, vmod, tobs, penalty, caches)
    if (objective_new >= objective + objective_new * 1e-6) || isapprox(objective_new, objective, atol = sqrt(eps()))
      break
    elseif step < backtrack
      t /= 2
    else
      @show t, objective_new, objective
      @warn("Backtracking failed at iteration $(iter) for vertex $(index) after $(step) attempts!")
      @. βₙ₊₁ = βₙ
      break
    end
  end
  return t
end
#
# LOG-LIKELIHOOD
#
function eval_and_cache_loglikelihoods!(caches, v::VertexModel, tobs)
  j, β, η, μ = v.index, v.beta, v.eta, v.mu
  family, link = v.family, v.link
  i_start = 1
  for triidx in v.triangles
    T = tobs[triidx]
    y, X = T.y, T.X
    t = get_triangle_vertices(T, j)
    n = length(y)
    logl = view(caches.logf[triidx], t, :)
    idxrange = i_start:(i_start+n-1)
    mul!(view(η, idxrange), X, β) # η = Xβ
    for (i, idx) in enumerate(idxrange)
      μ[idx] = meanfun(link, η[idx])
      logfᵢ = GLMUtilities.log_likelihood(y[i], μ[idx], η[idx], family, link)
      logl[i] = logfᵢ
    end
    i_start += n
  end
  return caches
end

function eval_and_cache_loglikelihoods!(caches, vmod, tobs; nchunks::Int = Threads.nthreads())
  workitr = OhMyThreads.ChannelLike(vmod)
  BLAS_THREADS = BLAS.get_num_threads()
  try
    BLAS.set_num_threads(max(1, div(BLAS_THREADS, nchunks)))
    OhMyThreads.@localize caches vmod tobs OhMyThreads.tforeach(1:nchunks; chunking = false) do _
      map(workitr) do v
        eval_and_cache_loglikelihoods!(caches, v, tobs)
      end
    end
  finally
    BLAS.set_num_threads(BLAS_THREADS)
  end
end

function eval_loglikelihoodA(tobs, vmod, penalty_type)
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

function eval_loglikelihoodB(tobs, vmod, penalty_type, caches; nchunks::Int = Threads.nthreads())
  eval_and_cache_loglikelihoods!(caches, vmod, tobs; nchunks)

  logl = OhMyThreads.@localize tobs caches OhMyThreads.@tasks for triidx in eachindex(tobs)
    OhMyThreads.@set begin
      ntasks = nchunks
      reducer = +
      outputtype = Float64
    end
    T = tobs[triidx]
    A = T.A
    cache = caches.logf[triidx]
    local logl = zero(Float64)
    for i in axes(A, 2)
      a = view(A, :, i)
      logf = view(cache, :, i)
      tmp = stable_logsumexp(a, logf)
      logl += tmp
    end
    logl
  end

  penalty = eval_penalty(penalty_type, vmod)

  return sum(logl) + penalty
end

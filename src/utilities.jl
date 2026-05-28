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
# BLAS TOOLS
#
macro safe_blas(code_block, options...)
  if isempty(options)
    nchunks = Threads.nthreads()
  else
    for option in options
      if Meta.isexpr(option, :(=)) && isequal(option.args[1], :nchunks)
        nchunks = option.args[2]
      end
    end
  end
  expr = quote
    BLAS_THREADS = BLAS.get_num_threads()
    BLAS.set_num_threads(max(1, div(BLAS_THREADS, $nchunks)))
    try
      $(code_block)
    finally
      BLAS.set_num_threads(BLAS_THREADS)
    end
  end
  return esc(expr)
end
#
# TRIANGULATION
#
function is_inside_mesh(Δ::Triangulation, s)
  triangle = find_triangle(Δ, s, concavity_protection = true)
 return all(>(0), triangle)
end

is_inside_mesh(Δ::Triangulation) = Base.Fix1(is_inside_mesh, Δ)
#
# DESCENT + LINESEARCH
#
function is_approx_decrease(f_new, f_old, rtol)
  return f_new <= f_old || isapprox(f_new, f_old, rtol = rtol)
end

function linesearch!(f, βₙ₊₁, βₙ, Δ, backtrack)
  # Backtracking line search
  t = 1.0
  objective = f(βₙ)
  for step in 0:backtrack
    @. βₙ₊₁ = βₙ - t * Δ
    objective_new = f(βₙ₊₁)
    if is_approx_decrease(objective_new, objective, sqrt(eps()))
      break
    elseif step < backtrack
      t /= 2
    else
      @warn("Backtracking failed after $(step) attempts!")
      @. βₙ₊₁ = βₙ
      break
    end
  end
  return t
end
#
# MISC
#
macro get_triple(A, i)
  A = esc(A)
  i = esc(i)
  return quote ($A[1,$i], $A[2,$i], $A[3,$i]) end
end
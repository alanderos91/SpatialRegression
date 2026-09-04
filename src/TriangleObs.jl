"""
    TriangleObs(triangle, idx, y, X, S, A, V)

Represents a collection of observations, `(y, X, S)`, that lie within a `triangle`.

Users should create `TriangleObs` instances using `create_triangle_obs()` instead of
invoking this type's constructors directly.

The following invariants hold for instances of `TriangleObs`:

- `triangle` is a tuple `(j,k,l)` of vertex indices in ascending order (`j < k < l`).
- `idx` contains indices into the parent arrays of `(y, X, S)`.
"""
struct TriangleObs
  triangle::Tuple{Int,Int,Int} # vertex indices (j,k,l) in ascending order
  idx::Vector{Int}    # set of indices into original data (y, X)
  y::Vector{Float64}  # response local to triangle
  X::Matrix{Float64}  # covariates local to triangle, stored in rows
  S::Matrix{Float64}  # Cartesian coordiantes, stored in columns
  A::Matrix{Float64}  # mixture weights as barycentric coordinates, stored in columns
  V::Matrix{Float64}  # vertices of triangle, stored in columns

  function TriangleObs(
    triangle::NTuple{3,Int},
    idx::Vector{Int},
    y::Vector{Float64},
    X::Matrix{Float64},
    S::Matrix{Float64},
    A::Matrix{Float64},
    V::Matrix{Float64}
  )
    #
    @assert triangle[1] < triangle[2] < triangle[3] "Encountered violation of triangle invariant j < k < l"
    @assert length(y) == length(idx)
    @assert length(y) == size(X, 1)
    @assert length(y) == size(S, 2)
    @assert length(y) == size(A, 2)
    @assert size(V) == (2, 3)
    new(triangle, idx, y, X, S, A, V)
  end
end

function Base.show(io::IO, tobs::TriangleObs)
  T = tobs.triangle
  n, p = size(tobs.X)
  print(io, "Triangle $(T)\n")
  print(io, "  number of samples: $(n)\n")
  print(io, "  number of features: $(p)\n")
end

struct TriangleObsHelper
  idx::Vector{Int}
  loc::Vector{NTuple{2,Float64}}
  id2loc::Dict{Int,Int}
end

function create_triobs_helper(tri)
  # Get points + mapping to solid vertices.
  # The mapping is needed to ensure vertex label = index into some array
  idx = each_solid_vertex(tri) |> collect |> sort
  loc = Vector{NTuple{2,Float64}}(undef, length(idx))
  id2loc = Dict{Int,Int}()

  # Matrix of vertices, stored along columns
  for (j, id) in enumerate(idx)
    v = DelaunayTriangulation.get_point(tri, id)
    loc[j] = (v[1], v[2])
    id2loc[id] = j
  end
  return TriangleObsHelper(idx, loc, id2loc)
end

"""
    create_triobs_set(y, X, S, tri::Triangulation; nchunks::Int = Threads.nthreads())

Create a `Vector` of `TriangleObs` instances from the spatial data `(y, X, S)` over mesh `tri`.

This effectively assigns each observation to a unique triangle within triangulation `tri`.

# Arguments

- `y`: The dependent variable or response, given as an `n` by `1` vector-like object.
- `X`: The independent variables or predictors, given as an `n` by `p` matrix-like object.
- `S`: The spatial (Cartesian) coordinates for each observation, given as an `2` by `n` matrix-like object.
- `tri`: A `Triangulation` implementing a mesh for a spatial domain.
 
# Optional

- `nchunks`: The number of parallel tasks to use in assigning each observation to a triangle.
  The default references the result of `Threads.nthreads()`.
"""
function create_triobs_set(y, X, S, tri::Triangulation; nchunks::Int = Threads.nthreads())
  helper = create_triobs_helper(tri)
  triobs = _create_triobs_set_with_helper_(helper, y, X, S, tri, nchunks)
  return triobs
end

function _assign_data_to_triangles(tri, id2vertex, S, nchunks)
  TriType = Tuple{Int,Int,Int}
  IdxType = Vector{Int}
  cases = ChannelLike(axes(S, 2))

  # Build mapping in parallel; don't care about sorting keys or indices yet.
  tmp = tmap(Dict{TriType,IdxType}, 1:nchunks; chunking = false) do _
    local tri2idx = Dict{TriType,IdxType}()
    for i in cases
      # Check whether i-th case lies inside any triangle of the mesh
      if iszero(i)
        triidx = (0, 0, 0)
      else
        s = view(S, :, i)
        j, k, l = find_triangle(tri, s, concavity_protection = true)
        triidx = sort((id2vertex[j], id2vertex[k], id2vertex[l]))
      end

      # Associate index i to triangle key (j,k,l)
      if !haskey(tri2idx, triidx)
        tri2idx[triidx] = Int[]
      end
      push!(tri2idx[triidx], i)
    end
    return tri2idx
  end
  
  # Aggregate results across tasks
  tri2idx = Dict{TriType,IdxType}()
  mergewith!(union!, tri2idx, tmp...)
  nonempty_triangles = length(keys(tri2idx))

  return nonempty_triangles, tri2idx
end

function _create_triobs_set_with_helper_(helper, y, X, S, tri, nchunks)
  vertex = helper.loc
  id2loc = helper.id2loc

  # Assign each observation to a triangle in parallel
  nonempty_triangles, tri2idx = _assign_data_to_triangles(tri, id2loc, S, nchunks)

  # Create TriangleObs for each 'active' triangle
  triobs = Vector{TriangleObs}(undef, nonempty_triangles)
  for (i, (triidx, idx)) in enumerate(tri2idx)
    sort!(idx) # mitigate random access patterns as much as possible
    Vₜ = [vertex[ijk][i] for i in 1:2, ijk in triidx]
    Sₜ = S[:, idx]
    Aₜ = barycentric(Sₜ, Vₜ)
    triobs[i] = TriangleObs(triidx, idx, y[idx], X[idx, :], Sₜ, Aₜ, Vₜ)
  end
  return triobs
end

"""
    get_triangle_vertices(Tobs::TriangleObs, index, vmod)

Match `index` to one of `(j, k, l)` and retrieve the vertices `(vⱼ, vₖ, vₗ)`.

Returns `pos` as one of `1`, `2`, or `3` along with a tuple of vertices.
"""
function get_triangle_vertices(Tobs::TriangleObs, index, vmod)
  j, k, l = Tobs.triangle
  pos = get_triangle_vertices(Tobs, index)
  return pos, (vmod[j], vmod[k], vmod[l])
end

function get_triangle_vertices(Tobs::TriangleObs, index)
  j, k, l = Tobs.triangle
  if index == j
    pos = 1
  elseif index == k
    pos = 2
  elseif index == l
    pos = 3
  else
    error("The index $(index) is not in the triangle $(Tobs.triangle).")
  end
  return pos
end

# check if triangle has vertex with index j as one of its vertices
has_vertex(T::TriangleObs, j::Int) = j in T.triangle

struct TriangleObsIterator{T}
  triobs::Vector{TriangleObs}
  subset::T
end

TriangleObsIterator(triobs::Vector{TriangleObs}) = TriangleObsIterator(triobs, eachindex(triobs))
TriangleObsIterator(triobs::Vector{TriangleObs}, subset::Base.OneTo{Int}) = TriangleObsIterator(triobs, UnitRange(subset))

ChunkSplitters.is_chunkable(::TriangleObsIterator) = true

Base.length(itr::TriangleObsIterator) = length(itr.subset)

function Base.eltype(itr::TriangleObsIterator)
  triobs = itr.triobs
  return Tuple{typeof(first(triobs).triangle), eltype(triobs)}
end

function Base.iterate(itr::TriangleObsIterator, state = 1)
  if state > length(itr.subset)
    return nothing
  else
    k = itr.subset[state]
    triobs = itr.triobs[k]
    triidx = triobs.triangle
    item = (triidx, triobs)
    return (item, state + 1)
  end
end

Base.firstindex(::TriangleObsIterator) = 1
Base.lastindex(itr::TriangleObsIterator) = length(itr.subset)
Base.view(itr::TriangleObsIterator, idx::UnitRange) = TriangleObsIterator(itr.triobs, @views itr.subset[idx])
Base.view(itr::TriangleObsIterator, idx::StepRange) = TriangleObsIterator(itr.triobs, @views itr.subset[idx])

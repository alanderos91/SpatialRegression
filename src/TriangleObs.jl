struct TriangleObs
  triangle::Tuple{Int,Int,Int} # indices in ascending order
  idx::Vector{Int}    # set of indices into original data (y, X)
  y::Vector{Float64}  # response local to triangle
  X::Matrix{Float64}  # covariates local to triangle
  A::Matrix{Float64}  # mixture weights as barycentric coordinates
  V::Matrix{Float64}  # vertices of triangle, stored in columns
end

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
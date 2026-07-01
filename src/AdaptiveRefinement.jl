module AdaptiveRefinement
using DelaunayTriangulation, Distances, NearestNeighbors

struct SizingField{NNT <: NNTree}
  knn::NNT
end

function build_sizing_field(data;
  metric::PreMetric = Euclidean(),
  kwargs...)
  # KDTree should suffice for 2D and 3D
  nnt = KDTree(data, metric)
  return SizingField(nnt)
end

function derive_lower_bound(nnt)
  _, d = allnn(nnt)
  mask = @. !iszero(d)
  return minimum(d[mask])
end

function (field::SizingField)(x, y)
  # compute distance to nearest point in the data
  _, dist = nn(field.knn, [x, y])
  return dist*dist
end

struct AdaptiveDataConstraint{F,T}
  field::F
  min_area::T
  scaling::T
end

function build_constraint(data;
  min_area::Real = 0.0,
  scaling::Real = 0.05,
  kwargs...)

  # Sanity check on sizing field parameters
  @assert min_area >= 0
  @assert scaling >= 0

  # Build SizingField from data
  field = build_sizing_field(data; kwargs...)

  # No minimum area => derive a lower bound from the data
  if iszero(min_area)
    lb = derive_lower_bound(field.knn)
  else
    lb = min_area
  end

  # Default scaling to scale of coordinates in data
  T = NearestNeighbors.dist_type_internal(field.knn)
  _min_area = T(lb)
  _scaling = T(scaling)

  return AdaptiveDataConstraint(field, _min_area, _scaling)
end

# this is what DelaunayTriangulation.jl sees
function (f::AdaptiveDataConstraint)(tri, T)
  adaptive_data_constraint(tri, T, f.field, f.min_area, f.scaling)
end

# implementation details
function adaptive_data_constraint(tri, T, field, min_area, scaling)
  i, j, k = triangle_vertices(T)
  p, q, r = get_point(tri, i, j, k)
  area = DelaunayTriangulation.triangle_area(p, q, r)
  cx, cy = DelaunayTriangulation.triangle_circumcenter(p, q, r, area)

  maximum_area = min_area + scaling * field(cx, cy)
  return area >= maximum_area
end

export SizingField, AdaptiveDataConstraint

end # end module
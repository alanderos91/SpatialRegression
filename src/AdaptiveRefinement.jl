module AdaptiveRefinement
using DelaunayTriangulation, Distances, NearestNeighbors
using StaticArrays

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
  return pi*minimum(d[mask])^2
end

function (field::SizingField)(x, y)
  # compute distance to nearest point in the data
  _, dist = nn(field.knn, SA[x, y])
  return dist*dist
end

struct AdaptiveDataConstraint{F,T}
  field::F
  lb::T
  scaling::T
end

function build_constraint(data;
  lb::Real = 0.0,
  scaling::Real = 0.05,
  kwargs...)

  # Sanity check on sizing field parameters
  @assert lb >= 0
  @assert scaling >= 0

  field = build_sizing_field(data; kwargs...)
  build_constraint(field; lb, scaling)
end

function build_constraint(field::SizingField;
  lb::Real = 0.0,
  scaling::Real = 0.05)

  # Sanity check on sizing field parameters
  @assert lb >= 0
  @assert scaling >= 0

  # No minimum area => derive a lower bound from the data
  T = NearestNeighbors.dist_type_internal(field.knn)
  if iszero(lb)
    _lb = T(derive_lower_bound(field.knn))
  else
    _lb = T(lb)
  end
  _scaling = T(scaling)

  return AdaptiveDataConstraint(field, _lb, _scaling)
end

# this is what DelaunayTriangulation.jl sees
function (f::AdaptiveDataConstraint)(tri, T)
  adaptive_data_constraint(tri, T, f.field, f.lb, f.scaling)
end

# implementation details
function adaptive_data_constraint(tri, T, field, lb, scaling)
  i, j, k = triangle_vertices(T)
  p, q, r = get_point(tri, i, j, k)
  area = DelaunayTriangulation.triangle_area(p, q, r)
  cx, cy = DelaunayTriangulation.triangle_centroid(p, q, r)

  maximum_area = lb + scaling * pi * field(cx, cy)
  return area >= maximum_area
end

export SizingField, AdaptiveDataConstraint

end # end module
module AdaptiveRefinement
using DelaunayTriangulation, Distances, NearestNeighbors
using StaticArrays
using DelaunayTriangulation: AbstractParametricCurve

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

function create_preprocess_mask(data, knn, cutoff)
  keep = trues(size(data, 2))
  for (i, s) in enumerate(eachcol(data))
    if !keep[i] continue end
    neighbors = inrange(knn, s, cutoff)
    for idx in neighbors
      if idx > i keep[idx] = false end
    end
  end
  return keep
end

struct MaxEdgeConstraint{T<:Real}
  max_edge::T
end

function build_max_edge_constraint(; max_edge::Real = Inf, kwargs...)
  return MaxEdgeConstraint(max_edge)
end

function (f::MaxEdgeConstraint)(tri, T)
  max_edge_constraint(tri, T, f.max_edge)
end

function max_edge_constraint(tri, T, max_edge)
  # get edges
  i, j, k = triangle_vertices(T)
  p, q, r = get_point(tri, i, j, k)

  # get edge lengths
  e1 = DelaunayTriangulation.dist(p, q)
  e2 = DelaunayTriangulation.dist(p, r)
  e3 = DelaunayTriangulation.dist(q, r)

  return max(e1, e2, e3) > max_edge
end

_format_boundary_info_(::Nothing) = nothing
_format_boundary_info_(bd::Vector{AbstractParametricCurve}) = [[c] for c in bd]
_format_boundary_info_(bd) = bd

function _sign(x, tol = eps(typeof(x)))
  absx = abs(x)
  return ifelse(absx < tol, zero(x), x / absx)
end

function _get_convex_hull_(points)
  ch = DelaunayTriangulation.convex_hull(points)
  return map(i -> copy(ch.points[i]), ch.vertices)
end

function _expand_convex_hull!_(points, offset)
  # assumes points represents a convex hull
  for i in eachindex(points)
    @. points[i] += offset * _sign(points[i])
  end
end

function check_mesh_options(opt::T) where T
  @assert hasfield(T, :offset)
  @assert hasfield(T, :max_edge)
  @assert hasfield(T, :cutoff)

  if opt.offset < 0
    throw(ArgumentError("The `offest` must be non-negative. Use `offset = 0.0` to use Von Neumann boundary conditions."))
  end

  if opt.max_edge <= 0
    throw(ArgumentError("The `max_edge` limit must be positive. Use `max_edge = Inf` to ignore maximum edge constraint."))
  end

  return nothing
end

function default_mesh_options()
  return (;
    offset = 0.0,
    max_edge = Inf,
    min_angle = 21.0,
    cutoff = 1e-12,
  )
end

function _expand_and_refine!_(mesh, opts, original_boundary)
  if opts.offset > 0
    # Extend domain with a buffer region
    # inner_index = DelaunayTriangulation.get_all_boundary_nodes(mesh)
    inner_index = Int[]
    for edge in DelaunayTriangulation.get_all_segments(mesh)
      for index in edge
        push!(inner_index, index)
      end
    end
    inner_boundary = Vector{Vector{Float64}}(undef, length(inner_index))
    points = DelaunayTriangulation.get_points(mesh)
    for (i, index) in enumerate(inner_index)
      inner_boundary[i] = copy(points[index])
    end
    ch_points = _get_convex_hull_(inner_boundary)
    S = hcat(ch_points...)
    mask = create_preprocess_mask(S, KDTree(S), opts.cutoff)
    ch_points = ch_points[mask]
    _expand_convex_hull!_(ch_points, opts.offset)
    mesh = DelaunayTriangulation.triangulate(
      unique!([deepcopy(points); ch_points]),
      segments = original_boundary,
    )
  end
  ours = (:offset, :cutoff, :max_edge)
  itr = (key => opts[key] for key in keys(opts) if !(key in ours))
  refine_kwargs = (; itr...)

  mesh = DelaunayTriangulation.refine!(mesh;
    custom_constraint = build_max_edge_constraint(; opts...),
    refine_kwargs...
  )

  return mesh
end

function build_mesh(;
    data = nothing,
    boundary = nothing,
    inner = nothing,
    outer = nothing,
    kwargs...,
  )
  # Sanity Checks
  if isnothing(data) && isnothing(boundary)
    throw(ArgumentError("At least one of `data` or `boundary` must be specified."))
  end
  # Now we know there *something* to triangulate.
  maybe_data = something(data, Vector{Float64}[])
  if !isempty(maybe_data) maybe_data = deepcopy(maybe_data) end
  dtboundary = _format_boundary_info_(boundary)

  # Validate mesh options: default behavior is von Neumann boundary conditions
  if isnothing(inner)
    iopt = default_mesh_options()
  else
    iopt = (; default_mesh_options()..., inner...)
  end
  if isnothing(outer)
    oopt = iopt
  else
    oopt = (; default_mesh_options()..., outer...)
  end
  check_mesh_options(iopt)
  check_mesh_options(oopt)

  # First pass: get boundary data for domain
  mesh1 = DelaunayTriangulation.triangulate(maybe_data;
    boundary_nodes = dtboundary,
    kwargs...,
  )
  original_boundary = DelaunayTriangulation.get_all_segments(mesh1) |> deepcopy

  # Second pass: refine the inner domain
  mesh2 = _expand_and_refine!_(mesh1, iopt, original_boundary)

  # Third pass: triangulate full domain
  mesh3 = _expand_and_refine!_(mesh2, oopt, original_boundary)

  return mesh3
end

export SizingField, AdaptiveDataConstraint

end # end module
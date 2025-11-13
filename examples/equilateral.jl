import Pkg
Pkg.activate(".")

using LinearAlgebra, Random, Statistics, Distributions
using DelaunayTriangulation
using SpatialRegression
using CairoMakie

# Vertices of equilateral triangle centered at (0, 0), inscribed in unit circle
# Create boundary points of triangle area; use CCW orientation
boundary_points = [[0.0, 1.0], [-sqrt(3)/2, -1/2], [sqrt(3)/2, -1/2]]

function symmetric_refinement(Δ)
  stats = statistics(Δ)
  N = stats.num_solid_vertices
  Δnew = refine!(deepcopy(Δ),
    min_area = stats.smallest_area / 4 - sqrt(eps()),
    max_area = stats.largest_area / 4 + sqrt(eps()),
    min_angle = 30.0,
    max_angle = 30.0,
  )
  DelaunayTriangulation.delete_ghost_triangles!(Δnew)
  return Δnew
end

Δ₀ = triangulate_convex(boundary_points, [1, 2, 3]; delete_ghosts = true)
triplot(Δ₀)

Δ₁ = symmetric_refinement(Δ₀)
triplot(Δ₁)

Δ₂ = symmetric_refinement(Δ₁)
triplot(Δ₂)

# Sample from the standard 2-simplex, then convert to Cartesian coords
n = 10 ^ 2
A = rand(Dirichlet(ones(3)), n)


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
    # min_area = stats.smallest_area / 4 - sqrt(eps()),
    max_area = stats.largest_area / 3 + sqrt(eps()),
    # min_angle = 30.0,
    # max_angle = 30.0,
  )
  return Δnew
end

Δ₀ = triangulate_convex(boundary_points, [1, 2, 3]; delete_ghosts = true)
triplot(Δ₀)

Δ₁ = symmetric_refinement(Δ₀)
triplot(Δ₁)

Δ₂ = symmetric_refinement(Δ₁)
triplot(Δ₂)

function simulate_data(n, Δ)
  # Sample from the standard 2-simplex, then convert to Cartesian coords
  A = rand(Dirichlet(ones(3)), n)
  V = [0 1; -sqrt(3)/2 -1/2; sqrt(3)/2 -1/2] |> Transpose |> Matrix{Float64}
  S = SpatialRegression.cartesian(A, V)

  # Simulate responses at each location according to true model
  X = randn(n, 2); X[:, 1] .= 1
  B = hcat([ [1.0, j*randn()] for j in 1:statistics(Δ).num_solid_vertices]...)
  y = zeros(n)
  for i in 1:n
    s = @views S[:, i]
    j, k, l = find_triangle(Δ, s) |> sort
    v1, v2, v3 = Δ.points[[j,k,l]]
    b1, b2, b3 = @views begin B[:, j], B[:, k], B[:, l] end
    a1, a2, a3 = SpatialRegression.barycentric(s, [v1 v2 v3])
    b = a1*b1 + a2*b2 + a3*b3
    y[i] = @views dot(X[i,:], b) + 1/sqrt(n) * randn()
  end
  return y, X, S, B
end

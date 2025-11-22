import Pkg
Pkg.activate("examples")

using LinearAlgebra, Random, Statistics, Distributions
using DelaunayTriangulation
using SpatialRegression
using CairoMakie
using DataFrames
using SpatialRegression.GLM

# Vertices of equilateral triangle centered at (0, 0), inscribed in unit circle
# Create boundary points of triangle area; use CCW orientation
boundary_points = [[0.0, 1.0], [-sqrt(3)/2, -1/2], [sqrt(3)/2, -1/2]]
init_area = 0.5 *
  norm(boundary_points[1] - 0.5*(boundary_points[2] + boundary_points[3])) *
  norm(boundary_points[3] - boundary_points[2])

function equilateral_refinement(Δ, max_area)
  stats = statistics(Δ)
  N = stats.num_solid_vertices
  Δnew = refine!(deepcopy(Δ),
    # min_area = stats.smallest_area / 4 - sqrt(eps()),
    max_area = max_area,
  )
  return Δnew
end

function btest(n, p, Δ)
  A = rand(Dirichlet(ones(3)), n)
  V = [0 1; -sqrt(3)/2 -1/2; sqrt(3)/2 -1/2] |> Transpose |> Matrix{Float64}
  S = SpatialRegression.cartesian(A, V)
  Bfun(x, y, k, n) = sin(4*(x+k/n*pi/8)) + cos(6*y)
  points = [[get_point(Δ, v)...] for v in each_solid_vertex(Δ)]
  B = [log10(j+1)*Bfun(s[1], s[2], j, p) for j in 1:p, s in points]
end

function simulate_data(n, p, Δ)
  # Sample from the standard 2-simplex, then convert to Cartesian coords
  A = rand(Dirichlet(ones(3)), n)
  V = [0 1; -sqrt(3)/2 -1/2; sqrt(3)/2 -1/2] |> Transpose |> Matrix{Float64}
  S = SpatialRegression.cartesian(A, V)

  # Simulate responses at each location according to true model
  Bfun(x, y, k, n) = sin(4*(x+k/n*pi/8)) + cos(6*y)
  points = [[get_point(Δ, v)...] for v in each_solid_vertex(Δ)]
  id2vertex = Dict{Int,Int}(id => j for (j, id) in enumerate(each_solid_vertex(Δ)))
  X = 1/p * randn(n, p); X[:, 1] .= 1
  B = [sqrt(j)*Bfun(s[1], s[2], j, p) for j in 1:p, s in points]
  y = zeros(n)
  for i in 1:n
    s = @views S[:, i]
    j, k, l = find_triangle(Δ, s) |> sort
    j, k, l = id2vertex[j], id2vertex[k], id2vertex[l]
    v1, v2, v3 = points[[j,k,l]]
    b1, b2, b3 = @views begin B[:, j], B[:, k], B[:, l] end
    a1, a2, a3 = SpatialRegression.barycentric(s, [v1 v2 v3])
    mixture = @views MixtureModel(
      [
        Normal(dot(X[i, :], b1), 0.5),
        Normal(dot(X[i, :], b2), 0.5),
        Normal(dot(X[i, :], b3), 0.5),
      ],
      [a1, a2, a3]
    )
    y[i] = rand(mixture)
  end
  return y, X, S, B
end

#
# TRIANGULATIONS
#
function get_tri_title(Δ)
  stats = statistics(Δ)
  return "$(stats.num_solid_triangles) triangles, $(stats.num_solid_vertices) vertices"
end

Δ₀ = triangulate_convex(boundary_points, [1, 2, 3]; delete_ghosts = false)
Δ₁ = equilateral_refinement(Δ₀, 0.01*init_area)
Δ₂ = equilateral_refinement(Δ₁, 0.001*init_area)
Δs = (Δ₀, Δ₁, Δ₂)

figtri = Figure(size = (400*length(Δs), 400))
for (j, Δ) in enumerate(Δs)
  ax = Axis(figtri[1,j], title = get_tri_title(Δ))
  triplot!(ax, Δ)
end
figtri

#
# SCENARIO 1: Uniform over domain
#
n, p = 10^4, 10

results1 = DataFrame(
  vertices = Int[],
  triangles = Int[],
  niter = Int[],
  objective = Float64[],
  time = Float64[],
  rmse_beta = Float64[],
  rmse_resp = Float64[],
)

fig1 = Figure(size = (400*length(Δs), 400))
for (j, Δ) in enumerate(Δs)
  Random.seed!(1903)
  stats = statistics(Δ)
  local y, X, S, B0 = simulate_data(n, p, Δ)

  ax = Axis(fig1[1,2*(j-1)+1], title = get_tri_title(Δ))
  plotdata = scatter!(ax, S[1,:], S[2,:], color = y, markersize = 4.0)
  Colorbar(fig1[1,2*j], plotdata, label = "Response, Y")

  timed_result = @timed SpatialRegression.fitmodel(y, X, S, Δ;
    family = Normal(),
    link = IdentityLink(),
    rho = 1.0,
    tol = 1e-6,
    backtrack = 100,
    maxiter = 10^4
  )
  println()
  niter, tobs, vmod = timed_result.value
  logl = SpatialRegression.eval_loglikelihood(tobs, vmod)
  timing = timed_result.time  
  B = hcat([v.beta for v in vmod]...)
  yhat = SpatialRegression.predict(X, S, vmod, Δ)
  rmse_beta = sqrt(mean(abs2, B - B0))
  rmse_resp = sqrt(mean(abs2, y - yhat))
  push!(results1,
    (length(vmod), stats.num_solid_triangles, niter, logl, timing, rmse_beta, rmse_resp)
  )
end
fig1
display(results1)



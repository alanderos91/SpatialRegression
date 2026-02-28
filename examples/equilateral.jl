import Pkg
Pkg.activate(".")

using LinearAlgebra, Random, Statistics, Distributions
using DelaunayTriangulation
using SpatialRegression
using CairoMakie
using DataFrames, PrettyTables
using SpatialRegression.GLM
using SpatialRegression: L2Squared, L1Approx

function equilateral_refinement(Δ, max_area)
  Δnew = refine!(deepcopy(Δ),
    max_area = max_area,
  )
  return Δnew
end

function create_component(family::Normal, link, η)
  σ = scale(family)
  μ = GLM.linkinv(link, η)
  return Normal(μ, σ)
end

function create_component(::Binomial, link, η)
  return Bernoulli(GLM.linkinv(link, η))
end

function create_component(::Poisson, link, η)
  return Poisson(GLM.linkinv(link, η))
end

function simulate_data(n, p, Δ, family, link)
  # Sample from the standard 2-simplex, then convert to Cartesian coords
  A = rand(Dirichlet(ones(3)), n)
  V = [0 1; -sqrt(3)/2 -1/2; sqrt(3)/2 -1/2] |> Transpose |> Matrix{Float64}
  S = SpatialRegression.cartesian(A, V)

  # Simulate responses at each location according to true model
  Bfun(x, y, k, n) = (1 - (1 - 2*(x+1/2*cos(pi*k/n)))^2) * (1 - (1 - 2*(y+1/2 + 1/2*sin(pi*k/n)))^2)
  vertices = [[get_point(Δ, v)...] for v in each_solid_vertex(Δ)]
  points = get_points(Δ)
  id2vertex = Dict{Int,Int}(id => j for (j, id) in enumerate(each_solid_vertex(Δ)))
  X = 1/p*rand(Uniform(-1, 1), n, p)
  X[:, 1] .= 1
  B = [Bfun(s[1], s[2], j-1, p) for j in 1:p, s in vertices]
  if family isa Poisson
    B[1, :] .= rand(Uniform(3, 4), size(B, 2))
    B[2:end, :] .*= 1/4
  else
    B[1, :] .= rand(Uniform(-3, 3), size(B, 2))
  end
  y = zeros(n)
  for i in 1:n
    s = @views S[:, i]
    j, k, l = find_triangle(Δ, s; concavity_protection = true) |> sort
    v1, v2, v3 = points[[j,k,l]]
    j, k, l = id2vertex[j], id2vertex[k], id2vertex[l]
    b1, b2, b3 = @views begin B[:, j], B[:, k], B[:, l] end
    a1, a2, a3 = SpatialRegression.barycentric(s, [v1 v2 v3])
    mixture = @views MixtureModel(
      [
        create_component(family, link, dot(X[i, :], b1)),
        create_component(family, link, dot(X[i, :], b2)),
        create_component(family, link, dot(X[i, :], b3)),
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

#
# PLOTS
#
function plot_compare_fitted(instances; markersize = 4.0, kwargs...)
  local W = 400; H = 250
  local NROW = 2; NSCENARIO = length(instances)

  # Initialize figure with column labels
  fig = Figure(size = (W * NSCENARIO, H * NROW))
  Label(fig[2,1], "Observed", font = :bold, rotation = pi/2, tellheight = false)
  Label(fig[3,1], "Predicted", font = :bold, rotation = pi/2, tellheight = false)
  cr = Observable((0.0, 1.0))
  for (j, prob) in enumerate(instances)
    # Retrieve information about the instance
    y, yhat, Δ, S = prob.response, prob.prediction, prob.triangulation, prob.location

    # Update minimum and maximum values for color scale
    cr[] = (min(minimum(y), minimum(yhat)), max(maximum(y), maximum(yhat)))

    # Add a row label with triangulation characteristics
    Label(fig[1,1+j], get_tri_title(Δ), font = :bold, tellwidth = false)

    # Add a panel for observed data
    ax = Axis(fig[2,1+j])
    scatter!(ax, S[1,:], S[2,:]; color = y, colorrange = cr, markersize, kwargs...)

    # Predicted
    ax = Axis(fig[3,1+j])
    scatter!(ax, S[1,:], S[2,:]; color = yhat, colorrange = cr, markersize, kwargs...)
  end
  Colorbar(fig[2:3, 2+NSCENARIO], colorrange = cr, vertical = true)
  return fig
end

#
# MISC
#
function init_table()
  return DataFrame(
    family = String[],
    vertices = Int[],
    triangles = Int[],
    niter = Int[],
    objective = Float64[],
    time = Float64[],
    rmse_beta = Float64[],
    rmse_resp = Float64[],
  )
end

function run_benchmark!(generate_data, results, Δ, family, link, seed, rho, penalty)
  # Sample from data generating function
  Random.seed!(seed)
  local y, X, S, B0 = generate_data(Δ)
  BACKTRACK = 100
  TOL = 1e-5

  # Precompile
  @timed SpatialRegression.fitmodel(y, X, S, Δ;
    family = family,
    link = link,
    penalty = penalty,
    rho = rho,
    tol = TOL,
    backtrack = BACKTRACK,
    maxiter = 10,
    nchunks = Threads.nthreads()
  )

  # Fit a model with our MM algorithm
  timed_result = @timed SpatialRegression.fitmodel(y, X, S, Δ;
    family = family,
    link = link,
    penalty = penalty,
    rho = rho,
    tol = TOL,
    backtrack = BACKTRACK,
    maxiter = 10^3,
    nchunks = Threads.nthreads()
  )

  # Collect results and write to DataFrame
  stats = statistics(Δ)
  niter, tobs, vmod, logl = timed_result.value
  timing = timed_result.time  
  B = hcat([v.beta for v in vmod]...)
  yhat = SpatialRegression.predict(X, S, vmod, Δ)
  rmse_beta = sqrt(mean(abs2, B - B0))
  rmse_resp = sqrt(mean(abs2, y - yhat))
  push!(results,
    (
      family |> typeof |> nameof |> string,
      length(vmod), stats.num_solid_triangles, 
      niter, logl, timing, rmse_beta, rmse_resp
    )
  )

  # Return information about this instance
  info = (;
    response = y,
    features = X,
    location = S,
    triangulation = Δ,
    prediction = yhat,
    niter, logl, timing, rmse_beta, rmse_resp,
  )
  return info
end

function run_benchmarks(scenario, seed)
  results = init_table()
  instances = NamedTuple[]
  for Δ in scenario.triangulations
    prob = run_benchmark!(scenario.data, results, Δ, scenario.family, scenario.link, seed, scenario.rho, scenario.penalty)
    push!(instances, prob)
  end
  return results, instances
end

function default_equilateral_domain()
  # Vertices of equilateral triangle centered at (0, 0), inscribed in unit circle
  # Create boundary points of triangle area; use CCW orientation
  return [[0.0, 1.0], [-sqrt(3)/2, -1/2], [sqrt(3)/2, -1/2]]
end

function init_triangulation(boundary_points = default_equilateral_domain())
  triangulate_convex(boundary_points, [1, 2, 3]; delete_ghosts = false)
end

function main()
  boundary_points = default_equilateral_domain()
  init_area = 0.5 *
  norm(boundary_points[1] - 0.5*(boundary_points[2] + boundary_points[3])) *
  norm(boundary_points[3] - boundary_points[2])

  Random.seed!(1903)
  Δ₀ = init_triangulation(boundary_points)
  Δ₁ = equilateral_refinement(Δ₀, 0.01*init_area)
  Δ₂ = equilateral_refinement(Δ₁, 0.001*init_area)
  Δs = (Δ₀, Δ₁, Δ₂);

  figtri = Figure(size = (400*length(Δs), 400));
  for (j, Δ) in enumerate(Δs)
    ax = Axis(figtri[1,j], title = get_tri_title(Δ))
    triplot!(ax, Δ)
  end
  figtri
  save("Figure-Meshes.pdf", figtri)

  # RUN BENCHMARKS
  n, p = 10^4, 10
  RHO = 5e-2
  seed = 1903
  penalties = (
    ("Ridge", L2Squared()),
    ("L1Smooth", L1Approx(sqrt(1e-8))),
  )

  for (penalty_name, penalty) in penalties
    scenarios = [
      #
      # SCENARIO 1: Uniform over domain, Normal response
      #
      (;
        name    = "Balanced_Normal",
        data    = Δ -> simulate_data(n, p, Δ, Normal(0.0, 0.1), IdentityLink()),
        family  = Normal(),
        link    = IdentityLink(),
        rho     = RHO,
        triangulations = Δs,
        penalty = penalty,
      ),
      #
      # SCENARIO 2: Uniform over domain, Binomial response
      #
      (;
        name    = "Balanced_Binomial",
        data    = Δ -> simulate_data(n, p, Δ, Binomial(), LogitLink()),
        family  = Binomial(),
        link    = LogitLink(),
        rho     = RHO,
        triangulations = Δs,
        penalty = penalty,
      ),
      #
      # SCENARIO 3: Uniform over domain, Poisson response
      #
      (;
        name    = "Balanced_Poisson",
        data    = Δ -> simulate_data(n, p, Δ, Poisson(), LogLink()),
        family  = Poisson(),
        link    = LogLink(),
        rho     = RHO,
        triangulations = Δs,
        penalty = penalty,
      ),
    ];

    fig = Figure[]
    tbl = DataFrame[]
    ins = []
    for scenario in scenarios
      results, instances = run_benchmarks(scenario, seed)
      figure = plot_compare_fitted(instances)
      save("Figure-$(scenario.name)-$(penalty_name).pdf", figure)
      push!(tbl, results)
      push!(fig, figure)
      push!(ins, instances)
    end

    foreach(display, fig)
    foreach(display, tbl)

    open("Table-Balanced-$(penalty_name).txt", "w") do io
      pretty_table(io, vcat(tbl...); backend = :latex)
    end
  end
end

# main()


import Pkg
Pkg.activate(".")

"""
Benchmarks over an equilateral spatial domain.

Main functions:

- `plot_meshes()`: Plot meshes used in the benchmarks.
- `get_meshes()`: Retrieve meshes used in the benchmarks.
- `run_balanced()`: Run benchmarks. Spatial data are sampled uniformly over domain.

"""
module Equilateral

using LinearAlgebra, Random, Statistics, Distributions, GLM
using DelaunayTriangulation
using SpatialRegression
using CairoMakie
using DataFrames, PrettyTables
using JLD2
using SpatialRegression: L2Squared, L1Approx, AsymmetricLaplace, ALD

function equilateral_refinement(Δ, max_area)
  Δnew = refine!(deepcopy(Δ),
    max_area = max_area,
  )
  return Δnew
end

function simulate_data(n, p, Δ, family, link)
  # Sample from the standard 2-simplex, then convert to Cartesian coords
  A = rand(Dirichlet(ones(3)), n)
  V = [0 1; -sqrt(3)/2 -1/2; sqrt(3)/2 -1/2] |> Transpose |> Matrix{Float64}
  S = SpatialRegression.cartesian(A, V)

  # Simulate responses at each location according to true model
  Bfun(x, y) = -10*sin(7*π*(x + 0.15)) + 10*sin(7*π*y)
  indices = each_solid_vertex(Δ) |> collect |> sort
  vertices = [[get_point(Δ, v)...] for v in indices]
  points = get_points(Δ)
  id2vertex = Dict{Int,Int}(id => j for (j, id) in enumerate(indices))
  X = 1/p*rand(Uniform(-1, 1), n, p)
  X[:, 1] .= 1
  B = [1//8 * Bfun(s[1], s[2]) for j in 1:p, s in vertices]
  if family isa Poisson
    # B[1, :] .= rand(Uniform(3, 4), size(B, 2))
    # B[2:end, :] .*= 1/4
    B .+= 3
  else
    # B[1, :] .= rand(Uniform(-3, 3), size(B, 2))
  end
  if family isa Normal
    Φ = rand(1*Beta(2.0, 5.0), length(vertices))
  elseif family isa AsymmetricLaplace
    Φ = rand(1*Beta(2.0, 10.0), length(vertices))
  else
    Φ = ones(length(vertices))
  end
  y = zeros(n)
  H = MixtureModel[]
  for i in 1:n
    s = @views S[:, i]
    j, k, l = find_triangle(Δ, s; concavity_protection = true) |> sort
    v1, v2, v3 = points[[j,k,l]]
    j, k, l = id2vertex[j], id2vertex[k], id2vertex[l]
    b1, b2, b3 = @views begin B[:, j], B[:, k], B[:, l] end
    a1, a2, a3 = SpatialRegression.barycentric(s, [v1 v2 v3])
    φ1, φ2, φ3 = Φ[j], Φ[k], Φ[l]
    mixture = @views MixtureModel(
      [
        SpatialRegression.create_component(family, link, dot(X[i, :], b1), φ1),
        SpatialRegression.create_component(family, link, dot(X[i, :], b2), φ2),
        SpatialRegression.create_component(family, link, dot(X[i, :], b3), φ3),
      ],
      [a1, a2, a3]
    )
    y[i] = rand(mixture)
    push!(H, mixture)
  end
  return y, X, S, B, Φ, H
end
#
# TRIANGULATIONS
#
function default_equilateral_domain()
  # Vertices of equilateral triangle centered at (0, 0), inscribed in unit circle
  # Create boundary points of triangle area; use CCW orientation
  return [[0.0, 1.0], [-sqrt(3)/2, -1/2], [sqrt(3)/2, -1/2]]
end

function init_triangulation(boundary_points = default_equilateral_domain())
  triangulate_convex(boundary_points, [1, 2, 3]; delete_ghosts = false)
end

function mesh_identifier(Δ)
  stats = statistics(Δ)
  return "mesh-$(stats.num_solid_vertices)-$(stats.num_solid_triangles)"
end

function get_meshes()
  boundary_points = default_equilateral_domain()
  init_area = 0.5 *
  norm(boundary_points[1] - 0.5*(boundary_points[2] + boundary_points[3])) *
  norm(boundary_points[3] - boundary_points[2])

  Random.seed!(1903)
  Δ₀ = init_triangulation(boundary_points)
  Δ₁ = equilateral_refinement(Δ₀, 0.01*init_area)
  Δ₂ = equilateral_refinement(Δ₁, 0.001*init_area)
  
  return (Δ₀, Δ₁, Δ₂)
end
#
# PLOTS
#
function get_tri_title(Δ)
  stats = statistics(Δ)
  return "$(stats.num_solid_triangles) triangles, $(stats.num_solid_vertices) vertices"
end

function plot_meshes()
  Δs = get_meshes()
  figtri = Figure(size = (400*length(Δs), 400));
  for (j, Δ) in enumerate(Δs)
    ax = Axis(figtri[1,j], title = get_tri_title(Δ))
    triplot!(ax, Δ)
  end
  save(joinpath("figures", "Equilateral-Meshes.pdf"), figtri)
  return nothing
end

function plot_fitted(datasets; markersize = 4.0, kwargs...)
  global MODEL_DIR
  sort!(datasets, by=extract_mesh_triangles)
  model_basenames = @. basename(datasets) |> splitext |> first
  model_filenames = mapreduce(fn -> filter(contains(fn), readdir(MODEL_DIR)), union, model_basenames)

  local W = 400
  local H = 400
  local NCOL = length(model_basenames)
  local NROW = div(length(model_filenames), NCOL)

  # Initialize figure with column labels
  fig = Figure(size = (W * NCOL, H * NROW))
  Label(fig[2,1], "Observed", font = :bold, rotation = pi/2, tellheight = false)

  cmap = :berlin
  cr = Observable((0.0, 1.0))
  for (j, dataset_name) in enumerate(model_basenames)
    # Retrieve data
    y, X, S, mesh = load_data(dataset_name*".jld2", "y", "X", "S", "mesh")

    # Update minimum and maximum values for color scale
    cr[] = (minimum(y), maximum(y))

    # Add a row label with triangulation characteristics
    Label(fig[1,1+j], get_tri_title(mesh), font = :bold, tellwidth = false)

    # Add a panel for observed data
    ax = Axis(fig[2,1+j])
    scatter!(ax, S[1,:], S[2,:]; color = y, colormap = cmap, colorrange = cr, markersize, kwargs...)

    # Predictions
    matching_models = filter(contains(dataset_name), model_filenames)
    sort!(matching_models, by=extract_penalty)
    for (k, model_name) in enumerate(matching_models)
      model, metadata = load_model(model_name, "model", "metadata")
      if j == 1
        Label(fig[2+k,1], metadata.penalty_name, font = :bold, rotation = pi/2, tellheight = false)
      end
      yhat = SpatialRegression.predict(X, S, model, mesh; kind = :mean)
      cr[] = (minimum(yhat), maximum(yhat))
      ax = Axis(fig[2+k,1+j])
      scatter!(ax, S[1,:], S[2,:]; color = yhat, colormap = cmap, colorrange = cr, markersize, kwargs...)
    end
  end
  Colorbar(fig[2:2+NROW, 2+NCOL], colormap = cmap, colorrange = cr, vertical = true)
  return fig
end
#
# MISC
#
get_penalty_name(::L2Squared) = "Ridge"
get_penalty_name(::L1Approx) = "L1Smooth"

function run_benchmark(scenario_name, Δ, y, X, S, family, link, penalty, opts)
  # Precompile
  @timed SpatialRegression.fitmodel(VertexGLM, y, X, S, Δ;
    family    = family,
    link      = link,
    penalty   = penalty,
    nu        = opts.nu,
    rho       = opts.rho,
    tol       = opts.tol,
    backtrack = opts.backtrack,
    maxiter   = 10,
    nchunks   = opts.nthreads,
  )

  # Fit a model with our MM algorithm
  timed_result = @timed SpatialRegression.fitmodel(VertexGLM, y, X, S, Δ;
    family    = family,
    link      = link,
    penalty   = penalty,
    nu        = opts.nu,
    rho       = opts.rho,
    tol       = opts.tol,
    backtrack = opts.backtrack,
    maxiter   = opts.maxiter,
    nchunks   = opts.nthreads,
  )

  # Collect results and save
  stats  = statistics(Δ)
  niter, model, loss = timed_result.value
  timing = timed_result.time
  metadata = (;
    scenario  = scenario_name,
    dims      = size(X),
    vertices  = stats.num_solid_vertices,
    triangles = stats.num_solid_triangles,
    mesh_id   = mesh_identifier(Δ),
    family, link, penalty,
    penalty_name = get_penalty_name(penalty),
    opts...,
    niter, loss, timing,
  )

  return model, metadata
end
#
# DATA GENERATION & PERSISTENCE
#
const DATA_DIR  = joinpath("data")
const MODEL_DIR = joinpath("models")
const FIG_DIR   = joinpath("figures")

function instantiate_directories()
  global DATA_DIR
  global MODEL_DIR
  global FIG_DIR
  for relative_dir in (DATA_DIR, MODEL_DIR, FIG_DIR)
    !isdir(relative_dir) && mkdir(relative_dir)
  end
end

function save_data(filename, args...)
  global DATA_DIR
  JLD2.save(joinpath(DATA_DIR, filename), args...)
end

function load_data(filename, args...)
  global DATA_DIR
  JLD2.load(joinpath(DATA_DIR, filename), args...)
end

function filename_data(name, (n, p), mesh_id, seed)
  "$(name)_dims-$(n)-$(p)_$(mesh_id)_seed-$(seed).jld2"
end

function retrieve_data(scenario, Δ, seed)
  filename = filename_data(
    scenario.name, scenario.dims, mesh_identifier(Δ), seed
  )
  if !isfile(joinpath(DATA_DIR, filename))
    let
      Random.seed!(seed)
      y, X, S, B, Φ, _ = scenario.data(Δ)
      save_data(
        filename,
        "y",  y,
        "X",  X,
        "S",  S,
        "B",  B,
        "Φ",  Φ,
        "mesh", Δ,
      )
    end
  end
  return load_data(filename, "y", "X", "S")
end

function save_model(model, metadata)
  global MODEL_DIR
  filename = filename_model(
    metadata.scenario, metadata.dims, metadata.mesh_id, metadata.seed, metadata.penalty_name
  )
  JLD2.save(joinpath(MODEL_DIR, filename),
    "model",    model,
    "metadata", metadata,
  )
end

function load_model(filename, args...)
  global MODEL_DIR
  JLD2.load(joinpath(MODEL_DIR, filename), args...)
end

function filename_model(name, (n, p), mesh_id, seed, penalty)
  "$(name)_dims-$(n)-$(p)_$(mesh_id)_seed-$(seed)_$(penalty).jld2"
end

function extract_mesh_triangles(filename)
  bn = basename(filename) |> splitext |> first
  parts = split(bn, '_')
  mesh_id = parts[findfirst(contains("mesh"), parts)]
  return parse(Int, split(mesh_id, '-')[3])
end

function extract_penalty(filename)
  bn = basename(filename) |> splitext |> first
  penalty_name = split(bn, '_') |> last
  return penalty_name == "Ridge" ? 1 : 2
end
#
# MAIN FUNCTIONS
#
function run_balanced(; bench=true, plot=true)
  instantiate_directories()

  n, p  = 4*10^4, 1
  rho   = 1.0
  nu    = 1.0
  tol   = 1e-6
  seed  = 1903
  maxiter   = 2*10^3
  nthreads  = 4
  backtrack = 100
  penalties = (L2Squared(), L1Approx(sqrt(1e-16)))

  scenarios = [
    #
    # SCENARIO 1: Uniform over domain, Normal response
    #
    (;
      name    = "Equilateral-Uniform-Normal",
      dims    = (n, p),
      data    = Δ -> simulate_data(n, p, Δ, Normal(), IdentityLink()),
      family  = Normal(),
      link    = IdentityLink(),
    ),
    #
    # SCENARIO 2: Uniform over domain, Binomial response
    #
    (;
      name    = "Equilateral-Uniform-Binomial",
      dims    = (n, p),
      data    = Δ -> simulate_data(n, p, Δ, Binomial(), LogitLink()),
      family  = Binomial(),
      link    = LogitLink(),
    ),
    #
    # SCENARIO 3: Uniform over domain, Poisson response
    #
    (;
      name    = "Equilateral-Uniform-Poisson",
      dims    = (n, p),
      data    = Δ -> simulate_data(n, p, Δ, Poisson(), LogLink()),
      family  = Poisson(),
      link    = LogLink(),
    ),
  ];

  meshes = get_meshes()
  opts = (; rho, nu, tol, maxiter, nthreads, backtrack)

  bench && for scenario in scenarios, Δ in meshes
    stats = statistics(Δ)
    vertices = stats.num_solid_vertices
    triangles = stats.num_solid_triangles
    y, X, S = retrieve_data(scenario, Δ, seed)
    for penalty in penalties
      @info "Running $(scenario.name)..." penalty = get_penalty_name(penalty) vertices triangles
      model, _metadata = run_benchmark(scenario.name, Δ, y, X, S, scenario.family, scenario.link, penalty, opts)
      metadata = (; _metadata..., seed)
      @info "Saving model..."
      save_model(model, metadata)
    end
  end

  plot && for scenario in scenarios
    global DATA_DIR
    global FIG_DIR
    datasets = filter(contains(scenario.name), readdir(DATA_DIR))
    fig = plot_fitted(datasets)
    save(joinpath(FIG_DIR, scenario.name*".pdf"), fig)
  end
end
end # module

#
# MAIN
#
if !isinteractive()
  Equilateral.plot_meshes()
  Equilateral.run_balanced()
end

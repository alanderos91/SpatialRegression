import Pkg
Pkg.activate(".")

module Horseshoe
# Constants for common arguments
const DEF_R0      = 0.1
const DEF_R       = 0.5
const DEF_L       = 3.0
const DEF_NTHETA  = 20

# Constants for figures
const DEFAULT_WIDTH = 400
const DEFAULT_HEIGHT = 300
const DEFAULT_FONTSIZE = 12

const FIG_DIR   = joinpath("figures")

using Distributions, Random
using DelaunayTriangulation
using SpatialRegression
using SpatialRegression: AdaptiveRefinement
using CairoMakie

DCategorical = Distributions.Categorical # to avoid annoying conflict with Makie

function boundary(; r0=DEF_R0, r=DEF_R, l=DEF_L, ntheta=DEF_NTHETA)
  rr = r + (r - r0)

  # First outer quarter-circle
  θ = range(pi, pi/2, length=ntheta)
  x = rr .* cos.(θ)
  y = rr .* sin.(θ)

  # Tube / arm section
  θ = range(pi/2, -pi/2, length=2*ntheta)
  x = vcat(x, (r - r0) .* cos.(θ) .+ l)
  y = vcat(y, (r - r0) .* sin.(θ) .+ r)

  # Inner quarter-circle
  θ = range(pi/2, pi, length=ntheta)
  x = vcat(x, r0 .* cos.(θ))
  y = vcat(y, r0 .* sin.(θ))

  # Reflect across x-axis to complete boundary
  x = vcat(x, reverse(x[1:end-1]))
  y = vcat(y, -reverse(y[1:end-1]))

  return (;x=x, y=y)
end

function boundary_curve(; r0=DEF_R0, r=DEF_R, l=DEF_L)
  rr = r + (r - r0)
  c0 = (0.0, 0.0)
  cu = (l, +r)
  cl = (l, -r)

  boundary_nodes = [
      LineSegment((  l, +rr), (0.0, +rr), l),
      CircularArc((0.0, +rr), (0.0, -rr), c0; positive=true),
      LineSegment((0.0, -rr), (  l, -rr), l),
      CircularArc((  l, -rr), (  l, -r0), cl; positive=true),
      LineSegment((  l, -r0), (0.0, -r0), l),
      CircularArc((0.0, -r0), (0.0, +r0), c0; positive=false),
      LineSegment((0.0, +r0), (  l, +r0), l),
      CircularArc((  l, +r0), (  l, +rr), cu),
  ]

  return boundary_nodes
end

function test(x, y; r0=DEF_R0, r=DEF_R, l=DEF_L, b=1.0)
  L = π * r / 2

  if x >= 0 && y > 0
    a = L + x
    d = y - r
  elseif x >= 0 && y <= 0
    a = -L - x
    d = -r - y
  else
    a = -atan(y/x) * r
    d = hypot(x, y) - r
  end

  outside = abs(d) > (r - r0) || (x > l && (x - l)^2 + d^2 > (r - r0)^2)

  return outside ? Inf : b * a + d^2
end

function sample(n; r0=DEF_R0, r=DEF_R, l=DEF_L)
  function sample_arms(l, r, rho)
    upper = rand(Bool)
    x = rand(Uniform(0, l))
    y = upper ? r : -r
    y += rand(Uniform(-rho, rho))
    return (x, y)
  end

  function sample_bend(r, rho)
    θ = π/2 + rand()*π
    Rin = r - rho
    Rout = r + rho
    R = sqrt(rand() * (Rout^2 - Rin^2) + Rin^2)
    x = R*cos(θ)
    y = R*sin(θ)
    return (x, y)
  end

  function sample_caps(l, r, rho)
    upper = rand(Bool)
    θ = rand()*π - π/2
    R = rho * sqrt(rand())
    x = l + R*cos(θ)
    y = upper ? r : -r
    y += R*sin(θ)
    return (x, y)
  end

  rho = r - r0
  probs = [
    2*2*rho*l,
    π/2 * ((r+rho)^2 - (r-rho)^2),
    π*rho^2, 
  ]
  total_area = sum(probs)
  probs .= probs / total_area
  Z = DCategorical(probs)

  S = zeros(2, n)
  for i in axes(S, 2)
    k = rand(Z)
    x, y = if k == 1
      sample_arms(l, r, rho)
    elseif k == 2
      sample_bend(r, rho)
    elseif k == 3
      sample_caps(l, r, rho)
    end
    S[1,i], S[2,i] = x, y
  end
  return S
end

function simulate_data(n; kwargs...)
  S = Horseshoe.sample(n; kwargs...)
  y = map(coord -> test(coord[1], coord[2]), eachcol(S))
  X = ones(n, 1)
  return y, X, S
end

function get_mesh(; data=nothing, r0=DEF_R0, r=DEF_R, l=DEF_L, kwargs...)
  nodes = Horseshoe.boundary_curve(; r0, r, l)
  init_points = [[l, +r], [0.0, +r], [(-0.1 + -0.9)/2, 0.0], [0.0, -r], [l, -r]]
  if isnothing(data)
    points = init_points
  else
    points = [init_points; data]
  end
  return triangulate(Vector{Float64}[]; boundary_nodes = [[r] for r in nodes], kwargs...)
end

function add_mesh_subplot!(ax, mesh, data)
  triplot!(ax, mesh)
  scatter!(ax, data[1,:], data[2,:], color = :red, marker = :x)
  hidespines!(ax)
  hidedecorations!(ax)
  return nothing
end

function get_test_points(mesh; m::Int=300, n::Int=150)
  xmin, xmax = -1, maximum(first, DelaunayTriangulation.get_points(mesh))
  ymin, ymax = -1, 1
  xm = range(xmin, xmax, length=m);
  ym = range(ymin, ymax, length=n);
  Sm = hcat([[xm[i], ym[j]] for i in eachindex(xm) for j in eachindex(ym)]...);
  idx = findall(>(0), [DelaunayTriangulation.dist(mesh, s) for s in eachcol(Sm)]);
  return Sm[:,idx]
end

function add_array_labels!(fig, scaling_grid, minimum_grid, minimum_derived, fontsize)
  m, n = length(scaling_grid), length(minimum_grid)

  # Row labels
  for i in 1:m
    label_text = "Scaling = $(scaling_grid[i])"
    Label(fig[i,0], label_text, rotation=pi/2, tellheight=false, fontsize=fontsize)
  end

  # Column labels
  for j in 1:n
    if j == 1
      label_text = "Minimum = $(minimum_derived)"
    else
      label_text = "Minimum = $(minimum_grid[j])"
    end
    Label(fig[m+1,j], label_text, tellwidth=false, fontsize=fontsize)
  end
end

function make_test_meshes(n, scaling_grid, minimum_grid;
  r0::Real=DEF_R0,
  r::Real=DEF_R,
  l::Real=DEF_L,
  ntheta::Real=DEF_NTHETA,
  make_plots::Bool=true,
  w::Real=DEFAULT_WIDTH,
  h::Real=DEFAULT_HEIGHT,
  fontsize::Real=12,
  )
  # Construct an initial triangulation, then sample points within that mesh.
  init_mesh = Horseshoe.get_mesh(; ntheta, r0, r, l)
  y, X, S = Horseshoe.simulate_data(n; r0, r, l)

  # Construct the sizing field directly to avoid re-building the KDTree
  H = AdaptiveRefinement.build_sizing_field(S)
  lb = AdaptiveRefinement.derive_lower_bound(H.knn)
  lb = round(lb, sigdigits = 4)

  scaling_grid = sort(scaling_grid) |> unique!
  minimum_grid = sort!([0.0; minimum_grid]) |> unique!
  m, n = length(scaling_grid), length(minimum_grid)
  
  if make_plots
    # Initialize plots
    fontsize = fontsize*sqrt(min(m, n))

    fig_meshes = Figure(size = (w*(n+1),h*(m+1)))
    ax_main = Axis(fig_meshes[0, 1:n],
      width = Relative(0.5),
      subtitle="Initial Triangulation",
      subtitlesize=fontsize,
      aspect = DataAspect()
    )
    ax_meshes = Matrix{Axis}(undef, m, n)

    # Add the initial triangulation
    add_mesh_subplot!(ax_main, init_mesh, S)
  else
    fig_meshes = nothing
  end

  # build mesh array for benchmarks
  meshes = Matrix{Any}(undef, m, n)
  for (i, alpha) in enumerate(scaling_grid), (j, h) in enumerate(minimum_grid)
    # Use a custom constraint w/ our sizing field to refine the initial mesh.
    constraint = AdaptiveRefinement.build_constraint(H; lb=h, scaling=alpha)
    mesh = DelaunayTriangulation.refine!(deepcopy(init_mesh);
      min_angle = 30.0,
      custom_constraint=constraint
    )
    meshes[i,j] = mesh

    if make_plots
      # Add the refined triangulation
      ax_meshes[i,j] = Axis(fig_meshes[i,j])
      add_mesh_subplot!(ax_meshes[i,j], mesh, S)
    end
  end

  if make_plots
    add_array_labels!(fig_meshes, scaling_grid, minimum_grid, lb, fontsize)
    resize_to_layout!(fig_meshes)
  end

  return (;
    mesh=meshes,
    fig=make_plots ? fig_meshes : nothing,
    lb=lb,
    sizing_field=H,
    params=(; ntheta, r0, r, l,),
    scaling_grid,
    minimum_grid,
    data=(; y, X, S)
  )
end

function benchmark_adaptive_refinement(nt, penalty, common_kwargs;
  w::Real=DEFAULT_WIDTH,
  h::Real=DEFAULT_HEIGHT,
  fontsize::Real=12,
  cmap::Any=Reverse(:Spectral))
  # Retrieve grids
  scaling_grid, minimum_grid = nt.scaling_grid, nt.minimum_grid
  m, n = length(scaling_grid), length(minimum_grid)

  # Initialize plot
  fontsize = fontsize*sqrt(min(m, n))
  fig = Figure(size = (w*n, h*m))
  ax = Matrix{Axis}(undef, m, n)
  cr = Observable(extrema(nt.data.y))

  for i in eachindex(scaling_grid), j in eachindex(minimum_grid)
    # Adapt sample to the mesh, as some points near the boundary may be excluded
    mesh = nt.mesh[i,j]
    stats = DelaunayTriangulation.statistics(mesh)
    dist = [DelaunayTriangulation.dist(mesh, s) for s in eachcol(nt.data.S)]
    idx = findall(>(0), dist)
    y, X, S = nt.data.y[idx], nt.data.X[idx,:], nt.data.S[:,idx]

    # Construct a grid of test points
    Sm = Horseshoe.get_test_points(mesh)

    # fit model to the data + mesh
    niter, model, nlogl = @time SpatialRegression.fitmodel(VertexGLM, y, X, S, mesh;
      penalty=penalty,
      common_kwargs...
    )
    println("""
    Fitted $(penalty) penalized model
      - vertices: $(stats.num_solid_vertices)
      - triangles: $(stats.num_solid_triangles)
      - edges: $(stats.num_solid_edges)
      - iterations: $(niter)
      - neg-loglik: $(nlogl)
    """)
    yhat = SpatialRegression.predict(
      ones(size(Sm, 2)), Sm, model, mesh,
      kind=:mean
    )
    ax[i,j] = Axis(fig[i,j])
    scatter!(ax[i,j], Sm[1,:], Sm[2,:], color = yhat, colormap=cmap, colorrange=cr)
    scatter!(ax[i,j], S[1,:], S[2,:], color = :white, marker = :x)
    hidespines!(ax[i,j])
    hidedecorations!(ax[i,j])

    # Update minimum and maximum values for color scale
    ymin, ymax = cr[]
    cr[] = (min(minimum(yhat), ymin), max(maximum(yhat), ymax))
  end

  Colorbar(fig[0, 1:n], colormap = cmap, colorrange = cr, vertical = false, label="Predicted Value", labelsize=fontsize, ticks=LinearTicks(9))
  add_array_labels!(fig, scaling_grid, minimum_grid, nt.lb, fontsize)
  resize_to_layout!(fig)

  return fig
end

function run_benchmark_adaptive(seed=1903)
  global FIG_DIR

  Random.seed!(seed)

  # Mesh Parameters
  n = 100                           # sample size
  ntheta = 6                        # number of points on arcs in domain
  sgrid = [0.001, 0.05, 0.1, 10.0]  # scaling parameter in sizing field
  mgrid = [0.0, 1e-2, 1e-1]         # lower bound on in sizing field

  # Create the mesh array
  hs = Horseshoe.make_test_meshes(n, sgrid, mgrid; ntheta)
  save(joinpath(FIG_DIR, "Horseshoe-Meshes.pdf"), hs.fig)

  rho_grid = [1e-6, 1e-3, 1e0]
  epsilon_grid = [1e-12, 1e-2, 1e-1]

  common_kwargs = (;
    nu=1.0,
    maxiter=10^3,
    tol=1e-6,
    backtrack=20,
    verbose=false
  )

  # L2Squared benchmarks
  for rho in rho_grid
    options = (; rho, common_kwargs...)
    fig = Horseshoe.benchmark_adaptive_refinement(hs, L2Squared(), options)
    save(joinpath(FIG_DIR, "Horseshoe-Adaptive_penalty=Ridge_rho=$(rho).pdf"), fig)
  end

  # L1Smooth benchmarks
  for rho in rho_grid, epsilon in epsilon_grid
    options = (; rho, common_kwargs...)
    fig = Horseshoe.benchmark_adaptive_refinement(hs, L1Approx(epsilon), options)
    save(joinpath(FIG_DIR, "Horseshoe-Adaptive_penalty=L1Smooth_rho=$(rho)_epsilon=$(epsilon).pdf"), fig)
  end
end

end # module

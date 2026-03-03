import Pkg
Pkg.activate(".")

"""
Benchmarks for 2019 HMDA data; CONUS only.

Main functions:

- `plot_meshes()`: Plot meshes used in the benchmarks.
- `get_meshes()`: Retrieve meshes used in the benchmarks.
- `run_yu2025()`: Run benchmarks. Uses model due to Yu, Wang, and Wang (2025).

"""
module HMDA

using DataFrames, CSV, Arrow, ZipFile, PrettyTables
using Tables, TableOperations
using CategoricalArrays
using LibGEOS, Shapefile
import GeometryOps as GO
import ArchGDAL as AD

using LinearAlgebra, Random, Statistics, StatsBase, StatsModels, Distributions, GLM
using DelaunayTriangulation
using SpatialRegression
using CairoMakie
using SpatialRegression: L2Squared, L1Approx

const INCLUDED_AGE_LEVELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
const EXCLUDED_DTI_VALUES = ["Exempt", "NA"]
const INCLUDED_ETHNICITY_LEVELS = ["Not Hispanic or Latino", "Hispanic or Latino"]
const INCLUDED_RACE_LEVELS = ["White", "Asian", "Black or African American"]
const EXCLUDED_STATE_CODES = ["AK", "HI", "PR", "VI", "NA"]
const INCLUDED_SEX_LEVELS = ["Male", "Female"]

# See Table C.3 in Supplementary Materials of Yu et al 2025.
# Data: https://s3.amazonaws.com/cfpb-hmda-public/prod/snapshot-data/2019/2019_public_lar_csv.zip
# Source: https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2019
# Data Dictionary: https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields
const COLUMN_SPEC = Dict(
  "activity_year"   => Union{Missing, Int16},
  "lei"             => Union{Missing, String31},
  "derived_msa_md"  => Union{Missing, Int},
  "state_code"      => Union{Missing, String3},
  "county_code"     => Union{Missing, String7},
  "census_tract"    => Union{Missing, String15},
  "derived_dwelling_category"       => Union{Missing, String},
  "business_or_commercial_purpose"  => Union{Missing, Int16},
  "loan_term"                       => Union{Missing, String},
  "loan_type"                       => Union{Missing, Int16},
  "derived_loan_product_type"       => Union{Missing, String31},
  "applicant_ethnicity_observed"    => Union{Missing, Int16},
  "derived_ethnicity"               => Union{Missing, String31},
  "applicant_race_observed" => Union{Missing, Int16},
  "derived_race"            => Union{Missing, String},
  "applicant_sex_observed"  => Union{Missing, Int16},
  "derived_sex"             => Union{Missing, String31},
  "applicant_age"           => Union{Missing, String7},     # Need to handle categories > age 64
  "income"                  => Union{Missing, String15},    # Income, thousands of dollars
  "debt_to_income_ratio"    => Union{Missing, String15},    # DTI
  #
  "tract_population"                  => Union{Missing, Int32},   # Tpop
  "tract_minority_population_percent" => Union{Missing, Float64}, # Tminority
  "tract_to_msa_income_percentage"    => Union{Missing, Int16},   # Tincome
  "ffiec_msa_md_median_family_income" => Union{Missing, Int32},   # Mincome
  "action_taken"                      => Union{Missing, Int8},    # 1-8; 3 indicates application was denied
)
const ARCHIVE_FILENAME = joinpath("data", "2019_public_lar_csv.zip")
const CSV_FILENAME = "2019_public_lar_csv.csv"
const ARROW_FILENAME = joinpath("data", "hmda2019.arrow")
const SHAPEFILE = joinpath("data", "cb_2019_us_tract_500k.zip")
const VSIZIP_SHAPEFILE = joinpath("/vsizip", "data", "cb_2019_us_tract_500k.zip", "cb_2019_us_tract_500k.shp")
const ARROW_CLEANED = joinpath("data", "hmda2019_clean.arrow")

function empty_factor(::Type{T}, levels::Vector{T}; ordered = false) where T
  return CategoricalVector{T}(undef, 0; levels = levels, ordered = ordered)
end

function included_in_study(row)
  global INCLUDED_AGE_LEVELS, EXCLUDED_DTI_VALUES, INCLUDED_ETHNICITY_LEVELS, INCLUDED_RACE_LEVELS, EXCLUDED_STATE_CODES, INCLUDED_SEX_LEVELS
  !any(ismissing, row) &&
    row.derived_msa_md != 0 &&
    row.business_or_commercial_purpose == 2 &&
    row.derived_sex ∈ INCLUDED_SEX_LEVELS &&
    row.applicant_age ∈ INCLUDED_AGE_LEVELS &&
    !isnothing(tryparse(Int, row.debt_to_income_ratio)) && # use numeric ONLY
    row.derived_ethnicity ∈ INCLUDED_ETHNICITY_LEVELS &&
    contains(row.derived_dwelling_category, "Single Family") &&
    row.derived_race ∈ INCLUDED_RACE_LEVELS &&
    row.state_code ∉ EXCLUDED_STATE_CODES &&
    row.loan_term != "Exempt" &&
    row.action_taken ∈ (1, 3) # 1 - Loan originated; 3 - Application denied
end

function hmda2019_csv2arrow()
  global COLUMN_SPEC, ARCHIVE_FILENAME, ARROW_FILENAME, CSV_FILENAME
  archive = ZipFile.Reader(ARCHIVE_FILENAME) # make sure to clean this up!
  file = filter(x -> x.name == CSV_FILENAME, archive.files) |> first
  itr = CSV.Rows(file,
    header = true,
    select = collect(keys(COLUMN_SPEC)),
    types = COLUMN_SPEC,
    reusebuffer = false,
    missingstring = ["", "NA", "Na", "na"],
  )
  open(ARROW_FILENAME, "w") do io
    Arrow.write(io, DataFrame(itr); compress = :zstd, file = true)
  end
  close(archive)
  return ARROW_FILENAME
end

function load_conus_shapefile()
  global SHAPEFILE
  shapefile = Shapefile.Table(SHAPEFILE) |> DataFrame
  transform!(shapefile, :geometry => ByRow(GO.centroid) => [:lat, :long])
  filter!(:STATEFP => !in([02, 15, 60, 66, 69, 72, 78]), shapefile)
  return shapefile
end

function hmda2019_clean()
  global ARROW_FILENAME, ARROW_CLEANED

  # get coordinates
  shapefile = load_conus_shapefile()
  select!(shapefile, [:GEOID, :lat, :long])

  # add coordinates to HMDA
  arrow = Arrow.Table(ARROW_FILENAME)
  tmp = TableOperations.filter(included_in_study, arrow) |> DataFrame
  tmp = innerjoin(tmp, shapefile, on = [:census_tract => :GEOID])

  hmda = DataFrame()

  # RESPONSE
  # convert response to 0/1 scale, 1 - loan denied
  hmda.Denial = tmp.action_taken .== 3
  # ---------------------------------------------------

  # LOAN INFO
  hmda.LoanTerm = parse.(Int16, tmp.loan_term)
  hmda.LoanType = CategoricalArrays.recode(tmp.loan_type,
    1   => "Conventional", # baseline
    2   => "FHA",
    3:4 => "VA-USDA"
  ) |> CategoricalVector

  # APPLICANT INFO
  hmda.Ethnicity = CategoricalArrays.recode(tmp.derived_ethnicity,
    "Hispanic or Latino"      => "HL",
    "Not Hispanic or Latino"  => "NHL" # baseline
  ) |> CategoricalVector
  hmda.Race = CategoricalArrays.recode(tmp.derived_race,
    "White"                     => "White", # baseline
    "Asian"                     => "Asian",
    "Black or African American" => "AA",
  ) |> CategoricalVector
  hmda.Sex = CategoricalArrays.recode(tmp.derived_sex,
    "Male"    => "Male", # baseline
    "Female"  => "Female",  
  ) |> CategoricalVector
  hmda.Age = CategoricalArrays.recode(tmp.applicant_age,
    "<25"   => "<25", # baseline
    "25-34" => "25-34",
    "35-44" => "35-44",
    "45-54" => "45-54",
    "55-64" => "55-64",
    "65-74" => ">64",
    ">74"   => ">64",
  ) |> CategoricalVector
  hmda.Income = parse.(Int, tmp.income)
  hmda.DTI = parse.(Int, tmp.debt_to_income_ratio)

  # CENSUS TRACT DEMOGRAPHIC CHARACTERISTICS
  hmda.Tpop = tmp.tract_population
  hmda.Tminority = tmp.tract_minority_population_percent
  hmda.Tincome = tmp.tract_to_msa_income_percentage
  hmda.Mincome = tmp.ffiec_msa_md_median_family_income

  # SPATIAL COORDINATES
  hmda.Lat = tmp.lat
  hmda.Long = tmp.long

  # only non-negative income values; reporting for this and DTI is confusing
  # see for example: https://mycomplianceresource.com/forums/topic/hmda-dti/
  filter!(:Income => >=(0), hmda)

  open(ARROW_CLEANED, "w") do io
    Arrow.write(io, hmda; compress = :zstd, file = true)
  end
  return ARROW_CLEANED
end

function extract_shapefile_boundary(; tol = 0.1)
  global VSIZIP_SHAPEFILE
  
  # STATEFP corresponding to Alaska, Hawaii, US territories
  excluded = [02, 15, 60, 66, 69, 72, 78]

  # Collect geometries for census tracts
  geoms = MultiPolygon[]
  AD.read(VSIZIP_SHAPEFILE) do dataset
    layer = AD.getlayer(dataset, 0)
    for feature in layer
      key = parse(Int, AD.getfield(feature, "STATEFP"))
      if key in excluded continue end
      geom = GI.convert(LibGEOS,
        AD.forceto(AD.getgeom(feature), AD.wkbMultiPolygon)
      )
      push!(geoms, geom)
    end
  end

  # Combine into a single MultiPolygon with a cascaded union
  multipoly = geoms |> GeometryCollection |> LibGEOS.unaryUnion
  
  # Keep only the largest geometry by area; this should be the US without tiny islands
  areas = [LibGEOS.area(LibGEOS.getGeometry(multipoly, i)) for i in 1:LibGEOS.numGeometries(multipoly)]
  out = LibGEOS.topologyPreserveSimplify(LibGEOS.getGeometry(multipoly, argmax(areas)), tol)
  return out
end

function save_boundary_csv(out::Polygon, label)
  # Save a plot to check!
  fig = CairoMakie.plot(out)
  CairoMakie.save("USBoundary_$(label).pdf", fig)
  coords = LibGEOS.boundary(out) |> LibGEOS.getCoordSeq |> LibGEOS.getCoordinates
  open("USBoundary_$(label).csv", "w") do io
    CSV.write(io, [], writeheader = true, header = ["x", "y"])
    for (x, y) in coords
      CSV.write(io, (; x = [x], y = [y]), append = true)
    end
  end
end

function load_conus_triangulation(filepath)
  boundary_points = [(row[1], row[2]) for row in eachrow(CSV.read(filepath, DataFrame))]
  boundary_unique = unique(boundary_points)
  lookup = Dict(key => i for (i, key) in enumerate(boundary_unique))
  curve = map(key -> lookup[key], boundary_points)
  tri = triangulate(boundary_unique, boundary_nodes = reverse(curve))
  init_area = get_area(tri)
  DelaunayTriangulation.refine!(tri, min_angle = 30.0, max_area = 0.001 * init_area)
  DelaunayTriangulation.add_ghost_triangles!(tri)
  return tri
end

function get_meshes()
  return (
    load_conus_triangulation(joinpath("USBoundary_tol=0.5.csv")),
    load_conus_triangulation(joinpath("USBoundary_tol=0.1.csv")),
    load_conus_triangulation(joinpath("USBoundary_tol=0.01.csv"))
  )
end

function plot_meshes()
  Δs = get_meshes()
  figtri = Figure(size = (400*length(Δs), 400));
  for (j, Δ) in enumerate(Δs)
    ax = Axis(figtri[1,j], title = get_tri_title(Δ))
    triplot!(ax, Δ)
  end
  figtri
  save(joinpath("figures", "HMDA-Meshes.pdf"), figtri)
  return nothing
end

function is_inside_mesh(Δ::Triangulation, s)
  triangle = find_triangle(Δ, s, concavity_protection = true)
 return all(>(0), triangle)
end

is_inside_mesh(Δ::Triangulation) = Base.Fix1(is_inside_mesh, Δ)

function formula_yu2025()
  return @formula(0 ~
    LoanTerm + LoanType +
    Ethnicity + Race + Sex + Age +
    log(Income) + log(DTI) + log(Tpop) + 
    Tminority + log(Tincome) + log(Mincome)
  )
end

function get_modelmatrix(tbl)
  formula = formula_yu2025()
  mf = ModelFrame(formula, tbl)
  return ModelMatrix(mf)
end

function load_data()
  tbl_full = Arrow.Table(joinpath("data", "hmda2019_clean.arrow")) |> DataFrame
  tbl_full = filter(row -> row.Income > 0 && row.DTI > 0 && row.Tpop > 0 && row.Tincome > 0 && row.Mincome > 0, tbl_full)
  return tbl_full
end

function get_subsample(tbl_full, sample_pct)
  n_full = nrow(tbl_full)
  n_sample = round(Int, n_full * sample_pct)
  mask = zeros(Bool, n_full)
  mask[1:n_sample] .= 1
  shuffle!(mask)
  return tbl_full[mask, :]
end

function run_benchmark(rho, penalty; seed = 1903, sample_pct = 0.1)
  Random.seed!(seed)
  penalty_names = Dict(
    L2Squared => "Ridge",
    L1Approx => "L1Smooth",
  )
  penalty_name = penalty_names[typeof(penalty)]
  tol = 1e-5

  # Load triangulations
  Δs = get_meshes()

  # Load data and drop cases with dubious values
  tbl_full = load_data()
  
  # Sample the data to get down to a more manageable size
  tbl = get_subsample(tbl_full, sample_pct)

  # Build model from DataFrame. Categorical variables saved as CategoricalVector.
  mm = get_modelmatrix(tbl)

  # Build response, model matrix, and spatial matrix
  X_sample = mm.m
  S_sample = [tbl.Lat tbl.Long] |> Transpose |> Matrix
  y_sample = Float64.(tbl.Denial)

  results = NamedTuple[]
  for Δ in Δs
    # Make sure locations lie in this version of the mesh!
    idx = map(is_inside_mesh(Δ), eachcol(S_sample))
    y, X, S = y_sample[idx], X_sample[idx, :], S_sample[:, idx]

    # Run the benchmark
    @timed SpatialRegression.fitmodel(y, X, S, Δ;
      family = Binomial(),
      link = LogitLink(),
      rho = rho,
      tol = tol,
      backtrack = 100,
      maxiter = 10,
      penalty = penalty,
      nchunks = Threads.nthreads()
    )

    timed_result = @timed SpatialRegression.fitmodel(y, X, S, Δ;
      family = Binomial(),
      link = LogitLink(),
      rho = rho,
      tol = tol,
      backtrack = 100,
      maxiter = 10^4,
      penalty = penalty,
      nchunks = Threads.nthreads()
    )

    # Make predictions and check how we did.
    n, p = sum(idx), size(X, 2) - 1
    niter, tobs, vmod, logl = timed_result.value
    timing = timed_result.time
    yhat = SpatialRegression.predict(X, S, vmod, Δ)
    rmse_resp = sqrt(mean(abs2, y - yhat))
    result = (;
      cases = n,
      features = p,
      response = y,
      location = S,
      triangulation = Δ,
      prediction = yhat,
      niter, logl, timing, rmse_resp,
      penalty = penalty_name
    )
    push!(results, result)
  end

  return results
end

#
# TRIANGULATIONS
#
function get_tri_title(Δ)
  stats = statistics(Δ)
  return "$(stats.num_solid_triangles) triangles, $(stats.num_solid_vertices) vertices"
end

function plot_fitted(instances; markersize = 4.0, kwargs...)
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

    # Retrive the boundary
    nodes = DelaunayTriangulation.get_boundary_nodes(Δ)
    pts = DelaunayTriangulation.get_points(Δ)
    path = pts[nodes]

    # Update minimum and maximum values for color scale
    cr[] = (min(minimum(y), minimum(yhat)), max(maximum(y), maximum(yhat)))

    # Add a row label with triangulation characteristics
    Label(fig[1,1+j], get_tri_title(Δ), font = :bold, tellwidth = false)

    # Add a panel for observed data
    ax = Axis(fig[2,1+j])
    lines!(ax, path, color = :black)
    scatter!(ax, S[1,:], S[2,:]; colormap = Reverse(:redblue), color = y, colorrange = cr, markersize, kwargs...)

    # Predicted
    ax = Axis(fig[3,1+j])
    lines!(ax, path, color = :black)
    scatter!(ax, S[1,:], S[2,:]; colormap = Reverse(:redblue), color = yhat, colorrange = cr, markersize, kwargs...)
  end
  Colorbar(fig[2:3, 2+NSCENARIO], colormap = Reverse(:redblue), colorrange = cr, vertical = true)
  return fig
end

function run_yu2025()
  penalty_names = Dict(
    L2Squared => "Ridge",
    L1Approx => "L1Smooth",
  )
  for penalty in (L2Squared(), L1Approx(sqrt(1e-8)))
    results = run_benchmark(1.0, penalty; sample_pct = 0.1, seed = 1903)
    penalty_name = penalty_names[typeof(penalty)]
    plot_fitted(results; markersize = 4)
    save(joinpath("figures", "HMDA-n=10pct-$(penalty_name).pdf"), current_figure())

    tbl = DataFrame()
    for r in results
      stats = statistics(r.triangulation)
      push!(
        tbl,
        (;
          triangles = stats.num_solid_triangles,
          vertices = stats.num_solid_vertices,
          n = r.cases,
          p = r.features,
          penalty = r.penalty,
          time = r.timing / 60,   # record in minutes
          niters = r.niter,
          logl = r.logl,
          rmse_resp = r.rmse_resp,
        )
      )
    end

    open(joinpath("tables", "HMDA-n=10pct-$(penalty_name).txt"), "w") do io
      pretty_table(io, tbl; backend = :latex)
    end
  end
end
end # module
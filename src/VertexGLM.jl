struct VertexGLM{D <: Distribution, L <: Link}
  family::D                   # distribution
  link::L                     # link function, g(μ) = η
  index::Int                  # vertex index, needed for consitency with triangulation
  part::Vector{UnitRange{Int}}# partition of indices into incident triangles
  d::Vector{Float64}          # IRLS weights, determined by GLM + mixture coefficients
  beta::Vector{Float64}       # coefficients, β
  gamma::Matrix{Float64}      # matrix of local average coefficients in MM algorithm, Γ
  eta::Vector{Float64}        # linear predictor, η = xᵀβ
  mu::Vector{Float64}         # mean parameter vector, μ = g⁻¹(η)
  workres::Vector{Float64}    # working residuals, (y - μ) g'(μ)
  neighbors::Vector{Int}      # index set representing neighboring vertices in penalty
  triangles::Vector{Int}      # index set representing incident triangles
  weights::Vector{Float64}    # penalty weights, w in ∑ w P(B)
  rho::Float64                # penalty coefficient
  beta_new::Vector{Float64}   # proposed update
end

function create_vertexmodel_set(family::D, link::L, tri::Triangulation, tobs::Vector{TriangleObs}, nvars; rho::Real = 1.0) where {D,L}
  mv = VertexGLM{D,L}[]
  
  # recreate mapping from triangulation labels to our labels
  id2vertex = Dict{Int,Int}()
  for (j, id) in enumerate(each_solid_vertex(tri))
    id2vertex[id] = j
  end
  
  # This assumes triangles in tobs are labeled using OUR scheme, not the one in the triangulation.
  wmax = 0.0
  for (j, id) in enumerate(each_solid_vertex(tri))
    # count the total number of samples incident with vertex j
    nsamples, part, triangles = 0, UnitRange{Int}[], Int[]
    for k in eachindex(tobs)
      if has_vertex(tobs[k], j)
        n = length(tobs[k].y)
        push!(part, (1+nsamples):(nsamples+n))
        push!(triangles, k)
        nsamples += n
      end
    end

    # determine the number of incident vertices
    neighbors = [id2vertex[u] for u in DelaunayTriangulation.iterated_neighbourhood(tri, id, 1)]
    sort!(neighbors)
    nneighbors = length(neighbors)

    # allocate!
    d = zeros(nsamples)
    beta = zeros(nvars)
    gamma = zeros(nvars, nneighbors)
    eta = zeros(nsamples)
    mu = zeros(nsamples)
    workres = zeros(nsamples)
    weights = ones(nneighbors)

    for (k, other) in enumerate(DelaunayTriangulation.iterated_neighbourhood(tri, id, 1))
      weights[k] = 1 / DelaunayTriangulation.dist(get_point(tri, id), get_point(tri, other))
      wmax = max(wmax, weights[k])
    end

    push!(mv, VertexGLM(family, link, j, part,
        d, beta, gamma, eta, mu, workres,
        neighbors, triangles, weights, rho, similar(beta)
      )
    )
  end
  
  # sort!(mv, by = v -> v.index)
  for v in mv
    v.weights .= v.weights / wmax
  end

  return mv
end

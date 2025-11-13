module SpatialRegression

using LinearAlgebra
using StaticArrays
using Distributions
using ForwardDiff
using DelaunayTriangulation

import Distributions: loglikelihood

import GLM
using GLM: GlmResp, Link, CauchitLink, CloglogLink,
  IdentityLink, InverseLink, InverseSquareLink,
  LogitLink, LogLink, NegativeBinomialLink,
  PowerLink, ProbitLink, SqrtLink,
  canonicallink

#
# COORDINATE CONVERSIONS
#
"""
Convert Cartesian coordinates `s` to barycentric coordinates with respect
to triangle vertices in `V`. Vertices are assumed to stored in columns.
"""
barycentric(s::AbstractVector, V) = barycentric!(zeros(eltype(s), 3), s, V)

function barycentric!(out, s::AbstractVector{T}, V::AbstractMatrix{T}) where T <: AbstractFloat
  @assert length(s) == 2
  A = SMatrix{3,3,T,9}( # remember: column major
    V[1,1], V[2,1], 1, # column 1
    V[1,2], V[2,2], 1, # column 2
    V[1,3], V[2,3], 1  # column 3
  )
  b = SVector{3,T}(s[1], s[2], 1)
  out .= A \ b
  return out
end

"""
Converts Cartesian coordinates in `S` (stored in columns) to barycentric coordinates, using triangle vertices in `V`.
"""
function barycentric(S::AbstractMatrix, V)
  out = zeros(eltype(S), 3, size(S, 2))
  for (o, s) in zip(eachcol(out), eachcol(S))
    barycentric!(o, s, V)
  end
  return out
end

"""
Convert barycentric coordinates `a` with respect to triangle vertices `V` to Cartesian coordinates.
Assumes vertices are stored in columns.
"""
cartesian(a::AbstractVector, V) = cartesian!(zeros(eltype(a), 2), a, V)
cartesian!(out, a::AbstractVector{T}, V::AbstractMatrix{T}) where T <: AbstractFloat = mul!(out, V, a)

"""
Converts barycentric coordinates in `A` (stored in columns) to Cartesian coordinates, using triangle vertices in `V.`
"""
function cartesian(A::AbstractMatrix, V)
  out = zeros(eltype(A), 2, size(A, 2))
  for (o, a) in zip(eachcol(out), eachcol(A))
    cartesian!(o, a, V)
  end
  return out
end

struct TriangleObs
  triangle::Tuple{Int,Int,Int} # indices in ascending order
  y::Vector{Float64}  # response local to triangle
  X::Matrix{Float64}  # covariates local to triangle
  A::Matrix{Float64}  # mixture weights as barycentric coordinates
  V::Matrix{Float64}  # vertices of triangle, stored in columns
end

# check if triangle has vertex with index j as one of its vertices
has_vertex(T::TriangleObs, j::Int) = j in T.triangle

function create_triobs_sets(y, X, S, tri)
  stats = statistics(tri)
  vertex = tri.points

  # Matrix of vertices, stored along columns
  V = Matrix{Float64}(undef, 2, stats.num_solid_vertices)
  for j in each_solid_vertex(tri)
    V[:, j] .= vertex[j]
  end

  # Retrieve each triangle, sorting indices in ascending order
  dict = Dict{Tuple{Int,Int,Int},Vector{Int}}()
  for (i, s) in enumerate(eachcol(S))
    tri_ind = sort(find_triangle(tri, s))
    if !haskey(dict, tri_ind)
      dict[tri_ind] = Int[]
    end
    index_set = dict[tri_ind]
    push!(index_set, i)
  end

  triobs = TriangleObs[]
  k = 0
  for (tri_ind, idx) in dict
    k += 1
    display(tri_ind)
    Vₖ = V[:, [tri_ind...]]
    A = barycentric(view(S, :, idx), Vₖ)
    push!(triobs, TriangleObs(tri_ind, y[idx], X[idx, :], A, Vₖ))
  end
  return triobs
end

struct VertexModel{D <: Distribution, L <: Link}
  d::D                      # distribution
  link::L                   # link function, g(μ) = η
  v::Vector{Float64}        # vertex coordinates
  index::Int                # vertex index, needed for consitency with triangulation
  beta::Vector{Float64}     # coefficients, β
  c::Vector{Float64}        # case weights, determined by GLM + mixture coefficients
  gamma::Matrix{Float64}    # matrix of local average coefficients in MM algorithm, Γ
  eta::Vector{Float64}      # linear predictor, η = xᵀβ
  mu::Vector{Float64}       # mean parameter vector, μ = g⁻¹(η)
  workres::Vector{Float64}  # working residuals, (y - μ) g'(μ)
  w::Vector{Float64}        # penalty weights, w in ∑ w P(B)
  rho::Float64              # penalty coefficient
end

function create_vertexmodel_set(d::D, link::L, tri::Triangulation, tobs::Vector{TriangleObs}, nvars; rho::Real = 1.0) where {D,L}
  mv = VertexModel{D,L}[]
  for j in each_solid_vertex(tri)
    # count the total number of samples incident with vertex j
    nsamples = 0
    for k in eachindex(tobs)
      if has_vertex(tobs[k], j)
        nsamples += length(tobs[k].y)
      end
    end

    # determine the number of incident vertices
    nneighbors = length(DelaunayTriangulation.iterated_neighbourhood(tri, j, 1))

    # allocate!
    beta = zeros(nvars)
    weights = zeros(nsamples)
    gamma = zeros(nvars, nneighbors)
    eta = zeros(nsamples)
    mu = zeros(nsamples)
    workres = zeros(nsamples)
    w = zeros(nneighbors)

    push!(mv, VertexModel(d, link, v, j, beta, weights, gamma, eta, mu, workres, w, rho))
  end
  return mv
end

export TriangleObs, create_triobs_sets,
  VertexModel, create_vertexmodel_set

end

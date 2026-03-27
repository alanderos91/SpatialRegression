struct SpatialVertexModel{V <: AbstractVertexModel, P <: AbstractPenalty, C} <: AbstractSpatialModel
  triobs::Vector{TriangleObs}
  vertex::Vector{V}
  penalty::P
  caches::C
end

function create_model(::Type{V}, y::Vector{T}, X::Matrix{T}, S::Matrix{T}, tri::Triangulation;
  penalty::P = L2Squared(),
  nchunks::Int = Threads.nthreads(),
  kwargs...
  ) where {V <: AbstractVertexModel, T <: Real, P <: AbstractPenalty}
  #
  nvars = size(X, 2)
  triobs = create_triobs_set(y, X, S, tri; nchunks = nchunks)
  vertex = create_vertex_set(V, tri, triobs, nvars; kwargs...)
  caches = build_caches(triobs, vertex, penalty, nvars, nchunks)
  return SpatialVertexModel(triobs, vertex, penalty, caches)
end
#
# ITERATION
#
eachobsindex(triobs::TriangleObs) = eachindex(triobs.y)

eachtriobs(triobs::Vector{TriangleObs}) = TriangleObsIterator(triobs)
eachtriobs(triobs::Vector{TriangleObs}, v::AbstractVertexModel) = TriangleObsIterator(triobs, v.triangles)
eachtriobs(model::SpatialVertexModel) = eachtriobs(model.triobs)
eachtriobs(model::SpatialVertexModel, v::AbstractVertexModel) = eachtriobs(model.triobs, v.triangles)

eachvertex(model::SpatialVertexModel) = model.vertex
#
# LOSS
#
function (f::SpatialVertexModel)(rho; kwargs...)
  loss = eval_loss(f.triobs, f.vertex, f.caches; kwargs...)
  penalty = eval_penalty(f.penalty, f.vertex)
  return loss + rho*penalty
end
#
# SURROGATE
#
struct VertexSurrogate{V <: AbstractVertexModel, P <: AbstractPenalty, C}
  index::Int
  model::SpatialVertexModel{V,P,C}
  rho::Float64
end

function (g::VertexSurrogate)(beta)
  j = g.index
  v = g.model.vertex[j]
  loss = eval_loss_surrogate(beta, v, g.model.triobs, g.model.caches)
  penalty = eval_penalty_surrogate(g.model.penalty, beta, v.weights, g.model.caches.gamma[j], v.beta)
  return loss + g.rho*penalty
end
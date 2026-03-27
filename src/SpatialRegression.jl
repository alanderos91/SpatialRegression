module SpatialRegression

using LinearAlgebra
using StaticArrays
using ForwardDiff
using DelaunayTriangulation
using OhMyThreads: @localize, @tasks, @set, tmap, tforeach, ChannelLike
using ChunkSplitters

# Imports for writing custom chunkable iterators
import Base:
  iterate, length, eltype,
  firstindex, lastindex,
  view

import ChunkSplitters: is_chunkable

# Isolate Distributions + GLM code in separate module.
# Any required imports are handled in the module and available here.
include("GLMUtilities.jl")
include("utilities.jl")
using .GLMUtilities

abstract type AbstractVertexModel end
abstract type AbstractPenalty end
abstract type AbstractSpatialModel end
include("TriangleObs.jl")
include("SpatialVertexModel.jl")
include("VertexGLM.jl")
export TriangleObs,
  VertexGLM,
  SpatialVertexModel

include("stable.jl")
include("penalty.jl")
export L2Squared, L1Approx

function initialize_coefficients!(tobs, vmod)
  for v in vmod
    initialize_coefficients!(v, tobs, vmod)
  end
end

function initialize_coefficients!(v::VertexGLM, tobs, vmod)
  # TODO: Local GLM init leads to poor numerical behavior downstream.
end

function fitmodel(::Type{V}, yfull, Xfull, Sfull, tri;
    maxiter::Int = 100,
    backtrack::Int = 5,
    tol::Real = 1e-3,
    rho::Real = 1.0,
    nchunks::Int = Threads.nthreads(),
    kwargs...
    # intercept = all(isequal(1), view(Xfull, :, 1)),
  ) where V <: AbstractVertexModel
  # Initialize
  nvars = size(Xfull, 2)
  model = f = create_model(V, yfull, Xfull, Sfull, tri; nchunks, kwargs...)
  update_caches!(model; nchunks)
  nlogl = f(rho; nchunks)
  nlogl_prev = zero(nlogl)
  iter = 0
  while iter < maxiter && abs(nlogl - nlogl_prev) > (1 + abs(nlogl_prev)) * tol
    iter += 1

    # Visit each vertex once to update local parameters
    workspace = ChannelLike(model.caches.workspace)
    workitr = ChannelLike(eachvertex(model))

    @safe_blas begin
      @localize iter model rho backtrack tforeach(1:nchunks; chunking = false) do _
        map(workspace) do wrk
          for v in workitr
            g = VertexSurrogate(v.index, model, rho)
            if isempty(v.triangles)
              update_empty_case!(model.penalty, v, v.weights, model.caches)
            else
              mm_update!(model.penalty, g, v, model.triobs, wrk, model.caches, backtrack)
            end
          end
        end
      end
    end nchunks=nchunks

    # Apply all updates
    for v in eachvertex(model)
      @. v.beta = v.beta_new
    end

    # Evaluate log-likelihood
    update_caches!(model; nchunks)
    nlogl_prev = nlogl
    nlogl = f(rho; nchunks)
  end
  return iter, model, nlogl
end

function predict(X, S, vmod, tri)
  id2vertex = Dict{Int,Int}(id => j for (j, id) in enumerate(each_solid_vertex(tri)))
  points = get_points(tri)
  yhat = zeros(size(X, 1))
  @views for i in axes(X, 1)
    s = S[:, i]
    x = X[i, :]
    j, k, l = find_triangle(tri, s; concavity_protection = true) |> sort
    p, q, r = points[j], points[k], points[l]
    j, k, l = id2vertex[j], id2vertex[k], id2vertex[l]
    v, u, w = vmod[j], vmod[k], vmod[l]
    a, b, c = barycentric(s, [p[1] q[1] r[1]; p[2] q[2] r[2]])
    yhat[i] = 
      a*GLMUtilities.meanfun(v.link, dot(x, v.beta)) +
      b*GLMUtilities.meanfun(u.link, dot(x, u.beta)) +
      c*GLMUtilities.meanfun(w.link, dot(x, w.beta))
  end
  return yhat
end

end

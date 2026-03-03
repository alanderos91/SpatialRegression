module SpatialRegression

using LinearAlgebra
using StaticArrays
using ForwardDiff
using DelaunayTriangulation
using OhMyThreads

# Isolate Distributions + GLM code in separate module.
# Any required imports are handled in the module and available here.
include("GLMUtilities.jl")
using .GLMUtilities

include("TriangleObs.jl")
include("VertexModel.jl")
export TriangleObs, create_triobs_sets,
  VertexModel, create_vertexmodel_set

include("utilities.jl")
include("stable.jl")
include("penalty.jl")
include("mm_update.jl")

function initialize_coefficients!(tobs, vmod)
  for v in vmod
    initialize_coefficients!(v, tobs, vmod)
  end
end

function initialize_coefficients!(v::VertexModel, tobs, vmod)
  # TODO: Local GLM init leads to poor numerical behavior downstream.
end

function fitmodel(yfull, Xfull, Sfull, tri;
    maxiter::Int = 100,
    backtrack::Int = 5,
    tol::Real = 1e-3,
    family::UnivariateDistribution = Normal(),
    link::Link = canonicallink(family),
    rho::Real = 1.0,
    penalty::PT = L2Squared(),
    nchunks::Int = Threads.nthreads(),
    # intercept = all(isequal(1), view(Xfull, :, 1)),
  ) where PT <: AbstractPenalty
  # Initialize
  nvars = size(Xfull, 2)
  tobs = create_triobs_sets(yfull, Xfull, Sfull, tri; nchunks = 4*nchunks)
  vmod = create_vertexmodel_set(family, link, tri, tobs, nvars; rho = rho)
  # initialize_coefficients!(tobs, vmod)
  logf_cache = (;
    logf = Dict(triidx => zeros(3, length(T.y)) for (triidx, T) in enumerate(tobs)),
  )
  logl = eval_loglikelihoodB(tobs, vmod, penalty, logf_cache; nchunks = nchunks)
  logl_prev = zero(logl)

  BLAS_THREADS = BLAS.get_num_threads()
  cache = [(zeros(nvars), zeros(nvars, nvars), zeros(nvars)) for _ in 1:nchunks]

  iter = 0
  while iter < maxiter && abs(logl - logl_prev) > (1 + abs(logl_prev)) * tol
    iter += 1

    # Update local averaged estimates, γₙⱼₖ
    eval_gamma!(vmod)

    # Visit each vertex once to update local parameters
    workspace = OhMyThreads.ChannelLike(cache)
    workitr = OhMyThreads.ChannelLike(vmod)
    try
      BLAS.set_num_threads(max(1, div(BLAS_THREADS, nchunks)))
      OhMyThreads.@localize iter tobs vmod nvars OhMyThreads.tforeach(1:nchunks; chunking = false) do _
        map(workspace) do (∇L, ∇²L, search_direction)

          map(workitr) do vⱼ
            if isempty(vⱼ.triangles)
              update_empty_case!(penalty, vⱼ, vⱼ.weights, vⱼ.gamma)
            else
              objective = eval_surrogate(vⱼ.beta, vⱼ.index, vⱼ.gamma, vⱼ.weights, vⱼ.rho, vmod, tobs, penalty, logf_cache)
              if isinf(objective) || isnan(objective)
                println("Vertex ", vⱼ.index, " Iteration ", iter)
                display(vⱼ.beta)
                error("Encountered unstable objective value $(objective) at iteration $(iter).")
              end
              mm_update!(penalty, vⱼ, vmod, tobs, (∇L, ∇²L, search_direction), logf_cache)
              linesearch!(
                vⱼ.beta_new,
                vⱼ.beta,
                search_direction,
                iter,
                objective,
                backtrack,
                vⱼ.index, vⱼ.gamma, vⱼ.weights, vⱼ.rho, vmod, tobs, penalty, logf_cache
              )
            end
          end
        end
      end
    finally
      BLAS.set_num_threads(BLAS_THREADS)
    end

    # Apply all updates
    for vⱼ in vmod
      @. vⱼ.beta = vⱼ.beta_new
    end

    # Evaluate log-likelihood
    logl_prev = logl
    logl = eval_loglikelihoodB(tobs, vmod, penalty, logf_cache; nchunks = nchunks)
  end
  return iter, tobs, vmod, logl
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

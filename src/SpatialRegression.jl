module SpatialRegression

using LinearAlgebra
using StaticArrays
using Distributions
using ForwardDiff
using DelaunayTriangulation
using OhMyThreads

import Distributions: loglikelihood

import GLM
using GLM: GlmResp, Link, CauchitLink, CloglogLink,
  IdentityLink, InverseLink, InverseSquareLink,
  LogitLink, LogLink, NegativeBinomialLink,
  PowerLink, ProbitLink, SqrtLink,
  canonicallink

include("TriangleObs.jl")
include("VertexModel.jl")
export TriangleObs, create_triobs_sets,
  VertexModel, create_vertexmodel_set

include("utilities.jl")
include("stable.jl")
include("penalty.jl")
include("mm_update.jl")

function eval_loglikelihoods(v, y, x)
  return (  # TODO: Make this use BLAS; i.e. η₁, η₂, η₃ = Bᵀx with Bᵀ 3 × p
    loglik_obs(v[1], y, x),
    loglik_obs(v[2], y, x),
    loglik_obs(v[3], y, x)
  )
end

function eval_loglikelihood(tobs, vmod, penalty_type::AbstractPenalty)
  logl = zero(Float64)
  for T in tobs
    A, y, X = T.A, T.y, T.X
    _, v = get_triangle_vertices(T, T.triangle[1], vmod)
    for i in eachindex(y)
      x = view(X, i, :)
      a = view(A, :, i)

      # evaluate log-likelihood at each vertex
      logf = eval_loglikelihoods(v, y[i], x)
      
      # shift log(∑ⱼ αⱼ exp(fⱼ)) by the most negative log-likelihood
      logl += stable_logsumexp(a, logf)
    end
  end

  penalty = eval_penalty(penalty_type, vmod)

  return logl + penalty
end

function initialize_coefficients!(tobs, vmod)
  for v in vmod
    initialize_coefficients!(v, tobs, vmod)
  end
end

function initialize_coefficients!(v::VertexModel, tobs, vmod)
  p = length(v.beta)
  for triidx in v.triangles
    T = tobs[triidx]
    j, k, l = T.triangle
    
    # Solve linear system associated with triangle
    η = GLM.linkfun.(v.link, T.y)
    X = [Diagonal(T.A[1,:])*T.X Diagonal(T.A[2,:])*T.X Diagonal(T.A[3,:])*T.X]
    β = [vmod[j].beta; vmod[k].beta; vmod[l].beta]
    β .= X \ η

    # Average solutions over incident triangles for each vertex
    for (r, u) in enumerate((vmod[j], vmod[k], vmod[l]))
      u.beta .+= 1/length(u.triangles) * β[p*(r-1)+1 : p*r]
    end
  end
  return nothing
end

function fitmodel(yfull, Xfull, Sfull, tri;
    maxiter::Int = 100,
    backtrack::Int = 5,
    tol::Real = 1e-3,
    family::UnivariateDistribution = Normal(),
    link::Link = GLM.canonicallink(family),
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
  logl = eval_loglikelihood(tobs, vmod, penalty)
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
            objective = eval_surrogate(vⱼ.beta, vⱼ.index, vⱼ.gamma, vⱼ.weights, vⱼ.rho, vmod, tobs, penalty)
            isinf(objective) && display(vⱼ.beta)
            mm_update!(penalty, vⱼ, vmod, tobs, (∇L, ∇²L, search_direction))
            linesearch!(
              vⱼ.beta_new,
              vⱼ.beta,
              search_direction,
              objective,
              backtrack,
              vⱼ.index, vⱼ.gamma, vⱼ.weights, vⱼ.rho, vmod, tobs, penalty
            )
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
    logl = eval_loglikelihood(tobs, vmod, penalty)
    # if logl < logl_prev
    #   rel = abs(logl - logl_prev) / (1 + abs(logl_prev))
    #   @show iter, logl, logl_prev, rel
    #   @warn "Detected increase in log-likelihood, likely due to instability in evaluating it at current estimate."
    #   break
    # end
    # rel = abs(logl - logl_prev) / (1 + abs(logl_prev))
    # @show iter, logl, logl_prev, rel
  end
  return iter, tobs, vmod
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
      a*GLM.linkinv(v.link, dot(x, v.beta)) +
      b*GLM.linkinv(u.link, dot(x, u.beta)) +
      c*GLM.linkinv(w.link, dot(x, w.beta))
  end
  return yhat
end

end

struct ResampleHelper
  model
  X
  S
  mixtures
end

function create_resampler(model, X, S, tri)
  indices = each_solid_vertex(tri) |> collect |> sort
  id2vertex = Dict{Int,Int}(id => j for (j, id) in enumerate(indices))
  points = get_points(tri)
  mixtures = Vector{MixtureModel}(undef, size(S, 2))
  @views for i in axes(X, 1)
    s = S[:, i]
    x = X[i, :]
    mixtures[i] = assemble_mixture(x, s, model, tri, id2vertex, points)
  end
  return ResampleHelper(model, X, S, mixtures)
end

function resample(rh::ResampleHelper)
  return rand.(rh.mixtures)
end
#
# Assumes (y, X, S) are the data used to fit the model
#
function bootstrap(model::SpatialVertexModel{V}, y, X, S, tri, options;
    refit::Bool = false,
    replicates::Int = 1000,
    alpha::Real = 0.05,
    keep::Bool = false,
  ) where V <: VertexGLM
  #
  nvertices = length(model.vertex)
  helper = create_resampler(model, X, S, tri)
  beta_b = zeros(size(X, 2), nvertices, replicates)
  scale_b = zeros(1, nvertices, replicates)

  alpha_lb = 1 - alpha/2
  alpha_ub = alpha/2
  for b in 1:replicates
    # use parametric bootstrap to generate bootstrap data
    print("[$(b) / $(replicates)] resample")
    y_b = @time resample(helper)

    if refit
      # need to determine smoothing parameters w/ cross validation
      error("Not yet implemented.")
    else
      # conditional bootstrap; here be dragons
      print("[$(b) / $(replicates)] fitmodel")
      niter, model_b, nlogl = @time fitmodel(VertexGLM, y_b, X, S, tri; options...)
    end

    # obtain differences between fitted parameters and bootstrap versions
    print("[$(b) / $(replicates)] record  ")
    @time for (j, v_b) in enumerate(eachvertex(model_b))
      @. beta_b[:,j,b] = v_b.beta
      scale_b[:,j,b] .= v_b.dispersion
    end
    println()
  end
  
  # helper to create output
  init_output(x) = (;
    bias = zeros(size(x, 1), size(x, 2)),
    lb   = zeros(size(x, 1), size(x, 2)),
    ub   = zeros(size(x, 1), size(x, 2)),
    data = keep ? x : nothing,
  )
  beta = init_output(beta_b)
  scale = init_output(scale_b)
  for (j, v) in enumerate(eachvertex(model))
    itr = ((beta, beta_b, :beta), (scale, scale_b, :dispersion))
    @views for (out, boot, param_name) in itr
      print("[$(j) / $(nvertices)] $(param_name)")
      @time begin
      # estimate the bias
      theta_b = mean(boot[:,j,:], dims = 2) |> vec
      theta_est = getfield(v, param_name)
      out.bias[:,j] .= theta_b .- theta_est

      # use quantiles of residuals determine a confidence interval
      d = boot[:,j,:] .- theta_est
      out.lb[:,j] .= theta_est .- mapslices(row -> quantile(row, alpha_lb), d, dims=2) |> vec
      out.ub[:,j] .= theta_est .- mapslices(row -> quantile(row, alpha_ub), d, dims=2) |> vec
      end
    end
  end
  results = (; model, replicates, refit, alpha, beta, scale)
  return results
end

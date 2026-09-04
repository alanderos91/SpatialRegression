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
    cv_options::NamedTuple = (;),
    refit::Bool = true,
    replicates::Int = 100,
    alpha::Real = 0.05,
    keep::Bool = false,
    progress = false,
  ) where V <: VertexGLM
  #
  refit && isempty(cv_options) &&
    throw(ArgumentError("Using `refit = true` requires careful consideration of cross validation settings. Reference `?SpatialRegression.cv` and specify settings with `cv_options = (<key> = <value>, ...)`."))

  nvertices = length(model.vertex)
  helper = create_resampler(model, X, S, tri)
  beta_b = zeros(size(X, 2), nvertices, replicates)
  scale_b = zeros(1, nvertices, replicates)
  rho_ = refit ? zeros(replicates) : [model.state.rho]
  nu_ = refit ? zeros(replicates) : [model.state.nu]

  # use parametric bootstrap to generate bootstrap data
  for b in 1:replicates
    y_b = resample(helper)
    tbeg = progress ? time() : 0.0
    r_b = bootstrap_replicate(model, y_b, X, S, tri, cv_options, options, refit)
    tend = progress ? time() : 0.0
    progress && println("Replicate $(lpad(b, ndigits(replicates))) / $(replicates): $(rpad(round(tend - tbeg, sigdigits = 4), 5)) seconds")
    if refit
      rho_[b] = r_b.model.state.rho
      nu_[b] = r_b.model.state.rho
    end
    for (j, v_b) in enumerate(eachvertex(r_b.model))
      @. beta_b[:,j,b] = v_b.beta
      scale_b[:,j,b] .= v_b.dispersion
    end
  end
  # compute ci from the bootstrap data
  beta, scale = bootstrap_ci(model, beta_b, scale_b, alpha, keep)
  results = (;
    model, replicates, refit, alpha, beta, scale, rho = rho_, nu = nu_
  )
  return results
end

function bootstrap_replicate(model, y_b, X, S, tri, cv_options, options, refit)
  # get information from model object
  family, link, penalty = get_family(model), get_link(model), model.penalty
  if refit
    # need to determine smoothing parameters w/ cross validation
    results = cv(VertexGLM, y_b, X, S, tri;
      cv_options..., options..., family, link, penalty,
    )
    rho_b, nu_b = results.best_rho, results.best_nu
  else
    # conditional bootstrap; here be dragons
    rho_b, nu_b = model.state.rho, model.state.nu
  end
  # fit the model
  niter, model_b, nlogl = fitmodel(VertexGLM, y_b, X, S, tri;
    options..., family, link, penalty, rho = rho_b, nu = nu_b,
  )
  return (; model = model_b, niter, nlogl,)
end

function bootstrap_ci(model, beta_b, scale_b, alpha, keep)
  # helper to create output
  init_output(x) = (;
    bias = zeros(size(x, 1), size(x, 2)),
    lb   = zeros(size(x, 1), size(x, 2)),
    ub   = zeros(size(x, 1), size(x, 2)),
    data = keep ? x : nothing,
  )
  beta = init_output(beta_b)
  scale = init_output(scale_b)
  alpha_lb = 1 - alpha/2
  alpha_ub = alpha/2
  itr = ((beta, beta_b, :beta), (scale, scale_b, :dispersion))
  for (j, v) in enumerate(eachvertex(model))
    @views for (out, boot, param_name) in itr
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
  return beta, scale
end

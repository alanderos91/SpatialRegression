struct VertexGLM{D <: UnivariateDistribution, L <: Link} <: AbstractVertexModel
  index::Int                  # vertex index, needed for consitency with triangulation
  part::Vector{UnitRange{Int}}# partition of indices into incident triangles
  triangles::Vector{Int}      # index set representing incident triangles
  neighbors::Vector{Int}      # index set representing neighboring vertices in penalty
  weights::Vector{Float64}    # penalty weights, w in ∑ w P(B)

  nobs::Int                   # number of incident observations
  nvar::Int                   # number of variables + intercept 
  family::D                   # distribution
  link::L                     # link function, g(μ) = η
  d::Vector{Float64}          # IRLS weights, determined by GLM + mixture coefficients
  beta::Vector{Float64}       # coefficients, β
  beta_new::Vector{Float64}   # proposed update
  eta::Vector{Float64}        # linear predictor, η = xᵀβ
  mu::Vector{Float64}         # mean parameter vector, μ = g⁻¹(η)
  workres::Vector{Float64}    # working residuals, (y - μ) g'(μ)
  dispersion::Float64         # may be transformed while fitting
end

# convenience accesors to hide implementation details
get_family(v::VertexGLM) = v.family
get_link(v::VertexGLM) = v.link
needs_dispersion(v::VertexGLM) = needs_dispersion(get_family(v))

get_family(model::SpatialVertexModel{V}) where V <: VertexGLM = get_family(first(model.vertex))
get_link(model::SpatialVertexModel{V}) where V <: VertexGLM = get_link(first(model.vertex))
needs_dispersion(model::SpatialVertexModel{V}) where V <: VertexGLM = needs_dispersion(first(model.vertex))

infer_vertex_type(::Type{VertexGLM}, ::D, ::L) where {D <: UnivariateDistribution, L <: Link} = VertexGLM{D,L}

function transform_parameters!(model::SpatialVertexModel{V}) where V <: VertexGLM
  if needs_dispersion(model)
    for v in eachvertex(model)
      phi = v.dispersion
      model.vertex[v.index] = Accessors.@set v.dispersion = 1 / phi
    end
  end
end

function create_vertex(::Type{VertexGLM{D,L}}, index, part, triangles, neighbors, weights, nobs, nvar;
  family::D = Normal(),
  link::L   = IdentityLink(),
  ) where {D <: UnivariateDistribution, L <: Link}
  #
  d = zeros(nobs)
  beta = zeros(nvar)
  beta_new = similar(beta)
  eta = zeros(nobs)
  mu = zeros(nobs)
  workres = zeros(nobs)
  dispersion = 1.0
  return VertexGLM(
    index, part, triangles, neighbors, weights,
    nobs, nvar, family, link, d, beta, beta_new,
    eta, mu, workres, dispersion
  )
end
#
# CACHES
#
function build_caches(triobs::Vector{TriangleObs}, vertex::Vector{V}, ::AbstractPenalty, nvar, nchunks) where V <: VertexGLM
  return (;
    # Store log-likelihood values for each triangle
    logf = init_cache_logf(triobs),
    # matrix of local average coefficients in MM algorithm, Γ
    gamma = init_cache_gamma(vertex, nvar),
    # ∇L, ∇²L, Δ
    workspace = [(zeros(nvar), zeros(nvar, nvar), zeros(nvar)) for _ in 1:nchunks],
  )
end

function update_caches!(model::SpatialVertexModel{<:VertexGLM}; nchunks::Int = Threads.nthreads())
  workitr = ChannelLike(eachvertex(model))
  @safe_blas begin
    tforeach(1:nchunks; chunking = false) do _
      for v in workitr
        update_cache_logf!(model.caches.logf, v, model.triobs)
        update_cache_gamma!(model.caches.gamma, v, model.vertex)
      end
    end
  end nchunks=nchunks
end

function update_caches_serial!(model::SpatialVertexModel{<:VertexGLM})
  for v in eachvertex(model)
    update_cache_logf!(model.caches.logf, v, model.triobs)
    update_cache_gamma!(model.caches.gamma, v, model.vertex)
  end
end
#
# SURROGATE
#
function eval_loss_surrogate(beta::Vector, v::VertexGLM, triobs::Vector{TriangleObs}, caches, xi)
  cache = caches.logf
  logl = zero(Float64)
  phi = v.dispersion
  j, family, link = v.index, v.family, v.link
  for (triidx, T) in eachtriobs(triobs, v)
    t = get_triangle_vertices(T, j)
    F = cache[triidx]
    for i in eachobsindex(T)
      x = view(T.X, i, :)
      alpha = @get_triple T.A i
      logf = @get_triple F i

      # Evaluate log-likelihood term at β
      eta = v.nvar > 1 ? dot(x, beta) : x[1]*beta[1]
      mu = meanfun(link, eta)
      logfⱼ = GLMUtilities.logpdf(T.y[i], mu, phi, eta, family, link)

      # Evaluate terms dependent on the anchor point βₙ
      zweight = stable_convex_weight(t, alpha, logf)
      tmp = zweight < eps() ? zero(zweight) : zweight * logfⱼ + zweight * (log(alpha[t]) + log(inv(zweight)))
      logl += tmp
    end
  end
  return -logl
end
#
# UPDATES
#
function initialize_coefficients!(v::VertexGLM, triobs)
  X = similar(v.beta, length(v.d), length(v.beta))
  y = similar(v.beta, length(v.d))
  for (k, idxrange) in zip(v.triangles, v.part)
    T = triobs[k]
    X[idxrange, :] .= T.X
    y[idxrange] .= T.y
  end
  if v.family isa ALD
    fitted = glm(X, y, Normal(), v.link)
  else
    fitted = glm(X, y, v.family, v.link)
  end
  v.beta .= coef(fitted)
  return v
end

function mm_update_coef!(penalty::AbstractPenalty, g::CoefficientSurrogate, v::VertexGLM, triobs::Vector{TriangleObs}, workspace, caches)
  # Setup local variables to match notation
  family = v.family
  link = v.link
  Γ = caches.gamma[v.index]
  d = v.d
  r = v.workres
  η = v.eta
  μ = v.mu
  w = v.weights
  ∇L, ∇²L, Δ = workspace

  fill!(∇L, 0); fill!(∇²L, 0)
  itr = zip(eachtriobs(triobs, v), v.part)
  phi = v.dispersion

  # Update the coefficients using IRWLS
  for ((triidx, T), idxrange) in itr
    y, X, A, t = T.y, T.X, T.A, get_triangle_vertices(T, v.index)
    F = caches.logf[triidx]
    for (i, idx) in zip(eachobsindex(T), idxrange)
      x = view(X, i, :)
      alpha = @get_triple A i
      logf = @get_triple F i
      zweight = stable_convex_weight(t, alpha, logf)
      dμdη = meanderiv(v.link, η[idx]) 
      r[idx] = (y[i] - μ[idx]) / dμdη
      d[idx] = zweight * stable_irls_weight(family, link, η[idx], μ[idx], dμdη)
      if iszero(zweight)
          d[idx] = one(zweight)
          r[idx] = zero(zweight)
      end

      # Evaluate gradient + Hessian
      if v.nvar > 1
        BLAS.axpy!(-d[idx]*r[idx] * dispfun(family, phi), x, ∇L)
        BLAS.syr!('U', d[idx] * dispfun(family, phi), x, ∇²L)
      else
        ∇L[1] = ∇L[1] - d[idx]*r[idx] * dispfun(family, phi) * x[1]
        ∇²L[1] = ∇²L[1] + d[idx] * dispfun(family, phi) * x[1]*x[1]
      end
    end
  end
  accumulate_penalty_derivs!(penalty, ∇L, ∇²L, v, Γ, g.opt.rho)
  if v.nvar > 1
    H = Symmetric(∇²L, :U)
    ldiv!(Δ, cholesky!(H), ∇L)
  else
    Δ[1] = ∇L[1] / ∇²L[1]
  end
  linesearch!(g, v.beta_new, v.beta, Δ, g.opt.backtrack)
  return v
end

function local_deviance(::UnivariateDistribution, v::VertexGLM, triobs::Vector{TriangleObs}, workspace, caches)
  error("local_deviance() not implemented for the $(v.family) case.")
end

function local_deviance(::Normal, v::VertexGLM, triobs::Vector{TriangleObs}, workspace, caches)
  # Setup local variables to match notation
  mu = v.mu

  itr = zip(eachtriobs(triobs, v), v.part)

  # Fix β, update φ
  n_j = rss = zero(eltype(v.beta))
  for ((triidx, T), idxrange) in itr
    y, A, t = T.y, T.A, get_triangle_vertices(T, v.index)
    F = caches.logf[triidx]
    for (i, idx) in zip(eachobsindex(T), idxrange)
      alpha = @get_triple A i
      logf = @get_triple F i
      zweight = stable_convex_weight(t, alpha, logf)
      rss += zweight * deviance(v.family, y[i], mu[idx])
      n_j += zweight
    end
  end

  return n_j, rss
end

function fitmodel(::Type{VertexGLM}, yfull, Xfull, Sfull, tri;
    family::D = Normal(),
    link::L = IdentityLink(),
    maxiter::Int = 100,
    backtrack::Int = 5,
    tol::Real = 1e-3,
    rho::Real = 1.0,
    nchunks::Int = Threads.nthreads(),
    verbose::Bool = false,
    kwargs...
    # intercept = all(isequal(1), view(Xfull, :, 1)),
  ) where {D <: UnivariateDistribution, L <: Link}
  # Initialize
  VT = infer_vertex_type(VertexGLM, family, link)
  model = create_model(VT, yfull, Xfull, Sfull, tri; nchunks, family, link, kwargs...)
  initialize!(model)

  opt = (; rho, backtrack, verbose, nchunks)
  _fitmodel_loop_(model, maxiter, tol, opt, true)
end

function _fitmodel_loop_(model::SpatialVertexModel{V}, maxiter, tol, opt, needs_transform) where V <: VertexGLM
  # unpacking + convenient definitions
  rho, verbose, nchunks = opt.rho, opt.verbose, opt.nchunks
  family = get_family(model)
  f = model

  update_phi = needs_dispersion(family)
  update_caches!(model; nchunks)

  # Main loop
  nlogl = f(rho; nchunks)
  nlogl_prev = zero(nlogl)
  iter = 0

  while iter < maxiter && abs(nlogl - nlogl_prev) > (1 + abs(nlogl_prev)) * tol
    iter += 1

    # Visit each vertex once to update local regression coefficients
    update_coefficients!(model, opt, nchunks)

    # Visit each vertex once to update auxiliary parameters
    update_phi && update_caches!(model; nchunks)
    update_phi && update_dispersion!(model, opt, nchunks)

    # Synchronize algorithm state and evaluate current model
    update_caches!(model; nchunks)

    # Check convergence
    nlogl_prev = nlogl
    nlogl = f(rho; nchunks)
    verbose && @show iter, nlogl, nlogl_prev - nlogl
    @assert is_approx_decrease(nlogl, nlogl_prev, sqrt(eps()))
  end

  # Report negative log-likelihood without penalty
  nlogl = eval_loss(model.triobs, model.vertex, model.caches; nchunks)

  # Transform parameters as needed
  needs_transform && transform_parameters!(model)

  # Reflect updated state in model
  model = Accessors.@set model.state =
    FittedState(true, needs_transform, rho)

  return iter, model, nlogl
end
#
# CROSS VALIDATION
#
function cv(::Type{VertexGLM}, yfull, Xfull, Sfull, tri;
    train_prop::Real = 0.8,
    nfolds::Int = 5,
    grid_rho::AbstractVector=[1e-6, 1e-3, 1.0],
    family::D = Normal(),
    link::L = IdentityLink(),
    maxiter::Int = 100,
    backtrack::Int = 5,
    tol::Real = 1e-3,
    nchunks::Int = Threads.nthreads(),
    verbose::Bool = false,
    penalty::P = L2Squared(),
    kwargs...
    # intercept = all(isequal(1), view(Xfull, :, 1)),
  ) where {D <: UnivariateDistribution, L <: Link, P <: AbstractPenalty}
  # Extract vertex data so we can quickly build up TriangleObs sets
  helper = create_triobs_helper(tri)
  tri_vertex_coords = DelaunayTriangulation.get_points(tri)

  # Shuffle data, then split data into CV subset and test subset.
  n = length(yfull)
  idx_shuffle = shuffleobs(1:n)
  aa, bb = splitobs(n; at = train_prop)
  idx_cv, idx_tes = getobs(idx_shuffle, aa), getobs(idx_shuffle, bb)

  # Buffers to help us materialize data.
  # We do our best to minimize the memory footprint.
  tmp_tra, tmp_val = kfolds(length(idx_cv), nfolds)
  max_size_tra = maximum(length, tmp_tra)
  max_size_val = maximum(length, tmp_val)
  max_size_tes = length(idx_tes)
  n_buf = max(max_size_tra, max_size_val, max_size_tes)
  y_tmp = similar(yfull, n_buf)
  y_buf = similar(yfull, n_buf)
  X_buf = similar(Xfull, (n_buf, size(Xfull, 2)))
  S_buf = similar(Sfull, (size(Sfull, 1), n_buf))
  dst = (y_buf, X_buf, S_buf)
  src = (yfull, Xfull, Sfull)

  # Setup before any CV or model fitting takes place
  itr = grid_rho
  nvals = length(itr)
  nvar = size(Xfull, 2)
  VT = infer_vertex_type(VertexGLM, family, link)
  PT = create_component(family, link, 0.0, 1.0) |> typeof
  buffers = (Vector{PT}(undef, 3), zeros(3))

  score_tra = zeros(nvals, nfolds)
  score_val = zeros(nvals, nfolds)
  score_tes = zeros(nvals, nfolds)

  niter_tra = zeros(Int, nvals, nfolds)
  nlogl_tra = zeros(nvals, nfolds)

  times_fit = zeros(nvals, nfolds)
  times_tra = zeros(nvals, nfolds)
  times_val = zeros(nvals, nfolds)
  times_tes = zeros(nvals, nfolds)

  result1 = (score_tra, score_val, score_tes)
  result2 = (times_tra, times_val, times_tes)

  for (k, (idx_tra, idx_val)) in enumerate(kfolds(idx_cv; k=nfolds))
    # iterator specifying each of the three subsets
    sub = (idx_tra, idx_val, idx_tes)

    # retrieve training data
    y_tra, X_tra, S_tra = materialize_data!(dst, src, idx_tra)

    # Initialize model with training data
    triobs = _create_triobs_set_with_helper_(helper, y_tra, X_tra, S_tra, tri, nchunks)
    vertex = create_vertex_set(VT, tri, triobs, nvar; family, link, kwargs...)
    caches = build_caches(triobs, vertex, penalty, nvar, nchunks)
    model = SpatialVertexModel(triobs, vertex, penalty, caches)
    initialize!(model)

    # Cross validate on penalty strength, rho
    for (i, rho) in enumerate(itr)
      verbose && println("Fold $(k), rho = $(round(rho, sigdigits=4))")
      # fit model with target hyperparameter + other fixed parameters
      opt = (; rho, backtrack, verbose=false, nchunks)
      fitstats = @timed _fitmodel_loop_(model, maxiter, tol, opt, false)

      # record model fit information
      niter_tra[i,k] = fitstats.value[1]
      nlogl_tra[i,k] = fitstats.value[3]
      times_fit[i,k] = fitstats.time

      # score model on different subsets
      for (idx, arr1, arr2) in zip(sub, result1, result2)
        y, X, S = materialize_data!(dst, src, idx)
        predstats = @timed eval_negloglik(y, X, S, model, tri; nchunks)
        arr1[i,k] = predstats.value
        arr2[i,k] = predstats.time
      end
    end
  end

  # assemble results
  cv_scores = mean(score_val, dims=2) |> vec
  idx = argmin(cv_scores)

  best_rho = grid_rho[idx]
  results = (;
    # cross validation settings
    idx_cv, idx_tes, nfolds, train_prop,
    # cross validation results
    score_tra, score_val, score_tes,
    times_fit, niter_tra, nlogl_tra,
    times_tra, times_val, times_tes,
    cv_scores,
    # optimal hyperparameter(s)
    grid_rho,
    best_score = cv_scores,
    best_rho,
    best_rho_index = idx,
  )

  return results
end

# Helper functions to materialize data from (y, X, S) tuples
function materialize_data!(dst, src, idx, obsdim)
  src_lazy = obsview(src, idx, obsdim)
  n = length(src_lazy)
  src_view = _get_view_(src_lazy, n, obsdim)
  dst_view = _get_view_(dst, n, obsdim)
  dst_view .= src_view
  return dst_view
end

function materialize_data!(dst, src, idx)
  y_ = materialize_data!(dst[1], src[1], idx, ObsDim(1))
  X_ = materialize_data!(dst[2], src[2], idx, ObsDim(1))
  S_ = materialize_data!(dst[3], src[3], idx, ObsDim(2))
  return y_, X_, S_
end

# Always guarantee linear indexing to ensure BLAS/LAPACK work
_get_view_(dst::MLUtils.ArrayObsView{1,<:Vector,<:AbstractVector}, _, _) = view(dst.data, dst.indices)
_get_view_(dst::MLUtils.ArrayObsView{1,<:Matrix,<:AbstractVector}, _, _) = view(dst.data, dst.indices, :)
_get_view_(dst::MLUtils.ArrayObsView{2,<:Matrix,<:AbstractVector}, _, _) = view(dst.data, :, dst.indices)
_get_view_(dst::Vector, n, ::ObsDim{1}) = view(dst, 1:n)
_get_view_(dst::Matrix, n, ::ObsDim{1}) = view(dst, 1:n, :)
_get_view_(dst::Matrix, n, ::ObsDim{2}) = view(dst, :, 1:n)
#
# COEFFICIENTS
#
function coefficients(s, m::SpatialVertexModel, mesh)
  indices = each_solid_vertex(mesh) |> collect |> sort
  id2vertex = Dict{Int,Int}(id => j for (j, id) in enumerate(indices))
  points = get_points(mesh)

  j = k = l = -1
  attempts = 0
  MAX_ATTEMPTS = 20
  while any(<(1), (j, k, l)) && attempts < MAX_ATTEMPTS
    attempts += 1
    j, k, l = find_triangle(mesh, s) |> sort
    # Try to detect the solid edges again; j may fail as < 0
    if DelaunayTriangulation.is_ghost_triangle((j, k, l))
      l, k, _ = DelaunayTriangulation.sort_triangle((j, k, l))
      j = DelaunayTriangulation.get_adjacent(mesh, k, l)
    end
    # If the previous attempt failed, try switching orientation
    if DelaunayTriangulation.is_ghost_triangle((j, k, l))
      j = DelaunayTriangulation.get_adjacent(mesh, l, k)
    end
  end

  if DelaunayTriangulation.is_ghost_triangle((j, k, l))
    error("Failed to find triangle containing s = $(s); got ($j, $k, $l).")
  end
  j, k, l = sort((j, k, l))
  p, q, r = points[j], points[k], points[l]
  j, k, l = id2vertex[j], id2vertex[k], id2vertex[l]
  vertex = (m.vertex[j], m.vertex[k], m.vertex[l])
  alpha = barycentric(s, [p[1] q[1] r[1]; p[2] q[2] r[2]])
  clamp!(alpha, 1e-12, 1.0)
  alpha .= alpha / sum(alpha)
  return alpha[1] * vertex[1].beta +
    alpha[2] * vertex[2].beta +
    alpha[3] * vertex[3].beta
end
#
# PREDICTION
#
function assemble_mixture(x, s, m, tri, id2vertex, points)
    j, k, l = find_triangle(tri, s; concavity_protection = true) |> sort
    p, q, r = points[j], points[k], points[l]
    j, k, l = id2vertex[j], id2vertex[k], id2vertex[l]
    vertex = (m.vertex[j], m.vertex[k], m.vertex[l])
    alpha = barycentric(s, [p[1] q[1] r[1]; p[2] q[2] r[2]])
    clamp!(alpha, 1e-12, 1.0)
    alpha .= alpha / sum(alpha)
    return MixtureModel(
      [create_component(v.family, v.link, dot(x, v.beta), v.dispersion) for v in vertex],
      alpha
    )
end

function assemble_mixture(family, link, x, s, B, Φ, tri, id2vertex, points)
    j, k, l = find_triangle(tri, s; concavity_protection = true) |> sort
    p, q, r = points[j], points[k], points[l]
    vertex_index = id2vertex[j], id2vertex[k], id2vertex[l]
    alpha = barycentric(s, [p[1] q[1] r[1]; p[2] q[2] r[2]])
    return @views MixtureModel(
      [create_component(family, link, dot(x, B[:,idx]), Φ[idx]) for idx in vertex_index],
      alpha
    )
end

function _assemble_mixture_(x, s, m, tri, id2vertex, points, buffers)
  prob, alpha = buffers
  j, k, l = find_triangle(tri, s; concavity_protection = true) |> sort
  p, q, r = points[j], points[k], points[l]
  j, k, l = id2vertex[j], id2vertex[k], id2vertex[l]
  vertex = (m.vertex[j], m.vertex[k], m.vertex[l])
  barycentric!(alpha, s, [p[1] q[1] r[1]; p[2] q[2] r[2]])
  clamp!(alpha, 1e-12, 1.0)
  alpha .= alpha / sum(alpha)
  for (i, v) in enumerate(vertex)
    prob[i] = create_component(v.family, v.link, dot(x, v.beta), v.dispersion)
  end
  return MixtureModel(prob, alpha)
end

function predict(X, S, m::SpatialVertexModel, tri; kind = :mean)
  yhat = zeros(size(X, 1))
  tmp = create_component(get_family(m), get_link(m), 0.0, 1.0)
  T = typeof(tmp)
  buffers = (Vector{T}(undef, 3), zeros(3))
  predict!(yhat, X, S, m, tri, buffers; kind)
end

function predict!(yhat, X, S, m, tri, buffers; kind = :mean)
  indices = each_solid_vertex(tri) |> collect |> sort
  id2vertex = Dict{Int,Int}(id => j for (j, id) in enumerate(indices))
  points = get_points(tri)
  _predict!_(yhat, X, S, m, tri, id2vertex, points, kind, buffers)
end

function _predict!_(yhat, X, S, m, tri, id2vertex, points, kind, buffers)
  !(m.state.is_transformed) && transform_parameters!(m)
  @views for i in axes(X, 1)
    s = S[:, i]
    x = X[i, :]
    h = _assemble_mixture_(x, s, m, tri, id2vertex, points, buffers)
    
    yhat[i] = if kind == :mean
      mean(h)
    elseif kind == :mode
      _search_mode_(h)
    end
  end
  !(m.state.is_transformed) && transform_parameters!(m)
  return yhat
end

function pit_residuals(y, X, S, m::SpatialVertexModel, tri)
  res = zeros(size(X, 1))
  tmp = create_component(get_family(m), get_link(m), 0.0, 1.0)
  T = typeof(tmp)
  buffers = (Vector{T}(undef, 3), zeros(3))
  pit_residuals!(res, y, X, S, m, tri, buffers)
end

function pit_residuals!(res, y, X, S, m, tri, buffers)
  @inline function lse_update(m, s, t)
    t == -Inf && return m, s  # zero-weight / underflowed term
    t <= m ? (m, s + exp(t - m)) : (t, s * exp(m - t) + one(s))
  end
  indices = each_solid_vertex(tri) |> collect |> sort
  id2vertex = Dict{Int,Int}(id => j for (j, id) in enumerate(indices))
  points = get_points(tri)
  phi = m.vertex[1].dispersion
  !(m.state.is_transformed) && transform_parameters!(m)
  @views for i in axes(X, 1)
    s = S[:, i]
    x = X[i, :]
    h = _assemble_mixture_(x, s, m, tri, id2vertex, points, buffers)
    mL, sL = -Inf, 0.0
    mU, sU = -Inf, 0.0
    for k in 1:3
      logalpha = log(h.prior.p[k])
      mL, sL = lse_update(mL, sL, logalpha + Distributions.logcdf(h.components[k], y[i]))
      mU, sU = lse_update(mU, sU, logalpha + Distributions.logccdf(h.components[k], y[i]))
    end
    logL = mL + log(sL)
    logU = mU + log(sU)
    res[i] = logL < logU ? Distributions.invlogcdf(Normal(), logL) : Distributions.invlogccdf(Normal(), logU)
  end
  !(m.state.is_transformed) && transform_parameters!(m)
  return res
end

function _search_mode_(h)
  # Initialize the predicted value using the weighted mean.
  y = mean(h)

  # Use an MM algorithm to search for a mode
  mu = [mean(d) for d in components(h)]
  logf = Distributions.logpdf.(components(h), y)
  logh = Distributions.logpdf(h, y)
  logh_prev = logh*10
  while abs(logh - logh_prev) > 1e-6 * (1 + abs(logh_prev))
    logh_prev = logh
    y = __mm_update_mode__(component_type(h), Distributions.probs(h), mu, logf)
    logf .= Distributions.logpdf.(components(h), y)
    logh = Distributions.logpdf(h, y)
  end

  return y
end

function __mm_update_mode__(::Type{Normal{T}}, alpha, mu, logf) where T <: Real
  num = den = zero(first(mu))
  for k in 1:3
    weight = stable_convex_weight(k, alpha, logf)
    num += weight * mu[k]
    den += weight # should be 1?
  end
  @assert den ≈ one(den)
  return num / den
end

# function __mm_update_mode__(::Union{Binomial,Bernoulli}, alpha, mu, logf)
#   a = b = zero(first(mu))
#   for k in 1:3
#     weight = stable_eval_mm_weight(k, alpha, logf)
#     a += weight * log(mu[k])
#     b += weight * log1p(-mu[k])
#   end
#   return a > b ? zero
# end

# function __mm_update_mode__(::Poisson, alpha, mu, logf)

# end

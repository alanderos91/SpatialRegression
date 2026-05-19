#
# HELPER FUNCTIONS
#
prox_pinball(r, tau, mu) = begin
  if r >= mu*tau
    r - mu*tau
  elseif -mu*(1-tau) < r < mu*tau
    zero(r)
  else # r <= -mu*(1-tau)
    r + mu*(1-tau)
  end
end
#
# CACHES
#
# function build_caches(triobs::Vector{TriangleObs}, vertex::Vector{V}, ::AbstractPenalty, nvars, nchunks) where V <: VertexQREG
#   return (;
#     # Store residuals and proximal values for each triangle
#     residuals = Dict(triidx => zeros(3, length(t.y)) for (triidx, t) in TriangleObsIterator(triobs)),
#     proximals = Dict(triidx => zeros(3, length(t.y)) for (triidx, t) in TriangleObsIterator(triobs)),
#     # matrix of local average coefficients in MM algorithm, Γ
#     gamma = Dict(v.index => zeros(nvars, length(v.neighbors)) for v in vertex),
#     # LHS and RHS for each worker
#     workspace = [(zeros(nvars, nvars), zeros(nvars)) for _ in 1:nchunks]
#   )
# end
#
# SURROGATE
#
# function eval_loss_surrogate(beta, v::V, triobs::Vector{TriangleObs}, caches, tau, mu) where V <: VertexQREG
#   zcache = caches.proximals
#   n = length(v.d)
#   n = ifelse(iszero(n), one(n), n)
#   loss = zero(Float64)
#   for (triidx, T) in eachtriobs(triobs, v)
#     t = get_triangle_vertices(T, v.index)
#     for i in eachobsindex(T)
#       y = T.y[i]
#       x = view(T.X, :, i)
#       a = view(T.A, :, i)
#       z = view(zcache[triidx], :, i)

#       r = y - dot(x, beta)
#       loss += a[t]/n * (pinball(z[t], tau) + 1//2*inv(mu) * abs2(r - z[t]))
#     end
#   end
#   return loss
# end
#
# UPDATES
#
function mm_update_coef!(penalty::AbstractPenalty, g::VertexSurrogate, v::VertexGLM{<:ALD}, triobs::Vector{TriangleObs}, workspace, caches, opt)
# Setup local variables to match notation
  Γ = caches.gamma[v.index]
  d = v.d
  μ = v.mu
  τ = v.family.τ
  rho = g.rho
  xi = opt.xi
  LHS, RHS, Δ = workspace
  
  fill!(LHS, 0); fill!(RHS, 0)
  itr = zip(eachtriobs(triobs, v), v.part)
  phi = v.extra_params[:dispersion]

  # Update the coefficients using IRWLS
  for ((triidx, T), idxrange) in itr
    y, X, A, t = T.y, T.X, T.A, get_triangle_vertices(T, v.index)
    F = caches.logf[triidx]
    for (i, idx) in zip(eachobsindex(T), idxrange)
      x = view(X, i, :)
      alpha = @get_triple A i
      logf = @get_triple F i
      zweight = stable_convex_weight(t, alpha, logf)
      r = (y[i] - μ[idx])
      dweight = zweight * phi * xi
      z = prox_pinball(r, τ, dweight)
      u = r - z
      d[idx] = inv(dweight)
      if iszero(zweight)
        d[idx] = one(zweight)
        u = zero(zweight)
      end
      # @assert zweight > 0 "MM Update w/ Zero Weight @ Vertex $(v.index)\n\tObs. Index: $(idx)\n\talpha = $(alpha)\n\tlogf = $(logf)\n\tResidual: $(r[idx])\n\tIRLS Weight: $(stable_irls_weight(family, link, η[idx], μ[idx], dμdη))"

      # Evaluate gradient + Hessian
      if v.nvar > 1
        BLAS.axpy!(d[idx]*u, x, RHS)
        BLAS.syr!('U', d[idx], x, LHS)
      else
        RHS[1] = RHS[1] - d[idx] * u * x[1]
        LHS[1] = LHS[1] + d[idx] * x[1]*x[1]
      end
    end
  end
  accumulate_penalty_terms!(penalty, RHS, LHS, v, Γ, rho)
  if v.nvar > 1
    H = Symmetric(LHS, :U)
    ldiv!(Δ, cholesky!(H), RHS)
  else
    Δ[1] = RHS[1] / LHS[1]
  end
  @. v.beta_new = v.beta - Δ
end

function mm_update_disp!(::ALD, g::VertexSurrogate, v::VertexGLM, triobs::Vector{TriangleObs}, workspace, caches, use_prior)
  # Setup local variables to match notation
  μ = v.mu

  itr = zip(eachtriobs(triobs, v), v.part)

  # Fix β, update φ
  num = den = zero(eltype(v.beta))
  for ((triidx, T), idxrange) in itr
    y, A, t = T.y, T.A, get_triangle_vertices(T, v.index)
    F = caches.logf[triidx]
    for (i, idx) in zip(eachobsindex(T), idxrange)
      alpha = @get_triple A i
      logf = @get_triple F i
      zweight = stable_convex_weight(t, alpha, logf)
      num += zweight * deviance(v.family, y[i], μ[idx])
      den += zweight
      # v.index == 1 && @show zweight, deviance(v.family, y[i], μ[idx])
    end
  end

  if use_prior
    phi0 = v.extra_params[:global_dispersion]
    nu = v.extra_params[:nu]
    if !isempty(v.triangles)
      phi = (den + nu) / (num + nu / phi0)
    else
      phi = phi0
    end
  else
    nu = v.extra_params[:nu]
    A = iszero(den) ? zero(den) : num / den
    B = zero(den)
    C = nu
    wsum = sum(v.weights)
    for (idx, k) in enumerate(v.neighbors)
      u = g.model.vertex[k]
      B_k = u.extra_params[:local_dispersion]
      B += v.weights[idx]/wsum * inv(B_k)
    end
    invphi = den / (den + C) * A + C / (den + C) * B
    phi = inv(invphi)
  end

  # g_curr = g(phi)
  # @assert g_curr < g_prev || abs(g_curr - g_prev) < sqrt(eps()) * (1 + abs(g_prev)) "\n\tVertex: $(v.index)\n\tNobs: $(v.nobs)\n\tPrevious: $(g_prev)\n\t Current: $(g_curr)\n\tDen/Num: $(den) / $(num)\n\tφ₀: $(phi_old)\n\tφ: $(phi)\n\tβ: $(v.beta)"

  v.extra_params[:dispersion] = isinf(phi) ? one(phi) : phi
end

function fitqreg(::Type{VertexGLM}, yfull, Xfull, Sfull, tri;
    maxiter::Int = 100,
    tol::Real = 1e-3,
    rho::Real = 1.0,
    quantile::Real = 0.5,
    smooth_max::Real = 10.0,
    smooth_min::Real = 0.1,
    smooth_itr::Real = 20,
    nchunks::Int = Threads.nthreads(),
    use_prior::Bool = false,
    kwargs...
    # intercept = all(isequal(1), view(Xfull, :, 1)),
  )
  # Initialize
  tau = quantile
  family, link = ALD(tau), IdentityLink() # need to propagate tau!
  VT = infer_vertex_type(VertexGLM, family, link; kwargs...)
  model = f = create_model(VT, yfull, Xfull, Sfull, tri; nchunks, family, link, kwargs...)
  update_phi = needs_dispersion(family)
  # initialize_coefficients!(model.triobs, model.vertex)
  update_caches!(model; nchunks)
  update_phi && update_pooling!(model, use_prior)
  nlogl = f(rho; nchunks, use_prior)
  nlogl_prev = zero(nlogl)
  iter = 0
  xi = smooth_max
  opt = (; rho = rho, xi = xi)
  while iter < maxiter && abs(nlogl - nlogl_prev) > (1 + abs(nlogl_prev)) * tol
    iter += 1

    # Visit each vertex once to update local regression coefficients
    update_coefficients!(model, opt, nchunks)
    opt = (; rho = rho, xi = xi)
    xi = iter > smooth_itr ? smooth_min : smooth_max * (smooth_min / smooth_max) ^ (iter / smooth_itr)
    update_phi && update_dispersion!(model, opt, nchunks, use_prior)

    update_caches!(model; nchunks)
    update_phi && update_pooling!(model, use_prior)
    nlogl_prev = nlogl
    nlogl = f(rho; nchunks, use_prior)
    @show iter, nlogl, nlogl_prev - nlogl
    @assert nlogl < nlogl_prev || abs(nlogl - nlogl_prev) < sqrt(eps()) * (1 + abs(nlogl_prev))
  end

  # Apply all updates
  if update_phi
    for v in eachvertex(model)
      phi = v.extra_params[:dispersion]
      v.extra_params[:dispersion] = 1 / phi
    end
  end

  return iter, model, nlogl
end

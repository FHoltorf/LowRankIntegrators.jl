struct RankAdaptiveUnconventionalAlgorithm_Params{sType, lType, kType}
    S_rhs # rhs of S step (core projected rhs)
    L_rhs # rhs of L step (range projected rhs)
    K_rhs # rhs of K step (corange projected rhs)
    S_kwargs
    L_kwargs
    K_kwargs
    S_alg::sType
    L_alg::lType
    K_alg::kType
    tol
    r_max::Int
end

struct RankAdaptiveUnconventionalAlgorithm{sType, lType, kType} <: AbstractDLRAlgorithm
    alg_params::RankAdaptiveUnconventionalAlgorithm_Params{sType, lType, kType}
end
function RankAdaptiveUnconventionalAlgorithm(tol; S_rhs = nothing, L_rhs = nothing, K_rhs = nothing,
                                             S_kwargs = Dict(), L_kwargs = Dict(), K_kwargs = Dict(),
                                             S_alg = Tsit5(), L_alg = Tsit5(), K_alg = Tsit5(), rmax = 2^62)
    params = RankAdaptiveUnconventionalAlgorithm_Params(S_rhs, L_rhs, K_rhs, S_kwargs, L_kwargs, K_kwargs, S_alg, L_alg, K_alg, tol, rmax)
    return RankAdaptiveUnconventionalAlgorithm(params)
end

struct RankAdaptiveUnconventionalAlgorithm_Cache{uType,
                                                 SIntegratorType,LIntegratorType,KIntegratorType,
                                                 SProbType, LProbType, KProbType,
                                                 yFunc,yType,dType} <: AbstractDLRAlgorithm_Cache
    US::Matrix{uType}
    Uhat::Matrix{uType}
    VS::Matrix{uType}
    Vhat::Matrix{uType}
    M::Matrix{uType}
    N::Matrix{uType}
    SIntegrator::SIntegratorType
    SProblem::SProbType
    LIntegrator::LIntegratorType
    LProblem::LProbType
    KIntegrator::KIntegratorType
    KProblem::KProbType
    r::Int # current rank
    tol::Float64
    r_max::Int
    y::yFunc
    ycurr::yType
    yprev::yType
    Δy::yType
    interpolation_cache::dType
end

function alg_cache(prob::MatrixDEProblem, alg::RankAdaptiveUnconventionalAlgorithm, u, dt; t0 = prob.tspan[1])
    # allocate memory for frequently accessed arrays
    tspan = (t0, t0+dt)
    r = rank(u)
    n, m = size(u)
    US = u.U*u.S
    VS = u.V*u.S'
    Uhat = zeros(n,2*r)
    Vhat = zeros(m,2*r)
    M = zeros(2*r,r)
    N = zeros(2*r,r)

    if isnothing(alg.alg_params.K_rhs)
        K_rhs = function (US, V, t)
                    return Matrix(prob.f(TwoFactorRepresentation(US,V),t)*V)
                end 
    else
        K_rhs = alg.alg_params.K_rhs
    end
    KProblem = ODEProblem(K_rhs, US, tspan, u.V)
    KIntegrator = init(KProblem, alg.alg_params.K_alg; save_everystep=false, alg.alg_params.K_kwargs...)
    if isnothing(alg.alg_params.L_rhs)
        L_rhs = function (VS, U, t)
                    return Matrix(prob.f(TwoFactorRepresentation(U,VS),t)'*U)
                end
    else
        L_rhs = alg.alg_params.L_rhs
    end
    
    LProblem = ODEProblem(L_rhs, VS, tspan, u.U)
    LIntegrator = init(LProblem, alg.alg_params.L_alg; save_everystep=false, alg.alg_params.L_kwargs...)
    if isnothing(alg.alg_params.S_rhs)
        S_rhs = function (S, (U,V), t)
                    return Matrix(U'*prob.f(SVDLikeRepresentation(U,S,V),t)*V)
                end
    else
        S_rhs = alg.alg_params.S_rhs
    end
    SProblem = ODEProblem(S_rhs, M*u.S*N', tspan, [Uhat, Vhat])
    SIntegrator = init(SProblem, alg.alg_params.S_alg; save_everystep=false, alg.alg_params.S_kwargs...)

    return RankAdaptiveUnconventionalAlgorithm_Cache(US, Uhat, VS, Vhat, M, N, 
                                                     SIntegrator, SProblem, LIntegrator, LProblem,
                                                     KIntegrator, KProblem, r, alg.alg_params.tol, alg.alg_params.r_max,
                                                     nothing, nothing, nothing, nothing, nothing)
end

function alg_cache(prob::MatrixDEProblem{fType, uType, tType}, 
                   alg::RankAdaptiveUnconventionalAlgorithm, u, dt; t0 = prob.tspan[1]) where
                   {fType <: ComponentFunction, uType, tType}
    # allocate memory for frequently accessed arrays
    tspan = (t0, t0+dt)
    r = rank(u)
    n, m = size(u)
    US = u.U*u.S
    VS = u.V*u.S'
    Uhat = zeros(n,2*r)
    Vhat = zeros(m,2*r)
    M = zeros(2*r,r)
    N = zeros(2*r,r)

    @unpack tol, rmin, rmax, 
            init_range, init_corange,
            selection_alg = alg.alg_params.interpolation

    row_idcs = index_selection(init_range, selection_alg)
    col_idcs = index_selection(init_corange, selection_alg)

    Π_corange = DEIMInterpolator(col_idcs, init_corange/init_corange[col_idcs,:])
    Π_range = DEIMInterpolator(row_idcs, init_range/init_range[row_idcs,:])
    Π = SparseFunctionInterpolator(prob.f, SparseMatrixInterpolator(Π_range, Π_corange))

    US = u.U*u.S
    QRK = qr(US)
    M = Matrix(QRK.Q)'*u.U

    VS = u.V*u.S'
    QRL = qr(VS)
    N = Matrix(QRL.Q)'*u.V

    if isnothing(alg.alg_params.K_rhs)
        K_rhs = function (dK, K, p, t)
                    Π_K, V0, params = p
                    Π_K(dK, TwoFactorRepresentation(K, V0), params, t)
                end
    else
        K_rhs = alg.alg_params.K_rhs
    end
    Π_K_corange = DEIMInterpolator(col_idcs,
                                    u.V'*Π.interpolator.corange.weights)'
    Π_K = SparseFunctionInterpolator(prob.f, Π_K_corange)
    p_K = (Π_K, u.V, ())
    KProblem = ODEProblem(K_rhs, US, tspan, p_K)
    KIntegrator = init(KProblem, alg.alg_params.K_alg; save_everystep=false, alg.alg_params.K_kwargs...)

    if isnothing(alg.alg_params.L_rhs)
        L_rhs = function (dL, L, p, t) 
                    Π_L, U0, params = p
                    Π_L(dL', TwoFactorRepresentation(U0,L), params, t)
                end
    else
        L_rhs = alg.alg_params.L_rhs
    end
    Π_L_range = DEIMInterpolator(row_idcs,
                                 u.U'*Π.interpolator.range.weights)
    Π_L = SparseFunctionInterpolator(prob.f, Π_L_range)
    p_L = (Π_L, u.U, ())
    LProblem = ODEProblem(L_rhs, VS, tspan, p_L)
    LIntegrator = init(LProblem, alg.alg_params.L_alg; save_everystep=false, alg.alg_params.L_kwargs...)

    if isnothing(alg.alg_params.S_rhs)
        S_rhs = function (dS, S, p, t)
                    Π_S, U1, V1, params = p
                    Π_S(dS, SVDLikeRepresentation(U1,S,V1), params, t)
                end
    else
        S_rhs = alg.alg_params.S_rhs
    end
    Π_S_mat = SparseMatrixInterpolator(row_idcs, col_idcs, 
                                        U_hat'*Π.interpolator.range.weights, 
                                        V_hat'*Π.interpolator.corange.weights)
    Π_S = SparseFunctionInterpolator(prob.f, Π_S_mat)
    p_S = (Π_S, U_hat, V_hat, ())

    SProblem = ODEProblem(S_rhs, M*u.S*N', tspan, p_S)
    SIntegrator = init(SProblem, alg.alg_params.S_alg; save_everystep=false, alg.alg_params.S_kwargs...)

    interpolation_cache = OnTheFlyInterpolation_Cache(alg.alg_params.interpolation, deepcopy(u),
                                                      Π, Π_K, Π_L, Π_S)
    return RankAdaptiveUnconventionalAlgorithm_Cache(US, Uhat, VS, Vhat, M, N, 
                                                     SIntegrator, SProblem, LIntegrator, LProblem,
                                                     KIntegrator, KProblem, r, alg.alg_params.tol, alg.alg_params.r_max,
                                                     nothing, nothing, nothing, nothing, interpolation_cache)
end

function alg_cache(prob::MatrixDataProblem, alg::RankAdaptiveUnconventionalAlgorithm, u, dt; t0 = prob.tspan[1])
    # creates caches for frequently used arrays by performing the first time step
    @unpack y = prob
    t0 = prob.tspan[1]
    n, r = size(u.U)
    m = size(u.V, 1)

    yprev = y isa AbstractArray ? deepcopy(y[1]) : y(t0) 
    ycurr = deepcopy(yprev)
    Δy = similar(yprev)
    
    US = zeros(n,r)
    VS = zeros(m,r)
    Uhat = zeros(n,2*r)
    Vhat = zeros(m,2*r)
    M = zeros(2*r,r)
    N = zeros(2*r,r)
    KIntegrator = MatrixDataIntegrator(Δy, US, I, u.V, 1)
    LIntegrator = MatrixDataIntegrator(Δy', VS, I, u.U, 1)
    SIntegrator = MatrixDataIntegrator(Δy, zeros(2r,2r), Uhat, Vhat, 1)

    return RankAdaptiveUnconventionalAlgorithm_Cache(US, Uhat, VS, Vhat, M, N, 
                                                     SIntegrator, nothing, LIntegrator, nothing,
                                                     KIntegrator, nothing, r, alg.alg_params.tol, alg.alg_params.r_max,
                                                     y, ycurr, yprev, Δy, nothing)
end

function alg_recache(cache::RankAdaptiveUnconventionalAlgorithm_Cache, alg::RankAdaptiveUnconventionalAlgorithm, u, t)
    @unpack SProblem, KProblem, LProblem, tol, r_max, y, yprev, ycurr, Δy, interpolation_cache = cache
    r = LowRankArithmetic.rank(u)
    n, m = size(u)

    US = zeros(n,r)
    Uhat = zeros(n,2r)
    M = zeros(2r,r)
    
    VS = zeros(m,r)
    Vhat = zeros(m,2r)
    N = zeros(2r,r)

    # the following sequence can probably somehow be handled via dispatch in a simpler way
    if isnothing(KProblem)
        KIntegrator = MatrixDataIntegrator(Δy, US, I, u.V, 1)
    elseif isnothing(interpolation_cache)
        KProblem = remake(KProblem, u0 = US, p = u.V, tspan = (t, t+1.0))
        KIntegrator = init(KProblem, alg.alg_params.K_alg; save_everystep=false, alg.alg_params.K_kwargs...)
    else
        KProblem = remake(KProblem, u0 = US, p = (interpolation_cache.Π_K, u.V, ()), tspan = (t, t+1.0))
        KIntegrator = init(KProblem, alg.alg_params.K_alg; save_everystep=false, alg.alg_params.K_kwargs...)
    end
    if isnothing(LProblem)
        LIntegrator = MatrixDataIntegrator(Δy', VS, I, u.U, 1)
    elseif isnothing(interpolation_cache)
        LProblem = remake(LProblem, u0 = VS, p = u.U, tspan = (t, t+1.0))
        LIntegrator = init(LProblem, alg.alg_params.L_alg; save_everystep=false, alg.alg_params.L_kwargs...)
    else
        LProblem = remake(LProblem, u0 = VS, p = (interpolation_cache.Π_L, u.U, ()), tspan = (t, t+1.0))
        LIntegrator = init(LProblem, alg.alg_params.L_alg; save_everystep=false, alg.alg_params.L_kwargs...)
    end

    if isnothing(SProblem)
        SIntegrator = MatrixDataIntegrator(Δy, zeros(2r,2r), Uhat, Vhat, 1)
    elseif isnothing(interpolation_cache)
        SProblem = remake(SProblem, u0 = zeros(2*r,2*r), p = [Uhat, Vhat], tspan = (t, t+1.0))
        SIntegrator = init(SProblem, alg.alg_params.S_alg; save_everystep=false, alg.alg_params.S_kwargs...)
    else
        SProblem = remake(SProblem, u0 = zeros(2*r,2*r), p = (interpolation_cache.Π_S, U_hat, V_hat, ()), tspan = (t, t+1.0))
        SIntegrator = init(SProblem, alg.alg_params.S_alg; save_everystep=false, alg.alg_params.S_kwargs...)
    end

    return RankAdaptiveUnconventionalAlgorithm_Cache(US, Uhat, VS, Vhat, M, N, SIntegrator, SProblem, LIntegrator, LProblem,
                                                     KIntegrator, KProblem, r, tol, r_max, y, ycurr, yprev, Δy, interpolation_cache)
end

function step!(integrator::DLRIntegrator, alg::RankAdaptiveUnconventionalAlgorithm, dt)
    @unpack u, t, iter, cache, probType = integrator
    u_new, rank_adjusted = rankadaptive_unconventional_step!(u, cache, t, dt, probType)
    if rank_adjusted
        integrator.u = u_new
        integrator.cache = alg_recache(cache, alg, u_new, t+dt)
    end
    integrator.t += dt
    integrator.iter += 1
end

function rankadaptive_unconventional_step!(u, cache, t, dt, ::Type{<:MatrixDataProblem})
    @unpack y, ycurr, yprev, Δy = cache
    update_data!(ycurr, y, t, dt)
    Δy .= ycurr - yprev
    yprev .= ycurr
    rankadaptive_unconventional_step!(u, cache, t, dt)
end

function rankadaptive_unconventional_step!(u, cache, t, dt, ::Type{<:MatrixDEProblem})
    rankadaptive_unconventional_step!(u, cache, t, dt)
end

function rankadaptive_unconventional_step!(u, cache, t, dt)
    @unpack r, r_max, tol, US, Uhat, VS, Vhat, M, N, KIntegrator, LIntegrator, SIntegrator = cache
    
    # K step
    mul!(US,u.U,u.S)
    set_u!(KIntegrator, US)
    step!(KIntegrator, dt, true)
    US .= KIntegrator.u
    Uhat[:, 1:r] .= US
    Uhat[:, r+1:end] .= u.U
    QRK = qr(Uhat)
    Uhat .= Matrix(QRK.Q)
    mul!(M,Uhat',u.U)
    
    # L step
    mul!(VS,u.V,u.S')
    set_u!(LIntegrator, VS)
    step!(LIntegrator, dt, true)
    VS .= LIntegrator.u
    Vhat[:, 1:r] .= VS
    Vhat[:, r+1:end] .= u.V 
    QRL = qr(Vhat)
    Vhat .= Matrix(QRL.Q)
    mul!(N,Vhat',u.V)
    
    set_u!(SIntegrator, M*u.S*N')
    step!(SIntegrator, dt, true)

    U, S, V = svd(SIntegrator.u)
    r_new = min(r_max, LowRankArithmetic.truncate_to_tolerance(S, tol))#, rel=true))
    if r_new == r 
        mul!(u.U,Uhat,U[:,1:r_new])
        u.S .= Matrix(Diagonal(S[1:r_new]))
        mul!(u.V,Vhat,V[:,1:r_new])
        return nothing, false
    else
        println("rank adjusted: new rank = $r_new")
        return SVDLikeRepresentation(Uhat*U[:,1:r_new], Matrix(Diagonal(S[1:r_new])), Vhat*V[:,1:r_new]), true
    end    
end
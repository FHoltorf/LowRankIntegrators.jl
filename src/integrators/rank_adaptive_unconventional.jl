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
    r_max
end

struct RankAdaptiveUnconventionalAlgorithm{sType, lType, kType} <: AbstractDLRAlgorithm
    alg_params::RankAdaptiveUnconventionalAlgorithm_Params{sType, lType, kType}
end
function RankAdaptiveUnconventionalAlgorithm(tol; S_rhs = nothing, L_rhs = nothing, K_rhs = nothing,
                                    S_kwargs = Dict(), L_kwargs = Dict(), K_kwargs = Dict(),
                                    S_alg = Tsit5(), L_alg = Tsit5(), K_alg = Tsit5(), r_max = Inf)
    params = RankAdaptiveUnconventionalAlgorithm_Params(S_rhs, L_rhs, K_rhs, S_kwargs, L_kwargs, K_kwargs, S_alg, L_alg, K_alg, tol, r_max)
    return RankAdaptiveUnconventionalAlgorithm(params)
end

struct RankAdaptiveUnconventionalAlgorithm_Cache{uType,SIntegratorType,LIntegratorType,KIntegratorType,yType} <: AbstractDLRAlgorithm_Cache
    US::Matrix{uType}
    Uhat::Matrix{uType}
    VS::Matrix{uType}
    Vhat::Matrix{uType}
    M::Matrix{uType}
    N::Matrix{uType}
    SIntegrator::SIntegratorType
    SProblem
    LIntegrator::LIntegratorType
    LProblem
    KIntegrator::KIntegratorType
    KProblem
    r::Int # current rank
    tol
    r_max 
    y
    ycurr::yType
    yprev::yType
    Δy::yType
end

#=
function alg_cache(prob::MatrixDEProblem, alg::UnconventionalAlgorithm, u, dt, t0 = prob.tspan[1])
    # the fist integration step is used to allocate memory for frequently accessed arrays
    US = u.U*u.S
    tspan = (t0,t0+dt)
    if isnothing(alg.alg_params.K_rhs)
        K_rhs = function (US, V, t)
                    return Matrix(prob.f(TwoFactorApproximation(US,V),t)*V)
                end 
    else
        K_rhs = alg.alg_params.K_rhs
    end
    KProblem = ODEProblem(K_rhs, US, tspan, u.V)
    KIntegrator = init(KProblem, alg.alg_params.K_alg; save_everystep=false, alg.alg_params.K_kwargs...)
    step!(KIntegrator, dt, true)
    Uhat = hcat(KIntegrator.u, US)
    QRK = qr!(Uhat)
    M = Matrix(QRK.Q)'*u.U
    
    if isnothing(alg.alg_params.L_rhs)
        L_rhs = function (VS, U, t)
                    return Matrix(prob.f(TwoFactorApproximation(U,VS),t)'*U)
                end
    else
        L_rhs = alg.alg_params.L_rhs
    end
    VS = u.V*u.S'
    LProblem = ODEProblem(L_rhs, VS, tspan, u.U)
    LIntegrator = init(LProblem, alg.alg_params.L_alg; save_everystep=false, alg.alg_params.L_kwargs...)
    step!(LIntegrator, dt, true)
    Vhat = hcat(LIntegrator.u, VS)
    QRL = qr!(Vhat)
    N = Matrix(QRL.Q)'*u.V
    
    if isnothing(alg.alg_params.S_rhs)
        S_rhs = function (S, (U,V), t)
                    return Matrix(U'*prob.f(SVDLikeApproximation(U,S,V),t)*V)
                end
    else
        S_rhs = alg.alg_params.S_rhs
    end
    SProblem = ODEProblem(S_rhs, M*u.S*N', tspan, (Uhat, Vhat))
    SIntegrator = init(SProblem, alg.alg_params.S_alg; save_everystep=false, alg.alg_params.S_kwargs...)
    step!(SIntegrator, dt, true)
    
    # truncated SVD
    U, S, V = svd(SIntegrator.u)
    update_rank!()
    r = truncate_to_tolerance(S, alg.alg_params.tol)
    u.U .= Uhat*U[:,1:r]
    u.S .= Matrix(Diagonal(S[1:r]))
    u.V .= Vhat*V[:,1:r]
    return UnconventionalAlgorithm_Cache(US, Uhat, VS, Vhat, M, N, QRK, QRL, 
                                         SIntegrator, LIntegrator, KIntegrator,
                                         nothing, nothing, nothing, nothing)
end
=#
function init(prob::MatrixDEProblem, alg::RankAdaptiveUnconventionalAlgorithm, dt)
    t0, tf = prob.tspan
    u = deepcopy(prob.u0)
    @assert tf > t0 "Integration in reverse time direction is not supported"
    # number of steps
    n = floor(Int,(tf-t0)/dt) + 1 
    # compute more sensible dt # rest will be handled via interpolation/save_at
    dt = (tf-t0)/(n-1)
    # initialize solution object
    sol = DLRSolution(Vector{typeof(prob.u0)}(undef, n), collect(range(t0, tf, length=n)))
    sol.Y[1] = deepcopy(prob.u0) # add initial point to solution object
    # initialize cache
    cache = alg_cache(prob, alg, u, t0)
    
    return DLRIntegrator(u, t0+dt, dt, sol, alg, cache, 0)   
end

function alg_cache(prob::MatrixDEProblem, alg::RankAdaptiveUnconventionalAlgorithm, u, t)
    # the fist integration step is used to allocate memory for frequently accessed arrays
    r = rank(u)
    n, m = size(u)
    US = zeros(n,r)
    VS = zeros(m,r)
    Uhat = zeros(n,2*r)
    Vhat = zeros(m,2*r)
    M = zeros(2*r,r)
    N = zeros(2*r,r)

    if isnothing(alg.alg_params.K_rhs)
        K_rhs = function (US, V, t)
                    return Matrix(prob.f(TwoFactorApproximation(US,V),t)*V)
                end 
    else
        K_rhs = alg.alg_params.K_rhs
    end
    KProblem = ODEProblem(K_rhs, US, (t,t+10.0), u.V)
    KIntegrator = init(KProblem, alg.alg_params.K_alg; save_everystep=false, alg.alg_params.K_kwargs...)
    step!(KIntegrator, 1e-9, true)
    set_t!(KIntegrator, t)
    if isnothing(alg.alg_params.L_rhs)
        L_rhs = function (VS, U, t)
                    return Matrix(prob.f(TwoFactorApproximation(U,VS),t)'*U)
                end
    else
        L_rhs = alg.alg_params.L_rhs
    end
    VS = u.V*u.S'
    LProblem = ODEProblem(L_rhs, VS, (t,t+10.0), u.U)
    LIntegrator = init(LProblem, alg.alg_params.L_alg; save_everystep=false, alg.alg_params.L_kwargs...)
    step!(LIntegrator, 1e-9, true)
    set_t!(LIntegrator, t)

    if isnothing(alg.alg_params.S_rhs)
        S_rhs = function (S, (U,V), t)
                    return Matrix(U'*prob.f(SVDLikeApproximation(U,S,V),t)*V)
                end
    else
        S_rhs = alg.alg_params.S_rhs
    end
    SProblem = ODEProblem(S_rhs, M*u.S*N', (t,t+10.0), (Uhat, Vhat))
    SIntegrator = init(SProblem, alg.alg_params.S_alg; save_everystep=false, alg.alg_params.S_kwargs...)
    step!(SIntegrator, 1e-9, true)
    set_t!(SIntegrator, t)

    return RankAdaptiveUnconventionalAlgorithm_Cache(US, Uhat, VS, Vhat, M, N, 
                                                     SIntegrator, SProblem, LIntegrator, LProblem,
                                                     KIntegrator, KProblem, r, alg.alg_params.tol, alg.alg_params.r_max,
                                                     nothing, nothing, nothing, nothing)
end

function alg_recache(cache::RankAdaptiveUnconventionalAlgorithm_Cache, alg::RankAdaptiveUnconventionalAlgorithm, u, t)
    @unpack SProblem, KProblem, LProblem, tol, r_max = cache
    r = rank(u)
    n, m = size(u)

    US = zeros(n,r)
    Uhat = zeros(n,2*r)
    M = zeros(2*r,r)
    
    VS = zeros(m,r)
    Vhat = zeros(m,2*r)
    N = zeros(2*r,r)

    KProblem = remake(KProblem, u0 = US, p = u.V)
    KIntegrator = init(KProblem, alg.alg_params.K_alg; save_everystep=false, alg.alg_params.K_kwargs...)
    step!(KIntegrator, 1e-9, true)
    set_t!(KIntegrator,t)

    LProblem = remake(LProblem, u0 = VS, p = u.U)
    LIntegrator = init(LProblem, alg.alg_params.L_alg; save_everystep=false, alg.alg_params.L_kwargs...)
    step!(LIntegrator, 1e-9, true)
    set_t!(LIntegrator,t)

    SProblem = remake(SProblem, u0 = zeros(2*r,2*r), p = (Uhat, Vhat))
    SIntegrator = init(SProblem, alg.alg_params.S_alg; save_everystep=false, alg.alg_params.S_kwargs...)
    step!(SIntegrator, 1e-9, true)
    set_t!(SIntegrator,t)

    return RankAdaptiveUnconventionalAlgorithm_Cache(US, Uhat, VS, Vhat, M, N, SIntegrator, SProblem, LIntegrator, LProblem,
                                                     KIntegrator, KProblem, r, tol, r_max, nothing, nothing, nothing, nothing)
end

function step!(integrator::DLRIntegrator, alg::RankAdaptiveUnconventionalAlgorithm, dt)
    @unpack u, t, iter, cache = integrator
    u_new, rank_adjusted = rankadaptive_unconventional_step!(u, cache, t, dt)
    if rank_adjusted
        integrator.u = u_new
        integrator.cache = alg_recache(cache, alg, u_new, t+dt)
    end
    integrator.t += dt
    integrator.iter += 1
end

function rankadaptive_unconventional_step!(u, cache, t, dt)
    @unpack r, r_max, tol, US, Uhat, VS, Vhat, M, N, KIntegrator, LIntegrator, SIntegrator, y, ycurr, yprev, Δy = cache
    
    if !isnothing(y)
        ycurr .= y(t+dt)
        Δy .= ycurr - yprev
        yprev .= ycurr
    end

    # K step
    US .= u.U*u.S
    set_u!(KIntegrator, US)
    step!(KIntegrator, dt, true)
    US .= KIntegrator.u
    Uhat[:, 1:r] .= US
    Uhat[:, r+1:end] .= u.U
    QRK = qr(Uhat)
    Uhat .= Matrix(QRK.Q)
    M .= Uhat'*u.U # need to figure out how to leave QRK.Q' as is
    
    
    # L step
    VS .= u.V*u.S'
    set_u!(LIntegrator, VS)
    step!(LIntegrator, dt, true)
    VS .= LIntegrator.u
    Vhat[:, 1:r] .= VS
    Vhat[:, r+1:end] .= u.V 
    QRL = qr(Vhat)
    Vhat .= Matrix(QRL.Q)
    N .= Vhat'*u.V
    
    set_u!(SIntegrator, M*u.S*N')
    step!(SIntegrator, dt, true)

    U, S, V = svd(SIntegrator.u)
    r_new = min(r_max, truncate_to_tolerance(S, tol))
    if r_new == r 
        u.U .= Uhat*U[:,1:r_new]
        u.S .= Matrix(Diagonal(S[1:r_new]))
        u.V .= Vhat*V[:,1:r_new]
        return nothing, false
    else
        return SVDLikeApproximation(Uhat*U[:,1:r_new], Matrix(Diagonal(S[1:r_new])), Vhat*V[:,1:r_new]), true
    end    
end

function truncate_to_tolerance(S, tol)
    s = 0
    r = length(S)
    for σ in reverse(S)
        s += σ^2
        if s > tol^2
            break
        end
        r -= 1
    end 
    return r
end 
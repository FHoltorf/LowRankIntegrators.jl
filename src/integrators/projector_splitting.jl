struct LieTrotterProjectorSplitting_Params{sType, lType, kType}
    S_rhs # rhs of S step (core projected rhs)
    L_rhs # rhs of L step (range projected rhs)
    K_rhs # rhs of K step (corange projected rhs)
    S_kwargs
    L_kwargs
    K_kwargs
    S_alg::sType
    L_alg::lType
    K_alg::kType
end

struct LieTrotterProjectorSplitting_Cache{uType,SIntegratorType,LIntegratorType,KIntegratorType,yType} <: AbstractDLRAlgorithm_Cache
    US::Matrix{uType}
    VS::Matrix{uType}
    QRK::LinearAlgebra.QRCompactWY{uType, Matrix{uType}}
    QRL::LinearAlgebra.QRCompactWY{uType, Matrix{uType}}
    SIntegrator::SIntegratorType
    LIntegrator::LIntegratorType
    KIntegrator::KIntegratorType
    y
    ycurr::yType
    yprev::yType
    Δy::yType
end

struct PrimalLieTrotterProjectorSplitting{sType, lType, kType} <: AbstractDLRAlgorithm
    alg_params::LieTrotterProjectorSplitting_Params{sType, lType, kType}
end
function PrimalLieTrotterProjectorSplitting(;S_rhs = nothing, L_rhs = nothing, K_rhs = nothing, 
                                             S_kwargs = Dict(), L_kwargs = Dict(), K_kwargs = Dict(),
                                             S_alg=Tsit5(), L_alg = Tsit5(), K_alg = Tsit5()) 
    params = LieTrotterProjectorSplitting_Params(S_rhs,L_rhs,K_rhs,S_kwargs,L_kwargs,K_kwargs,S_alg,L_alg,K_alg)
    return PrimalLieTrotterProjectorSplitting(params)
end 

struct DualLieTrotterProjectorSplitting{sType, lType, kType} <: AbstractDLRAlgorithm
    alg_params::LieTrotterProjectorSplitting_Params{sType, lType, kType}
end
function DualLieTrotterProjectorSplitting(;S_rhs = nothing, L_rhs = nothing, K_rhs = nothing,
                                           S_kwargs = Dict(), L_kwargs = Dict(), K_kwargs = Dict(),
                                           S_alg=Tsit5(), L_alg = Tsit5(), K_alg = Tsit5())
    
    params = LieTrotterProjectorSplitting_Params(S_rhs,L_rhs,K_rhs,S_kwargs,L_kwargs,K_kwargs,S_alg,L_alg,K_alg)
    return DualLieTrotterProjectorSplitting(params)          
end

struct StrangProjectorSplitting{sType, lType, kType} <: AbstractDLRAlgorithm
    alg_params::LieTrotterProjectorSplitting_Params{sType, lType, kType}
end
function StrangProjectorSplitting(;S_rhs = nothing, L_rhs = nothing, K_rhs = nothing,
                                   S_kwargs = Dict(), L_kwargs = Dict(), K_kwargs = Dict(),
                                   S_alg=Tsit5(), L_alg = Tsit5(), K_alg = Tsit5())
    params = LieTrotterProjectorSplitting_Params(S_rhs,L_rhs,K_rhs,S_kwargs,L_kwargs,K_kwargs,S_alg,L_alg,K_alg)
    return StrangProjectorSplitting(params)          
end

struct StrangProjectorSplitting_Cache <: AbstractDLRAlgorithm_Cache
    primal_cache::LieTrotterProjectorSplitting_Cache
    dual_cache::LieTrotterProjectorSplitting_Cache
end

function alg_cache(prob::MatrixDEProblem, alg::PrimalLieTrotterProjectorSplitting, u, dt, t0 = prob.tspan[1])
    # the fist integration step is used to allocate memory for frequently accessed arrays
    US = u.U*u.S
    tspan = (t0,t0+dt)
    
    if isnothing(alg.alg_params.K_rhs)
        alg.alg_params.K_rhs = function (US, V, t)
                    return prob.f(US*V',t)*V
                end 
    end
    KProblem = ODEProblem(alg.alg_params.K_rhs, US, tspan, u.V)
    KIntegrator = init(KProblem, alg.alg_params.K_alg, save_everystep=false, alg.alg_params.K_kwargs...)
    step!(KIntegrator, dt, true)
    US .= KIntegrator.u
    QRK = qr!(US)
    u.U .= Matrix(QRK.Q) 

    if isnothing(alg.alg_params.S_rhs)
        alg.alg_params.S_rhs = function (S, p, t)
                    return -p[1]'*prob.f(p[1]*S*p[2]',t)*p[2]
                end
    end    
    SProblem = ODEProblem(alg.alg_params.S_rhs, QRK.R, tspan, (u.U, u.V))
    SIntegrator = init(SProblem, alg.alg_params.S_alg, save_everystep=false, alg.alg_params.S_kwargs...)
    step!(SIntegrator, dt, true)
    VS = u.V*SIntegrator.u'

    if isnothing(alg.alg_params.L_rhs)
        alg.alg_params.L_rhs = function (VS, U, t)
                    return prob.f(U*VS',t)'*U
                end
    end
    LProblem = ODEProblem(alg.alg_params.L_rhs, VS, tspan, u.U)
    LIntegrator = init(LProblem, alg.alg_params.L_alg, save_everystep=false, alg.alg_params.L_kwargs...)
    step!(LIntegrator, dt, true)
    VS .= LIntegrator.u
    QRL = qr!(VS)
    u.V .= Matrix(QRL.Q)
    u.S .= QRL.R'
    return LieTrotterProjectorSplitting_Cache(US, VS, QRK, QRL, 
                                              SIntegrator, LIntegrator, KIntegrator,
                                              nothing, nothing, nothing, nothing)
end

function alg_cache(prob::MatrixDEProblem, alg::DualLieTrotterProjectorSplitting, u, dt, t0 = prob.tspan[1])
    # the fist integration step is used to allocate memory for frequently accessed arrays
    VS = u.V*u.S'
    tspan = (t0,t0+dt)
    
    if isnothing(alg.alg_params.L_rhs)
        alg.alg_params.L_rhs = function (VS, U, t)
                    return prob.f(U*VS',t)'*U
                end
    end
    LProblem = ODEProblem(alg.alg_params.L_rhs, VS, tspan, u.U)
    LIntegrator = init(LProblem, alg.alg_params.L_alg, save_everystep=false, alg.alg_params.L_kwargs...)
    step!(LIntegrator, dt, true)
    VS .= LIntegrator.u
    QRL = qr!(VS)
    u.V .= Matrix(QRL.Q)
    
    if isnothing(alg.alg_params.S_rhs)
        alg.alg_params.S_rhs = function (S, (U,V), t)
                    return -U'*prob.f(U*S*V',t)*V
                end 
    end
    SProblem = ODEProblem(alg.alg_params.S_rhs, Matrix(QRL.R'), tspan, (u.U, u.V))
    SIntegrator = init(SProblem, alg.alg_params.S_alg, save_everystep=false, alg.alg_params.S_kwargs...)
    step!(SIntegrator, dt, true)
    US = u.U*SIntegrator.u

    if isnothing(alg.alg_params.K_rhs)
        alg.alg_params.K_rhs = function (US, V, t)
                    return prob.f(US*V',t)*V
                end
    end
    KProblem = ODEProblem(alg.alg_params.K_rhs, US, tspan, u.V)
    KIntegrator = init(KProblem, alg.alg_params.K_alg, save_everystep=false, alg.alg_params.K_kwargs...)
    step!(KIntegrator, dt, true)
    US .= KIntegrator.u
    QRK = qr!(US)
    u.U .= Matrix(QRK.Q) 
    u.S .= QRK.R

    return LieTrotterProjectorSplitting_Cache(US, VS, QRK, QRL, 
                                              SIntegrator, LIntegrator, KIntegrator,
                                              nothing, nothing, nothing, nothing)
end

function alg_cache(prob::MatrixDEProblem, alg::StrangProjectorSplitting, u, dt)
    primal_cache = alg_cache(prob, PrimalLieTrotterProjectorSplitting(alg.alg_params), u, dt/2)
    dual_cache = alg_cache(prob, DualLieTrotterProjectorSplitting(alg.alg_params), u, dt/2, prob.tspan[1] + dt/2)
    return StrangProjectorSplitting_Cache(primal_cache, dual_cache)
end

function alg_cache(prob::MatrixDataProblem, alg::algType,u,dt) where algType <: Union{PrimalLieTrotterProjectorSplitting, DualLieTrotterProjectorSplitting}
    # creates caches for frequently used arrays by performing the first time step
    @unpack y = prob
    t0 = prob.tspan[1]

    yprev = y(t0)
    ycurr = deepcopy(yprev)
    Δy = similar(yprev)
    US = u.U*u.S
    KIntegrator = MatrixDataIntegrator(Δy, US, I, u.V, 1)
    QRK = qr(US)     
    
    SIntegrator = MatrixDataIntegrator(Δy, QRK.R, u.U, u.V, -1)
    VS = u.V*SIntegrator.u'

    LIntegrator = MatrixDataIntegrator(Δy', VS, I, u.U, 1)
    QRL = qr(VS)
    return LieTrotterProjectorSplitting_Cache(US, VS, QRK, QRL, 
                                              SIntegrator, LIntegrator, KIntegrator,
                                              y, ycurr, yprev, Δy)
end

function alg_cache(prob::MatrixDataProblem, alg::StrangProjectorSplitting, u, dt)
    primal_cache = alg_cache(prob, PrimalLieTrotterProjectorSplitting(alg.alg_params), u, dt/2)
    dual_cache = alg_cache(prob, DualLieTrotterProjectorSplitting(alg.alg_params), u, dt/2)
    return StrangProjectorSplitting_Cache(primal_cache, dual_cache)
end

function init(prob::MatrixDEProblem, alg::algType, dt) where algType <: Union{StrangProjectorSplitting, 
                                                                              PrimalLieTrotterProjectorSplitting, 
                                                                              DualLieTrotterProjectorSplitting}
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
    cache = alg_cache(prob, alg, u, dt)
    sol.Y[2] = deepcopy(u) # add first step to solution object
    return DLRIntegrator(u, t0+dt, dt, sol, alg, cache, 1)   
end

function init(prob::MatrixDataProblem, alg::algType, dt) where algType <: Union{StrangProjectorSplitting,
                                                                                PrimalLieTrotterProjectorSplitting, 
                                                                                DualLieTrotterProjectorSplitting}
    t0, tf = prob.tspan
    @assert tf > t0 "Integration in reverse time direction is not supported"
    u = deepcopy(prob.u0)
    # number of steps
    n = floor(Int,(tf-t0)/dt) + 1 
    # compute more sensible dt # rest will be handled via interpolation/save_at
    dt = (tf-t0)/(n-1)
    # initialize solution object
    sol = DLRSolution(Vector{typeof(prob.u0)}(undef, n), collect(range(t0, tf, length=n)))
    # initialize cache
    cache = alg_cache(prob, alg, u, dt)
    sol.Y[1] = deepcopy(prob.u0)
    return DLRIntegrator(u, t0, dt, sol, alg, cache, 0)
end

function primal_LT_step!(u, cache, t, dt)
    @unpack US, VS, QRK, QRL, KIntegrator, SIntegrator, LIntegrator, y, ycurr, yprev, Δy = cache

    if !isnothing(y) # should be done via dispatch
        ycurr .= y(t+dt)
        Δy .= ycurr - yprev
        yprev .= ycurr
    end

    # K step
    US .= u.U*u.S
    set_u!(KIntegrator, US)
    step!(KIntegrator, dt, true)
    US .= KIntegrator.u
    QRL = qr!(US)
    u.U .= Matrix(QRL.Q)  

    # S step
    set_u!(SIntegrator, QRL.R) 
    step!(SIntegrator, dt, true)
    
    # L step
    VS .= u.V*SIntegrator.u'
    set_u!(LIntegrator, VS)
    step!(LIntegrator, dt, true)
    VS .= LIntegrator.u
    QRL = qr!(VS)
    u.V .= Matrix(QRL.Q)
    u.S .= QRL.R'
end

function dual_LT_step!(u, cache, t, dt)
    @unpack US, VS, QRK, QRL, KIntegrator, SIntegrator, LIntegrator, y, ycurr, yprev, Δy = cache
    if !isnothing(y) # should be done via dispatch
        ycurr .= y(t+dt)
        Δy .= ycurr - yprev
        yprev .= ycurr
    end

    # L step
    VS .= u.V*u.S'
    set_u!(LIntegrator, VS)
    step!(LIntegrator, dt, true)
    VS .= LIntegrator.u
    QRL = qr!(VS)
    u.V .= Matrix(QRL.Q)
    
    # S step
    set_u!(SIntegrator, Matrix(QRL.R'))
    step!(SIntegrator, dt, true)
    
    # K step
    US .= u.U*SIntegrator.u
    set_u!(KIntegrator, US)
    step!(KIntegrator, dt, true)
    US .= KIntegrator.u
    QRK = qr!(US)
    u.U .= Matrix(QRK.Q) 
    u.S .= QRK.R
end

function step!(integrator::DLRIntegrator, alg::PrimalLieTrotterProjectorSplitting, dt)
    @unpack u, t, iter, cache = integrator
    primal_LT_step!(u, cache, t, dt)
    integrator.t += dt
    integrator.iter += 1
end

function step!(integrator::DLRIntegrator, alg::DualLieTrotterProjectorSplitting, dt)
    @unpack u, t, iter, cache = integrator
    dual_LT_step!(u, cache, t, dt)
    integrator.t += dt
    integrator.iter += 1
end

function step!(integrator::DLRIntegrator, alg::StrangProjectorSplitting, dt)
    @unpack u, t, iter, cache = integrator
    @unpack primal_cache, dual_cache = cache
    primal_LT_step!(u, primal_cache, t, dt/2)
    dual_LT_step!(u, dual_cache, t + dt/2, dt/2)
    integrator.t += dt
    integrator.iter += 1
end


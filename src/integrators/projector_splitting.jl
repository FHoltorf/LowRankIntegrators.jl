struct PrimalLieTrotter end
struct DualLieTrotter end
struct Strang end

struct ProjectorSplitting_Params{sType, lType, kType}
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

struct ProjectorSplitting_Cache{uType,SIntegratorType,LIntegratorType,KIntegratorType,yType} <: AbstractDLRAlgorithm_Cache
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

struct ProjectorSplitting{oType, sType, lType, kType} <: AbstractDLRAlgorithm
    order::oType
    alg_params::ProjectorSplitting_Params{sType, lType, kType}
end

function ProjectorSplitting(order = PrimalLieTrotter();S_rhs = nothing, L_rhs = nothing, K_rhs = nothing, 
                             S_kwargs = Dict(), L_kwargs = Dict(), K_kwargs = Dict(),
                             S_alg=Tsit5(), L_alg = Tsit5(), K_alg = Tsit5()) 
    params = ProjectorSplitting_Params(S_rhs,L_rhs,K_rhs,S_kwargs,L_kwargs,K_kwargs,S_alg,L_alg,K_alg)
    return ProjectorSplitting(order, params)
end 

function alg_cache(prob::MatrixDEProblem, alg::ProjectorSplitting, u, dt; t0 = prob.tspan[1])
    # allocate memory for frequently accessed arrays
    tspan = (t0,t0+dt)

    US = u.U*u.S
    VS = u.V*u.S'
    QRK = qr(US)
    QRL = qr(VS)
    
    if isnothing(alg.alg_params.K_rhs)
        K_rhs = function (US, V, t)
                    return Matrix(prob.f(TwoFactorRepresentation(US,V),t)*V)
                end
    else
        K_rhs = alg.alg_params.K_rhs
    end
    KProblem = ODEProblem(K_rhs, US, tspan, u.V)
    KIntegrator = init(KProblem, alg.alg_params.K_alg, save_everystep=false, alg.alg_params.K_kwargs...)
    
    if isnothing(alg.alg_params.S_rhs)
        S_rhs = function (S, (U,V), t)
                    return Matrix(-U'*prob.f(SVDLikeRepresentation(U,S,V),t)*V)
                end
    else
        S_rhs = alg.alg_params.S_rhs
    end
    SProblem = ODEProblem(S_rhs, QRK.R, tspan, (u.U, u.V))
    SIntegrator = init(SProblem, alg.alg_params.S_alg, save_everystep=false, alg.alg_params.S_kwargs...)

    if isnothing(alg.alg_params.L_rhs)
        L_rhs = function (VS, U, t)
                    return Matrix(prob.f(TwoFactorRepresentation(U,VS),t)'*U)
                end
    else
        L_rhs = alg.alg_params.L_rhs
    end
    LProblem = ODEProblem(L_rhs, VS, tspan, u.U)
    LIntegrator = init(LProblem, alg.alg_params.L_alg, save_everystep=false, alg.alg_params.L_kwargs...)
    
    return ProjectorSplitting_Cache(US, VS, QRK, QRL, 
                                    SIntegrator, LIntegrator, KIntegrator,
                                    nothing, nothing, nothing, nothing)
end

function alg_cache(prob::MatrixDataProblem, alg::ProjectorSplitting, u, dt; t0 = prob.tspan[1])
    # creates caches for frequently used arrays by performing the first time step
    @unpack y = prob
    
    yprev = y isa AbstractArray ? deepcopy(y[1]) : y(t0)
    ycurr = deepcopy(yprev)
    Δy = similar(yprev)
    VS = u.V*u.S'
    US = u.U*u.S
    QRK = qr(US)     
    QRL = qr(VS)
    KIntegrator = MatrixDataIntegrator(Δy, US, I, u.V, 1)
    SIntegrator = MatrixDataIntegrator(Δy, QRK.R, u.U, u.V, -1)
    LIntegrator = MatrixDataIntegrator(Δy', VS, I, u.U, 1)
    
    return ProjectorSplitting_Cache(US, VS, QRK, QRL, 
                                    SIntegrator, LIntegrator, KIntegrator,
                                    prob.y, ycurr, yprev, Δy)
end

# function init(prob::AbstractDLRProblem, alg::ProjectorSplitting, dt)
#     t0, tf = prob.tspan
#     @assert tf > t0 "Integration in reverse time direction is not supported"
#     u = deepcopy(prob.u0)
#     sol = init_sol(dt, t0, tf, prob.u0)
#     cache = alg_cache(prob, alg, u, dt, t0 = t0)
#     sol.Y[1] = deepcopy(prob.u0) # add initial point to solution object
#     return DLRIntegrator(u, t0, dt, sol, alg, cache, typeof(prob), 0)   
# end

function primal_LT_step!(u, cache, t, dt, ::Type{<:MatrixDataProblem})
    @unpack y, ycurr, yprev, Δy = cache
    update_data!(ycurr, y, t, dt)
    Δy .= ycurr - yprev
    yprev .= ycurr
    primal_LT_step!(u, cache, t, dt)
end

function primal_LT_step!(u, cache, t, dt, ::Type{<:MatrixDEProblem})
    primal_LT_step!(u, cache, t, dt)
end

function primal_LT_step!(u, cache, t, dt)
    @unpack US, VS, QRK, QRL, KIntegrator, SIntegrator, LIntegrator = cache

    # K step
    mul!(US,u.U,u.S)
    set_u!(KIntegrator, US)
    step!(KIntegrator, dt, true)
    US .= KIntegrator.u
    QRL = qr!(US)
    u.U .= Matrix(QRL.Q)  

    # S step
    set_u!(SIntegrator, QRL.R) 
    step!(SIntegrator, dt, true)
    
    # L step
    mul!(VS,u.V,SIntegrator.u')
    set_u!(LIntegrator, VS)
    step!(LIntegrator, dt, true)
    VS .= LIntegrator.u
    QRL = qr!(VS)
    u.V .= Matrix(QRL.Q)
    u.S .= QRL.R'
end

function dual_LT_step!(u, cache, t, dt, ::Type{<:MatrixDataProblem})
    @unpack y, ycurr, yprev, Δy = cache
    update_data!(ycurr, y, t, dt) 
    Δy .= ycurr - yprev
    yprev .= ycurr
    dual_LT_step!(u, cache, t, dt)
end

function dual_LT_step!(u, cache, t, dt, ::Type{<:MatrixDEProblem})
    dual_LT_step!(u, cache, t, dt)
end

function dual_LT_step!(u, cache, t, dt)
    @unpack US, VS, QRK, QRL, KIntegrator, SIntegrator, LIntegrator = cache

    # L step
    mul!(VS,u.V,u.S')
    set_u!(LIntegrator, VS)
    step!(LIntegrator, dt, true)
    VS .= LIntegrator.u
    QRL = qr!(VS)
    u.V .= Matrix(QRL.Q)
    
    # S step
    set_u!(SIntegrator, Matrix(QRL.R'))
    step!(SIntegrator, dt, true)
    
    # K step
    mul!(US,u.U,SIntegrator.u)
    set_u!(KIntegrator, US)
    step!(KIntegrator, dt, true)
    US .= KIntegrator.u
    QRK = qr!(US)
    u.U .= Matrix(QRK.Q) 
    u.S .= QRK.R
end

function step!(integrator::DLRIntegrator, ::ProjectorSplitting{PrimalLieTrotter,S,L,K}, dt) where {S,L,K}
    @unpack u, t, iter, cache, probType = integrator
    primal_LT_step!(u, cache, t, dt, probType)
    integrator.t += dt
    integrator.iter += 1
end

function step!(integrator::DLRIntegrator, ::ProjectorSplitting{DualLieTrotter,S,L,K}, dt) where {S,L,K}
    @unpack u, t, iter, cache, probType = integrator
    dual_LT_step!(u, cache, t, dt, probType)
    integrator.t += dt
    integrator.iter += 1
end

function step!(integrator::DLRIntegrator, ::ProjectorSplitting{Strang,S,L,K}, dt) where {S,L,K}
    @unpack u, t, iter, cache, probType = integrator
    primal_LT_step!(u, cache, t, dt/2, probType)
    dual_LT_step!(u, cache, t + dt/2, dt/2, probType)
    integrator.t += dt
    integrator.iter += 1
end


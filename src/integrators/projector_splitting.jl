struct PrimalLieTrotter end
struct DualLieTrotter end
struct Strang end

struct ProjectorSplitting_Params{sType, lType, kType, iType}
    S_rhs # rhs of S step (core projected rhs)
    L_rhs # rhs of L step (range projected rhs)
    K_rhs # rhs of K step (corange projected rhs)
    S_kwargs
    L_kwargs
    K_kwargs
    S_alg::sType
    L_alg::lType
    K_alg::kType
    interpolation::iType
end

struct ProjectorSplitting_Cache{uType,
                                SIntegratorType,LIntegratorType,KIntegratorType,
                                yFunc,yType,dType} <: AbstractDLRAlgorithm_Cache
    US::Matrix{uType}
    VS::Matrix{uType}
    QRK::LinearAlgebra.QRCompactWY{uType, Matrix{uType}}
    QRL::LinearAlgebra.QRCompactWY{uType, Matrix{uType}}
    SIntegrator::SIntegratorType
    LIntegrator::LIntegratorType
    KIntegrator::KIntegratorType
    y::yFunc
    ycurr::yType
    yprev::yType
    Δy::yType
    interpolation_cache::dType
end

struct ProjectorSplitting{oType, sType, lType, kType} <: AbstractDLRAlgorithm
    order::oType
    alg_params::ProjectorSplitting_Params{sType, lType, kType}
end

function ProjectorSplitting(order = PrimalLieTrotter(), deim = nothing;S_rhs = nothing, L_rhs = nothing, K_rhs = nothing, 
                            S_kwargs = Dict(), L_kwargs = Dict(), K_kwargs = Dict(),
                            S_alg=Tsit5(), L_alg = Tsit5(), K_alg = Tsit5()) 
    params = ProjectorSplitting_Params(S_rhs,L_rhs,K_rhs,
                                       S_kwargs,L_kwargs,K_kwargs,
                                       S_alg,L_alg,K_alg,
                                       deim)
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
    KIntegrator = init(KProblem, alg.alg_params.K_alg; save_everystep=false, alg.alg_params.K_kwargs...)
    step!(KIntegrator, 0.01*dt, true)
    set_t!(KIntegrator, t0)
    
    if isnothing(alg.alg_params.S_rhs)
        S_rhs = function (S, (U,V), t)
                    return Matrix(-U'*prob.f(SVDLikeRepresentation(U,S,V),t)*V)
                end
    else
        S_rhs = alg.alg_params.S_rhs
    end
    SProblem = ODEProblem(S_rhs, QRK.R, tspan, (u.U, u.V))
    SIntegrator = init(SProblem, alg.alg_params.S_alg; save_everystep=false, alg.alg_params.S_kwargs...)
    step!(SIntegrator, 0.01*dt, true)
    set_t!(SIntegrator, t0)

    if isnothing(alg.alg_params.L_rhs)
        L_rhs = function (VS, U, t)
                    return Matrix(prob.f(TwoFactorRepresentation(U,VS),t)'*U)
                end
    else
        L_rhs = alg.alg_params.L_rhs
    end
    LProblem = ODEProblem(L_rhs, VS, tspan, u.U)
    LIntegrator = init(LProblem, alg.alg_params.L_alg; save_everystep=false, alg.alg_params.L_kwargs...)
    step!(LIntegrator, 0.01*dt, true)
    set_t!(LIntegrator, t0)

    return ProjectorSplitting_Cache(US, VS, QRK, QRL, 
                                    SIntegrator, LIntegrator, KIntegrator,
                                    nothing, nothing, nothing, nothing, nothing)
end

function alg_cache(prob::MatrixDEProblem{fType, uType, tType}, 
                   alg::ProjectorSplitting, u, dt; t0 = prob.tspan[1]) where
                   {fType <: ComponentFunction, uType, tType}
    tspan = (t0,t0+dt)

    @unpack tol, rmin, rmax, 
            init_range, init_corange,
            selection_alg = alg.alg_params.interpolation

    row_idcs = index_selection(init_range, selection_alg)
    col_idcs = index_selection(init_corange, selection_alg)

    Π_corange = DEIMInterpolator(col_idcs, init_corange/init_corange[col_idcs,:])
    Π_range = DEIMInterpolator(row_idcs, init_range/init_range[row_idcs,:])
    Π = SparseFunctionInterpolator(prob.f, SparseMatrixInterpolator(Π_range, Π_corange))
    
    US = u.U*u.S
    VS = u.V*u.S'
    QRK = qr(US)
    QRL = qr(VS)
    
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
    step!(KIntegrator,0.01*dt,true)
    set_t!(KIntegrator, t0)

    if isnothing(alg.alg_params.S_rhs)
        S_rhs = function (dS, S, p, t)
                    Π_S, U1, V1, params = p
                    Π_S(dS, SVDLikeRepresentation(U1,S,V1), params, t) 
                end
    else
        S_rhs = alg.alg_params.S_rhs
    end
    Π_S_mat = SparseMatrixInterpolator(row_idcs, col_idcs, 
                                        u.U'*Π.interpolator.range.weights, 
                                        u.V'*Π.interpolator.corange.weights)
    Π_S = SparseFunctionInterpolator(prob.f, Π_S_mat)
    p_S = (Π_S, u.U, u.V, ())

    SProblem = ODEProblem(S_rhs, QRK.R, tspan, p_S)
    SIntegrator = init(SProblem, alg.alg_params.S_alg; save_everystep=false, alg.alg_params.S_kwargs...)
    step!(SIntegrator,0.01*dt,true)
    set_t!(SIntegrator, t0)

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
    step!(LIntegrator,0.01*dt,true)
    set_t!(LIntegrator, t0)

    
    interpolation_cache = OnTheFlyInterpolation_Cache(alg.alg_params.interpolation, deepcopy(u),
                                                      Π, Π_K, Π_L, Π_S)
    return ProjectorSplitting_Cache(US, VS, QRK, QRL, 
                                    SIntegrator, LIntegrator, KIntegrator,
                                    nothing, nothing, nothing, nothing, interpolation_cache)
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
                                    prob.y, ycurr, yprev, Δy, nothing)
end

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

function primal_LT_step!(u, cache, t, dt, ::Type{<:MatrixDEProblem{<:ComponentFunction}})
    primal_LT_deim_step!(u, cache, t, dt)
end

function primal_LT_deim_step!(u, cache, t, dt)
    @unpack US, VS, QRK, QRL, 
            KIntegrator, LIntegrator, SIntegrator,
            interpolation_cache = cache
    @unpack params, u_prev, Π, Π_L, Π_K, Π_S = interpolation_cache

    if params.update_scheme == :avg_flow
        u_prev.U = copy(u.U)
        u_prev.S = copy(u.S)
        u_prev.V = copy(u.V)
    end

    # K step
    mul!(US,u.U,u.S)
    set_u!(KIntegrator, US)
    step!(KIntegrator, dt, true)
    US .= KIntegrator.u
    QRK = qr!(US)
    u.U .= Matrix(QRK.Q)  

    # S step
    mul!(Π_S.interpolator.range.weights, u.U', Π.interpolator.range.weights, -1.0, 0)
    set_u!(SIntegrator, QRK.R)
    step!(SIntegrator, dt, true)

    # L step
    mul!(Π_L.interpolator.weights, u.U', Π.interpolator.range.weights)
    mul!(VS,u.V,SIntegrator.u')
    set_u!(LIntegrator, VS)
    step!(LIntegrator, dt, true)
    VS .= LIntegrator.u
    QRL = qr!(VS)
    u.V .= Matrix(QRL.Q)
    u.S .= QRL.R'

    update_interpolation!(interpolation_cache, u, t, dt)
end

function dual_LT_step!(u, cache, t, dt, ::Type{<:MatrixDEProblem{<:ComponentFunction}})
    dual_LT_deim_step!(u, cache, t, dt)
end

function dual_LT_deim_step!(u, cache, t, dt)
    @unpack US, VS, QRK, QRL, 
            KIntegrator, LIntegrator, SIntegrator,
            interpolation_cache = cache
    @unpack params, u_prev, Π, Π_L, Π_K, Π_S = interpolation_cache

    if params.update_scheme == :avg_flow
        u_prev.U = copy(u.U)
        u_prev.S = copy(u.S)
        u_prev.V = copy(u.V)
    end

    # L step
    mul!(VS,u.V,u.S')
    set_u!(LIntegrator, VS)
    step!(LIntegrator, dt, true)
    VS .= LIntegrator.u
    QRL = qr!(VS)
    u.V .= Matrix(QRL.Q)
    
    # S step
    mul!(Π_S.interpolator.corange.weights, u.V', Π.interpolator.corange.weights, -1, 0)
    set_u!(SIntegrator, Matrix(QRL.R'))
    step!(SIntegrator, dt, true)
    
    # K step
    mul!(US,u.U,SIntegrator.u)
    mul!(Π_K.interpolator.parent.weights, u.V', Π.interpolator.corange.weights)
    set_u!(KIntegrator, US)
    step!(KIntegrator, dt, true)
    US .= KIntegrator.u
    QRK = qr!(US)
    u.U .= Matrix(QRK.Q) 
    u.S .= QRK.R

    update_interpolation!(interpolation_cache, u, t, dt)
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
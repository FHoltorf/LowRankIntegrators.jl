struct UnconventionalAlgorithm_Params{sType, lType, kType, iType} 
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

struct UnconventionalAlgorithm{sType, lType, kType} <: AbstractDLRAlgorithm
    alg_params::UnconventionalAlgorithm_Params{sType, lType, kType}
end
function UnconventionalAlgorithm(; S_rhs = nothing, L_rhs = nothing, K_rhs = nothing,
                                   S_kwargs = Dict(), L_kwargs = Dict(), K_kwargs = Dict(),
                                   S_alg = Tsit5(), L_alg = Tsit5(), K_alg = Tsit5())
    params = UnconventionalAlgorithm_Params(S_rhs, L_rhs, K_rhs, 
                                            S_kwargs, L_kwargs, K_kwargs,
                                            S_alg, L_alg, K_alg, 
                                            nothing)
    return UnconventionalAlgorithm(params)
end
function UnconventionalAlgorithm(deim; S_rhs = nothing, L_rhs = nothing, K_rhs = nothing,
                                 S_kwargs = Dict(), L_kwargs = Dict(), K_kwargs = Dict(),
                                 S_alg = Tsit5(), L_alg = Tsit5(), K_alg = Tsit5())
    params = UnconventionalAlgorithm_Params(S_rhs, L_rhs, K_rhs, 
                                                S_kwargs, L_kwargs, K_kwargs,
                                                S_alg, L_alg, K_alg, 
                                                deim)
    return UnconventionalAlgorithm(params)
end

struct UnconventionalAlgorithm_Cache{uType,
                                     SIntegratorType,LIntegratorType,KIntegratorType,
                                     yFunc, yType, dType} <: AbstractDLRAlgorithm_Cache
    US::Matrix{uType}
    VS::Matrix{uType}
    M::Matrix{uType}
    N::Matrix{uType}
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

rank_DEIM(cache::UnconventionalAlgorithm_Cache) = rank_DEIM(cache.interpolation_cache)
rank_DEIM(::Nothing) = 0
rank_DEIM(cache::OnTheFlyInterpolation_Cache) = rank(cache.Π)[1]
interpolation_indices(cache::UnconventionalAlgorithm_Cache) = interpolation_indices(cache.interpolation_cache)
interpolation_indices(::Nothing) = (Int[],Int[])
interpolation_indices(cache::OnTheFlyInterpolation_Cache) = interpolation_indices(cache.Π)

function alg_cache(prob::MatrixDEProblem, alg::UnconventionalAlgorithm, u, dt; t0 = prob.tspan[1])
    # allocate memory for frequently accessed arrays
    tspan = (t0,t0+dt)

    US = u.U*u.S
    QRK = qr(US)
    M = Matrix(QRK.Q)'*u.U
    
    VS = u.V*u.S'
    QRL = qr(VS)
    N = Matrix(QRL.Q)'*u.V
    
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
    SProblem = ODEProblem(S_rhs, M*u.S*N', tspan, (u.U, u.V))
    SIntegrator = init(SProblem, alg.alg_params.S_alg; save_everystep=false, alg.alg_params.S_kwargs...)
    
    return UnconventionalAlgorithm_Cache(US, VS, M, N, QRK, QRL, 
                                         SIntegrator, LIntegrator, KIntegrator,
                                         nothing, nothing, nothing, nothing, nothing)
end


function alg_cache(prob::MatrixDEProblem{fType, uType, tType}, 
                   alg::UnconventionalAlgorithm, u, dt; t0 = prob.tspan[1]) where
                   {fType <: ComponentFunction, uType, tType}
    # allocate memory for frequently accessed arrays
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
                                       u.U'*Π.interpolator.range.weights, 
                                       u.V'*Π.interpolator.corange.weights)
    Π_S = SparseFunctionInterpolator(prob.f, Π_S_mat)
    p_S = (Π_S, u.U, u.V, ())

    SProblem = ODEProblem(S_rhs, M*u.S*N', tspan, p_S)
    SIntegrator = init(SProblem, alg.alg_params.S_alg; save_everystep=false, alg.alg_params.S_kwargs...)
    
    interpolation_cache = OnTheFlyInterpolation_Cache(alg.alg_params.interpolation, deepcopy(u),
                                                   Π, Π_K, Π_L, Π_S)
    return UnconventionalAlgorithm_Cache(US, VS, M, N, QRK, QRL, 
                                         SIntegrator, LIntegrator, KIntegrator,
                                         nothing, nothing, nothing, nothing, interpolation_cache)
end


function alg_cache(prob::MatrixDataProblem, alg::UnconventionalAlgorithm, u, dt; t0 = prob.tspan[1])
    n, r = size(u.U)
    m = size(u.V, 1)
    @unpack y = prob

    yprev = y isa AbstractArray ? deepcopy(y[1]) : y(t0)
    ycurr = deepcopy(yprev)
    Δy = similar(yprev)
    US = zeros(eltype(u.U),n,r)
    QRK = qr(US)     
    M = zeros(eltype(u.U),r,r)
    VS = zeros(eltype(u.V), m, r)
    QRL = qr(VS)
    N = zeros(eltype(u.U),r,r)
    KIntegrator = MatrixDataIntegrator(Δy, US, I, u.V, 1)
    LIntegrator = MatrixDataIntegrator(Δy', VS, I, u.U, 1)   
    SIntegrator = MatrixDataIntegrator(Δy, M*u.S*N', u.U, u.V, 1)

    return UnconventionalAlgorithm_Cache(US, VS, M, N, QRK, QRL, 
                                         SIntegrator, LIntegrator, KIntegrator,
                                         prob.y, ycurr, yprev, Δy, nothing)
end

function unconventional_step!(u, cache, t, dt, ::Type{<:MatrixDataProblem})
    @unpack y, ycurr, yprev, Δy = cache
    update_data!(ycurr, y, t, dt)
    Δy .= ycurr - yprev
    yprev .= ycurr
    unconventional_step!(u, cache, t, dt)
end

function unconventional_step!(u, cache, t, dt, ::Type{<:MatrixDEProblem})
    unconventional_step!(u, cache, t, dt)
end

function unconventional_step!(u, cache, t, dt, ::Type{<:MatrixDEProblem{<:ComponentFunction}})
    unconventional_deim_step!(u, cache, t, dt)
end

function unconventional_step!(u, cache, t, dt)
    @unpack US, VS, M, N, QRK, QRL, 
            KIntegrator, LIntegrator, SIntegrator = cache
    
    # K step
    mul!(US, u.U, u.S)
    set_u!(KIntegrator, US)
    step!(KIntegrator, dt, true)
    US .= KIntegrator.u
    QRK = qr!(US)
    mul!(M, Matrix(QRK.Q)', u.U)
    
    # L step
    mul!(VS, u.V, u.S')
    set_u!(LIntegrator, VS)
    step!(LIntegrator, dt, true)
    VS .= LIntegrator.u
    QRL = qr!(VS)
    mul!(N,Matrix(QRL.Q)',u.V)
    u.V .= Matrix(QRL.Q)
    u.U .= Matrix(QRK.Q)
    
    set_u!(SIntegrator, M*u.S*N')
    step!(SIntegrator, dt, true)
    u.S .= SIntegrator.u
end

function unconventional_deim_step!(u, cache, t, dt)
    @unpack US, VS, M, N, QRK, QRL, 
            KIntegrator, LIntegrator, SIntegrator,
            interpolation_cache = cache
    @unpack params, u_prev, Π, Π_L, Π_K, Π_S = interpolation_cache

    if params.update_scheme == :avg_flow
        u_prev.U = copy(u.U)
        u_prev.S = copy(u.S)
        u_prev.V = copy(u.V)
    end

    # K step
    mul!(US, u.U, u.S)
    set_u!(KIntegrator, US)
    step!(KIntegrator, dt, true)
    US .= KIntegrator.u
    QRK = qr!(US)
    mul!(M, Matrix(QRK.Q)', u.U)
    
    # L step
    mul!(VS, u.V, u.S')
    set_u!(LIntegrator, VS)
    step!(LIntegrator, dt, true)
    VS .= LIntegrator.u
    QRL = qr!(VS)
    mul!(N,Matrix(QRL.Q)',u.V)
    u.V .= Matrix(QRL.Q)
    u.U .= Matrix(QRK.Q)
    
    # update core interpolator
    mul!(Π_S.interpolator.range.weights, u.U', Π.interpolator.range.weights)
    mul!(Π_S.interpolator.corange.weights, u.V', Π.interpolator.corange.weights)
    
    # integration
    set_u!(SIntegrator, M*u.S*N')
    step!(SIntegrator, dt, true)
    u.S .= SIntegrator.u

    update_interpolation!(interpolation_cache, u, t, dt)
end

function step!(integrator::DLRIntegrator, ::UnconventionalAlgorithm, dt)
    @unpack u, t, iter, cache, probType = integrator
    unconventional_step!(u, cache, t, dt, probType)
    integrator.t += dt
    integrator.iter += 1
end
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

struct SparseInterpolation
    selection_alg
    tol
    rmin
    rmax
    init_range
    init_corange
end

function SparseInterpolation(selection_alg, init_range, init_corange;
                             rmin = 1, rmax = min(size(init_range,1), size(init_corange,1)), tol = eps(Float64))
    return SparseInterpolation(selection_alg, tol, rmin, rmax, init_range, init_corange)
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

struct UnconventionalDEIM_Cache
    params::SparseInterpolation
    Π::SparseFunctionInterpolator
    Π_K::SparseFunctionInterpolator
    Π_L::SparseFunctionInterpolator
    Π_S::SparseFunctionInterpolator
end

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
                   {fType <: ComponentFunction}
    # allocate memory for frequently accessed arrays
    tspan = (t0,t0+dt)
    
    @unpack tol, rmin, rmax, 
            init_range, init_corange,
            selection_alg = alg.alg_params.interpolation

    row_idcs = index_selection(init_range, selection_alg)
    col_idcs = index_selection(init_corange, selection_alg)

    Π_corange = DEIMInterpolator(col_idcs, init_range/init_range[col_idcs,:])
    Π_range = DEIMInterpolator(row_idcs, init_corange/init_corange[row_idcs,:])
    Π = SparseFunctionInterpolator(F, SparseMatrixInterpolator(Π_range, Π_corange))
    
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
                                   V0'*Π.interpolator.corange.weights)'
    Π_K = SparseFunctionInterpolator(prob.F, Π_K_corange)
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
                                 U0'*Π.interpolator.range.weights)
    Π_K = SparseFunctionInterpolator(prob.F, Π_L_range)
    p_K = (Π_L, u.U, ())
    LProblem = ODEProblem(L_rhs, VS, tspan, u.U)
    LIntegrator = init(LProblem, alg.alg_params.L_alg; save_everystep=false, alg.alg_params.L_kwargs...)
    
    if isnothing(alg.alg_params.S_rhs)
        S_rhs = function (dS, S, p, t)
                    Π_S, U1, V1, params = p
                    Π_S(dS, SVDLikeRepresentation(U1,S,V1), params, t)
                end
    else
        S_rhs = alg.alg_params.S_rhs
    end
    Π_S_mat = SparseMatrixInterpolator(row_dics, col_idcs, 
                                       u.U'*Π.interpolator.range.weights, 
                                       u.V'*Π.interpolator.corange.weights)
    Π_S = SparseFunctionInterpolator(prob.F, Π_S_mat)
    p_S = (Π_S, u.U, u.V, ())

    SProblem = ODEProblem(S_rhs, M*u.S*N', tspan, p_S)
    SIntegrator = init(SProblem, alg.alg_params.S_alg; save_everystep=false, alg.alg_params.S_kwargs...)
    
    interpolation_cache = UnconventionalDEIM_Cache(alg.alg_params.interpolation, Π, Π_K, Π_L, Π_S)
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

function init(prob::AbstractDLRProblem, alg::UnconventionalAlgorithm, dt)
    t0, tf = prob.tspan
    @assert tf > t0 "Integration in reverse time direction is not supported"
    u = deepcopy(prob.u0)
    # initialize solution 
    sol = init_sol(dt, t0, tf, prob.u0)
    # initialize cache
    cache = alg_cache(prob, alg, u, dt, t0 = t0)
    sol.Y[1] = deepcopy(prob.u0) 
    return DLRIntegrator(u, t0, dt, sol, alg, cache, typeof(prob), 0)   
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
    @unpack Π, Π_L, Π_K, Π_S = interpolation_cache

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
    mul!(Π_S.range.weights, u.U', Π.interpolator.range.weights)
    mul!(Π_S.corange.weights, u.V', Π.interpolator.corange.weights)
    
    # integration
    set_u!(SIntegrator, M*u.S*N')
    step!(SIntegrator, dt, true)
    u.S .= SIntegrator.u

    update_interpolation!(interpolation_cache, u)
end

function update_interpolation!(interpolation_cache, u)
    @unpack params, Π, Π_K, Π_L, Π_S = interpolation_cache
    @unpack selection_method, tol, rmin, rmax = params

    eval!(Π_L, u, (), SIntegrator.t) # rows
    eval!(Π_K, u, (), SIntegrator.t) # columns
    VF = truncated_svd(Π_L.cache.rows, tol=tol, rmin=rmin, rmax=rmax).V # corange from rows
    UF = truncated_svd(Π_K.cache.cols, tol=tol, rmin=rmin, rmax=rmax).U # range from cols
    
    # find interpolation indices
    row_idcs = index_selection(UF, selection_method)
    col_idcs = index_selection(VF, selection_method)
    
    # find interpolation weights
    @views range_weights = UF/UF[row_idcs,:]
    @views corange_weights = VF/VF[col_idcs,:]
    
    # new interpolators
    range = DEIMInterpolator(row_idcs, range_weights)
    projected_range = DEIMInterpolator(row_idcs, u.U'*range_weights)
    corange = DEIMInterpolator(col_idcs, corange_weights)
    projected_corange = DEIMInterpolator(row_idcs, u.V'*corange_weights)

    # update function interpolators
    update_interpolator!(Π_L, projected_range)
    update_interpolator!(Π_K, projected_corange')
    update_interpolator!(Π_S, SparseMatrixInterpolator(projected_range, projected_corange))
    update_interpolator!(Π, SparseMatrixInterpolator(range, corange))
end

function step!(integrator::DLRIntegrator, ::UnconventionalAlgorithm, dt)
    @unpack u, t, iter, cache, probType = integrator
    unconventional_step!(u, cache, t, dt, probType)
    integrator.t += dt
    integrator.iter += 1
end

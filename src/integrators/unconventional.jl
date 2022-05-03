struct UnconventionalAlgorithm_Params{sType, lType, kType} 
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

struct UnconventionalAlgorithm{sType, lType, kType} <: AbstractDLRAlgorithm
    alg_params::UnconventionalAlgorithm_Params{sType, lType, kType}
end
function UnconventionalAlgorithm(; S_rhs = nothing, L_rhs = nothing, K_rhs = nothing,
                                   S_kwargs = Dict(), L_kwargs = Dict(), K_kwargs = Dict(),
                                   S_alg = Tsit5(), L_alg = Tsit5(), K_alg = Tsit5())
    params = UnconventionalAlgorithm_Params(S_rhs, L_rhs, K_rhs, S_kwargs, L_kwargs, K_kwargs, S_alg, L_alg, K_alg)
    return UnconventionalAlgorithm(params)
end

struct UnconventionalAlgorithm_Cache{uType,SIntegratorType,LIntegratorType,KIntegratorType,yType} <: AbstractDLRAlgorithm_Cache
    US::Matrix{uType}
    VS::Matrix{uType}
    M::Matrix{uType}
    N::Matrix{uType}
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
                                         nothing, nothing, nothing, nothing)
end

function alg_cache(prob::MatrixDataProblem, alg::UnconventionalAlgorithm,u,dt; t0 = prob.tspan[1])
    n, r = size(u.U)
    m = size(u.V, 1)

    yprev = prob.y(t0)
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
                                         prob.y, ycurr, yprev, Δy)
end

function init(prob::AbstractDLRProblem, alg::UnconventionalAlgorithm, dt)
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
    cache = alg_cache(prob, alg, u, dt, t0 = t0)
    sol.Y[1] = deepcopy(prob.u0) 
    return DLRIntegrator(u, t0, dt, sol, alg, cache, typeof(prob), 0)   
end

function unconventional_step!(u, cache, t, dt, ::Type{<:MatrixDataProblem})
    @unpack y, ycurr, yprev, Δy = cache
    ycurr .= y(t+dt)
    Δy .= ycurr - yprev
    yprev .= ycurr
    unconventional_step!(u, cache, t, dt)
end

function unconventional_step!(u, cache, t, dt, ::Type{<:MatrixDEProblem})
    unconventional_step!(u, cache, t, dt)
end

function unconventional_step!(u, cache, t, dt)
    @unpack US, VS, M, N, QRK, QRL, KIntegrator, LIntegrator, SIntegrator = cache
    
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

function step!(integrator::DLRIntegrator, ::UnconventionalAlgorithm, dt)
    @unpack u, t, iter, cache, probType = integrator
    unconventional_step!(u, cache, t, dt, probType)
    integrator.t += dt
    integrator.iter += 1
end
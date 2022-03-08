# greedy approach to fit 
struct GreedyIntegrator_Cache
    Y
    X
    XZ
    XV
    XU
    ZIntegrator
end 

struct GreedyIntegrator_Params
    Z_alg
    Z_kwargs 
end

struct GreedyIntegrator <: AbstractDLRAlgorithm 
    alg_params::GreedyIntegrator_Params
end
function GreedyIntegrator(; Z_alg = Tsit5(), Z_kwargs = Dict{Symbol,Any}())
    params = GreedyIntegrator_Params(Z_alg, Z_kwargs)
    return GreedyIntegrator(params)
end

function alg_cache(prob::MatrixDataProblem, alg::GreedyIntegrator, u::TwoFactorApproximation, dt)
    X = zeros(size(u))
    r = rank(u)
    n = size(X,1)
    XZ = zeros(n,r)
    return GreedyIntegrator_Cache(prob.y, X, XZ, nothing, nothing, nothing)
end

function alg_cache(prob::MatrixDataProblem, alg::GreedyIntegrator, u::SVDLikeApproximation, dt)
    X = zeros(size(u))
    r = rank(u)
    n, m = size(X)
    XV = zeros(n,r)
    XU = zeros(m,r) 
    return GreedyIntegrator_Cache(prob.y, X, nothing, XV, XU, nothing)
end

function alg_cache(prob::MatrixHybridProblem, alg::GreedyIntegrator, u, dt)
    XZ = zeros(size(u,1), rank(u))
    ZProblem = ODEProblem(prob.f, u.Z, prob.tspan, u.U)
    ZIntegrator = init(ZProblem, alg.alg_params.Z_alg; save_everystep=false, alg.alg_params.Z_kwargs...)
    return GreedyIntegrator_Cache(prob.y, nothing, XZ, nothing, nothing, ZIntegrator)
end

function init(prob::MatrixDataProblem, alg::GreedyIntegrator, dt)
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
    return DLRIntegrator(u, t0, dt, sol, alg, cache, typeof(prob), 0)
end

function init(prob::MatrixHybridProblem, alg::GreedyIntegrator, dt)
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
    return DLRIntegrator(u, t0, dt, sol, alg, cache, typeof(prob), 0)
end

function greedy_step!(u::TwoFactorApproximation, cache, t, dt, ::Type{<:MatrixHybridProblem})
    @unpack Z, U = u
    @unpack Y, ZIntegrator, XZ = cache
    step!(ZIntegrator, dt, true) 
    Z .= ZIntegrator.u
    mul!(XZ, Y(t+dt), Z)
    Q, _, P = svd(XZ)
    mul!(U, Q, P') 
end

function greedy_step!(u::TwoFactorApproximation, cache, t, dt, ::Type{<:MatrixDataProblem})
    @unpack Z, U = u 
    @unpack Y, X, XZ = cache
    X .= Y(t+dt)
    mul!(Z, X', U)
    mul!(XZ, X, Z)
    Q, _, P = svd(XZ)
    mul!(U, Q, P') 
end

function greedy_step!(u::SVDLikeApproximation, cache, t, dt, ::Type{<:MatrixDataProblem})
    @unpack U, S, V = u 
    @unpack Y, X, XV, XU = cache
    X .= Y(t+dt)
    mul!(XV,X,V)
    mul!(XU,X',U)
    U .= Matrix(qr(XV).Q)
    V .= Matrix(qr(XU).Q)
    mul!(XV,X,V)
    mul!(S,U',XV)
end

function step!(integrator::DLRIntegrator, ::GreedyIntegrator, dt)
    @unpack u, t, iter, cache, probType = integrator
    greedy_step!(u, cache, t, dt, probType)
    integrator.t += dt
    integrator.iter += 1
end
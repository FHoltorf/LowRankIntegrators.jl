# greedy approach to fit 
struct GreedyIntegrator_Cache <: AbstractDLRAlgorithm_Cache
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

function alg_cache(prob::MatrixDataProblem, alg::GreedyIntegrator, u::TwoFactorRepresentation, dt; t0 = prob.tspan[1])
    X = zeros(size(u))
    r = rank(u)
    n = size(X,1)
    XZ = zeros(n,r)
    return GreedyIntegrator_Cache(prob.y, X, XZ, nothing, nothing, nothing)
end

function alg_cache(prob::MatrixDataProblem, alg::GreedyIntegrator, u::SVDLikeRepresentation, dt; t0 = prob.tspan[1])
    X = zeros(size(u))
    r = rank(u)
    n, m = size(X)
    XV = zeros(n,r)
    XU = zeros(m,r) 
    return GreedyIntegrator_Cache(prob.y, X, nothing, XV, XU, nothing)
end

function alg_cache(prob::MatrixHybridProblem, alg::GreedyIntegrator, u, dt; t0 = prob.tspan[1])
    XZ = zeros(size(u,1), rank(u))
    X = zeros(size(u)...)
    ZProblem = ODEProblem(prob.f, u.Z, prob.tspan, u.U)
    ZIntegrator = init(ZProblem, alg.alg_params.Z_alg; save_everystep=false, alg.alg_params.Z_kwargs...)
    return GreedyIntegrator_Cache(prob.y, X, XZ, nothing, nothing, ZIntegrator)
end

function init(prob::MatrixHybridProblem, alg::GreedyIntegrator, dt)
    t0, tf = prob.tspan
    @assert tf > t0 "Integration in reverse time direction is not supported"
    u = deepcopy(prob.u0)
    # initialize solution 
    sol = init_sol(dt, t0, tf, prob.u0)
    # initialize cache
    cache = alg_cache(prob, alg, u, dt)
    sol.Y[1] = deepcopy(prob.u0)
    return DLRIntegrator(u, t0, dt, sol, alg, cache, typeof(prob), 0)
end

function greedy_step!(u::TwoFactorRepresentation, cache, t, dt, ::Type{<:MatrixHybridProblem})
    @unpack Z, U = u
    @unpack X, Y, ZIntegrator, XZ = cache
    step!(ZIntegrator, dt, true) 
    Z .= ZIntegrator.u
    update_data!(X, Y, t, dt)
    mul!(XZ, X, Z)
    Q, _, P = svd(XZ)
    mul!(U, Q, P') 
end

function greedy_step!(u::TwoFactorRepresentation, cache, t, dt, ::Type{<:MatrixDataProblem})
    @unpack Z, U = u 
    @unpack Y, X, XZ = cache
    update_data!(X,Y,t,dt)
    mul!(Z, X', U)
    mul!(XZ, X, Z)
    Q, _, P = svd(XZ)
    mul!(U, Q, P') 
end

function greedy_step!(u::SVDLikeRepresentation, cache, t, dt, ::Type{<:MatrixDataProblem})
    @unpack U, S, V = u 
    @unpack Y, X, XV, XU = cache
    update_data!(X,Y,t,dt)
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
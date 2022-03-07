# greedy approach to fit 
struct GreedyIntegrator <: AbstractDLRAlgorithm end

struct GreedyIntegrator_Cache
    Y
    X
    XZ
    XV
    XU
end 

function GreedyIntegrator_Cache(y, tspan, u::TwoFactorApproximation)
    X = y(tspan[1])
    r = rank(u)
    n = size(X,1)
    XZ = zeros(n,r)
    return GreedyIntegrator_Cache(y, X, XZ, nothing, nothing)
end

function GreedyIntegrator_Cache(y, tspan, u::SVDLikeApproximation)
    X = y(tspan[1])
    r = rank(u)
    n, m = size(X)
    XV = zeros(n,r)
    XU = zeros(m,r) 
    return GreedyIntegrator_Cache(y, X, nothing, XV, XU)
end

function alg_cache(prob::MatrixDataProblem, alg::GreedyIntegrator, u, dt)
    return GreedyIntegrator_Cache(prob.y, prob.tspan, u)
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
    return DLRIntegrator(u, t0, dt, sol, alg, cache, 0)
end

function greedy_step!(u::TwoFactorApproximation, cache, t, dt)
    @unpack Z, U = u 
    @unpack Y, X = cache
    X .= Y(t)
    mul!(Z, X', U)
    Q, _, P = svd(X'*Z)
    mul!(U, Q, P') 
end

function greedy_step!(u::SVDLikeApproximation, cache, t, dt)
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
    @unpack u, t, iter, cache = integrator
    greedy_step!(u, cache, t, dt)
    integrator.t += dt
    integrator.iter += 1
end
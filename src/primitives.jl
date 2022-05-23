abstract type AbstractDLRProblem end
abstract type AbstractDLRSolution end
abstract type AbstractDLRIntegrator end
abstract type AbstractDLRAlgorithm end
abstract type AbstractDLRAlgorithm_Cache end

"""
    Problem of approximating the solution of a matrix differential equation
    dy/dt = f(t,y) with y(0) = y₀ on [t0,tf] with a low rank factorization 
    y(t) ≈ u(t) = U(t)S(t)V(t)' where U(t) and V(t) are orthonormal bases approximating 
    range and co-range of y(t). 
"""
mutable struct MatrixDEProblem{fType, uType, tType} <: AbstractDLRProblem 
    f::fType
    u0::uType
    tspan::Tuple{tType, tType}
end

"""
    Problem of tracking the low rank decomposition u(t) = U(t)S(t) V(t)' (or U(t)Z(t)') of a 
    time-dependent (or streamed) matrix y(t) with t ∈ [t_0, t_f].
"""
mutable struct MatrixDataProblem{yType, uType, tType} <: AbstractDLRProblem 
    y::yType
    u0::uType
    tspan::Tuple{tType, tType}
end
function MatrixDataProblem(y::AbstractArray, u0)
    return MatrixDataProblem(y, u0, (1,length(y)))
end

"""
    Problem of identifying optimal projection basis U(t) such that we get good reconstruction
    y(t) = U(t) Z(t)' where dZ/dt = f(Z,U,t).
"""
mutable struct MatrixHybridProblem{yType, fType, uType, tType} <: AbstractDLRProblem
    y::yType
    f::fType
    u0::uType
    tspan::Tuple{tType, tType}
end

"""
    Solution object that tracks the evolution of a low rank approximation
"""
mutable struct DLRSolution{solType, tType} <: AbstractDLRSolution
    Y::Vector{solType}
    t::Vector{tType}
end

"""
    Integrator computing solution to a dynamic low rank approximation problem
"""
mutable struct DLRIntegrator{uType, tType, aType, cType, pType} <: AbstractDLRIntegrator
    u::uType
    t::tType
    dt::tType
    sol::DLRSolution{uType,tType}
    alg::aType
    cache::cType
    probType::pType
    iter::Int
end

"""
    solves the given problem with the specified algorithm and step size
"""
function solve(prob::AbstractDLRProblem, alg::AbstractDLRAlgorithm, dt)
    integrator = init(prob, alg, dt)
    T = prob.tspan[2] - prob.tspan[1]
    while (prob.tspan[2]-integrator.t)/T > 1e-8 
        step!(integrator, alg, dt)
        update_sol!(integrator)
    end
    return integrator.sol
end
function solve(prob::MatrixDataProblem, alg::AbstractDLRAlgorithm)
    @assert typeof(prob.y) <: AbstractArray "If the data is not provided as array, integration stepsize needs to be specified"
    return solve(prob, alg, 1)
end

function update_sol!(integrator::AbstractDLRIntegrator)
    if integrator.iter <= length(integrator.sol.Y) - 1
        integrator.sol.Y[integrator.iter + 1] = deepcopy(integrator.u)
        integrator.sol.t[integrator.iter + 1] = integrator.t
    else
        push!(integrator.sol.Y, deepcopy(integrator.u))
        push!(integrator.sol.t, integrator.t)
    end
end
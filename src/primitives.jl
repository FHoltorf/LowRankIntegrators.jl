
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
mutable struct MatrixDEProblem{uType, tType} <: AbstractDLRProblem 
    f
    u0::uType
    tspan::Tuple{tType, tType}
end

MatrixDEProblem(f, u0, tspan) = MatrixDEProblem(f, nothing, nothing, nothing, u0, tspan)
MatrixDEProblem(corange_projected_f, range_projected_f, core_projected_f, u0, tspan) = MatrixDEProblem(nothing, corange_projected_f, range_projected_f, core_projected_f, u0, tspan)

"""
    Problem of tracking the low rank decomposition u(t) = U(t)S(t) V(t)' of a 
    time-dependent (or streamed) matrix y(t) with t ∈ [t_0, t_f].
"""
mutable struct MatrixDataProblem{uType, tType} <: AbstractDLRProblem 
    y
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
mutable struct DLRIntegrator{uType, tType, aType, cType} <: AbstractDLRIntegrator
    u::uType
    t::tType
    dt::tType
    sol::DLRSolution{uType,tType}
    alg::aType
    cache::cType
    iter::Int
end

"""
    solves the given problem with the specified algorithm and step size
"""
function solve(prob::AbstractDLRProblem, alg::AbstractDLRAlgorithm, dt)
    integrator = init(prob, alg, dt)
    T = prob.tspan[2] - prob.tspan[1]
    while (prob.tspan[2]-integrator.t)/T > 1e-8 # robust to round-off-errors but need to find something actually rigorous? Maybe for loop with N = round(T/dt)? That won't work well with adaptive time stepping in the future.
        step!(integrator, alg, dt)
        update_sol!(integrator, dt)
    end
    return integrator.sol
end

function update_sol!(integrator::AbstractDLRIntegrator, dt)
    if integrator.iter <= length(integrator.sol.Y) - 1
        integrator.sol.Y[integrator.iter + 1] = deepcopy(integrator.u)
        integrator.sol.t[integrator.iter + 1] = integrator.t
    else
        push!(integrator.sol.Y, deepcopy(integrator.u))
        push!(integrator.sol.t, integrator.t)
    end
end
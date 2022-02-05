
abstract type AbstractDLRProblem end
abstract type AbstractDLRSolution end
abstract type AbstractDLRIntegrator end
abstract type AbstractLowRankApproximation end
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
    The factors of an SVD like approximation are an orthogonal matrix U which represents the range of the matrix,
    an orthogonal matrix V which represents the co-range of the matrix and a square core-matrix S which reprepresents
    the map from the co-range to the range. To recover the full matrix one only needs to take the product U*S*V'.
"""
mutable struct SVDLikeApproximation{uType, sType, vType} <: AbstractLowRankApproximation
    U::uType
    S::sType
    V::vType
end 

"""

"""
mutable struct LowRankApproximation{uType, zType} <: AbstractLowRankApproximation
    U::uType
    Z::zType
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
    return integrator
end
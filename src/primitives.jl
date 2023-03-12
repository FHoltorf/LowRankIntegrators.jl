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
    r::Vector{Int}
    r_DEIM::Vector{Int}
    idcs_DEIM::Vector{Tuple{Vector{Int}, Vector{Int}}}
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
function solve(prob::AbstractDLRProblem, alg::AbstractDLRAlgorithm, dt; save_increment::Int=1)
    println("Initialize integrator ...")
    integrator, t_int, save_idcs = init(prob, alg, dt, save_increment)
    println("... initialization complete. Start Integration ...")
    dt = step(t_int)
    disp_digits = abs(round(Int, log10(save_increment*dt)))
    k = 2
    #while (prob.tspan[2]-integrator.t)/T > 1e-8 
    while integrator.iter < length(t_int) - 1
        step!(integrator, alg, dt)
        if integrator.iter + 1 == save_idcs[k] 
            update_sol!(integrator, k)
            k += 1 
            println("t = $(round(integrator.t,sigdigits=disp_digits))")
        end
    end
    println("... integration complete.")
    return integrator.sol
end
function solve(prob::MatrixDataProblem, alg::AbstractDLRAlgorithm)
    @assert typeof(prob.y) <: AbstractArray "If the data is not provided as array, integration stepsize needs to be specified"
    return solve(prob, alg, 1)
end

function init(prob::AbstractDLRProblem, alg::AbstractDLRAlgorithm, dt, save_increment::Int)
    t0, tf = prob.tspan
    @assert tf > t0 "Integration in reverse time direction is not supported"
    u = deepcopy(prob.u0)
    # initialize solution 
    sol, t_int, save_idcs = init_sol(dt, t0, tf, prob.u0, save_increment)
    # initialize cache
    cache = alg_cache(prob, alg, u, dt, t0 = t0)
    sol.Y[1] = deepcopy(prob.u0) 
    return DLRIntegrator(u, t0, dt, sol, alg, cache, typeof(prob), 0), t_int, save_idcs  
end

function init_sol(dt, t0, tf, u0, save_increment) 
    n = floor(Int,(tf-t0)/dt) + 1 
    dt = (tf-t0)/(n-1)
    t_int = t0:dt:tf
    save_idcs = collect(1:save_increment:length(t_int))
    if save_idcs[end] != length(t_int)
        push!(save_idcs, length(t_int)) 
    end
    n_eff = length(save_idcs)
    # initialize & return solution object
    Y = Vector{typeof(u0)}(undef, n_eff)
    r = Vector{Int}(undef, n_eff)
    r_deim = Vector{Int}(undef, n_eff)
    t = collect(range(t0, tf, length=n_eff))
    interpolation_idcs = Vector{Tuple{Vector{Int},Vector{Int}}}(undef, n_eff)
    return DLRSolution(Y, t, r, r_deim, interpolation_idcs), t_int, save_idcs  
end

function init_sol(dt::Int, t0, tf, u0) 
    # initialize solution object
    n = floor(Int,(tf-t0)/dt) + 1 
    dt = (tf-t0)/(n-1)
    t_int = t0:dt:tf
    save_idcs = 1:save_increment:length(t_int)
    if save_idcs[end] != length(t_int)
        push!(save_idcs, length(t_int)) 
    end
    n_eff = length(save_idcs)
    Y = Vector{typeof(u0)}(undef, n_eff)
    r = Vector{Int}(undef, n_eff)
    r_deim = Vector{Int}(undef, n_eff)
    interpolation_idcs = Vector{Tuple{Vector{Int},Vector{Int}}}(undef, n_eff)
    return DLRSolution(Y, steps, r, r_deim, interpolation_idcs), t_int, save_idcs
end

function update_sol!(integrator::AbstractDLRIntegrator, idx)
    #if integrator.iter <= length(integrator.sol.Y) - 1
    integrator.sol.Y[idx] = deepcopy(integrator.u)
    integrator.sol.t[idx] = integrator.t
    integrator.sol.r[idx] = LowRankArithmetic.rank(integrator.u)
    integrator.sol.r_DEIM[idx] = rank_DEIM(integrator.cache)
    integrator.sol.idcs_DEIM[idx] = interpolation_indices(integrator.cache)
    #else
    #     push!(integrator.sol.Y, deepcopy(integrator.u))
    #     push!(integrator.sol.t, integrator.t)
    #     push!(integrator.sol.r[integrator.iter + 1], LowRankArithmetic.rank(integrator.u))
    #     push!(integrator.sol.r_DEIM[integrator.iter + 1], rank_DEIM(integrator.cache))
    #     push!(integrator.sol.r_DEIM[integrator.iter + 1], indices(integrator.cache))
    # end
end
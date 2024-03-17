using ConcreteStructs, LowRankArithmetic, ProgressMeter, UnPack
"""
    ($TYPEDEF)

    To register a new LowRankRetraction simply subtype it as `LowRankRetraction`.
"""
abstract type LowRankRetraction end

"""
    ($TYPEDEF)

    To register a new ExtendedLowRankRetraction simply subtype it as `LowRankRetraction`.
"""
abstract type ExtendedLowRankRetraction <: LowRankRetraction end 

"""
    ($TYPEDEF)

    For efficient computations we need caches. If you register a cache for your new retraction,
    subtype it with `LowRankRetractionCache`. There a few things that need to have dispatches.

    state(cache::YourLowRankRetractionCache)
    update_cache!(cache::YourLowRankRetractionCache, SA::SparseApproximation)

    Every `LowRankRetractionCache` should have a dispatch for `state(cache)` which should
    return the low-rank approximation representing the current state X̂.
"""
abstract type LowRankRetractionCache end

"""
    ($TYPEDEF)

    For efficient computations we need caches. If you register a cache for your new retraction,
    subtype it with `LowRankStepperCache`. There a few things that need to have dispatches.

    state(cache::YourLowRankStepperCache)
    update_cache!(cache::YourLowRankStepperCache, SA::SparseApproximation)

    Every `LowRankStepperCache` should have a dispatch for `state(cache)` which should
    return the low-rank approximation representing the current state X̂.
"""
abstract type LowRankStepperCache end


abstract type AbstractLowRankModel end
"""
    $(TYPEDEF)

    need to subtype every factored low-rank model with this type!
    
    When `isinplace = true`, then one must implement a dispatch for the function
    `F!(model::YourCustomModelType, dX, X, t)` where X is a `TwoFactorRepresentation` or a 
    `SVDLikeRepresentation` and dX is a `TwoFactorRepresentation`. 

    When `isinplace = false`, then one must implement a dispatch for the function
    `F(model::YourCustomModelType, X, t)` where X is a `TwoFactorRepresentation` or a 
    `SVDLikeRepresentation`. `F` must return a `TwoFactorRepresentation`. 
"""
abstract type FactoredLowRankModel{isinplace} <: AbstractLowRankModel end 

"""
    $(TYPEDEF)

    need to subtype every model that shall be approximated via on-the-fly sparse interpolation with this type!
    
    When `isinplace = true`, then one must implement a dispatch for the functions

    * `rows!(model::YourCustomModelType, drows, X, t,rows)` --> inplace evaluation of the matrix rows at indices `rows`. 
    * `columns!(model::YourCustomModelType, dcols, X, t,cols)` --> inplace evaluation of the matrix columns at indices `columns`. 
    * `elements!(model::YourCustomModelType, dels, X, t,rows,cols)` --> inplace evaluation of the matrix elements at indices `rows`x`columns`. 

    `rows` and `cols` above are vectors of integers referencing the relevant rows and columns. `X` will be given either  as
    `TwoFactorRepresentation` or `SVDLikeRepresentation`.

    When `isinplace = false`, then one must implement a dispatch for the functions

    * `rows!(drows, X, t,rows)` --> inplace evaluation of the matrix rows at indices `rows`. 
    * `columns!(dcols, X, t,cols)` --> inplace evaluation of the matrix columns at indices `columns`. 
    * `elements!(dels, X, t,rows,cols)` --> inplace evaluation of the matrix elements at indices `rows`x`columns`. 

    `rows` and `cols` above are vectors of integers referencing the relevant rows and columns. `X` will be given either  as
    `TwoFactorRepresentation` or `SVDLikeRepresentation`.
"""
abstract type SparseLowRankModel{isinplace} <: AbstractLowRankModel end 

abstract type AbstractMatrixProblem end
"""
    $(TYPEDEF)

    describes a matrix-valued initial value problem.
"""
@concrete struct MatrixDEProblem <: AbstractMatrixProblem
    model
    X0 
    tspan::Tuple{Float64,Float64}   
    dims::Tuple{Int,Int}
end

function MatrixDEProblem(model, X0, tspan)
    if !(model isa AbstractLowRankModel)
        error("The supplied model is not an `AbstractLowRankModel'.
               Please ensure to subtype your model struct as `AbstractLowRankModel'!")
    end
    MatrixDEProblem(model, X0, tspan, size(X0))
end

"""
    $(TYPEDEF)

    solution to a dynamical low-rank approximation problem.

    Contains 
        * `t` = time points visited during integration
        * `X` = state at every time point visited during integration
        * `r` = rank at every time point visited during integration
        * `approximation_indices` = indices used to approximate the right-hand side if the model is a `SparseLowRankModel`
"""
@concrete mutable struct DLRASolution
    t
    X 
    r
    approximation_indices
end
# routines for initializing and updating the solution, depending on whether sparse interpolation is used or not
function initialize_sol(prob::MatrixDEProblem, SA)
    if SA isa SparseApproximation
        @unpack sparse_approximator = SA
        return DLRASolution([prob.tspan[1]], 
                    [deepcopy(prob.X0)], 
                    [rank(prob.X0)],
                    [(indices(sparse_approximator.range), indices(sparse_approximator.corange))])
    else
        return DLRASolution([prob.tspan[1]], 
                 [deepcopy(prob.X0)], 
                 [rank(prob.X0)],
                 missing)
    end
end
function update_sol!(sol::DLRASolution, cache, t_, SA)
    @unpack t, X, r, approximation_indices = sol
    X_new = state(cache)
    push!(t, t_)
    push!(X, deepcopy(X_new))
    push!(r, rank(X_new))
    if SA isa SparseApproximation
        @unpack sparse_approximator = SA
        push!(approximation_indices, (copy(indices(sparse_approximator.range)), 
                                      copy(indices(sparse_approximator.corange))))
    end
end

"""
    supertype for all integration routines
"""
abstract type LowRankStepper end

function interleave_grids(t_grid, t_save)
    t_grid_extended = sort(unique(vcat(t_grid, t_save)))
    save_idcs = Int[]
    k = 1
    for (i,t) in enumerate(t_grid_extended)
        if t == t_save[k] 
            push!(save_idcs, i)
            k += 1
        end
    end
    return t_grid_extended, save_idcs
end
save_grid(t_grid, ::Missing, tspan) = t_grid, 1:length(t_grid)
function save_grid(t_grid, dt_save::Number, tspan)
    t0, tf = tspan
    t_save = collect(t0:dt_save:tf)
    if !(tf == tsave[end])
        push!(t_save, tf)
    end
    return interleave_grids(t_grid, t_save)
end
function save_grid(t_grid, t_save::AbstractVector, tspan)
    t0, tf = tspan
    @assert issorted(t_save) "list of saved timepoints must be sorted"
    @assert t_save[1] >= t0 "saved timepoint cannot be before t0"
    @assert t_save[end] <= tf "saved timepoint cannot be after tf"
    t_save = collect(t_save)
    if t_save[1] != t0
        pushfirst!(t0, t_save)
    end
    if t_save[end] != tf
        push!(tf, t_save)
    end
    return interleave_grids(t_grid, t_save)
end
function solve(prob::MatrixDEProblem, h::Number, stepper::LowRankStepper, SA = missing;
               saveat = missing)
    t0, tf = prob.tspan
    t_grid = collect(t0:h:tf)
    if !(t_grid[end] == tf)
        push!(t_grid, tf)
    end
    solve(prob, t_grid, stepper, SA; saveat=saveat)
end

function solve(prob::MatrixDEProblem, t_grid::AbstractVector, stepper::LowRankStepper, SA = missing;
               saveat=missing)
    SA = deepcopy(SA) # do not alter input!

    @unpack model, tspan, X0 = prob
    t0, tf = tspan
    
    t_grid, save_idcs = save_grid(t_grid, saveat, tspan)

    @assert t_grid[1] == t0 "The first entry of integration time grid must 
                            coincide with initial time as specified in the problem."

    @assert t_grid[end] == tf "The last entry of integration time grid must 
                              coincide with initial time as specified in the problem."

    @assert issorted(t_grid) || issorted(tgrid, rev=true) "The integration time grid must be sorted."
    
    sol = initialize_sol(prob, SA)
    cache = initialize_cache(prob, stepper, SA)

    n_steps = length(t_grid) - 1 
    progressmeter = Progress(n_steps)

    k = 2
    @inbounds for i in 1:n_steps
        t = t_grid[i]
        t_new = t_grid[i+1]
        h = t_new - t
        
        # take step with stepsize h
        retracted_step!(cache, model, t, h, stepper, SA)
    
        # postprocess the state
        postprocess_step!(cache, model, t_new)

        # update sparse approximation if necessary
        if !ismissing(SA)
            update_sparse_approximation!(SA, model, cache, t_new)        
        end

        # update solution
        if i + 1 == save_idcs[k]
            update_sol!(sol, cache, t_new, SA)
            k += 1
        end
        
        # update progress meter
        next!(progressmeter)
        flush(stdout)
    end

    sol
end


struct ProjectedEuler{RType <: LowRankRetraction} <: LowRankStepper
    R::RType
end

@concrete struct EulerCache <: LowRankStepperCache
    X   # state
    dΠX # tangent space projection
    dX  # model
    sparse_cache 
    retraction_cache 
end
state(cache::EulerCache) = cache.X 

@concrete mutable struct SparseTangentCache
    W_rows # needs to be changed in size if sparse interpolation is adapted
    W_cols # needs to be changed in size if sparse interpolation is adapted
    dXV
    dXV_
    dXᵀU 
    UᵀdXV
end

function initialize_sparse_tangent_cache(X, SA)
    @unpack range, corange = SA.sparse_approximator
    r = rank(X)
    n, m = size(X)
    W_rows = zeros(r, length(indices(range)))
    W_cols = zeros(r, length(indices(corange)))
    dXV = zeros(n, r)
    dXV_ = similar(dXV)
    dXᵀU = zeros(m, r)
    UᵀdXV = zeros(r, r)
    SparseTangentCache(W_rows, W_cols, dXV, dXV_, dXᵀU, UᵀdXV)
end

function initialize_cache(prob, stepper::ProjectedEuler{<:ExtendedLowRankRetraction}, SA)
    @unpack X0, model, tspan = prob
    X = deepcopy(X0)
    dΠX = missing
    dX = evaluate_model(X, model, tspan[1], SA)
    if !ismissing(SA)
        sparse_cache = initialize_sparse_tangent_cache(X, SA)  
    else
        sparse_cache = missing 
    end
    retraction_cache = initialize_cache(X, dX, stepper.R)
    EulerCache(X, dΠX,  dX, sparse_cache, retraction_cache)
end

function initialize_cache(prob, stepper::ProjectedEuler{<:LowRankRetraction}, SA)
    X = deepcopy(prob.X0)
    dΠX = evaluate_tangent_model(X, model, t, SA)
    dX = missing
    if !ismissing(SA)
        sparse_cache = initialize_sparse_tangent_cache(X, SA)  
    else
        sparse_cache = missing 
    end
    retraction_cache = initialize_cache(X, dX, stepper.R)
    EulerCache(X, dΠX, dX, sparse_cache, retraction_cache)
end

function retracted_step!(cache, model, t, h, stepper::ProjectedEuler{<:ExtendedLowRankRetraction}, SA)
    evaluate_model!(cache, model, t, SA)
    cache.dX.S .*= h
    retract!(cache, cache.dX, stepper.R)
end

function retracted_step!(cache, model, t, h, stepper::ProjectedEuler{<:LowRankRetraction}, SA)
    evaluate_tangent_model!(cache, model, t, SA)
    cache.dΠX.S .*= h
    retract!(cache, cache.dΠX, stepper.R)
end
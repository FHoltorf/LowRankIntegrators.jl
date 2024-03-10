@concrete struct DefaultLowRankRetractionCache <: LowRankRetractionCache
    X  
    PQ
end

function initialize_cache(prob::MatrixDEProblem, ::LowRankRetraction, SA)  
    X = deepcopy(prob.X0)
    if !ismissing(SA)
        PQ = SA.sparse_approximator
    else
        PQ = missing
    end
    DefaultLowRankRetractionCache(X, PQ)
end

state(cache::DefaultLowRankRetractionCache) = cache.X

retract(X,V,R::LowRankRetraction) = error("Please implement dispatch for `retract(X,V,$(typeof(R)))`")
retract(X,V,R::ExtendedLowRankRetraction) = error("Please implement dispatch for `retract(X,V,$(typeof(R)))`")

function retracted_step!(cache, model, t, h, R::LowRankRetraction, SA)
    X = state(cache)
    dX = evaluate_tangent_model(cache, model, t)
    Xnew = retract(X, h*dX, R)
    orthonormalize!(Xnew)
    copyto!(X.U, Xnew.U)
    copyto!(X.V, Xnew.V)
    copyto!(X.S, Xnew.S)
end
function retracted_step!(cache, model, t, h, R::ExtendedLowRankRetraction, SA)
    X = state(cache)
    dX = evaluate_rate(cache, model, t)
    Xnew = retract(X, h*dX, R)
    orthonormalize!(Xnew)
    copyto!(X.U, Xnew.U)
    copyto!(X.V, Xnew.V)
    copyto!(X.S, Xnew.S)
end
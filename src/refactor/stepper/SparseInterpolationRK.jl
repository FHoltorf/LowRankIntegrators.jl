struct SparseInterpolationRK{RType <: ExtendedLowRankRetraction, T <: ButcherTableau} <: LowRankStepper
    R::RType
    tableau::T
end

@concrete struct SparseInterpolationRKCache <: LowRankStepperCache
    X
    ks # low rank factorizations => s times
    ηs # intermediate points => (s-1) times (the first one is X)
end
state(cache::SparseInterpolationRKCache) = cache.X

function initialize_cache(prob, SIRK::SparseInterpolationRK, SA::SparseApproximation)
    @unpack sparse_approximator = SA
    @unpack elements = cache(sparse_approximator)
    X = deepcopy(prob.X0)
    s = SIRK.tableau.s
    ks = [deepcopy(elements) for i in 1:s]
    ηs = [deepcopy(X) for i in 1:s]
    return SparseInterpolationRKCache(X, ks, ηs)
end

function retracted_step!(cache::SparseInterpolationRKCache, model::SparseLowRankModel, t, h, PRK::SparseInterpolationRK, SA::SparseApproximation)
    @unpack R, tableau = PRK
    @unpack a, b, c, s = tableau
    @unpack X, ks, ηs = cache
    @unpack sparse_approximator = SA 

    UF, VF = weights(sparse_approximator)

    ηs[1] = X
    evaluate_elements!(ks[1], sparse_approximator, model, ηs[1], t + c[1]*h)
    for i in 2:s
        # compute intermediate slope dη
        slope_core = sum((h*a[i-1][j])*ks[i] for j in 1:i-1)
        dη = SVDLikeRepresentation(UF, slope_core, VF)

        # compute intermediate step
        # make this cached in the long term (should be straightforward)
        ηs[i] = retract(X, dη, R)
        
        # compute new slope 
        evaluate_elements!(ks[i], sparse_approximator, model, ηs[i], t + c[i]*h)
    end
    # final step 
    slope_core = sum((h*b[j]) * ks[j] for j in 1:s)
    dX = SVDLikeRepresentation(UF, slope_core, VF)
    Xnew = retract(X, dX, R)
    
    # update
    copyto!(X.U, Xnew.U)
    copyto!(X.S, Xnew.S)
    copyto!(X.V, Xnew.V)
end

function update_cache!(cache::SparseInterpolationRKCache, SA::SparseApproximation)
    # adjust upon rank adaptation!
    nothing 
end



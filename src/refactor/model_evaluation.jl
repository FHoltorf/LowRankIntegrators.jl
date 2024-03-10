# evaluate projection of model into tangent space
"""
    X.U and X.V must be orthogonal for this to work
"""
evaluate_tangent_model(X, model::FactoredLowRankModel{false}, t, SA) = evaluate_tangent_model(X, model, t, SA)
function evaluate_tangent_model(X, model::FactoredLowRankModel{false}, t)
    dX = F(model,X,t)

    # the stuff below can all be cached
    dXV = dX*X.V
    dXᵀU = dX'*X.U
    UᵀdXV = U'*dXᵀU

    ΠdX_range = [X.U dXV]
    ΠdX_core = [I -UᵀdXV; zeros(r,r) I]
    ΠdX_corange = [dXᵀU X.V]
    return SVDLikeRepresentation(ΠdX_range, ΠdX_core, ΠdX_corange)
end
function evaluate_tangent_model(X, model::SparseLowRankModel{false}, t, SA)
    @unpack sparse_approximator = SA
    @unpack range, corange = sparse_approximator
    @unpack elements, rows, columns = cache(sparse_approximator)
    
    # the projectors and everything after the evaluations can be cached
    VᵀQ = SparseProjector(X.V'*weights(corange), indices(corange), columns)
    UᵀP = SparseProjector(X.U'*weights(range), indices(range), rows)
    UᵀP_QᵀV = SparseMatrixApproximator(UᵀP, VᵀQ) 

    dXV = evaluate_approximator(VᵀQ', model, X, t)
    dXᵀU = evaluate_approximator(UᵀP, model, X, t)
    UᵀdXV = evaluate_approximator(UᵀP_QᵀV, model, X, t)

    ΠdX_range = [X.U dXV]
    ΠdX_core = [I -UᵀdXV; zeros(r,r) I]
    ΠdX_corange = [dXᵀU X.V]
    return SVDLikeRepresentation(ΠdX_range, ΠdX_core, ΠdX_corange)
end
function evaluate_tangent_model(X, model::SparseLowRankModel{true}, t, SA)
    @unpack sparse_approximator = SA
    @unpack range, corange = sparse_approximator
    @unpack elements, rows, columns = cache(sparse_approximator)

    # everything below can be cached
    VᵀQ = SparseProjector(X.V'*weights(corange), indices(corange), columns)
    UᵀP = SparseProjector(X.U'*weights(range), indices(range), rows)
    UᵀP_QᵀV = SparseMatrixApproximator(UᵀP, VᵀQ) 

    # improve 
    r = rank(X)
    n, m = size(X)
    dXV = zeros(n, r)
    dXᵀU = zeros(m, r)
    UᵀdXV = zeros(r, r)
    evaluate_approximator!(dXV, VᵀQ', model, X, t)
    evaluate_approximator!(dXᵀU, UᵀP, model, X, t)
    evaluate_approximator!(UᵀdXV, UᵀP_QᵀV, model, X, t)

    ΠdX_range = [X.U dXV]
    ΠdX_core = [I -UᵀdXV; zeros(r,r) I]
    ΠdX_corange = [dXᵀU X.V]
    return SVDLikeRepresentation(ΠdX_range, ΠdX_core, ΠdX_corange)
end

# factored models
function evaluate_model(cache, model::FactoredLowRankModel{false}, t)
    X = state(cache)
    dX = F(model, X, t)
    return dX
end
function evaluate_model(cache, model::SparseLowRankModel{false}, t)
    X = state(cache)
    dX = evaluate_approximator_factored(cache.PQ, model, X, t)
    return dX
end
function evaluate_model(cache, model::SparseLowRankModel{true}, t)
    X = state(cache)
    PQ = cache.PQ

    @unpack elements, rows, columns = cache(PQ)
    row_idcs, col_idcs = indices(PQ)
    elements!(model, elements, X, t, row_idcs, col_idcs)
    dX = SVDLikeRepresentation(weights(PQ.range), elements, weights(PQ.corange))
    
    return dX
end

# Tangent space projection
"""
    X.U and X.V must be orthogonal for this to work
"""
function tangent_space_projection(X::SVDLikeRepresentation, dX)
    @unpack U, S, V = X
    r = rank(X)

    dXᵀU = dX'*U
    dXV = dX*V
    UᵀdXV = range'*V
    
    ΠdX_range = [X.U dXV]
    ΠdX_core = [I -UᵀdXV; zeros(r,r) I]
    ΠdX_corange = [dXᵀU X.V]
   
    return SVDLikeRepresentation(ΠdX_range, ΠdX_core, ΠdX_corange) 
end

#=
function evaluate_projected_range(cache, model::FactoredLowRankModel{false}, t)
    X = state(cache)
    dX = F(model, X, t)
    return tangent_space_projection(X, dX)
end

function evaluate_rate(cache, model::FactoredLowRankModel{false}, t)
    X = state(cache)
    dX = F(model, X, t)
    return dX
end

# sparse models
function evaluate_projected_rate(cache, model::SparseLowRankModel{false}, t)
    X = state(cache)
    dX = evaluate_approximator_factored(cache.PQ, model, X, t)
    return tangent_space_projection(X, dX)
end
function evaluate_projected_rate(cache, model::SparseLowRankModel{true}, t)
    X = state(cache)
    PQ = cache.PQ

    @unpack elements, rows, columns = cache(PQ)
    row_idcs, col_idcs = indices(PQ)
    elements!(model, elements, X, t, row_idcs, col_idcs)
    dX = SVDLikeRepresentation(weights(PQ.range), elements, weights(PQ.corange))
    
    return tangent_space_projection(X, dX)
end

function evaluate_rate(cache, model::SparseLowRankModel{false}, t)
    X = state(cache)
    dX = evaluate_approximator_factored(cache.PQ, model, X, t)
    return dX
end
function evaluate_rate(cache, model::SparseLowRankModel{true}, t)
    X = state(cache)
    PQ = cache.PQ

    @unpack elements, rows, columns = cache(PQ)
    row_idcs, col_idcs = indices(PQ)
    elements!(model, elements, X, t, row_idcs, col_idcs)
    dX = SVDLikeRepresentation(weights(PQ.range), elements, weights(PQ.corange))
    
    return dX
end
=#
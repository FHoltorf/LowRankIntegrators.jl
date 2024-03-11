# evaluate projection of model into tangent space
"""
    X.U and X.V must be orthogonal for this to work
"""
evaluate_tangent_model(X::SVDLikeRepresentation, model::FactoredLowRankModel{false}, t, SA) = evaluate_tangent_model(X, model, t, SA)
function evaluate_tangent_model(X::SVDLikeRepresentation, model::FactoredLowRankModel{false}, t)
    dX = F(model,X,t)

    # the stuff below can all be cached
    dXV = dX*X.V
    dXᵀU = dX'*X.U
    UᵀdXV = dXᵀU'*X.V

    ΠdX_range = [X.U dXV]
    ΠdX_core = [I -UᵀdXV; zeros(r,r) I]
    ΠdX_corange = [dXᵀU X.V]
    return SVDLikeRepresentation(ΠdX_range, ΠdX_core, ΠdX_corange)
end
function evaluate_tangent_model(X::SVDLikeRepresentation, model::SparseLowRankModel, t, SA)
    @unpack sparse_approximator = SA
    @unpack range, corange = sparse_approximator
    @unpack elements, rows, columns = cache(sparse_approximator)
    
    evaluate_oracles!(sparse_approximator, model, X, t)
    
    W_rows = X.U'*weights(range)
    W_cols = X.V'*weights(corange)
    
    dXV = columns*W_cols'
    dXᵀU = rows'*W_rows'
    UᵀdXV = W_rows*elements*W_cols'
    
    r = rank(X)
    ΠdX_range = [X.U dXV]
    ΠdX_core = [I -UᵀdXV; zeros(r,r) I]
    ΠdX_corange = [dXᵀU X.V]
    return SVDLikeRepresentation(ΠdX_range, ΠdX_core, ΠdX_corange)
end

# for a cache
function evaluate_tangent_model!(cache, model::SparseLowRankModel, t, SA)
    @unpack X, dΠX = cache
    @unpack sparse_approximator = SA
    @unpack range, corange = sparse_approximator 
    @unpack rows, columns, elements = cache(sparse_approximator)
    @unpack W_rows, W_cols, dXV, dXᵀU, UᵀdXV, dXV_  = cache

    evaluate_oracles!(sparse_approximator, model, X, t)

    mul!(W_rows, X.U', weights(range))
    mul!(W_cols, X.V', weights(corange))
    
    mul!(dXV, columns, W_cols')
    mul!(dXᵀU, rows', W_rows)
    mul!(dXV_, elements, W_cols')
    mul!(UᵀdXV, W_rows, XV_, -1, 0)

    r = rank(X)
    @views copyto!(dΠX.U[:,1:r], X.U)
    @views copyto!(dΠX.U[:,r+1:2r], dXV)
    
    @views copyto!(dΠX.V[:,1:r], dXᵀU)
    @views copyto!(dΠX.V[:,r+1:2r], X.V)
    
    @views copyto!(dΠX.S[1:r,r+1:2r], UᵀdXV)
end
evaluate_tangent_model!(cache, model::FactoredLowRankModel{false}, t) = 
                                                evaluate_tangent_model!(cache, model, t, missing)
function evaluate_tangent_model!(cache, model::FactoredLowRankModel{false}, t, SA)
    @unpack X, dΠX = cache
    dX = F(model,X,t)
    
    dXV = dX*X.V
    dXᵀU = dX'*X.U
    UᵀdXV = -U'*dXᵀU
    
    # need to implement routines like these cleverly 
    # mul!(dXV, dX, X.V)
    # mul!(dXᵀU, dX', X.U)
    # mul!(UᵀdXV, dXᵀU, X.V, -1, 0)

    r = rank(X)
    @views copyto!(dΠX.U[:,1:r], X.U)
    @views copyto!(dΠX.U[:,r+1:2r], dXV)
    
    @views copyto!(dΠX.V[:,1:r], dXᵀU)
    @views copyto!(dΠX.V[:,r+1:2r], X.V)
    
    @views copyto!(dΠX.S[1:r,r+1:2r], UᵀdXV)
end

# factored models
function evaluate_model(X::SVDLikeRepresentation, model::FactoredLowRankModel{false}, t)
    dX = F(model, X, t)
    return dX
end
function evaluate_model(X::SVDLikeRepresentation, model::SparseLowRankModel{false}, t, SA)
    @unpack sparse_approximator = SA
    dX = evaluate_approximator_factored(sparse_approximator, model, X, t)
    return dX
end
function evaluate_model(X::SVDLikeRepresentation, model::SparseLowRankModel{true}, t, SA)
    @unpack sparse_approximator = SA
    @unpack range, corange = sparse_approximator
    @unpack elements, rows, columns = cache(sparse_approximator)

    row_idcs, col_idcs = indices(sparse_approximator)
    elements!(model, elements, X, t, row_idcs, col_idcs)
    dX = SVDLikeRepresentation(weights(range), elements, weights(corange))
    return dX
end

function evaluate_model!(cache, model::FactoredLowRankModel{false}, t)
    @unpack X, dX = cache
    dX_ = F(model, X, t)
    dX.U .= dX_.U
    dX.S .= dX_.S
    dX.V .= dX_.V
end
function evaluate_model!(cache, model::SparseLowRankModel{false}, t, SA)
    @unpack X, dX = cache
    @unpack sparse_approximator = SA
    dX_ = evaluate_approximator_factored(sparse_approximator, model, X, t)
    dX.U .= dX_.U
    dX.S .= dX_.S
    dX.V .= dX_.V
end
function evaluate_model!(cache, model::SparseLowRankModel{true}, t, SA)
    @unpack X, dX = cache
    @unpack sparse_approximator = SA
    @unpack range, corange = sparse_approximator

    row_idcs, col_idcs = indices(sparse_approximator)
    elements!(model, dX.S, X, t, row_idcs, col_idcs)
    copyto!(dX.U, weights(range)) # might not be necessary
    copyto!(dX.V, weights(corange)) # might not be necessary

    # could be enough to put the references to weights and
    # elements into dX and then call evaluate_elements! or so
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
    UᵀdXV = dXᵀU'*V
    
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
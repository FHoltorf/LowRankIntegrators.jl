struct KSLRetraction <: AbstractLowRankRetraction end
struct LSKRetraction <: AbstractLowRankRetraction end
struct KSLLSKRetraction <: AbstractLowRankRetraction end

@concrete struct KSLCache <: AbstractLowRankRetractionCache
    K0
    L0
    dK
    dL
    dS
    X
    sparse_approximator_cache
end
state(cache::KSLCache) = cache.X

initialize_cache(prob::MatrixDEProblem, R::KSLRetraction, SA) = 
                    initialize_cache(prob, prob.model, R, SA)
initialize_cache(prob::MatrixDEProblem, ::LSKRetraction, SA) = 
                    initialize_cache(prob, prob.model, KSLRetraction(), SA)
initialize_cache(prob::MatrixDEProblem, ::KSLLSKRetraction, SA) = 
                    initialize_cache(prob, prob.model, KSLRetraction(), SA)

function initialize_cache(prob, ::AbstractLowRankModel, R::KSLRetraction, SA)
    @unpack model, X0, tspan, dims = prob
    r = rank(X0)
    X = deepcopy(X0)
    K0 = zeros(dims[1], r)
    L0 = zeros(dims[2], r)
    dK = similar(K0)
    dL = similar(L0)
    dS = similar(X.S)
    if !ismissing(SA)
        sparse_approximator_cache = initialize_sparse_approximation_cache(SA, X0, R)
    else
        sparse_approximator_cache = missing
    end
    KSLCache(K0, L0, dK, dL, dS, X, sparse_approximator_cache)
end

@concrete struct KSLSparseApproximatorCache
    PK # sparse projector for K step 
    PL # sparse projector for L step
    PS # sparse approximator for S step
end

function initialize_sparse_approximation_cache(SA::SparseApproximation, X0, ::KSLRetraction)
    @unpack sparse_approximator = SA
    @unpack range, corange = sparse_approximator

    col_cache = similar(cache(sparse_approximator).columns)
    row_cache = similar(cache(sparse_approximator).rows)
    PK = SparseProjector(X0.V'*weights(corange), indices(corange), col_cache)
    PL = SparseProjector(X0.U'*weights(range), indices(range), row_cache)
    PS = SparseMatrixApproximator(deepcopy(PL), deepcopy(PK))
    
    KSLSparseApproximatorCache(PK, PL, PS)
end

function retracted_step!(cache, model::AbstractLowRankModel, t, h, ::KSLRetraction, SA)
    K_step!(cache, model, t, h, SA)
    S_step!(cache, model, t, h, SA)
    L_step!(cache, model, t, h, SA)
end

function retracted_step!(cache, model::AbstractLowRankModel, t, h, ::LSKRetraction, SA)
    L_step!(cache, model, t, h, SA)
    S_step!(cache, model, t, h, SA)
    K_step!(cache, model, t, h, SA)
end

function retracted_step!(cache, model::FactoredLowRankModel, t, h, ::KSLLSKRetraction, SA)
    retracted_step!(cache, model, t, h/2, KSLRetraction(), SA)
    retracted_step!(cache, model, t, h/2, LSKRetraction(), SA)
end

function retracted_step!(cache, model::SparseLowRankModel, t, h, ::KSLLSKRetraction, SA)
    retracted_step!(cache, model, t, h/2, KSLRetraction(), SA)
    update_cache!(cache, SA)
    retracted_step!(cache, model, t + h/2, h/2, LSKRetraction(), SA)
end

# for outofplace factored model 
function K_step!(cache::KSLCache, model::FactoredLowRankModel{false}, t, h, ::Missing)
    @unpack X, K0 = cache

    mul!(K0, X.U, X.S)
    K0 .+= h*Matrix(F(model, TwoFactorRepresentation(K0, X.V), t)*X.V)
    QRK = qr!(K0) 
    X.S .= QRK.R
    X.U .= Matrix(QRK.Q)
end
function S_step!(cache::KSLCache, model::FactoredLowRankModel{false}, t, h, ::Missing)
    @unpack X = cache
    dS = Matrix(X.U'*F(model, X, t)*X.V)
    X.S .-= h*dS
end
function L_step!(cache::KSLCache, model::FactoredLowRankModel{false}, t, h, ::Missing)
    @unpack X, L0 = cache
    mul!(L0, X.V, X.S')
    L0 .+= h*Matrix(F(model, TwoFactorRepresentation(X.U, L0), t)'*X.U)
    QRL = qr!(L0) 
    X.V .= Matrix(QRL.Q) # mul!(V1, QRL.Q, I[1:size(L0,2), 1:size(L0,2)])
    X.S .= QRL.R'
end


# for inplace sparse low rank model
function K_step!(cache::KSLCache, model::SparseLowRankModel{true}, t, h, SA::SparseApproximation)
    @unpack X, K0, dK, sparse_approximator_cache = cache
    @unpack PK = sparse_approximator_cache 
    mul!(K0, X.U, X.S)

    evaluate_approximator!(dK, PK', model, TwoFactorRepresentation(K0,X.V), t)

    K0 .+= h*dK
    QRK = qr!(K0) 
    X.U .= Matrix(QRK.Q) 
    X.S .= QRK.R
end
function S_step!(cache::KSLCache, model::SparseLowRankModel{true}, t, h, SA::SparseApproximation)
    @unpack X, dS, sparse_approximator_cache = cache
    @unpack PS = sparse_approximator_cache
    @unpack sparse_approximator = SA

    mul!(PS.range.weights, X.U', sparse_approximator.range.weights)
    mul!(PS.corange.weights, X.V', sparse_approximator.corange.weights)

    evaluate_approximator!(dS, PS, model, X, t)
    X.S .-= h*dS
end
function L_step!(cache::KSLCache, model::SparseLowRankModel{true}, t, h, SA::SparseApproximation)
    @unpack X, L0, dL, sparse_approximator_cache = cache
    @unpack PL = sparse_approximator_cache
    mul!(L0, X.V, X.S')

    # maybe make better by storing L', cache misses etc.?
    evaluate_approximator!(dL', PL, model, TwoFactorRepresentation(X.U,L0), t)
    
    L0 .+= h*dL
    QRL = qr!(L0) 
    X.V .= Matrix(QRL.Q) 
    X.S .= QRL.R'
end

# needs to be adapted if rank adaptation is considered 
# need to recreate the projectors
function update_cache!(cache::KSLCache, SA::SparseApproximation)
    @unpack X, sparse_approximator_cache = cache
    @unpack PK, PL, PS = sparse_approximator_cache
    @unpack sparse_approximator = SA
    
    mul!(PK.weights, X.V', sparse_approximator.corange.weights)
    PK.indices .= sparse_approximator.corange.indices

    mul!(PL.weights, X.U', sparse_approximator.range.weights)
    PL.indices .= sparse_approximator.range.indices 

    PS.range.indices .= sparse_approximator.range.indices 
    PS.corange.indices .= sparse_approximator.corange.indices 
end

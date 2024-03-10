struct KLSRetraction <: ExtendedLowRankRetraction end

@concrete struct KLSCache <: LowRankRetractionCache
    K0
    L0
    U1
    S1
    V1
    dK
    dL
    dS
    M
    N
    X
    sparse_approximator_cache
end
state(cache::KLSCache) = cache.X

initialize_cache(prob::MatrixDEProblem, R::KLSRetraction, SA) = 
                    initialize_cache(prob, prob.model, R, SA)

function initialize_cache(prob, ::AbstractLowRankModel, R::KLSRetraction, SA)
    @unpack model, X0, tspan, dims = prob
    r = rank(X0)
    X = deepcopy(X0)
    K0 = zeros(dims[1], r)
    L0 = zeros(dims[2], r)
    S1 = similar(X.S)
    U1 = similar(X.U)
    V1 = similar(X.V)
    dK = similar(K0)
    dL = similar(L0)
    dS = similar(S1)
    M = similar(X.S)
    N = similar(X.S)
    if !ismissing(SA)
        sparse_approximator_cache = initialize_sparse_approximation_cache(SA, X0, R)
    else
        sparse_approximator_cache = missing
    end
    KLSCache(K0, L0, U1, S1, V1, dK, dL, dS, M, N, X, sparse_approximator_cache)
end

@concrete struct KLSSparseApproximatorCache
    PK # sparse projector for K step 
    PL # sparse projector for L step
    PS # sparse approximator for S step
end

function initialize_sparse_approximation_cache(SA::SparseApproximation, X0, ::KLSRetraction)
    @unpack sparse_approximator = SA
    @unpack range, corange = sparse_approximator

    col_cache = similar(cache(sparse_approximator).columns)
    row_cache = similar(cache(sparse_approximator).rows)
    PK = SparseProjector(X0.V'*weights(corange), indices(corange), col_cache)
    PL = SparseProjector(X0.U'*weights(range), indices(range), row_cache)
    PS = SparseMatrixApproximator(deepcopy(PL), deepcopy(PK))
    
    KLSSparseApproximatorCache(PK, PL, PS)
end

function retracted_step!(cache, model::AbstractLowRankModel, t, h, ::KLSRetraction, SA)
    K_step!(cache, model, t, h, SA)
    L_step!(cache, model, t, h, SA)
    
    # update bases
    @unpack X, U1, V1 = cache
    X.U .= U1
    X.V .= V1

    S_step!(cache, model, t, h, SA)
end

# for outofplace factored model 
function K_step!(cache::KLSCache, model::FactoredLowRankModel{false}, t, h, ::Missing)
    @unpack X, K0, U1, M= cache

    mul!(K0, X.U, X.S)
    K0 .+= h*Matrix(F(model, TwoFactorRepresentation(K0, X.V), t)*X.V)
    QRK = qr!(K0) 
    U1 .= Matrix(QRK.Q) # mul!(V1, QRK.Q, I[1:size(K0,2), 1:size(K0,2)])
    mul!(M, U1', X.U) 
end
function L_step!(cache::KLSCache, model::FactoredLowRankModel{false}, t, h, ::Missing)
    @unpack X, L0, V1, N = cache
    mul!(L0, X.V, X.S')
    L0 .+= h*Matrix(F(model, TwoFactorRepresentation(X.U, L0), t)'*X.U)
    QRL = qr!(L0) 
    V1 .= Matrix(QRL.Q) # mul!(V1, QRL.Q, I[1:size(L0,2), 1:size(L0,2)])
    mul!(N,V1',X.V) 
end
function S_step!(cache::KLSCache, model::FactoredLowRankModel{false}, t, h, ::Missing)
    @unpack X, S1, M, N = cache

    mul!(S1, M, X.S)
    mul!(X.S, S1, N') 
    S1 .= X.S
    S1 .+= h*Matrix(X.U'*F(model, X, t)*X.V)
    X.S .= S1
end

# for inplace sparse low rank model
function K_step!(cache::KLSCache, model::SparseLowRankModel{true}, t, h, SA::SparseApproximation)
    @unpack X, K0, dK, U1, M, sparse_approximator_cache = cache
    @unpack PK = sparse_approximator_cache 
    mul!(K0, X.U, X.S)

    evaluate_approximator!(dK, PK', model, TwoFactorRepresentation(K0,X.V), t)

    K0 .+= h*dK
    QRK = qr!(K0) 
    U1 .= Matrix(QRK.Q) 
    mul!(M, U1', X.U) 
end
function L_step!(cache::KLSCache, model::SparseLowRankModel{true}, t, h, SA::SparseApproximation)
    @unpack X, L0, dL, V1, N, sparse_approximator_cache = cache
    @unpack PL = sparse_approximator_cache
    mul!(L0, X.V, X.S')

    # maybe make better by storing L', cache misses etc.?
    evaluate_approximator!(dL', PL, model, TwoFactorRepresentation(X.U,L0), t)
    
    L0 .+= h*dL
    QRL = qr!(L0) 
    V1 .= Matrix(QRL.Q) 
    mul!(N, V1', X.V) 
end
function S_step!(cache::KLSCache, model::SparseLowRankModel{true}, t, h, SA::SparseApproximation)
    @unpack X, M, N, S1, dS, sparse_approximator_cache = cache
    @unpack PS = sparse_approximator_cache
    @unpack sparse_approximator = SA

    mul!(S1, M, X.S)
    mul!(X.S, S1, N') 
    S1 .= X.S 

    mul!(PS.range.weights, X.U', sparse_approximator.range.weights)
    mul!(PS.corange.weights, X.V', sparse_approximator.corange.weights)

    evaluate_approximator!(dS, PS, model, X, t)
    S1 .+= h*dS
    X.S .= S1
end

# needs to be adapted if rank adaptation is considered 
# need to recreate the projectors
function update_cache!(cache::KLSCache, SA::SparseApproximation)
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

function retract(X, dX, ::KLSRetraction)
    #K-step
    K = X.U*X.S
    K .+= Matrix(dX*X.V)
    QRK = qr!(K0) 
    U1 = Matrix(QRK.Q) 
    M = U1'*X.U
   
    #L-step
    L = X.V*X.S'
    L .+= Matrix(dX'*X.U)
    QRL = qr!(L0) 
    V1 = Matrix(QRL.Q) 
    N = V1'*X.V

    #S-step
    S1 = M*X.S*N'
    S1 += Matrix(U1'*dX*V1)
    return SVDLikeRepresentation(U1, S1, V1)
end

function retract!(cache, dX, ::KLSRetraction)
    @unpack X, L0, dL, K0, dK, U1, S1, V1, dS, M, N = cache

    #K-step
    mul!(K0, X.U, X.S)
    K0 .+= Matrix(dX*X.V) # can avoid by caching cleverly
    QRK = qr!(K0) 
    U1 .= Matrix(QRK.Q) 
    mul!(M, U1', X.U)
   
    #L-step 
    mul!(L0, X.V, X.S')
    L0 .+= Matrix(dX'*X.U) # can avoid by caching cleverly
    QRL = qr!(L0) 
    V1 .= Matrix(QRL.Q) 
    mul!(N, V1',X.V)
    
    #S-step
    mul!(S1, X.S, N')
    mul!(X.S, M, S1)

    # Update
    X.S .+= Matrix(U1'*dX*V1)
    X.V .= V1
    X.U .= U1
end
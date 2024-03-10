using OrdinaryDiffEq
import OrdinaryDiffEq: step!, set_u!, init

@concrete struct BUGRetraction <: LowRankRetraction 
    S_alg
    K_alg
    L_alg
    S_options
    K_options
    L_options
end

BUGRetraction(alg; kwargs...) = BUGRetraction(alg, alg, alg, kwargs, kwargs, kwargs)
BUGRetraction(; S_stepper = Tsit5(),
                       K_stepper = Tsit5(),
                       L_stepper = Tsit5(),
                       S_options = Dict{Symbol, Any}(),
                       K_options = Dict{Symbol, Any}(),
                       L_options = Dict{Symbol, Any}()) = BUGRetraction(S_stepper, K_stepper, L_stepper,
                                                                        S_options, K_options, L_options)

@concrete struct BUGCache <: LowRankRetractionCache
    K0
    L0
    U1
    S1
    V1
    M
    N
    L_integrator
    S_integrator
    K_integrator
    X
    sparse_approximation_cache
end
state(cache::BUGCache) = cache.X

initialize_cache(prob::MatrixDEProblem, R::BUGRetraction, SA) = 
                    initialize_cache(prob, prob.model, R, SA)

function initialize_cache(prob, ::FactoredLowRankModel{false}, R::BUGRetraction, ::Missing)
    @unpack model, X0, tspan, dims = prob
    @unpack S_alg, K_alg, L_alg, S_options, K_options, L_options = R

    t0, tf = tspan

    X = deepcopy(X0)
    K0 = X.U*X.S
    L0 = X.V*X.S'
    U1 = similar(X.U)
    S1 = similar(X.S)
    V1 = similar(X.V)
    M = similar(X.S)
    N = similar(X.S)
    
    K_rhs = (K, (V, model), t) -> Matrix(F(model, TwoFactorRepresentation(K,V),t)*V)
    K_problem = ODEProblem(K_rhs, K0, tspan, (X.V,model))
    K_integrator = init(K_problem, K_alg; save_everystep=false, K_options...)
    step!(K_integrator, 1e-5, true)
    set_t!(K_integrator, t0)

    S_rhs = (S, (U, V, model), t) -> Matrix(U'*F(model, SVDLikeRepresentation(U,S,V),t)*V)
    S_problem = ODEProblem(S_rhs, X.S, tspan, (X.U, X.V, model))
    S_integrator = init(S_problem, S_alg; save_everystep=false, S_options...)
    step!(S_integrator, 1e-5, true)
    set_t!(S_integrator, t0)

    L_rhs = (L, (U,model), t) -> Matrix(F(model, TwoFactorRepresentation(U,L), t)'*U)
    L_problem = ODEProblem(L_rhs, L0, tspan, (X.U, model))
    L_integrator = init(L_problem, L_alg; save_everystep=false, L_options...)
    step!(L_integrator, 1e-5, true)
    set_t!(L_integrator, t0)

    BUGCache(K0, L0, U1, S1, V1, M, N, L_integrator, S_integrator, K_integrator, X, missing)
end

function initialize_cache(prob, ::SparseLowRankModel{true}, R::BUGRetraction, SA)
    @unpack model, X0, tspan, dims = prob
    @unpack S_alg, K_alg, L_alg, S_options, K_options, L_options = R

    t0, tf = tspan

    sparse_approximation_cache = initialize_sparse_approximation_cache(SA, X0, R)
    @unpack PL, PK, PS = sparse_approximation_cache

    X = deepcopy(X0)
    K0 = X.U*X.S
    L0 = X.V*X.S'
    U1 = similar(X.U)
    S1 = similar(X.S)
    V1 = similar(X.V)
    M = similar(X.S)
    N = similar(X.S)
    
    K_rhs = function (dK, K, (PK, V, model), t)
                evaluate_approximator!(dK, PK, model, TwoFactorRepresentation(K, V), t)
                nothing
            end
    K_problem = ODEProblem(K_rhs, K0, tspan, (PK', X.V, model))
    K_integrator = init(K_problem, K_alg; save_everystep=false, K_options...)
    step!(K_integrator, 1e-5, true)
    set_t!(K_integrator, t0)

    S_rhs = function (dS, S, (PS, U, V, model), t)
        evaluate_approximator!(dS, PS, model, SVDLikeRepresentation(U, S, V), t)
        nothing
    end
    S_problem = ODEProblem(S_rhs, X.S, tspan, (PS, X.U, X.V, model))
    S_integrator = init(S_problem, S_alg; save_everystep=false, S_options...)
    step!(S_integrator, 1e-5, true)
    set_t!(S_integrator, t0)

    L_rhs = function (dL, L, (PL, U, model), t)
        evaluate_approximator!(dL', PL, model, TwoFactorRepresentation(U, L), t)
    end
    L_problem = ODEProblem(L_rhs, L0, tspan, (PL, X.U, model))
    L_integrator = init(L_problem, L_alg; save_everystep=false, L_options...)
    step!(L_integrator, 1e-5, true)
    set_t!(L_integrator, t0)

    BUGCache(K0, L0, U1, S1, V1, M, N, L_integrator, S_integrator, K_integrator, X, sparse_approximation_cache)
end

@concrete struct BUGSparseApproximatorCache
    PK # sparse projector for K step 
    PL # sparse projector for L step
    PS # sparse approximator for S step
end

function initialize_sparse_approximation_cache(SA::SparseApproximation, X0, ::BUGRetraction)
    @unpack sparse_approximator = SA
    @unpack range, corange = sparse_approximator

    col_cache = similar(cache(sparse_approximator).columns)
    row_cache = similar(cache(sparse_approximator).rows)
    PK = SparseProjector(X0.V'*weights(corange), indices(corange), col_cache)
    PL = SparseProjector(X0.U'*weights(range), indices(range), row_cache)
    PS = SparseMatrixApproximator(deepcopy(PL), deepcopy(PK))
    
    BUGSparseApproximatorCache(PK, PL, PS)
end

function retracted_step!(cache, model::AbstractLowRankModel, t, h, ::BUGRetraction, SA)
    K_step!(cache, model, t, h, SA)
    L_step!(cache, model, t, h, SA)

    @unpack X, U1, V1 = cache
    X.U .= U1
    X.V .= V1

    S_step!(cache, model, t, h, SA)
end

# for outofplace factored model 
function K_step!(cache::BUGCache, model, t, h, SA)
    @unpack X, K0, U1, M, K_integrator = cache

    mul!(K0, X.U, X.S)
    set_u!(K_integrator, K0)
    step!(K_integrator, h, true)
    K0 .= K_integrator.u # maybe can save this
    QRK = qr!(K0)
    U1 .= Matrix(QRK.Q) 
    mul!(M, U1', X.U)
end
function L_step!(cache::BUGCache, model, t, h, SA)
    @unpack X, L0, V1, N, L_integrator = cache

    mul!(L0,X.V,X.S')
    set_u!(L_integrator, L0)
    step!(L_integrator, h, true)
    L0 .= L_integrator.u # maybe can save this
    QRL = qr!(L0)
    V1 .= Matrix(QRL.Q)
    mul!(N, V1', X.V)
end
function S_step!(cache::BUGCache, model, t, h, SA)
    @unpack X, S1, M, N, S_integrator = cache
    
    mul!(S1, M, X.S)
    mul!(X.S, S1, N') 
    set_u!(S_integrator, X.S) 
    step!(S_integrator, h, true)
    X.S .= S_integrator.u
end
function S_step!(cache::BUGCache, model::SparseLowRankModel, t, h, SA)
    @unpack X, S1, M, N, S_integrator, sparse_approximation_cache = cache
    @unpack PS = sparse_approximation_cache
    @unpack sparse_approximator = SA
    
    mul!(PS.range.weights, X.U', sparse_approximator.range.weights)
    mul!(PS.corange.weights, X.V', sparse_approximator.corange.weights)


    mul!(S1, M, X.S)
    mul!(X.S, S1, N') 
    set_u!(S_integrator, X.S) 
    step!(S_integrator, h, true)
    X.S .= S_integrator.u
end

# needs to be adapted if rank adaptation is considered 
# need to recreate the projectors
function update_cache!(cache::BUGCache, SA::SparseApproximation)
    @unpack X, sparse_approximation_cache = cache
    @unpack PK, PL, PS = sparse_approximation_cache
    @unpack sparse_approximator = SA
    
    mul!(PK.weights, X.V', sparse_approximator.corange.weights)
    PK.indices .= sparse_approximator.corange.indices

    mul!(PL.weights, X.U', sparse_approximator.range.weights)
    PL.indices .= sparse_approximator.range.indices 

    PS.range.indices .= sparse_approximator.range.indices 
    PS.corange.indices .= sparse_approximator.corange.indices 
end

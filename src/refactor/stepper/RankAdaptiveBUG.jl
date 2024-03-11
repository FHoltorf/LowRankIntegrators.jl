using OrdinaryDiffEq
import OrdinaryDiffEq: step!, set_u!, init

@concrete struct RankAdaptiveBUGStepper <: LowRankStepper 
    rmin::Int
    rmax::Int
    atol::Float64
    rtol::Float64
    S_alg
    K_alg
    L_alg
    S_options
    K_options
    L_options
end

function RankAdaptiveBUGStepper(rmin::Int, rmax::Int, alg = Tsit5(); rtol = 1e-8, atol = 1e-8, 
                            S_stepper = alg,
                            K_stepper = alg,
                            L_stepper = alg,
                            S_options = Dict{Symbol,Any}(),
                            K_options = Dict{Symbol,Any}(),
                            L_options = Dict{Symbol,Any}(),
                            kwargs...) 

    S_options = isempty(S_options) ? kwargs : S_options
    K_options = isempty(K_options) ? kwargs : K_options
    L_options = isempty(L_options) ? kwargs : L_options

    RankAdaptiveBUGStepper(rmin, rmax, atol, rtol, S_stepper, K_stepper, L_stepper, S_options, K_options, L_options)
end

@concrete mutable struct RankAdaptiveBUGCache <: LowRankStepperCache
    K0
    L0
    Uhat
    MS
    MSN
    Vhat
    M
    N
    L_integrator
    S_integrator
    K_integrator
    L_rhs
    S_rhs
    K_rhs
    X
    sparse_approximation_cache
end

state(cache::RankAdaptiveBUGCache) = cache.X

initialize_cache(prob::MatrixDEProblem, R::RankAdaptiveBUGStepper, SA) = 
                    initialize_cache(prob.X0, prob.tspan, prob.model, R, SA)

function initialize_cache(X0, tspan, model::FactoredLowRankModel{false}, R::RankAdaptiveBUGStepper, ::Missing)
    @unpack S_alg, K_alg, L_alg, S_options, K_options, L_options = R

    t0 = tspan[1]

    X = deepcopy(X0)
    r = rank(X.S)
    K0 = X.U*X.S
    L0 = X.V*X.S'
    Uhat = zeros(size(X.U,1), 2r)
    MS = zeros(2r,r)
    MSN = zeros(2r,2r)
    Vhat = zeros(size(X.V,1), 2r)
    M = zeros(2r, r)
    N = similar(M)
    
    K_rhs = (K, (V, model), t) -> Matrix(F(model, TwoFactorRepresentation(K,V),t)*V)
    K_problem = ODEProblem(K_rhs, K0, tspan, (X.V,model))
    K_integrator = init(K_problem, K_alg; save_everystep=false, K_options...)
    step!(K_integrator, 1e-5, true)
    set_t!(K_integrator, t0)

    S_rhs = (S, (U, V, model), t) -> Matrix(U'*F(model, SVDLikeRepresentation(U,S,V),t)*V)
    S_problem = ODEProblem(S_rhs, S1, tspan, (Uhat, Vhat, model))
    S_integrator = init(S_problem, S_alg; save_everystep=false, S_options...)
    step!(S_integrator, 1e-5, true)
    set_t!(S_integrator, t0)

    L_rhs = (L, (U,model), t) -> Matrix(F(model, TwoFactorRepresentation(U,L), t)'*U)
    L_problem = ODEProblem(L_rhs, L0, tspan, (X.U, model))
    L_integrator = init(L_problem, L_alg; save_everystep=false, L_options...)
    step!(L_integrator, 1e-5, true)
    set_t!(L_integrator, t0)

    RankAdaptiveBUGCache(K0, L0, Uhat, MS, MSN, Vhat, M, N, L_integrator, S_integrator, K_integrator,
                         L_rhs, S_rhs, K_rhs, X, missing)
end

function initialize_cache(X0, tspan, model::SparseLowRankModel{true}, R::RankAdaptiveBUGStepper, SA)
    @unpack S_alg, K_alg, L_alg, S_options, K_options, L_options = R
    
    sparse_approximation_cache = initialize_sparse_approximation_cache(SA, X0, R)
    @unpack PL, PK, PS = sparse_approximation_cache
    
    t0 = tspan[1]

    X = deepcopy(X0)
    r = rank(X)

    K0 = X.U*X.S
    L0 = X.V*X.S'
    Uhat = hcat(X.U, zeros(size(X.U,1), r))
    Vhat = hcat(X.V, zeros(size(X.V,1), r))
    M = zeros(2r, r)
    MS = similar(M)
    N = similar(M)
    MSN = [X.S zeros(r,r);
           zeros(r,r) zeros(r,r)]
    
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
    S_problem = ODEProblem(S_rhs, MSN, tspan, (PS, Uhat, Vhat, model))
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

    RankAdaptiveBUGCache(K0, L0, Uhat, MS, MSN, Vhat, M, N, L_integrator, S_integrator, K_integrator,
                         L_rhs, S_rhs, K_rhs, X, sparse_approximation_cache)
end

function adapt_cache!(cache::RankAdaptiveBUGCache, model, X_new, t, R::RankAdaptiveBUGStepper, SA)    
    cache.X = X_new
    r = rank(X_new)

    cache.K0 = X_new.U*X_new.S
    cache.L0 = X_new.V*X_new.S'
    cache.Uhat = hcat(X_new.U, zeros(size(X_new.U,1), r))
    cache.Vhat = hcat(X_new.V, zeros(size(X_new.V,1), r))
    cache.M = zeros(2r, r)
    cache.MS = similar(cache.M)
    cache.N = similar(cache.M)
    cache.MSN = [X_new.S zeros(r,r);
                 zeros(r,r) zeros(r,r)]

    cache.sparse_approximation_cache = initialize_sparse_approximation_cache(SA, X_new, R)
    @unpack PL, PK, PS = cache.sparse_approximation_cache
             
    K_problem = ODEProblem(cache.K_rhs, cache.K0, (t, t+1e-5), (PK', X_new.V, model))
    cache.K_integrator = init(K_problem, R.K_alg; save_everystep=false, R.K_options...)
    step!(cache.K_integrator, 1e-5, true)
    set_t!(cache.K_integrator, t)

    S_problem = ODEProblem(cache.S_rhs, cache.MSN, (t, t+1e-5), (PS, cache.Uhat, cache.Vhat, model))
    cache.S_integrator = init(S_problem, R.S_alg; save_everystep=false, R.S_options...)
    step!(cache.S_integrator, 1e-5, true)
    set_t!(cache.S_integrator, t)

    L_problem = ODEProblem(cache.L_rhs, cache.L0, (t, t+1e-5), (PL, X_new.U, model))
    cache.L_integrator = init(L_problem, R.L_alg; save_everystep=false, R.L_options...)
    step!(cache.L_integrator, 1e-5, true)
    set_t!(cache.L_integrator, t)

end

@concrete struct RankAdaptiveBUGSparseApproximatorCache
    PK # sparse projector for K step 
    PL # sparse projector for L step
    PS # sparse approximator for S step
end

function initialize_sparse_approximation_cache(SA::SparseApproximation, X0, ::RankAdaptiveBUGStepper)
    @unpack sparse_approximator = SA
    @unpack range, corange = sparse_approximator

    r = rank(X0)

    col_cache = similar(cache(sparse_approximator).columns)
    row_cache = similar(cache(sparse_approximator).rows)
    PK = SparseProjector(X0.V'*weights(corange), indices(corange), col_cache)
    PL = SparseProjector(X0.U'*weights(range), indices(range), row_cache)
    
    PScorange = SparseProjector(zeros(2r, size(weights(corange),2)), indices(corange))
    PSrange = SparseProjector(zeros(2r, size(weights(range),2)), indices(range))
    PS = SparseMatrixApproximator(PSrange, PScorange)
    
    RankAdaptiveBUGSparseApproximatorCache(PK, PL, PS)
end


function retracted_step!(cache, model::AbstractLowRankModel, t, h, R::RankAdaptiveBUGStepper, SA)
    K_step!(cache, model, t, h, SA)
    L_step!(cache, model, t, h, SA)
    S_step!(cache, model, t, h, SA)

    @unpack X, MSN, Uhat, Vhat = cache 
    U, S, V = svd!(MSN)
    
    r = rank(X)
    r_new = rank_by_tol(S, R.rmin, R.rmax, R.atol, R.rtol)
    if r == r_new
        @views mul!(X.U,Uhat,U[:,1:r])
        X.S .= Diagonal(S[1:r])
        @views mul!(X.V,Vhat,V[:,1:r])
    else
        @views X_new = SVDLikeRepresentation(Uhat*U[:,1:r_new], 
                                             Matrix(Diagonal(S[1:r_new])), 
                                             Vhat*V[:,1:r_new])
        adapt_cache!(cache, model, X_new, t+h, R, SA)
    end
end

# for outofplace factored model 
function K_step!(cache::RankAdaptiveBUGCache, model, t, h, SA)
    @unpack X, K0, Uhat, M, K_integrator = cache

    r = rank(X)
    mul!(K0, X.U, X.S)
    set_u!(K_integrator, K0)
    step!(K_integrator, h, true)
    @views Uhat[:,1:r] .= X.U 
    @views Uhat[:,r+1:2r] .= K_integrator.u 
    QRK = qr!(Uhat)
    Uhat .= Matrix(QRK.Q) 
    mul!(M, Uhat', X.U)
end
function L_step!(cache::RankAdaptiveBUGCache, model, t, h, SA)
    @unpack X, L0, Vhat, N, L_integrator = cache

    r = rank(X)
    mul!(L0,X.V,X.S')
    set_u!(L_integrator, L0)
    step!(L_integrator, h, true)
    @views Vhat[:,1:r] .= X.V
    @views Vhat[:,r+1:2r] .= L_integrator.u 
    QRL = qr!(Vhat)
    Vhat .= Matrix(QRL.Q) 
    mul!(N, Vhat', X.V)
end
function S_step!(cache::RankAdaptiveBUGCache, model, t, h, SA)
    @unpack X, MS, MSN, M, N, S_integrator = cache
    
    mul!(MS, M, X.S)
    mul!(MSN, MS, N') 
    set_u!(S_integrator, MSN) 
    step!(S_integrator, h, true)
    MSN .= S_integrator.u
end
function S_step!(cache::RankAdaptiveBUGCache, model::SparseLowRankModel, t, h, SA)
    @unpack X, MS, MSN, M, N, Uhat, Vhat, S_integrator, sparse_approximation_cache = cache
    @unpack PS = sparse_approximation_cache
    @unpack sparse_approximator = SA
    
    mul!(PS.range.weights, Uhat', sparse_approximator.range.weights)
    mul!(PS.corange.weights, Vhat', sparse_approximator.corange.weights)

    mul!(MS, M, X.S)
    mul!(MSN, MS, N') 
    set_u!(S_integrator, MSN) 
    step!(S_integrator, h, true)
    MSN .= S_integrator.u
end

# needs to be adapted if rank adaptation is considered 
# need to recreate the projectors
function update_cache!(cache::RankAdaptiveBUGCache, SA::SparseApproximation)
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

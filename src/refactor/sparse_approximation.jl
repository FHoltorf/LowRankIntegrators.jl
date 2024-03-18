import LinearAlgebra: adjoint, rank
import Base.size, Base.*

using Clustering

"""
    $(TYPEDEF)

    Oblique projector that interpolates or approximates based on sparse evaluations.
"""
struct SparseProjector{wType,cType} 
    indices::Vector{Int}
    weights::wType
    cache::cType
    function SparseProjector(weights, indices::Vector{Int}, cache = missing)
        new{typeof(weights), typeof(cache)}(indices, weights, cache)
    end
end
 

*(P::SparseProjector, A::AbstractMatrix) = P.weights * A[P.indices,:]
*(P::SparseProjector, A::AbstractVector) = P.weights * A[P.indices]

"""
    $(TYPEDEF)

    Lazy adjoint for `SparseProjector`s.
"""
struct AdjointSparseProjector{P <: SparseProjector} 
    parent::P
end

*(A::AbstractMatrix, P::AdjointSparseProjector) = A[:,P.parent.indices] * P.parent.weights'


"""
    $(TYPEDSIGNATURES)

    lazy adjoint for `SparseProjector`s. We mean adjoint in the sense that the approximation is 
    applied to the corange. 
"""
adjoint(P::SparseProjector) = AdjointSparseProjector(P)
adjoint(P::AdjointSparseProjector) = P.parent

"""
    $(TYPEDEF)

    Sparse approximator for matrices encoded by approximators for range and corange.
"""
struct SparseMatrixApproximator{R <: SparseProjector, C <: SparseProjector, cType} 
    range::R
    corange::C
    cache::cType
end

function SparseMatrixApproximator(row_idcs::Vector{Int}, col_idcs::Vector{Int},
                                  range_weights, corange_weights; make_cache = true)
    range = SparseProjector(range_weights, row_idcs)
    corange = SparseProjector(corange_weights, col_idcs)
    
    if make_cache
        T = promote_type(eltype(range_weights), eltype(corange_weights))
        n_rows, n_cols = length(row_idcs), length(col_idcs)
        n, m = size(range_weights, 1), size(corange_weights,1)
        cache = SparseMatrixApproximatorCache{T}(zeros(T, n_rows, m),
                                                zeros(T, n, n_cols),
                                                zeros(T, n_rows, n_cols))
    else
        cache = missing
    end

    SparseMatrixApproximator(range, corange, cache)
end

function SparseMatrixApproximator(range::SparseProjector, corange::SparseProjector; make_cache = true)
    if make_cache
        range_weights, corange_weights = weights(range), weights(corange)
        n_rows, n_cols = length(indices(range)), length(indices(corange))
        T = promote_type(eltype(range_weights), eltype(corange_weights))
        n, m = size(range_weights, 1), size(corange_weights,1)
        cache = SparseMatrixApproximatorCache{T}(zeros(T, n_rows, m),
                                                zeros(T, n, n_cols),
                                                zeros(T, n_rows, n_cols))
    else
        cache = missing
    end

    SparseMatrixApproximator(range, corange, cache)
end



function *(P::SparseMatrixApproximator, A::AbstractMatrix)
    @unpack range, corange = P
    
    range.weights * A[range.indices,corange.indices] * corange.weights'
end

"""
    $(TYPEDEF)
 
    Auxiliary cache for inplace updates.
"""
@concrete struct SparseMatrixApproximatorCache{T}
    rows::Matrix{T}
    columns::Matrix{T}
    elements::Matrix{T}
end

"""
    $(TYPEDSIGNATURES)

    helper function to query the approximation indices.
"""
indices(P::SparseProjector) = P.indices
indices(P::AdjointSparseProjector) = P.parent.indices
indices(P::SparseMatrixApproximator) = (indices(P.range), indices(P.corange))

"""
    $(TYPEDSIGNATURES)

    helper function to query the approximation indices.
"""
weights(P::SparseProjector) = P.weights
weights(P::AdjointSparseProjector) = P.parent.weights
weights(P::SparseMatrixApproximator) = (weights(P.range), weights(P.corange))

"""
    $(TYPEDSIGNATURES)

    helper function to query the approximation indices.
"""
cache(P::SparseProjector) = P.cache
cache(P::AdjointSparseProjector) = P.parent.cache
cache(P::SparseMatrixApproximator) = P.cache


"""
    $(TYPEDSIGNATURES)

    outofplace evaluation of a `SparseLowrankModel` at approximators.
"""
function evaluate_approximator(P::SparseProjector, model::SparseLowRankModel{false}, X, t)
    R = rows(model, X, t, indices(P))
    weights(P) * R
end

function evaluate_approximator(P::AdjointSparseProjector, model::SparseLowRankModel{false}, X, t)
    C = columns(model, X, t, indices(P))
    C * weights(P)'
end

function evaluate_approximator(P::SparseMatrixApproximator, model::SparseLowRankModel{false}, X, t)
    E = elements(model, X, t, P.range.indices, indices(P)...)
    W, V = weights(P)
    W * E * V'
end

"""
    $(TYPEDSIGNATURES)

    out-of-place evaluation of sparse approximators in factored form.
"""
function evaluate_approximator_factored(P::SparseProjector, model::SparseLowRankModel{false}, X, t)
    R = rows(model, X, t, indices(P))
    TwoFactorRepresentation(weights(P), R')
end

function evaluate_approximator_factored(P::AdjointSparseProjector, model::SparseLowRankModel{false}, X, t)
    C = columns(model, X, t, indices(P))
    TwoFactorRepresentation(C, weights(P))
end

function evaluate_approximator_factored(P::SparseMatrixApproximator, model::SparseLowRankModel{false}, X, t)
    E = elements(model, X, t, P.range.indices, indices(P)...)
    W, V = weights(P)
    SVDLikeRepresentation(W, E, V)
end

"""
    $(TYPEDSIGNATURES)

    inplace evaluation of a `SparseLowrankModel` at approximators.
"""
function evaluate_approximator!(dX, P::SparseProjector, model::SparseLowRankModel{true}, X, t)
    rows!(model, cache(P), X, t, indices(P))
    mul!(dX, weights(P), cache(P))
end
function evaluate_approximator!(dX, P::SparseProjector, model::SparseLowRankModel{false}, X, t)
    rows = rows(model, X, t, indices(P))
    mul!(dX, weights(P), rows)
end

function evaluate_approximator!(dX, P::AdjointSparseProjector, model::SparseLowRankModel{true}, X, t)
    columns!(model, cache(P), X, t, indices(P))
    mul!(dX, cache(P), weights(P)')
end
function evaluate_approximator!(dX, P::AdjointSparseProjector, model::SparseLowRankModel{false}, X, t)
    columns = columns(model, X, t, indices(P))
    mul!(dX, columns, weights(P)')
end

function evaluate_approximator!(dX, P::SparseMatrixApproximator, model::SparseLowRankModel{true}, X, t)
    @unpack elements, rows, columns = cache(P)
    row_idcs, col_idcs = indices(P)
    elements!(model, elements, X, t, row_idcs, col_idcs)
    if size(rows,2) <= size(columns,1)
        mul!(rows, elements, weights(P.corange)')
        mul!(dX, weights(P.range), rows)
    else
        mul!(columns, elements, weights(P.corange)')
        mul!(dX, weights(P.range), rows)
    end
end
function evaluate_approximator!(dX, P::SparseMatrixApproximator, model::SparseLowRankModel{false}, X, t)
    @unpack elements, rows, columns = cache(P)
    row_idcs, col_idcs = indices(P)
    elements .= elements(model, X, t, row_idcs, col_idcs)
    if size(rows,2) <= size(columns,1)
        mul!(rows, elements, weights(P.corange)')
        mul!(dX, weights(P.range), rows)
    else
        mul!(columns, elements, weights(P.corange)')
        mul!(dX, weights(P.range), rows)
    end
end

"""
    $(TYPEDSIGNATURES)
    
    inplace evaluation of sparse approximator in factored form 
"""
function evaluate_approximator_factored!(dX::TwoFactorRepresentation, P::SparseProjector, model::SparseLowRankModel{true}, X, t)
    rows!(model, dX.Z', X, t, indices(P))
    copyto!(dX.U, weights(P))
end
function evaluate_approximator_factored!(dX::TwoFactorRepresentation, P::AdjointSparseProjector, model::SparseLowRankModel{true}, X, t)
    columns!(model, dX.U, X, t, indices(P))
    copyto!(dX.Z, weights(P))
end
function evaluate_approximator_factored!(dX::SVDLikeRepresentation, P::SparseMatrixApproximator, model::SparseLowRankModel{true}, X, t)
    row_idcs, col_idcs = indices(P)
    elements!(model, dX.S, X, t, row_idcs, col_idcs)
    copyto!(dX.U, weights(P.range))
    copyto!(dX.V, weights(P.corange))
end

"""
    $(TYPEDSIGNATURES)

    inplace (in cache) evaluation of columns, rows, elements oracles
"""
function evaluate_oracles!(P::SparseMatrixApproximator, model::SparseLowRankModel{true}, X, t)
    @unpack elements, rows, columns = cache(P)
    @unpack range, corange = P

    # rows/columns evaluation could be done in parallel
    columns!(model, columns, X, t, indices(corange))
    rows!(model, rows, X, t, indices(range))
    @views copyto!(elements, rows[:, indices(corange)])
end

function evaluate_elements!(P::SparseMatrixApproximator, model::SparseLowRankModel{true}, X, t)
    @unpack elements, rows, columns = cache(P)
    elements!(model, elements, X, t, indices(P)...)
end
function evaluate_elements!(els::AbstractMatrix, P::SparseMatrixApproximator, model::SparseLowRankModel{true}, X, t)
    elements!(model, els, X, t, indices(P)...)
end

function evaluate_rows!(P::SparseMatrixApproximator, model::SparseLowRankModel{true}, X, t)
    @unpack elements, rows, columns = cache(P)
    @unpack range, corange = P
    rows!(model, rows, X, t, indices(range))
end
function evaluate_rows!(rows::AbstractMatrix, P::SparseMatrixApproximator, model::SparseLowRankModel{true}, X, t)
    @unpack range, corange = P
    rows!(model, rows, X, t, indices(range))
end

function evaluate_columns!(P::SparseMatrixApproximator, model::SparseLowRankModel{true}, X, t)
    @unpack elements, rows, columns = cache(P)
    @unpack range, corange = P
    columns!(model, columns, X, t, indices(corange))
end
function evaluate_columns!(columns::AbstractMatrix,P::SparseMatrixApproximator, model::SparseLowRankModel{true}, X, t)
    @unpack range, corange = P
    columns!(model, columns, X, t, indices(corange))
end

function evaluate_oracles!(P::SparseMatrixApproximator, model::SparseLowRankModel{false}, X, t)
    @unpack elements, rows, columns = cache(P)
    @unpack range, corange = P

    # rows/columns evaluation could be done in parallel
    columns .= columns(model, X, t, indices(corange))
    rows .= rows(model, X, t, indices(range))
    @views copyto!(elements, rows[:, indices(corange)])
end

function evaluate_elements!(P::SparseMatrixApproximator, model::SparseLowRankModel{false}, X, t)
    @unpack elements, rows, columns = cache(P)
    elements .= elements(model, X, t, indices(P)...)
end
function evaluate_elements!(els::AbstractMatrix, P::SparseMatrixApproximator, model::SparseLowRankModel{false}, X, t)
    els .= elements(model, X, t, indices(P)...)
end

function evaluate_rows!(P::SparseMatrixApproximator, model::SparseLowRankModel{false}, X, t)
    @unpack elements, rows, columns = cache(P)
    @unpack range, corange = P
    rows .= rows(model, X, t, indices(range))
end
function evaluate_rows!(rows::AbstractMatrix, P::SparseMatrixApproximator, model::SparseLowRankModel{false}, X, t)
    @unpack range, corange = P
    rows .= rows(model, X, t, indices(range))
end

function evaluate_columns!(P::SparseMatrixApproximator, model::SparseLowRankModel{false}, X, t)
    @unpack elements, rows, columns = cache(P)
    @unpack range, corange = P
    columns .= columns(model, X, t, indices(corange))
end
function evaluate_columns!(columns::AbstractMatrix, P::SparseMatrixApproximator, model::SparseLowRankModel{false}, X, t)
    @unpack range, corange = P
    columns .= columns(model, X, t, indices(corange))
end


"""
    $(TYPEDSIGNATURES)

    returns the rank of a sparse projector
"""
rank(P::SparseProjector) = length(weights(P))
rank(P::AdjointSparseProjector) = rank(P.parent)
rank(P::SparseMatrixApproximator) = min(rank(P.range), rank(P.corange))


"""
    $(TYPEDEF)

    abstract type characterizing index selection algorithms for sparse approximation.
"""
abstract type IndexSelectionAlgorithm end

"""
    $(TYPEDEF)

    DEIM index selection algorithm.
"""
struct DEIM <: IndexSelectionAlgorithm 
    tol::Float64
    elasticity::Float64
    rmin::Int
    rmax::Int
end

DEIM( ;tol = eps(Float64), rmin = 1, rmax = 2^62, elasticity = 0.1) = DEIM(tol, elasticity, rmin, rmax)

"""
    $(TYPEDEF)

    QDEIM index selection algorithm.
"""
struct QDEIM <: IndexSelectionAlgorithm
    tol::Float64
    elasticity::Float64
    rmin::Int
    rmax::Int
end

QDEIM( ;tol = eps(Float64), rmin = 1, rmax = 2^62, elasticity = 0.1) = QDEIM(tol, elasticity, rmin, rmax)

"""
    $(TYPEDEF)

    LDEIM index selection algorithm. Selects `n` indices. 
"""
struct LDEIM <: IndexSelectionAlgorithm
    tol::Float64
    elasticity::Float64
    rmin::Int
    rmax::Int
end

LDEIM( ;tol = eps(Float64), rmin = 1, rmax = 2^62, elasticity = 0.1) = LDEIM(tol, elasticity, rmin, rmax)


"""
    $(TYPEDEF)

    GappyPOD+E index selection algorithm.
"""
struct GappyPODE <: IndexSelectionAlgorithm 
    m::Int
end

"""
    $(TYPEDSIGNATURES)

    returns approximation indices for sparse approximation. Supports DEIM and QDEIM index selection.
"""
function index_selection(U, ::DEIM)
    m = size(U, 2)
    indices = Vector{Int}(undef, m)
    @inbounds @views begin
        r = abs.(U[:, 1])
        indices[1] = argmax(r)
        for l in 2:m
            Uₗ = U[:, 1:(l - 1)]
            P = indices[1:(l - 1)]
            PᵀUₗ = Uₗ[P, :]
            uₗ = U[:, l]
            Pᵀuₗ = uₗ[P, :]
            c = vec(PᵀUₗ \ Pᵀuₗ)
            mul!(r, Uₗ, c)
            @. r = abs(uₗ - r)
            indices[l] = argmax(r)
        end
    end
    return indices
end

function index_selection(U,::QDEIM)
    QR = qr(U', ColumnNorm())
    return QR.p[1:size(U,2)]
end

function index_selection(U, r::Int, ::LDEIM)   
    n, k = size(U)
    if k > r
        return index_selection(view(U, :, 1:r), DEIM())
    else
        Ψ = copy(U)
        idcs = zeros(Int, r)
        @views for i in 1:k-1
            idcs[i] = argmax(Iterators.map(abs, Ψ[:,i]))
            corr = Ψ[idcs[1:i],1:i] \ Ψ[idcs[1:i], i+1]
            Ψ[:,i+1] -= Ψ[:,1:i] * corr
        end
        @views idcs[k] = argmax(Iterators.map(abs, Ψ[:,k]))
        @views l = [i in idcs ? -1.0 : norm(Ψ[i,:]) for i in 1:n]
        idcs[k+1:end] = partialsortperm(l, 1:r-k, by=-)
        return idcs
    end
end

index_selection(U, ::LDEIM) = index_selection(U, DEIM())

function index_selection(U, S::AbstractVector, alg::IndexSelectionAlgorithm)
    @assert size(U,2) == length(S) "Number of columns provided must coincide with number of singular values."

    cutoff = length(S)
    # for interpolation rank adaption this needs to be revisited
    # currently the rank can at most increase
    # also we need a routine to update the necessary caches for 
    # the sparse_approximator ...
    # also for SparseInterpolationRK this is critical.
    for σ in reverse(S)
        if σ > alg.tol
            break
        end
        cutoff -= 1
    end

    if cutoff < alg.rmin 
        r = alg.rmin
    elseif cutoff > alg.rmax
        r = alg.rmax
    else
        r = cutoff
    end

    return index_selection(U[:,1:r], alg)
end

function index_selection(U, S::AbstractVector, alg::LDEIM)
    @assert size(U,2) == length(S) "Number of columns provided must coincide with number of singular values."

    if S[end] > alg.tol 
        r = min(alg.rmax, length(S)+1)
    else
        cutoff = findlast(x -> x > alg.tol*alg.elasticity, S)
        r = isnothing(cutoff) ? alg.rmin : cutoff
    end 
    return index_selection(U, r, alg)
end

function index_selection(U, gpode::GappyPODE)
    indices = index_selection(U, DEIM())
    for k in 1:gpode.m
        _, S, W = svd(U[indices, :])
        g = S[end-1]^2 - S[end]^2
        Ub = W'*U'
        u = vec(sum(Ub .^ 2, dims = 1))
        r = g .+ u
        r .-= sqrt.(abs.((g .+ u).^2 .- 4 * g * Ub[end,:].^2))
        candidates = sortperm(r, rev = true)
        idx = findfirst(x -> !(x in indices), candidates)
        push!(indices, idx)
    end
    return indices
end

"""
    $(TYPEDEF)

    type for carrying all necessary information for continuous sparse approximation 
    to be applied for dynamical low rank approximation.
"""
@concrete mutable struct SparseApproximation
    selection_alg
    update_scheme
    UF
    VF
    sparse_approximator 
end

function SparseApproximation(selection_alg::IndexSelectionAlgorithm, 
                             UF::Matrix,
                             VF::Matrix;
                             update_scheme = :last_iterate)
    @assert norm(UF'*UF - I) ≤ 1e-8 "range estimates must be near orthogonal!"
    @assert norm(VF'*VF - I) ≤ 1e-8 "range estimates must be near orthogonal!"

    row_indices = index_selection(UF, selection_alg)
    col_indices = index_selection(VF, selection_alg)

    @views range_weights = UF/UF[row_indices,:]
    @views corange_weights = VF/VF[col_indices,:]
    
    sparse_approximator = SparseMatrixApproximator(row_indices, col_indices, range_weights, corange_weights)
    
    SparseApproximation(selection_alg, update_scheme, UF, VF, sparse_approximator)
end

function approximate_ranges!(SA::SparseApproximation, model, cache, t)
    @unpack update_scheme, selection_alg, sparse_approximator, UF, VF = SA
    X = state(cache)
    if update_scheme == :avg_flow
        error("average flpw clustering needs to be implemented")
    elseif update_scheme == :last_iterate
        evaluate_rows!(sparse_approximator, model, X, t)
        evaluate_columns!(sparse_approximator, model, X, t)
        UF .= svd!(sparse_approximator.cache.columns).U[:, 1:size(UF,2)]
        VF .= Matrix(svd!(sparse_approximator.cache.rows).V[:, 1:size(VF,2)])
    elseif update_scheme == :kmeans
        projected_cols = X.S*X.V'
        projected_rows = X.S'*X.U'
        n_cols = length(sparse_approximator.corange.indices)
        n_rows = length(sparse_approximator.range.indices)
        
        col_centers = get_clusters(projected_cols,n_cols)
        row_centers = get_clusters(projected_rows,n_rows)

        rows!(model, sparse_approximator.cache.rows, X, t, row_centers)
        columns!(model, sparse_approximator.cache.columns, X, t, col_centers)
        UF .= svd!(sparse_approximator.cache.columns).U
        VF .= Matrix(svd!(sparse_approximator.cache.rows).V)
    else 
        error("Update scheme $(update_scheme) not supported")
    end
end

function get_clusters(X, k)
    res = kmeans(X, k)  
    idcs = [findmin(x -> norm(x - res.centers[:,k]), eachcol(X))[2] for k in 1:size(res.centers,2)]
    return idcs
end

function update_sparse_approximation!(SA::SparseApproximation, model, cache, t)
    @unpack selection_alg, sparse_approximator = SA
    approximate_ranges!(SA, model, cache, t)
    row_indices = index_selection(SA.UF, selection_alg)
    col_indices = index_selection(SA.VF, selection_alg)
    
    # e.g. update_ranges!() 
    sparse_approximator.range.indices .= row_indices
    sparse_approximator.corange.indices .= col_indices
    sparse_approximator.range.weights .= SA.UF/SA.UF[row_indices,:]
    sparse_approximator.corange.weights .= SA.VF/SA.VF[col_indices,:]
    
    update_cache!(cache, SA) 
end


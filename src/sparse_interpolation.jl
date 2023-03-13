import LinearAlgebra: adjoint, rank
import Base.size

"""
    $(TYPEDEF)

    Abstract type identifying sparse interpolation methods.
"""
abstract type SparseInterpolator end

"""
    $(TYPEDEF)

    Abstract type identifying caches for sparse interpolation methods.
"""
abstract type AbstractInterpolatorCache end

"""
    $(TYPEDEF)
 
    Auxiliary cache for potentially fast in place updates.
"""
struct InterpolatorCache{T} <: AbstractInterpolatorCache
    rows::Matrix{T}
    cols::Matrix{T}
    elements::Matrix{T}
end

function resize_rows(IC::InterpolatorCache{T}, n_rows::Int) where T
    return InterpolatorCache(Matrix{T}(undef, n_rows, size(IC.rows,2)), IC.cols, IC.rows)
end

function resize_cols(IC::InterpolatorCache{T}, n_cols::Int) where T
    return InterpolatorCache(IC.rows, Matrix{T}(undef, size(IC.cols,2), n_cols), IC.rows)
end

"""
    $(TYPEDEF)

    type identifying (vector- or matrix-valued) functions which may be evaluated in components 
    (individual rows, columns or even elements).

    #Fields
        * rows!         function that evaluates (dx, x, p, t, row_idcs) -> F(x,p,t)[row_idcs, :] 
                        and stores the result in dx, i.e., evaluation of isolated rows
        * cols!         function that evaluates (dx, x, p, t, col_idcs) -> F(x,p,t)[:, col_idcs] 
                        and stores the result in dx, i.e., evaluation of isolated columns
        * elements!     function that evaluates (dx, x, p, t, row_idcs, col_idcs) -> F(x,p,t)[row_idcs, col_idcs] 
                        and stores the result in dx, i.e., evaluation of isolated entries of a given index grid
        * n_rows        number of rows of the function output
        * n_cols        number of columns of the function output
"""
struct ComponentFunction{FR,FC,FE}
    rows!::FR
    cols!::FC
    elements!::FE
    n_rows::Int
    n_cols::Int
end

function ComponentFunction(n_rows, n_cols = 1; rows = nothing, cols=nothing, elements=nothing)
    ComponentFunction(rows, cols, elements, n_rows, n_cols)
end

"""
    $(TYPEDEF)

    Discrete Empirical Interpolation Method (DEIM) Interpolators.
"""
struct DEIMInterpolator{T} <: SparseInterpolator 
    interpolation_idcs::Vector{Int}
    weights::Matrix{T}
end

function (Π::DEIMInterpolator)(A::AbstractMatrix)
    return Π.weights * A[Π.interpolation_idcs,:]
end

function (Π::DEIMInterpolator)(A::AbstractVector)
    return Π.weights * A[Π.interpolation_idcs,:]
end

"""
    $(TYPEDEF)

    Lazy adjoint for `DEIMInterpolator`s.
"""
struct AdjointDEIMInterpolator{P <: DEIMInterpolator} <: SparseInterpolator
    parent::P
end

function (Π::AdjointDEIMInterpolator)(A::AbstractMatrix)
    return A[:,Π.parent.interpolation_idcs] * Π.parent.weights'
end

"""
    $(TYPEDSIGNATURES)

    lazy adjoint for `DEIMInterpolator`s. We mean adjoint in the sense that the interpolation is 
    applied to the corange. 
"""
adjoint(Π::DEIMInterpolator) = AdjointDEIMInterpolator(Π)
adjoint(Π::AdjointDEIMInterpolator) = Π.parent

"""
    $(TYPEDEF)

    Sparse interpolator for matrices encoded by interpolators for range and corange.
"""
struct SparseMatrixInterpolator{R <: SparseInterpolator, C <: SparseInterpolator} <: SparseInterpolator 
    range::R
    corange::C
end

function SparseMatrixInterpolator(row_idcs::Vector{Int}, col_idcs::Vector{Int},
                                  range_weights::Matrix{T}, corange_weights::Matrix{T}) where T
    range = DEIMInterpolator(row_idcs, range_weights)
    corange = DEIMInterpolator(col_idcs, corange_weights)
    return SparseMatrixInterpolator(range, corange)
end


function (Π::SparseMatrixInterpolator)(A::AbstractMatrix)
    @unpack range, corange = Π
    return range.weights * A[range.interpolation_idcs,corange.interpolation_idcs] * corange.weights'
end

"""
    $(TYPEDEF)

    sparse interpolation for vector-valued function of the form `F!(dx,x,p,t)`. 
    All computation are assumed to be in-place. 
        
    #Fields 
        * `F`               `ComponentFunction` to be sparsely interpolated 
        * `interpolator`    `SparseInterpolator` that carries information for how to interpolate 
                            rows and columns (specific type determines how and which interpolation is performed)
        * `cache`           cache for auxiliary memory used for evaluation of rows!
    
    If interpolator is a `DEIMInterpolator` then it is assumed that this is intended to interpolate only for the range
    If interpolator is a `AdjointDEIMInterpolator` then it is assumed that this intended to interpolate only for the corange
    If interpolator is a `SparseMatrixInterpolator` then both range and corange are interpolated
"""
mutable struct SparseFunctionInterpolator{IP <: SparseInterpolator, fType <: ComponentFunction, T} <: SparseInterpolator
    F::fType
    interpolator::IP
    cache::InterpolatorCache{T}
end

function SparseFunctionInterpolator(F, interpolator::DEIMInterpolator;
                                    output::Type = Float64)
    r_rows = rank(interpolator)
    cache = InterpolatorCache(Matrix{output}(undef, r_rows, F.n_cols),
                              Matrix{output}(undef, 0, 0),
                              Matrix{output}(undef, 0, 0))
    return SparseFunctionInterpolator(F, interpolator, cache)
end

function SparseFunctionInterpolator(F, interpolator::AdjointDEIMInterpolator;
                                    output::Type = Float64)
    r_cols = rank(interpolator)
    cache = InterpolatorCache(Matrix{output}(undef, 0, 0),
                              Matrix{output}(undef, F.n_rows, r_cols),
                              Matrix{output}(undef, 0, 0))
    return SparseFunctionInterpolator(F, interpolator, cache)
end

function SparseFunctionInterpolator(F, interpolator::SparseMatrixInterpolator;
                                    output::Type = Float64)
    r_rows, r_cols = rank(interpolator.range), rank(interpolator.corange)
    dim_range = size(interpolator.range.weights, 1)
    dim_corange = size(interpolator.corange.weights, 1)
    cache = InterpolatorCache(Matrix{output}(undef, r_rows, dim_corange),
    Matrix{output}(undef, dim_range, r_cols),
    Matrix{output}(undef, r_rows, r_cols))
    return SparseFunctionInterpolator(F, interpolator, cache)
end 
"""
    $(TYPEDSIGNATURES)

    helper function querying the interpolation indices.
"""
interpolation_indices(Π::DEIMInterpolator) = Π.interpolation_idcs
interpolation_indices(Π::AdjointDEIMInterpolator) = Π.parent.interpolation_idcs
interpolation_indices(Π::SparseMatrixInterpolator) = (interpolation_indices(Π.range), interpolation_indices(Π.corange))
interpolation_indices(Π::SparseFunctionInterpolator) = interpolation_indices(Π.interpolator)


"""
    $(TYPEDSIGNATURES)

    makes updates to interpolator information in `SparseFunctionInterpolator` and reinitializes any cached memory.
"""
function update_interpolator!(Π::SparseFunctionInterpolator, interpolator::DEIMInterpolator)
    Π.interpolator = interpolator
    r_rows = rank(interpolator)
    T = eltype(Π.cache.cols)
    Π.cache = InterpolatorCache(Matrix{T}(undef, r_rows, Π.F.n_cols),
                                 Matrix{T}(undef, 0, 0),
                                 Matrix{T}(undef, 0, 0))
end

function update_interpolator!(Π::SparseFunctionInterpolator, interpolator::AdjointDEIMInterpolator)
    Π.interpolator = interpolator
    r_cols = rank(interpolator)
    T = eltype(Π.cache.cols)
    Π.cache = InterpolatorCache(Matrix{T}(undef, 0, 0),
                                 Matrix{T}(undef, Π.F.n_rows, r_cols),
                                 Matrix{T}(undef, 0, 0))
end

function update_interpolator!(Π::SparseFunctionInterpolator, interpolator::SparseMatrixInterpolator)
    Π.interpolator = interpolator
    r_rows, r_cols = rank(interpolator.range), rank(interpolator.corange)
    dim_range = size(interpolator.range.weights, 1)
    dim_corange = size(interpolator.corange.weights, 1)
    T = eltype(Π.cache.cols)
    Π.cache = InterpolatorCache(Matrix{T}(undef, r_rows, dim_corange),
                                Matrix{T}(undef, dim_range, r_cols),
                                Matrix{T}(undef, r_rows, r_cols))
end

"""
    $(TYPEDSIGNATURES)

    in-place evaluation of interpolation indices.
"""
function eval!(Π::SparseFunctionInterpolator{IP, fType, T}, x, p, t) where {IP <: DEIMInterpolator, fType, T}
    @unpack cache = Π
    range = Π.interpolator
    Π.F.rows!(cache.rows, x, p, t, range.interpolation_idcs)
end

function eval!(Π::SparseFunctionInterpolator{IP, fType, T}, x, p, t) where {IP <: AdjointDEIMInterpolator, fType, T}
    @unpack cache = Π
    corange = Π.interpolator.parent
    Π.F.cols!(cache.cols, x, p, t, corange.interpolation_idcs)
end

function eval!(Π::SparseFunctionInterpolator{IP, fType, T}, x, p, t) where {IP <: SparseMatrixInterpolator, fType, T}
    @unpack range, corange = Π.interpolator
    @unpack cache = Π
    Π.F.elements!(cache.elements, x, p, t, range.interpolation_idcs, corange.interpolation_idcs)
end

function eval_rows!(dx, Π::SparseFunctionInterpolator, x, p, t, idcs)
    Π.F.rows!(dx, x, p, t, idcs)
end

function eval_cols!(dx, Π::SparseFunctionInterpolator, x, p, t, idcs)
    Π.F.cols!(dx, x, p, t, idcs)
end

"""
    $(TYPEDSIGNATURES)

    functor for inplace evaluation of `SparseFunctionInterpolator`s. 
"""
function (Π::SparseFunctionInterpolator{IP, fType, T})(dx, x, p, t) where {IP <: DEIMInterpolator, fType, T}
    eval!(Π, x, p, t)
    mul!(dx, Π.interpolator.weights, Π.cache.rows)
end

function (Π::SparseFunctionInterpolator{IP, fType, T})(dx, x, p, t) where {IP <: AdjointDEIMInterpolator, fType, T}
    eval!(Π, x, p, t)
    mul!(dx, Π.cache.cols, Π.interpolator.parent.weights')
end

function (Π::SparseFunctionInterpolator{IP, fType, T})(dx, x, p, t) where {IP <: SparseMatrixInterpolator, fType, T}
    eval!(Π, x, p, t)
    mul!(Π.cache.rows, Π.cache.elements, Π.interpolator.corange.weights')
    mul!(dx, Π.interpolator.range.weights, Π.cache.rows)
end


"""
    $(TYPEDSIGNATURES)

    returns rank of sparse interpolators.
"""
rank(Π::DEIMInterpolator) = length(Π.interpolation_idcs)
rank(Π::AdjointDEIMInterpolator) = rank(Π.parent)
rank(Π::SparseMatrixInterpolator) = min(rank(Π.range), rank(Π.corange))
rank(Π::SparseFunctionInterpolator) = rank(Π.interpolator)

"""
    $(TYPEDSIGNATURES)

    returns dimension of sparse interpolators.
"""
dim(Π::DEIMInterpolator) = rank(Π)
dim(Π::AdjointDEIMInterpolator) = rank(Π.parent)
dim(Π::SparseMatrixInterpolator) = (rank(Π.range), rank(Π.corange))
dim(Π::SparseFunctionInterpolator) = dim(Π.interpolator)

"""
    $(TYPEDSIGNATURES)

    returns size of interpolations.
"""
size(Π::DEIMInterpolator) = size(Π.weights,1)
size(Π::AdjointDEIMInterpolator) = size(Π.parent)
size(Π::SparseMatrixInterpolator) = (size(Π.range.weights,1), size(Π.corange.weights,1))
size(Π::SparseFunctionInterpolator) = size(Π.interpolator)

"""
    $(TYPEDEF)

    abstract type characterizing index selection algorithms for sparse interpolation.
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

function DEIM( ;tol = eps(Float64), rmin = 1, rmax = 2^62, elasticity = 0.1)
    return DEIM(tol, elasticity, rmin, rmax)
end

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

function QDEIM( ;tol = eps(Float64), rmin = 1, rmax = 2^62, elasticity = 0.1)
    return QDEIM(tol, elasticity, rmin, rmax)
end

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

function LDEIM( ;tol = eps(Float64), rmin = 1, rmax = 2^62, elasticity = 0.1)
    return LDEIM(tol, elasticity, rmin, rmax)
end

"""
    $(TYPEDSIGNATURES)

    returns interpolation indices for sparse interpolation. Supports DEIM and QDEIM index selection.
"""
function index_selection(U,::DEIM)
    m = size(U, 2)
    indices = Vector{Int}(undef, m)
    @inbounds @views begin
        r = abs.(U[:, 1])
        indices[1] = argmax(r)
        for l in 2:m
            U = U[:, 1:(l - 1)]
            P = indices[1:(l - 1)]
            PᵀU = U[P, :]
            uₗ = U[:, l]
            Pᵀuₗ = uₗ[P, :]
            c = vec(PᵀU \ Pᵀuₗ)
            mul!(r, U, c)
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

function index_selection(U::Matrix, r::Int, ::LDEIM)   
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

function index_selection(U::Matrix, ::LDEIM)
    return index_selection(U, DEIM())
end

function index_selection(U::Matrix, S::Vector, alg::IndexSelectionAlgorithm)
    @assert size(U,2) == length(S) "Number of columns provided must coincide with number of singular values."

    cutoff = length(S)
    for σ in reverse(S)
        if σ > alg.tol
            break
        end
        cutoff -= 1
    end
    r = max(alg.rmin, min(alg.rmax, cutoff))
    return index_selection(U[:,1:r], alg)
end

function index_selection(U::Matrix, S::Vector, alg::LDEIM)
    @assert size(U,2) == length(S) "Number of columns provided must coincide with number of singular values."

    if S[end] > alg.tol 
        r = min(alg.rmax, length(S)+1)
        return index_selection(U, r, alg)
    else
        cutoff = findlast(x -> x > alg.tol*alg.elasticity, S)
        if isnothing(cutoff)
            cutoff = 1
        end
        return index_selection(U, cutoff, alg)
    end
end


"""
    $(TYPEDEF)

    type for carrying all necessary information for continuous sparse interpolation 
    to be applied for dynamical low rank approximation.
"""
struct SparseInterpolation
    selection_alg
    update_scheme
    tol
    rmin
    rmax
    init_range
    init_corange
end

function SparseInterpolation(selection_alg, init_range, init_corange; update_scheme = :last_iterate,
                             rmin = 1, rmax = min(size(init_range,1), size(init_corange,1)), tol = eps(Float64))
    return SparseInterpolation(selection_alg, update_scheme, tol, rmin, rmax, init_range, init_corange)
end
using DocStringExtensions, UnPack, LinearAlgebra

import LinearAlgebra.adjoint
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

    makes updates to interpolator information in `SparseFunctionInterpolator` and reinitializes any cached memory.
"""
function update_interpolator!(Π::SparseFunctionInterpolator, interpolator::DEIMInterpolator)
    Π.interpolator = interpolator
    r_rows = rank(interpolator)
    T = eltype(Π.cache.cols)
    Π.cache = InterpolationCache(Matrix{T}(undef, r_rows, F.n_cols),
                                 Matrix{T}(undef, 0, 0),
                                 Matrix{T}(undef, 0, 0))
end

function update_interpolator!(Π::SparseFunctionInterpolator, interpolator::AdjointDEIMInterpolator)
    Π.interpolator = interpolator
    r_cols = rank(interpolator)
    T = eltype(Π.cache.cols)
    Π.cache = InterpolationCache(Matrix{T}(undef, 0, 0),
                                 Matrix{T}(undef, F.n_rows, r_cols),
                                 Matrix{T}(undef, 0, 0))
end

function update_interpolator!(Π::SparseFunctionInterpolator, interpolator::SparseMatrixInterpolator)
    Π.interpolator = interpolator
    r_rows, r_cols = rank(interpolator.range), rank(interpolator.corange)
    dim_range = size(interpolator.range.interpolator, 1)
    dim_corange = size(interpolator.corange.weights, 1)
    T = eltype(Π.cache.cols)
    Π.cache = InterpolationCache(Matrix{T}(undef, r_rows, dim_corange),
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
struct DEIM end

"""
    $(TYPEDEF)

    QDEIM index selection algorithm.
"""
struct QDEIM end

"""
    $(TYPEDSIGNATURES)

    returns interpolation indices for sparse interpolation. Supports DEIM and QDEIM index selection.
"""
function index_selection(U::Matrix,::DEIM)
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

function index_selection(U::Matrix,::QDEIM)
    QR = qr(U', ColumnNorm())
    return QR.p[1:size(U,2)]
end

# a few lil tests
#=
function rows!(dx,x,p,t,idcs)
    @views dx .= x[idcs, :]
end
function cols!(dx,x,p,t,idcs)
    @views dx .= x[:, idcs]
end
function elements!(dx,x,p,t,idcsA, idcsB)
    @views dx .= x[idcsA, idcsB]
end

F = ComponentFunction(rows!, cols!, elements!, 5, 8)
A = Matrix(reshape(1.0:5*8, (5, 8)))

row_idcs = [1, 5, 2]
range_weights = zeros(eltype(A), 5, 3)
range_weights[row_idcs,1:3] .= Matrix{eltype(A)}(I,3,3)

col_idcs = [1, 8, 3, 6]
corange_weights = zeros(eltype(A),8, 4)
corange_weights[col_idcs,1:4] .= Matrix{eltype(A)}(I,4,4)

Π_range = DEIMInterpolator(row_idcs,range_weights)
Π_corange = DEIMInterpolator(col_idcs,corange_weights)
Π_mat = SparseMatrixInterpolator(Π_range, Π_corange)

test = all(Π_range(A)[row_idcs,:] .== A[row_idcs,:])
test = all(Π_corange'(A)[:,col_idcs] .== A[:,col_idcs])

F_range_int = SparseFunctionInterpolator(F, Π_range, output = eltype(A))
F_corange_int = SparseFunctionInterpolator(F, Π_corange', output = eltype(A))
F_int = SparseFunctionInterpolator(F, Π_mat, output = eltype(A))

eval!(F_range_int, A, (), 0.0)
test = all(F_range_int.cache.rows .== A[row_idcs,:])
eval!(F_corange_int, A, (), 0.0)
test = all(F_corange_int.cache.cols .== A[:,col_idcs])
eval!(F_int, A, (), 0.0)
test = all(F_int.cache.elements .== A[row_idcs,col_idcs])

dA = similar(A)
F_range_int(dA, A, (), 0.0)
test = all(dA[row_idcs, :] .== A[row_idcs, :])
F_corange_int(dA, A, (), 0.0)
test = all(dA[:, col_idcs] .== A[:, col_idcs])
F_int(dA, A, (), 0.0)
test = all(dA[row_idcs, col_idcs] .== A[row_idcs, col_idcs])

using BenchmarkTools
# non-allocating!
parms = ()
t = 0.0
@btime $F_range_int($dA, $A, $params, $t)
@btime $F_corange_int($dA, $A, $params, $t)
@btime $F_int($dA, $A, $params, $t)
=#
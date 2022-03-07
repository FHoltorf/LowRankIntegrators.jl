using Combinatorics
import Base: +, -, *, size, Matrix, getindex, hcat, vcat, axes, broadcasted, BroadcastStyle
import LinearAlgebra: rank, adjoint, svd

abstract type AbstractLowRankApproximation end

"""
    X = U*S*V'

    SVD like factorization of a low rank matrix into factors U ∈ ℝⁿˣʳ, S ∈ ℝʳˣʳ, V ∈ ℝᵐˣʳ. 
    The columns of U span the range of X, the colomuns of V span the co-range of X and S describes the map between
    co-range and range. The factorization is non-unique and U and V need not be orthogonal! 
    However, often U and V are chosen orthogonal to allow for cheap (pseudo)inversion. In that case, note that
    orthogonality of U and V is not guaranteed to and in fact will rarely be preserved under the operations that
    are supported (multiplication, addition, etc.)
"""
mutable struct SVDLikeApproximation{uType, sType, vType} <: AbstractLowRankApproximation
    U::uType
    S::sType
    V::vType
end 

"""
    X = U*Z'

    Two factor factorization of a low rank matrix into factors U ∈ ℝⁿˣʳ and Z ∈ ℝᵐˣʳ. 
    U should span the range of X while Z spans the co-range. The factorization is non-unique and
    U need not be orthogonal! However, U is often chosen orthogonal for cheap (pseudo)inversion. In that case,
    note that orthogonality of U is not guaranteed to and in fact will rarely be preserved under the operations that
    are supported (multiplication, addition, etc.). In order to reorthonormalize U, simply call `orthonormalize!(X, alg)`
    where alg refers to the algorithm used to compute the orthonormalization: 
        * GradientDescent() 
        * QR()
        * SVD()
"""
mutable struct TwoFactorApproximation{uType, zType} <: AbstractLowRankApproximation
    U::uType
    Z::zType                              
end

rank(LRA::SVDLikeApproximation) = size(LRA.S,1)
size(LRA::SVDLikeApproximation) = (size(LRA.U,1), size(LRA.V,1))
size(LRA::SVDLikeApproximation, ::Val{1}) = size(LRA.U,1)
size(LRA::SVDLikeApproximation, ::Val{2}) = size(LRA.V,1)
size(LRA::SVDLikeApproximation, i::Int) = size(LRA, Val(i))
Matrix(LRA::SVDLikeApproximation) = LRA.U*LRA.S*LRA.V'
getindex(LRA::SVDLikeApproximation, i::Int, j::Int) = sum(LRA.U[i,k]*sum(LRA.S[k,s]*LRA.V[j,s] for s in 1:rank(LRA)) for k in 1:rank(LRA)) # good enough for now
getindex(LRA::SVDLikeApproximation, i, j::Int) = SVDLikeApproximation(LRA.U[i,:], LRA.S, LRA.V[[j],:])  # good enough for now
getindex(LRA::SVDLikeApproximation, i::Int, j) = SVDLikeApproximation(LRA.U[[i],:], LRA.S, LRA.V[j,:])  # good enough for now
getindex(LRA::SVDLikeApproximation, i, j) = SVDLikeApproximation(LRA.U[i,:], LRA.S, LRA.V[j,:])  # good enough for now
getindex(LRA::SVDLikeApproximation, ::Colon, j::AbstractVector) = SVDLikeApproximation(LRA.U, LRA.S, LRA.V[j,:])  # good enough for now
getindex(LRA::SVDLikeApproximation, ::Colon, j::Int) = SVDLikeApproximation(LRA.U, LRA.S, LRA.V[[j],:])  # good enough for now
getindex(LRA::SVDLikeApproximation, i::AbstractVector, ::Colon) = SVDLikeApproximation(LRA.U[i,:], LRA.S, LRA.V)  # good enough for now
getindex(LRA::SVDLikeApproximation, i::Int, ::Colon) = SVDLikeApproximation(LRA.U[[i],:], LRA.S, LRA.V)  # good enough for now

hcat(A::SVDLikeApproximation, B::SVDLikeApproximation) = SVDLikeApproximation(hcat(A.U, B.U), blockdiagonal(A.S, B.S), blockdiagonal(A.V, B.V))
vcat(A::SVDLikeApproximation, B::SVDLikeApproximation) = SVDLikeApproximation(blockdiagonal(A.U, B.U), blockdiagonal(A.S, B.S), hcat(A.V, B.V))

rank(LRA::TwoFactorApproximation) = size(LRA.U, 2)

size(LRA::TwoFactorApproximation) = (size(LRA.U,1), size(LRA.Z,1))
size(LRA::TwoFactorApproximation, ::Val{1}) = size(LRA.U,1)
size(LRA::TwoFactorApproximation, ::Val{2}) = size(LRA.Z,1)
size(LRA::TwoFactorApproximation, i::Int) = size(LRA, Val(i))

axes(LRA::AbstractLowRankApproximation) = map(Base.oneto, size(LRA))

Matrix(LRA::TwoFactorApproximation) = LRA.U*LRA.Z'

getindex(LRA::TwoFactorApproximation, i::Int, j::Int) = sum(LRA.U[i,k]*LRA.Z[j,k] for k in 1:rank(LRA)) # good enough for now
getindex(LRA::TwoFactorApproximation, i, j::Int) = TwoFactorApproximation(LRA.U[i,:], LRA.Z[[j],:])
getindex(LRA::TwoFactorApproximation, i::Int, j) = TwoFactorApproximation(LRA.U[[i],:], LRA.Z[j,:])
getindex(LRA::TwoFactorApproximation, i, j) = TwoFactorApproximation(LRA.U[i,:], LRA.Z[j,:])
getindex(LRA::TwoFactorApproximation, ::Colon, j::AbstractVector) = TwoFactorApproximation(LRA.U, LRA.Z[j,:])
getindex(LRA::TwoFactorApproximation, i::AbstractVector, ::Colon) = TwoFactorApproximation(LRA.U[i,:], LRA.Z)
getindex(LRA::TwoFactorApproximation, ::Colon, j::Int) = TwoFactorApproximation(LRA.U, LRA.Z[[j],:])
getindex(LRA::TwoFactorApproximation, i::Int, ::Colon) = TwoFactorApproximation(LRA.U[[i],:], LRA.Z)

hcat(A::TwoFactorApproximation, B::TwoFactorApproximation) = TwoFactorApproximation(hcat(A.U, B.U), blockdiagonal(A.Z, B.Z))
vcat(A::TwoFactorApproximation, B::TwoFactorApproximation) = TwoFactorApproximation(blockdiagonal(A.U, B.U), hcat(A.Z, B.Z))

# simple support of adjoints, probably not ideal though
adjoint(LRA::TwoFactorApproximation) = TwoFactorApproximation(conj(LRA.Z),conj(LRA.U)) 
adjoint(LRA::SVDLikeApproximation) = TwoFactorApproximation(conj(LRA.V),LRA.S',conj(LRA.U)) 

# Is the following alternative better?
# *(A::SVDLikeApproximation, B::SVDLikeApproximation) = SVDLikeApproximation(A.U, A.S*(A.V'*B.U)*B.S, B.V)
# it would preserve orthonormality of range/co-range factors but make core rectangular and increase the storage cost unnecessarily.

# converting between both representations
TwoFactorApproximation(A::SVDLikeApproximation) = TwoFactorApproximation(A.U, A.V*A.S')
function SVDLikeApproximation(A::TwoFactorApproximation) 
    U, S, V = svd(A.Z)
    return SVDLikeApproximation(A.U*V, S', U)
end

## Multiplication
*(A::AbstractMatrix, B::SVDLikeApproximation) = SVDLikeApproximation(A*B.U, B.S, B.V)
*(A::SVDLikeApproximation, B::AbstractMatrix) = SVDLikeApproximation(A.U, A.S, B'*A.V)
*(A::SVDLikeApproximation, ::UniformScaling) = A
*(::UniformScaling, A::SVDLikeApproximation) = A
function *(A::SVDLikeApproximation, B::SVDLikeApproximation)
    if rank(A) ≤ rank(B)
        return SVDLikeApproximation(A.U, A.S, B.V*B.S'*(B.U'*A.V))
    else
        return SVDLikeApproximation(A.U*A.S*(A.V'*B.U), B.S, B.V)
    end
end
*(A::SVDLikeApproximation, α::Number) = SVDLikeApproximation(A.U, α*A.S, A.V) 
*(α::Number, A::SVDLikeApproximation) = SVDLikeApproximation(A.U, α*A.S, A.V)
*(A::SVDLikeApproximation, v::AbstractVector) = A.U*(A.S*(A.V'*v))
*(v::AbstractVector, A::SVDLikeApproximation) = ((v*A.U)*A.S)*A.V'

*(A::AbstractMatrix, B::TwoFactorApproximation) = TwoFactorApproximation(A*B.U, B.Z)
*(A::TwoFactorApproximation, B::AbstractMatrix) = TwoFactorApproximation(A.U, B'*A.Z)
*(A::TwoFactorApproximation, ::UniformScaling) = A
*(::UniformScaling, A::TwoFactorApproximation) = A
function *(A::TwoFactorApproximation,B::TwoFactorApproximation)
    if rank(A) ≤ rank(B) # minimize upper bound on rank
        return TwoFactorApproximation(A.U, B.Z*(B.U'*A.Z))
    else
        return TwoFactorApproximation(A.U*(A.Z'*B.U), B.Z)
    end
end
*(A::TwoFactorApproximation, α::Number) = TwoFactorApproximation(A.U, α*A.Z) 
*(α::Number, A::TwoFactorApproximation) = TwoFactorApproximation(A.U, α*A.Z)
*(A::TwoFactorApproximation, v::AbstractVector) = A.U*(A.Z'*v)
*(v::AbstractVector, A::TwoFactorApproximation) = (v*A.U)*A.Z'

# default to TwoFactorApproximation
*(A::TwoFactorApproximation, B::SVDLikeApproximation) = A*TwoFactorApproximation(B)
*(A::SVDLikeApproximation, B::TwoFactorApproximation) = TwoFactorApproximation(B)*A

## Addition
# just for convenience:
function blockdiagonal(A::T1, B::T2) where {T1, T2 <: Union{AbstractMatrix, AbstractVector}}
    n1,m1 = size(A,1), size(A,2)
    n2,m2 = size(B,1), size(B,2)
    C = zeros(eltype(A), n1+n2, m1+m2)
    C[1:n1, 1:m1] .= A
    C[n1+1:end, m1+1:end] .= B
    return C
end

+(A::SVDLikeApproximation, B::SVDLikeApproximation) = SVDLikeApproximation(hcat(A.U, B.U), blockdiagonal(A.S, B.S),hcat(A.V, B.V))
+(A::AbstractMatrix, B::SVDLikeApproximation) = A + Matrix(B)
+(A::SVDLikeApproximation, B::AbstractMatrix) = Matrix(A) + B
-(A::SVDLikeApproximation, B::SVDLikeApproximation) = SVDLikeApproximation(hcat(A.U, B.U), blockdiagonal(A.S, -B.S),hcat(A.V, B.V))
-(A::AbstractMatrix, B::SVDLikeApproximation) = A - Matrix(B)
-(A::SVDLikeApproximation, B::AbstractMatrix) = Matrix(A) - B

+(A::TwoFactorApproximation, B::TwoFactorApproximation) = TwoFactorApproximation(hcat(A.U, B.U), hcat(A.Z, B.Z))
+(A::AbstractMatrix, B::TwoFactorApproximation) = A + Matrix(B)
+(A::TwoFactorApproximation, B::AbstractMatrix) = Matrix(A) + B
-(A::TwoFactorApproximation, B::TwoFactorApproximation) = TwoFactorApproximation(hcat(A.U, B.U), hcat(A.Z, -B.Z))
-(A::TwoFactorApproximation, B::AbstractMatrix) = Matrix(A) - B
-(A::AbstractMatrix, B::TwoFactorApproximation) = A - Matrix(B)

# default to TwoFactorApproximation
+(A::TwoFactorApproximation, B::SVDLikeApproximation) = A+TwoFactorApproximation(B)
+(A::SVDLikeApproximation, B::TwoFactorApproximation) = B+A
-(A::TwoFactorApproximation, B::SVDLikeApproximation) = A-TwoFactorApproximation(B)
-(A::SVDLikeApproximation, B::TwoFactorApproximation) = TwoFactorApproximation(A)-B

# elementwise product
function hadamard(A::SVDLikeApproximation, B::SVDLikeApproximation) 
    @assert size(A) == size(B) "elementwise product is only defined between matrices of equal dimension"
    rA, rB = rank(A), rank(B)
    r_new = rA*rB
    U = ones(eltype(A.U), size(A,1), r_new)
    V = ones(eltype(A.V), size(A,2), r_new)
    S = ones(eltype(A.S), r_new, r_new)
    AUcols = [@view A.U[:,i] for i in 1:rA]
    AVcols = [@view A.V[:,i] for i in 1:rA]
    BUcols = [@view B.U[:,i] for i in 1:rB]
    BVcols = [@view B.V[:,i] for i in 1:rB]
    k = 0
    for r1 in 1:rA, r2 in 1:rB
        k += 1
        U[:,k] = AUcols[r1] .* BUcols[r2]
        V[:,k] = AVcols[r1] .* BVcols[r2]
        l = 0
        for k1 in 1:rA, k2 in 1:rB
            l += 1
            S[k,l] = A.S[r1,k1]*B.S[r2,k2]
        end
    end
    return SVDLikeApproximation(U,S,V)
end

function hadamard(A::TwoFactorApproximation, B::TwoFactorApproximation) 
    @assert size(A) == size(B) "elementwise product is only defined between matrices of equal dimension"
    rA, rB = rank(A), rank(B)
    r_new = rA*rB
    U = ones(eltype(A.U), size(A,1), r_new)
    Z = ones(eltype(A.Z), size(A,2), r_new)
    AUcols = [@view A.U[:,i] for i in 1:rA]
    AZcols = [@view A.Z[:,i] for i in 1:rA]
    BUcols = [@view B.U[:,i] for i in 1:rB]
    BZcols = [@view B.Z[:,i] for i in 1:rB]
    k = 0
    for r1 in 1:rA, r2 in 1:rB
        k += 1
        U[:,k] = AUcols[r1] .* BUcols[r2]
        Z[:,k] = AZcols[r1] .* BZcols[r2]
    end
    return TwoFactorApproximation(U,Z)
end

# default to TwoFactorApproximation
hadamard(A::TwoFactorApproximation, B::SVDLikeApproximation) = hadamard(A, TwoFactorApproximation(B))
hadamard(A::SVDLikeApproximation, B::TwoFactorApproximation) = hadamard(TwoFactorApproximation(A), B)

# catch all case
hadamard(A::AbstractLowRankApproximation, B::AbstractMatrix) = Matrix(A) .* B
hadamard(A::AbstractMatrix, B::AbstractLowRankApproximation) = Matrix(A) .* B
function hadamard(A,B)
    return A .* B
end

# elementwise power
function elpow(A::SVDLikeApproximation, d::Int)
    @assert d >= 1 "elementwise power operation 'elpow' only defined for positive powers"
    r = rank(A)
    r_new = binomial(r+d-1, d)

    # the following sequence is not ideal but cheap under the premise of low rank, need to be improved though
    U, S, V = svd(A.S)
    A.U .= A.U*U
    A.V .= A.V*V 
    A.S .= Matrix(Diagonal(S))

    Ucols = [@view A.U[:,i] for i in 1:r]
    Vcols = [@view A.V[:,i] for i in 1:r]
    U = ones(eltype(A.U), size(A,1), r_new)
    V = ones(eltype(A.V), size(A,2), r_new)
    S = zeros(eltype(A.S), r_new, r_new)
    k = 0 
    multi_exps = multiexponents(r,d) 
    for exps in multi_exps
        k += 1
        S[k,k] += multinomial(exps...)
        for j in 1:r
            if exps[j] != 0
                U[:, k] .*= Ucols[j].^exps[j]
                V[:, k] .*= Vcols[j].^exps[j]
                S[k,k] *= A.S[j,j]^exps[j]
            end
        end
    end
    return SVDLikeApproximation(U,S,V)
end

function elpow(A::TwoFactorApproximation, d::Int)
    @assert d >= 1 "elementwise power operation 'elpow' only defined for positive powers"
    r = rank(A)
    r_new = binomial(r+d-1, d)
    Ucols = [@view A.U[:,i] for i in 1:r]
    Zcols = [@view A.Z[:,i] for i in 1:r]
    U = ones(eltype(A.U), size(A,1), r_new)
    Z = ones(eltype(A.Z), size(A,2), r_new)
    k = 0 
    for exps in multiexponents(r,d)
        k += 1
        Z[:, k] .*= multinomial(exps...) 
        for j in 1:r
            if exps[j] > 0
                U[:, k] .*= Ucols[j].^exps[j]
                Z[:, k] .*= Zcols[j].^exps[j]
            end
        end
    end
    return TwoFactorApproximation(U, Z)
end

function elpow(A,d::Int)
    return A.^d
end

# more exotic operations
function add_to_cols(LRA::TwoFactorApproximation, v::AbstractVector)
    return LRA + TwoFactorApproximation(v, ones(eltype(v), size(LRA, 2)))
end

function multiply_cols(LRA::TwoFactorApproximation, v::AbstractVector)
    return hadamard(LRA, TwoFactorApproximation(v, ones(eltype(v), size(LRA, 2))))
end

function add_to_cols(LRA::SVDLikeApproximation, v::AbstractVector)
    return LRA + SVDLikeApproximation(v, ones(eltype(v), 1, 1), ones(eltype(v), size(LRA, 2)))
end

function multiply_cols(LRA::SVDLikeApproximation, v::AbstractVector)
    return hadamard(LRA, SVDLikeApproximation(v, ones(eltype(v), 1, 1), ones(eltype(v), size(LRA, 2))))
end

function add_to_cols(A, v::AbstractVector)
    return A .+ v
end

function multiply_cols(A, v::AbstractVector)
    return p .* A
end

function add_to_rows(LRA::TwoFactorApproximation, v::AbstractVector)
    return LRA + TwoFactorApproximation(ones(eltype(v), size(LRA, 1)), v)
end

function multiply_rows(LRA::TwoFactorApproximation, v::AbstractVector)
    return hadamard(LRA, TwoFactorApproximation(ones(eltype(v), size(LRA, 1)), v))
end

function add_to_rows(LRA::SVDLikeApproximation, v::AbstractVector)
    return LRA + SVDLikeApproximation(ones(eltype(v), size(LRA, 1)), ones(eltype(v), 1, 1), v)
end

function multiply_rows(LRA::SVDLikeApproximation, v::AbstractVector)
    return hadamard(LRA, SVDLikeApproximation(ones(eltype(v), size(LRA, 1)), ones(eltype(v), 1, 1), v))
end

function add_to_rows(A, v)
    return A .+ v'
end

function multiply_rows(A, v)
    return A .* v'
end

function multiply_rows(A, v::Number)
    return v*A 
end

function add_scalar(LRA::SVDLikeApproximation, α::Number)
    return LRA + SVDLikeApproximation(ones(eltype(α), size(LRA, 1)), [α], ones(eltype(α), size(LRA,2)))
end

function add_scalar(LRA::TwoFactorApproximation, α::Number)
    return LRA + TwoFactorApproximation(ones(eltype(α), size(LRA, 1)), α*ones(eltype(α), size(LRA,2)))
end

function add_scalar(A, α::Number)
    return A .+ α
end

# support broadcasting operations
broadcasted(::typeof(*), A::AbstractLowRankApproximation, B::AbstractLowRankApproximation) = hadamard(A,B)
broadcasted(::typeof(*), A::AbstractLowRankApproximation, b::AbstractVector) = multiply_cols(A,b)
broadcasted(::typeof(*), A::AbstractLowRankApproximation, b::Adjoint{<:Number, <:AbstractVector}) = multiply_rows(A, transpose(b))
broadcasted(::typeof(*), A::AbstractLowRankApproximation, b::Transpose{<:Number, <:AbstractVector}) = multiply_rows(A, transpose(b))
broadcasted(::typeof(*), A::AbstractLowRankApproximation, B::AbstractMatrix) = Matrix(A) .* B
broadcasted(::typeof(*), A::AbstractMatrix, B::AbstractLowRankApproximation) = A .* Matrix(B)

broadcasted(::typeof(+), A::AbstractLowRankApproximation, α::Number) = add_scalar(A, α)
broadcasted(::typeof(+), α::Number, A::AbstractLowRankApproximation) = add_scalar(A, α)
broadcasted(::typeof(+), A::AbstractLowRankApproximation, b::AbstractVector) = add_to_cols(A, b)
broadcasted(::typeof(+), A::AbstractLowRankApproximation, b::Adjoint{<:Number, <:AbstractVector}) = add_to_rows(A, transpose(b))
broadcasted(::typeof(+), A::AbstractLowRankApproximation, b::Transpose{<:Number, <:AbstractVector}) = add_to_rows(A, transpose(b))

broadcasted(::typeof(+), b::AbstractVector, A::AbstractLowRankApproximation) = add_to_cols(A, b)
broadcasted(::typeof(+), b::Adjoint{<:Number, <:AbstractVector}, A::AbstractLowRankApproximation) = add_to_rows(A, transpose(b))
broadcasted(::typeof(+), b::Transpose{<:Number, <:AbstractVector}, A::AbstractLowRankApproximation) = add_to_rows(A, transpose(b))

broadcasted(::typeof(^), A::AbstractLowRankApproximation, d::Int) = elpow(A, d)
broadcasted(::typeof(Base.literal_pow), ::typeof(^), A::AbstractLowRankApproximation, ::Val{d}) where d = elpow(A, d)

# rounding 
function svd(A::SVDLikeApproximation, alg=QR())
    orthonormalize!(A, alg)
    U_, S_, V_ = svd(A.S)
    return SVDLikeApproximation(A.U*U_, S_, A.V*V_)
end

function svd(A::TwoFactorApproximation, alg=QR())
    orthonormalize!(A, alg)
    U_, S_, V_ = svd(A.Z)
    return SVDLikeApproximation(A.U*U_, S_, V_)
end

#=
function truncated_svd(A::SVDLikeApproximation, r, alg=QR())
    orthonormalize!(A, alg)
    tsvd = truncated_svd(A.S, r)
    return SVDLikeApproximation(A.U*tsvd, tsvd.S, A.V*V_)
end

function svd(A::TwoFactorApproximation, r, alg=QR())
    orthonormalize!(A, alg)
     = truncated_svd(A.Z, r)
    return SVDLikeApproximation(A.U*U_, S_, V_)
end
=#
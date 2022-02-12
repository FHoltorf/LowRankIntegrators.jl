using Combinatorics
import Base: +, -, *, size, Matrix, getindex
import LinearAlgebra: rank

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

rank(LRA::TwoFactorApproximation) = size(LRA.U, 2)
size(LRA::TwoFactorApproximation) = (size(LRA.U,1), size(LRA.Z,1))
size(LRA::TwoFactorApproximation, ::Val{1}) = size(LRA.U,1)
size(LRA::TwoFactorApproximation, ::Val{2}) = size(LRA.Z,1)
size(LRA::TwoFactorApproximation, i::Int) = size(LRA, Val(i))
Matrix(LRA::TwoFactorApproximation) = LRA.U*LRA.Z'
getindex(LRA::TwoFactorApproximation, i::Int, j::Int) = sum(LRA.U[i,k]*LRA.Z[j,k] for k in 1:rank(LRA)) # good enough for now

# Is the following alternative better?
# *(A::SVDLikeApproximation, B::SVDLikeApproximation) = SVDLikeApproximation(A.U, A.S*(A.V'*B.U)*B.S, B.V)
# it would preserve orthonormality of range/co-range factors but make core rectangular and increase the storage cost unnecessarily.

## Multiplication
*(A::AbstractMatrix, B::SVDLikeApproximation) = SVDLikeApproximation(A*A.U, B.S, B.V)
*(A::SVDLikeApproximation, B::AbstractMatrix) = SVDLikeApproximation(A*A.U, B.S, B.V)
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

## Addition
# just for convenience:
function blockdiagonal(A::AbstractMatrix, B::AbstractMatrix) 
    n1,m1 = size(A)
    n2,m2 = size(B)
    C = zeros(eltype(A), n1+n2,m1+m2)
    C[1:n1, 1:m2] .= A
    C[n1+1:end, m2+1:end] .= B
    return C
end

+(A::SVDLikeApproximation, B::SVDLikeApproximation) = SVDLikeApproximation(hcat(A.U, B.U), blockdiagonal(A.S, B.S),hcat(A.V, B.V))
+(A::AbstractMatrix, B::SVDLikeApproximation) = A + Matrix(B)
+(A::SVDLikeApproximation, B::AbstractMatrix) = Matrix(A) + B
-(A::SVDLikeApproximation, B::SVDLikeApproximation) = SVDLikeApproximation(hcat(A.U, B.U), Diagonal(A.S, -B.S),hcat(A.V, B.V))
-(A::AbstractMatrix, B::SVDLikeApproximation) = A - Matrix(B)
-(A::SVDLikeApproximation, B::AbstractMatrix) = Matrix(A) - B

+(A::TwoFactorApproximation, B::TwoFactorApproximation) = TwoFactorApproximation(hcat(A.U, B.U), hcat(A.Z, B.Z))
+(A::AbstractMatrix, B::TwoFactorApproximation) = A + Matrix(B)
+(A::TwoFactorApproximation, B::AbstractMatrix) = Matrix(A) + B
-(A::TwoFactorApproximation, B::TwoFactorApproximation) = TwoFactorApproximation(hcat(A.U, B.U), hcat(A.Z, -B.Z))
-(A::TwoFactorApproximation, B::AbstractMatrix) = Matrix(A) - B
-(B::AbstractMatrix, A::TwoFactorApproximation) = A - Matrix(B)

# elementwise product
function elprod(A::SVDLikeApproximation, B::SVDLikeApproximation) 
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

function elprod(A::TwoFactorApproximation, B::TwoFactorApproximation) 
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
# catch all case
function elprod(A,B)
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

function elpower(A,d::Int)
    return A.^d
end

# more exotic operations
function add_to_cols(LRA::TwoFactorApproximation, v::AbstractVector)
    return LRA + TwoFactorApproximation(v, ones(eltype(v), size(LRA, 2)))
end

function add_to_cols(LRA::SVDLikeApproximation, v::AbstractVector)
    return LRA + SVDLikeApproximation(v, ones(eltype(v), 1, 1), ones(eltype(v), size(LRA, 2)))
end

function add_to_cols(A, v::AbstractVector)
    return A .+ v
end

function add_to_rows(LRA::TwoFactorApproximation, v::AbstractVector)
    return LRA + TwoFactorApproximation(ones(eltype(v), size(LRA, 1)), v)
end

function add_to_rows(LRA::SVDLikeApproximation, v::AbstractVector)
    return LRA + SVDLikeApproximation(ones(eltype(v), size(LRA, 1)), ones(eltype(v), 1), v)
end

function add_to_rows(A, v::AbstractVector)
    return A .+ v'
end

function add_scalar(LRA::SVDLikeApproximation, α::Number)
    return LRA + SVDLikeApproximation(ones(eltype(α), size(LRA, 1)), [α], ones(eltype(α), size(LRA,2)))
end

function add_scalar(LRA::TwoFactorApproximation, α::Number)
    return LRA + LowRankApproximation(ones(eltype(α), size(LRA, 1)), α*ones(eltype(α), size(LRA,2)))
end

function add_scalar(A, α::Number)
    return A .+ α
end
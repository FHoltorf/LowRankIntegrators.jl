using Combinatorics
import Base: +, -, *, size, Matrix
import LinearAlgebra: rank

rank(LRA::SVDLikeApproximation) = size(LRA.S, 2)ß
Matrix(LRA::SVDLikeApproximation) = LRA.U*LRA.S*LRA.V'
size(LRA::SVDLikeApproximation) = (size(LRA.U,1), size(LRA.V,1))
size(LRA::SVDLikeApproximation, ::Val{1}) = size(LRA.U,1)
size(LRA::SVDLikeApproximation, ::Val{2}) = size(LRA.V,1)
size(LRA::SVDLikeApproximation, i::Int) = size(LRA, Val(i))

rank(LRA::LowRankApproximation) = size(LRA.U, 2)
Matrix(LRA::LowRankApproximation) = LRA.U*LRA.Z'
size(LRA::LowRankApproximation) = (size(LRA.U,1), size(LRA.Z,1))
size(LRA::LowRankApproximation, ::Val{1}) = size(LRA.U,1)
size(LRA::LowRankApproximation, ::Val{2}) = size(LRA.Z,1)
size(LRA::LowRankApproximation, i::Int) = size(LRA, Val(i))

*(A::Matrix, B::LowRankApproximation) = LowRankApproximation(A*B.U, B.Z)
*(A::LowRankApproximation, B::Matrix) = LowRankApproximation(A.U, B'*A.Z)
function *(A::LowRankApproximation,B::LowRankApproximation)
    if rank(A) ≤ rank(B) # minimize upper bound on rank
        return LowRankApproximation(A.U, B.Z*(B.U'*A.Z))
    else
        return LowRankApproximation(A.U*(A.Z'*B.U), B.Z)
    end
end
*(A::LowRankApproximation, α::Number) = LowRankApproximation(A.U, α*A.Z) 
*(α::Number, A::LowRankApproximation) = LowRankApproximation(A.U, α*A.Z)

+(A::LowRankApproximation, B::LowRankApproximation) = LowRankApproximation(hcat(A.U, B.U), hcat(A.Z, B.Z))
+(A::Matrix, B::LowRankApproximation) = A + full(B)
+(A::LowRankApproximation, B::Matrix) = full(A) + B

-(A::LowRankApproximation, B::LowRankApproximation) = LowRankApproximation(hcat(A.U, B.U), hcat(A.Z, -B.Z))

function elprod(A::LowRankApproximation, B::LowRankApproximation) 
    @assert size(A) == size(B) "elementwise product only defined between matrices of equal dimension"
    rA, rB = rank(A), rank(B)
    r_new = rA*rB
    U = ones(size(A,1), r_new)
    Z = ones(size(A,2), r_new)
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
    return LowRankApproximation(U,Z)
end

function elprod(A,B)
    return A .* B
end

function elpow(A::LowRankApproximation, d::Int)
    @assert d >= 1 "elementwise power operation 'elpow' only defined for positive powers"
    r = rank(A)
    r_new = binomial(r+d-1, d)
    Ucols = [@view A.U[:,i] for i in 1:r]
    Zcols = [@view A.Z[:,i] for i in 1:r]
    U = ones(size(A,1), r_new)
    Z = ones(size(A,2), r_new)
    k = 0 
    for exps in multiexponents(r,d)
        k += 1
        Z[:, k] .*= multinomial(exps...) 
        for j in 1:r
            U[:, k] .*= Ucols[j].^exps[j]
            Z[:, k] .*= Zcols[j].^exps[j]
        end
    end
    return LowRankApproximation(U, Z)
end

function elpower(A,d::Int)
    return A.^d
end
@concrete struct SVDRetraction <: AbstractLowRankRetraction
    sparse_approximator
end

@concrete struct LieTrotterRetraction <: AbstractLowRankRetraction
    sparse_approximator 
    order
end

@concrete struct StrangRetraction <: AbstractLowRankRetraction
    sparse_approximator 
end

@concrete struct KSLRetraction <: AbstractLowRankRetraction
    sparse_approximator 
end

# Example



# useful?
function computeQ!(A)
    m, n = size(A)
    k = min(m, n)
    tau = zeros(eltype(A), k)
    LAPACK.geqrf!(A, tau)
    LAPACK.orgqr!(A, tau, k)
    return A
end

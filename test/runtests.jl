using LinearAlgebra, LowRankIntegrators
using Test

X = randn(10,10)
test = svd(X)
algs = [QDEIM(), DEIM(), LDEIM()]
for alg in algs
    idcs = index_selection(test.U[:,1:5], alg)
    @test length(idcs) == 5
end
#include("data_agnostic_approximation.jl")
#include("data_driven_approximation.jl")
#include("data_informed_approximation.jl")
#include("data_agnostic_DEIM_approximation.jl")
#include("sparse_interpolation.jl")

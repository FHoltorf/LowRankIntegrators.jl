using LinearAlgebra

@testset "Low Rank Algebra" begin
    U1 = randn(100, 4)
    Z1 = randn(1000, 4)

    U2 = randn(100, 2)
    Z2 = randn(1000, 2)

    A = U1*Z1'
    lr_A = LowRankApproximation(U1,Z1)


    B = U2*Z2'
    lr_B = LowRankApproximation(U2,Z2)

    @test norm(full(lr_A + lr_B) - (A + B)) < 1e-10

    U3, Z3 = randn(1000, 7), randn(128, 7)
    C = U3*Z3'
    lr_C = LowRankApproximation(U3, Z3)
    @test norm(full(lr_A*lr_C) - A*C) < 1e-10

    # testing elementwise product
    U4, Z4 = randn(100, 2), randn(1000,2)
    D = U4*Z4'
    lr_D = LowRankApproximation(U4,Z4)
    @test norm(full(elprod(lr_A, lr_D)) - A .* D) < 1e-10

    # testing elementwise power
    for d in 1:5
        @test norm(full(elpow(lr_A, d)) - A.^d) < 1e-8
    end
end
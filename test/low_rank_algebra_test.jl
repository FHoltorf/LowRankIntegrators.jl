using LinearAlgebra

@testset "Two Factor Low Rank Algebra" begin
    U1 = randn(100, 4)
    Z1 = randn(1000, 4)

    U2 = randn(100, 2)
    Z2 = randn(1000, 2)

    A = U1*Z1'
    lr_A = TwoFactorApproximation(U1,Z1)

    B = U2*Z2'
    lr_B = TwoFactorApproximation(U2,Z2)

    # testing addition
    @test norm(Matrix(lr_A + lr_B) - (A + B)) < 1e-10

    U3, Z3 = randn(1000, 7), randn(128, 7)
    C = U3*Z3'
    lr_C = TwoFactorApproximation(U3, Z3)
    full_factor = randn(1000, 100)

    # testing product
    @test norm(Matrix(lr_A*lr_C) - A*C) < 1e-10
    @test norm(Matrix(lr_A*full_factor) - A*full_factor) < 1e-10
    @test norm(Matrix(full_factor'*lr_A) - full_factor'*A)

    # testing elementwise product
    U4, Z4 = randn(100, 2), randn(1000,2)
    D = U4*Z4'
    lr_D = TwoFactorApproximation(U4,Z4)
    @test norm(Matrix(elprod(lr_A, lr_D)) - A .* D) < 1e-10

    # testing elementwise power
    for d in 1:5
        @test norm(Matrix(elpow(lr_A, d)) - A.^d) < 1e-8
    end

    # test sum 
    lr_matrices = [TwoFactorApproximation(randn(100, 5), randn(100, 5)) for i in 1:5]
    full_representations = [Matrix(lr_matrix) for lr_matrix in lr_matrices]
    @test norm(Matrix(sum(lr_matrices)) - sum(full_representations)) < 1e-8

    # test prod
    lr_matrices = [TwoFactorApproximation(randn(100, 5), randn(100, 5)) for i in 1:5]
    full_representations = [Matrix(lr_matrix) for lr_matrix in lr_matrices]    
    @test norm(Matrix(prod(lr_matrices)) -  prod(full_representations)) < 1e-7
end
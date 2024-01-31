@testset "Data compression" begin
    N = 100
    Ws = []
    for k in 1:2
        W = randn(N,N)
        for i in 1:N
            W[i,i] = 0
            for j in 1:i
                W[i,j] = - W[j,i]
            end
        end
        push!(Ws, W)
    end
    D = Diagonal([2.0^(-j) for j in 1:N])
    Y(t) = exp(t*Ws[1])*exp(t)*D*exp(t*Ws[2])
    Y0 = Y(0.0)
    Yf = Y(1.0)
    X0 = truncated_svd(Y0, tol=1e-4)
    prob = MatrixDataProblem(Y, X0, (0.0, 1.0))
    data = [Y(t) for t in range(0, 1.0, step = 0.01)]
    discrete_prob = MatrixDataProblem(data, X0)
    solvers = [UnconventionalAlgorithm(), 
               ProjectorSplitting(PrimalLieTrotter()), 
               ProjectorSplitting(DualLieTrotter()), 
               RankAdaptiveUnconventionalAlgorithm(1e-8, rmax = 20),
               GreedyIntegrator()]
    for solver in solvers 
        sol = LowRankIntegrators.solve(prob, solver, 1e-2)
        discrete_sol = LowRankIntegrators.solve(discrete_prob, solver)
        @test all(Matrix(sol.Y[end]) .≈ Matrix(discrete_sol.Y[end]))
    end
end
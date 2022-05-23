using LowRankIntegrators, SparseArrays

@testset "Burgers equation" begin
    n = 1000 # spatial discretization

    l = π # length of spatial domain
    Δx = l/n # step size
    x_range = Δx/2:Δx:l-Δx/2 # uniform grid

    # boundary conditions
    left(i) = i > 1 ? i - 1 : n
    right(i) = i < n ? i + 1 : 1

    # discretized diff operators
    # laplacian (times viscosity)
    Δ = spzeros(n, n)
    ν = 0.005
    for i in 1:n
        Δ[i,left(i)] = ν/Δx^2
        Δ[i,i] = -2ν/Δx^2
        Δ[i,right(i)] = ν/Δx^2
    end

    # gradient
    ∇ = spzeros(n, n)
    for i in 1:n
        ∇[i,left(i)] = -1/2/Δx
        ∇[i,right(i)] = 1/2/Δx
    end

    function burgers(ρ, (∇,Δ), t)
        return Δ*ρ - (∇*ρ) .* ρ 
    end

    # uncertainty range
    m = 20 # parameter realizations scale as m^2
    σ = [0.5,0.5]
    ξ_range = [(ξ_1,ξ_2) for ξ_1 in range(-1,1,length=m), ξ_2 in range(-1,1,length=m)];

    #initial condition
    ub(x) = 0.5*(exp.(cos.(x)) .- 1.5).*sin.(x .+ 2π*0.37) # deterministic initial condition
    uprime(x,ξ,σ) = σ[1]*ξ[1]*sin.(2π*x) .+ σ[2]*ξ[2]*sin.(3π*x) # stochastic fluctuation
    ρ0_mat = hcat([ub(x_range) + uprime(x_range, ξ, σ) for ξ in ξ_range]...) # full rank initial condition
    r = 5 # approximation rank
    lr_ρ0 = truncated_svd(ρ0_mat, r); # intial condition


    dt = 1e-2 # time step 
    lr_prob = MatrixDEProblem((ρ,t) -> burgers(ρ, (∇,Δ), t),  lr_ρ0, (0.0, 1.0)) # defines the matrix differential equation problem
    solvers = [UnconventionalAlgorithm(),
               ProjectorSplitting(PrimalLieTrotter()), 
               ProjectorSplitting(DualLieTrotter()), 
               ProjectorSplitting(Strang()),
               RankAdaptiveUnconventionalAlgorithm(1e-4, rmax=10)]
    for solver in solvers
        lr_sol = LowRankIntegrators.solve(lr_prob, solver, dt) # solves the low rank approximation
    end
end
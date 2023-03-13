@testset "Data informed compression - Burgers example" begin
    n = 1000 # spatial discretization
    using DifferentialEquations, LinearAlgebra, SparseArrays, LowRankIntegrators
    m = 10

    r = 15 # approximation rank
    l = π # length of spatial domain
    Δx = l/n # step size
    dt = 10e-3
    x_range = Δx/2:Δx:l-Δx/2 # uniform grid
    t_range = 0:dt:1.0

    # uncertainty realizations
    σ = [0.5,0.5]
    ξs = [(ξ_1,ξ_2) for ξ_1 in range(-1,1,length=m), ξ_2 in range(-1,1,length=m)];

    # generate coefficient matrices and store as global variables
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



    # generate the data used for comparison
    ub(x) = 0.5*(exp.(cos.(x)) .- 1.5).*sin.(x .+ 2π*0.37) # deterministic initial condition
    uprime(x,ξ,σ) = σ[1]*ξ[1]*sin.(2π*x) .+ σ[2]*ξ[2]*sin.(3π*x) # stochastic fluctuation

    function burgers(ρ, (Δ,∇), t)
        return Δ*ρ - (∇*ρ) .* ρ 
    end
    function burgers!(dρ, ρ, (∇,Δ,C,D), t)
        mul!(C, ∇, ρ)
        mul!(dρ, Δ, ρ)
        D .= C .* ρ
        dρ .-= D
    end
    C, D = ones(n), ones(n)
    true_sols = [zeros(n,m^2) for i in 1:length(t_range)]
    ode_prob = ODEProblem(burgers!, ones(n), (0.0, 1.0), (∇,Δ,C,D))
    for (i,ξ) in enumerate(ξs)
        println("Uncertainty realization $i")
        _prob = remake(ode_prob, u0 = ub(x_range) + uprime(x_range, ξ, σ))
        _sol = Array(DifferentialEquations.solve(_prob, saveat=t_range))
        for k in 1:length(t_range)
            true_sols[k][:,i] .= _sol[:,k]      
        end
    end

    # Greedy: data_informed
    X0 = truncated_svd(hcat(true_sols[1], true_sols[2]), 15)
    U0 = X0.U
    Z0 = true_sols[1]'*U0
    function interpolate_data(t, data, t_range, dt) 
        idx = min(floor(Int64, t/dt) + 1, length(t_range))
        return data[idx]
    end

    FZ(Z,U,t) = Matrix(burgers(TwoFactorRepresentation(U,Z), (Δ,∇),t)' * U)
    data_informed_prob = MatrixHybridProblem(t -> interpolate_data(t, true_sols, t_range, dt), FZ, TwoFactorRepresentation(U0, Z0), (0.0, 1.0))
    data_informed_sol = LowRankIntegrators.solve(data_informed_prob, GreedyIntegrator(), dt)
    @test norm(true_sols[end] - data_informed_sol.Y[end])/norm(true_sols[end]) <= 0.1
end
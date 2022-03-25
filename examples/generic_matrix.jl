using LowRankIntegrators, Plots
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
Y(t) =  exp(t*Ws[1])*exp(t)*D*exp(t*Ws[2])
Y0 = Y(0.0)
Yf = Y(1.0)

# initialization of low rank approximation based on the SVD
r_range = [4,8,16]
dt_range = [0.1,0.05,0.01,0.005,0.0001]
tol = 1e-8 # needed for rank adaptation
S_kwargs, K_kwargs, L_kwargs = Dict(:dt => 0.001), Dict(:dt => 0.001), Dict(:dt => 0.001)
alg = Euler()
solvers = [PrimalLieTrotterProjectorSplitting(), StrangProjectorSplitting(), UnconventionalAlgorithm(), RankAdaptiveUnconventionalAlgorithm(tol, r_max = 20)]
error = Dict()
for r in r_range
    X0 = truncated_svd(Y0, r)
    prob = MatrixDataProblem(Y, X0, (0.0, 1.0))
    for dt in dt_range
        for solver in solvers
            sol = LowRankIntegrators.solve(prob, solver, dt)
            error[(r,dt,solver)] = norm(Matrix(sol.Y[end]) - Yf)
        end
    end
end

# visualization of data
solver_names = Dict(solvers[1] => "Lie Trotter Projector Splitting", 
                    solvers[2] => "Strang Projector Splitting",
                    solvers[3] => "Unconventional Algorithm",
                    solvers[4] => "Rank Adaptive Unconventional Algorithm")
error_comparison = plot(yscale = :log, xscale = :log, xlabel = "step size", ylabel = "error", legend = :bottomleft)

colors = Dict(solvers[1] => :blue, solvers[2] => :red, solvers[3] => :green, solvers[4] => :dodgerblue)
marker_range = [:v, :hex, :o, :star, :cross, :^]
markers = Dict(r_range[k] => marker_range[k] for k in 1:length(r_range))
for r in r_range
    for solver in solvers
        try 
            plot!(error_comparison, dt_range, [error[(r, dt, solver)] for dt in dt_range], label = (r == r_range[1] ? solver_names[solver] : nothing), linestyle=:dash, marker=markers[r], color = colors[solver])
        catch
            continue
        end
    end
end
display(error_comparison)
using LinearAlgebra, DifferentialEquations, LowRankIntegrators, SparseArrays, Plots, DiffEqSensitivity, Zygote

ub(x) = 0.5*(exp.(cos.(x)) .- 1.5).*sin.(x .+ 2π*0.37)
uprime(x,ξ,σ) = σ[1]*ξ[1]*sin.(2π*x) .+ σ[2]*ξ[2]*sin.(3π*x)

# just some generic parameter definition (can ignore)
m = 100 # parameter realizations
n_x = 1000 # spatial discretization
r = 10 # approximation rank

l = π # length of spatial domain
Δx = l/n_x # step size
x_range = Δx/2:Δx:l-Δx/2 # uniform grid

# periodic bcs
left(i) = i > 1 ? i - 1 : n_x
right(i) = i < n_x ? i + 1 : 1

# Laplacian 
const L = spzeros(n_x, n_x)
for i in 1:n_x
    L[i,left(i)] = 1/Δx^2
    L[i,i] = -2/Δx^2
    L[i,right(i)] = 1/Δx^2
end

# first order derivative
const G = spzeros(n_x, n_x)
for i in 1:n_x
    G[i,left(i)] = -1/2/Δx
    G[i,right(i)] = 1/2/Δx
end

# parameter value
function full_burgers_1D!(dρ, ρ, ν, t)
    dρ .= ν*(L*ρ) - ρ .* (G*ρ) 
end


ξ_range = [(x,y) for x in range(-1,1,length=8), y in range(-1,1,length=8)]
σ = [0.5,0.5]

dt = 0.001
save_times = 0:dt:1.0
sol_by_parameter = Dict()
for ξ in ξ_range
    prob = ODEProblem(full_burgers_1D!, ub(x_range) + uprime(x_range, ξ, σ), (0, 10.0), 0.005)
    sol = Array(DifferentialEquations.solve(prob, saveat = save_times))
    sol_by_parameter[ξ] = sol 
end

sol_by_time = Matrix[]
for (i,t) in enumerate(save_times)
    push!(sol_by_time, hcat([sol_by_parameter[ξ][:,i] for ξ in ξ_range]...) )
end

function DO_eqns(dU, dZ, U, Z, (ρ, dρ), t)
    ρ .= U*Z'
    full_burgers_1D!(dρ, ρ, 0.005, t)
    dU .= (I - U*U')*dρ*Z*pinv(Z'*Z, atol = 1e-4)
    dZ .= dρ'*U
end

ρ = zeros(n_x, 64)
dρ = similar(ρ)
function DO_eqns_non_allocating!(dU, dZ, U, Z, t) 
    DO_eqns(dU, dZ, U, Z, (ρ,dρ), t)
end

r = 8
U, S, V = svd(sol_by_time[1])
X0 = LowRankApproximation(U[:,1:r], V[:,1:r]*Matrix(Diagonal(S[1:r])))
solver = DOAlgorithm(DirectTimeMarching(), DO_rhs = DO_eqns_non_allocating!)
lr_prob = MatrixDEProblem(nothing, X0, (0.0, 1.0))
@time lr_sol = LowRankIntegrators.solve(lr_prob, solver, dt).sol

plot(save_times, [norm(Matrix(lr_sol.Y[i]) - sol_by_time[i])^2/64^2 for i in 1:length(lr_sol.Y)])
using DifferentialEquations, LinearAlgebra, LowRankIntegrators, LaTeXStrings, GLMakie
# Reaction Network
# ∅ -> A -> ∅, [k1/(1+(B/θ)^3), k2*A]
# ∅ -> B -> ∅, [k3/(1+A), k4*B]

# computing the propensities
function props(k1, k2, k3, k4, θ, N)
    ax = [zeros(N) for i in 1:4]
    ay = [zeros(N) for i in 1:4]
    for x in 1:N
        ax[1][x] = 1
        ax[2][x] = k2*(x-1)
        ax[3][x] = k3/(1+(x-1))
        ax[4][x] = 1
    end
    for y in 1:N
        ay[1][y] = k1/(1+((y-1)/θ)^3) 
        ay[2][y] = 1
        ay[3][y] = 1
        ay[4][y] = k4*(y-1)
    end
    return ax, ay
end

# shift operators:
# let A be an n×n matrix. Then,
# shift(x,n)*A shifts rows by x rows down and padds the first x rows with 0s
# A*shift(x,n)' shifts cols by x cols right and padds the x leftmost cols with 0s
function shift(x,n)
    S = zeros(n,n)
    if x > 0
        S[1+x:end,1:end-x] = Matrix(I,n-x, n-x)
    end
    if x < 0
        S[1:end+x,1-x:end] = Matrix(I,n+x,n+x)
    end
    if x == 0
        S = I
    end
    return S
end

## simulation parameters
const N = 300
const Nrxn = 4
const ν = [(1,0), (-1,0), (0,1), (0,-1)]
k1 = 30.0
k2 = 1
k3 = 10
k4 = 1
θ = 1

ax, ay = props(k1,k2,k3,k4,θ,N) # propensity factors
Srows = [shift(ν[r][1],N) for r in 1:Nrxn] # shift operators
Scols = [shift(ν[r][2],N)' for r in 1:Nrxn] # shift operators
A = [Srows[r]*TwoFactorApproximation(ax[r],ay[r])*Scols[r] for r in 1:Nrxn] # coefficient matrices
Asum = TwoFactorApproximation(hcat(ax...), hcat(ay...)) # coefficient matrix

# chemical master equation
function cme(P, (A,Asum,Srows,Scols), t) 
    return sum(A[r] .* (Srows[r]*P*Scols[r]) for r in 1:Nrxn) - Asum .* P
end 

# Gaussian initial condition
x0 = 20
y0 = 20
Q = qr(randn(2,2)).Q
D = Q*Diagonal([rand()/20, rand()/20])*Q'
P0(x,y) = exp(-[x-x0; y-y0]'*D*[x-x0; y-y0])
P_init = [P0(x,y) for x in 1:N, y in 1:N]
P_init ./= sum(P_init)
Tf = 10.0

dt = 2e-2
prob = ODEProblem(cme, P_init, (0, Tf), (Matrix.(A), Matrix(Asum), Srows, Scols))
t_direct = @elapsed sol = DifferentialEquations.solve(prob, Tsit5(), saveat = dt)

P_lr = truncated_svd(P_init, ϵ = 1e-6)
@show rank(P_lr)
lr_prob = MatrixDEProblem((x,t) -> cme(x, (A, Asum, Srows, Scols), t), P_lr, (0.0, Tf))
t_lr = @elapsed lr_sol = LowRankIntegrators.solve(lr_prob, UnconventionalAlgorithm(), dt)

@show t_direct/t_lr

# comparison movie
fig = Figure()
ax_lra = Axis(fig[1,1], xlabel = "A", ylabel = "B", title = "full")
ax_full = Axis(fig[1,2], xlabel = "A", ylabel = "B", title = "low rank")
ax_error = Axis(fig[2,1:2], yscale = log10, xlabel="time", ylabel = "error")
limits!(ax_error, 0, Tf, 1e-8, 1e-0)
display(fig)

N_relevant = 40
sol_lra = Observable(Matrix(lr_sol.Y[1][1:N_relevant, 1:N_relevant]))
sol_full = Observable(sol.u[1][1:N_relevant, 1:N_relevant])
error = Observable([norm(sol.u[1] - Matrix(lr_sol.Y[1]))])
ts = Observable([0.0])
Makie.heatmap!(ax_lra, sol_lra)
Makie.heatmap!(ax_full, sol_full)
Makie.lines!(ax_error, ts, error, linewidth = 2, color = :black)

record(fig, "markov_chain.mp4", enumerate(lr_sol.t); framerate = 24) do (i,t)
    sol_lra[] = Matrix(lr_sol.Y[i][1:N_relevant,1:N_relevant])
    sol_full[] = sol(t)[1:N_relevant,1:N_relevant]
    push!(error.val, maximum(abs.(Matrix(lr_sol.Y[i][1:N_relevant, 1:N_relevant]) - sol(t)[1:N_relevant, 1:N_relevant])))
    push!(ts.val, t)
    ts[] = ts[] 
    error[] = error[]
end
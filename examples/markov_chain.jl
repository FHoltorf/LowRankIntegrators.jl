using DifferentialEquations, LinearAlgebra, LowRankIntegrators, GLMakie
# Reaction Network
# ∅ -> A -> ∅, [k1/(1+(B/θ)^3), k2*A]
# ∅ -> B -> ∅, [k3/(1+A), k4*B]

# computing the propensities
const N = 50
const Nrxn = 4
k1 = 30.0
k2 = 1
k3 = 10
k4 = 1
θ = 1
const ν = [(1,0), (-1,0), (0,1), (0,-1)]
function coeff_cache(k1, k2, k3, k4, θ)
    ax = [zeros(N) for i in 1:4]
    ay = [zeros(N) for i in 1:4]
    A_sum = zeros(N,N)
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
    for x in 1:N, y in 1:N
        A_sum[x,y] = sum(ay[i][y]*ax[i][x] for i in 1:4)
    end
    return ax, ay, A_sum
end

const ax, ay, A_sum = coeff_cache(k1, k2, k3, k4, θ)

function γx(U,ax,ay,(α,β),y,k,j) 
    if y - β > N || y - β < 1
        return 0.0
    else
        return ay[y-β]*sum(U[x,k]*U[x-α,j]*ax[x-α] for x in max(1,1+α):min(N,N+α))
    end 
end

function γx(U,A,(α,β),y,k,j)
    if y - β > N || y - β < 1
        return 0.0
    else
        return sum(U[x,k]*U[x-α,j]*A[x-α,y-β] for x in max(1,1+α):min(N,N+α))
    end 
end

function γy(V,ax,ay,(α,β),x,k,j)
    if x - α < 1 || x - α > N
        return 0.0
    else
        return ax[x-α]*sum(V[y,k]*V[y-β,j]*ay[y-β] for y in max(1,1+β):min(N,N+β))
    end
end

function γy(V,A,(α,β),x,k,j)
    if x - α < 1 || x - α > N
        return 0.0
    else
        return sum(V[y,k]*V[y-β,j]*A[x-α,y-β] for y in max(1,1+β):min(N,N+β))
    end
end

function γx_(U,ax,(α,β),k,j)
    return sum(U[x,k]*U[x-α,j]*ax[x-α] for x in max(1,1+α):min(N,N+α))
end

function γy_(V,ay,(α,β),k,j)
    return sum(V[y,k]*V[y-β,j]*ay[y-β] for y in max(1,1+β):min(N,N+β))
end


function K_step!(dK, K, V, t)
    n, r = size(dK)
    for x in 1:n, k in 1:r
        dK[x,k] = 0
        for j in 1:r 
            for i in 1:Nrxn
                if x - ν[i][1] >= 1 && x - ν[i][1] <= N
                    dK[x,k] += K[x-ν[i][1],j]*γy(V, ax[i], ay[i], ν[i], x, k, j)
                end
            end
            dK[x,k] -= K[x,j]*γy(V, A_sum, (0,0), x, k, j)
        end
    end
end

function L_step!(dL, L, U, t)
    m, r = size(dL)
    for y in 1:m, k in 1:r
        dL[y,k] = 0
        for j in 1:r
            for i in 1:Nrxn 
                if y - ν[i][2] >= 1 && y - ν[i][2] <= N
                    dL[y,k] += L[y-ν[i][2],j]*γx(U, ax[i], ay[i], ν[i], y, k, j) 
                end
            end
            dL[y,k] -= L[y,j]*γx(U, A_sum, (0,0), y, k, j)
        end
    end
end

function S_step!(dS, S, (U,V), t)
    r = size(dS,1)
    for i in 1:r, j in 1:r
        dS[i,j] = 0
        for m in 1:r, n in 1:r
            dS[i,j] += sum(S[m,n]*(γx_(U,ax[o],ν[o],i,m)*γy_(V,ay[o],ν[o],j,n)-γx_(U,ax[o],(0,0),i,m)*γy_(V,ay[o],(0,0),j,n)) for o in 1:Nrxn) 
        end
    end
end

function S_backwards_step!(dS, S, (U,V), t)
    r = size(dS,1)
    for i in 1:r, j in 1:r
        dS[i,j] = 0
        for m in 1:r, n in 1:r
            dS[i,j] -= sum(S[m,n]*(γx_(U,ax[o],ν[o],i,m)*γy_(V,ay[o],ν[o],j,n)-γx_(U,ax[o],(0,0),i,m)*γy_(V,ay[o],(0,0),j,n)) for o in 1:Nrxn) 
        end
    end
end

function cme!(dP, P, p, t)
    for x in 1:N, y in 1:N
        dP[x,y] = 0
        for i in 1:Nrxn
            if x - ν[i][1] > 0 && y - ν[i][2] > 0 && x - ν[i][1] < N+1 && y - ν[i][2] < N+1
                dP[x,y] += ax[i][x-ν[i][1]]*ay[i][y-ν[i][2]]*P[x-ν[i][1],y-ν[i][2]]
            end
        end
        dP[x,y] -= A_sum[x,y]*P[x,y]
    end
end

# Gaussian
x0 = 10
y0 = 10
Q = qr(randn(2,2)).Q
D = Q*Diagonal([rand()/10, rand()/10])*Q'
P0(x,y) = exp(-[x-x0; y-y0]'*D*[x-x0; y-y0])
P_init = [P0(x,y) for x in 1:N, y in 1:N]
P_init ./= sum(P_init)

prob = ODEProblem(cme!, P_init, (0, 2.0))
@time sol = DifferentialEquations.solve(prob, Euler(), dt = 0.01)

U, Σ, V = svd(P_init)
rank_0 = findfirst(x -> x < 1e-6, Σ)
X0 = SVDLikeApproximation(U[:,1:rank_0], Matrix(Diagonal(Σ[1:rank_0])), V[:,1:rank_0])

Solver = UnconventionalAlgorithm(S_rhs = S_step!, L_rhs = L_step!, K_rhs = K_step!,
                                 S_alg = Euler(), L_alg = Euler(), K_alg = Euler(),
                                 S_kwargs = Dict(:dt => 0.01), L_kwargs = Dict(:dt => 0.01), K_kwargs = Dict(:dt => 0.01)) #PrimalLieTrotterProjectorSplitting(S_rhs = S_backwards_step!, L_rhs = L_step!, K_rhs = K_step!)
prob = MatrixDEProblem(nothing, X0, (0.0,2.0))
@time integrator = LowRankIntegrators.solve(prob, Solver, 0.01)

# movie comparison
fig = Figure()
ax_lra = Axis(fig[1,1])
ax_full = Axis(fig[1,2])
ax_error = Axis(fig[2,1:2], yscale = log10)
limits!(ax_error, 0, 2.0, 1e-8, 1e-0)
display(fig)

sol_lra = Observable(full(integrator.sol.Y[1]))
sol_full = Observable(sol.u[1])
error = Observable([norm(sol.u[1] - full(integrator.sol.Y[1]))])
ts = Observable([0.0])
Makie.heatmap!(ax_lra, sol_lra)
Makie.heatmap!(ax_full, sol_full)
Makie.lines!(ax_error, ts, error, linewidth = 2, color = :black)
for (i,t) in enumerate(integrator.sol.t)
    sol_lra[] = full(integrator.sol.Y[i])
    sol_full[] = sol(t)
    push!(error.val, norm(full(integrator.sol.Y[i]) - sol(t)))
    push!(ts.val, t)
    ts[] = ts[]
    error[] = error[]
    sleep(0.03) 
end
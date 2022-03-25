# implements basic capabilities for DO integrators
abstract type AbstractRetraction end
struct DirectTimeMarching <: AbstractRetraction end 
struct ExponentialRetraction <: AbstractRetraction end #to-do
struct AlgebraicSVD <: AbstractRetraction end #to-do
struct ContinuousSVD <: AbstractRetraction end #to-do

struct DirectTimeMarching_Params
    DO_rhs # rhs of DO equations of this form: f!(dU, dZ, U, Z, t) 
    DO_alg
    DO_kwargs
    pinv_tol 
    ϵ_orth
    μ_orth
    maxiter_orth
end

struct DirectTimeMarching_Cache
    DOIntegrator
    X
    n
    m
    ϵ_orth
    μ_orth
    maxiter_orth
end

mutable struct DOAlgorithm{ρType} <: AbstractDLRAlgorithm
    ρ::ρType
    alg_params
end

function DOAlgorithm(ρ::DirectTimeMarching; 
                     DO_rhs = nothing, DO_alg = Tsit5(), DO_kwargs = Dict(),
                     pinv_tol = 1e-8, ϵ_orth = 1e-8, μ_orth = 0.1, maxiter_orth = 100)
    params = DirectTimeMarching_Params(DO_rhs, DO_alg, DO_kwargs, pinv_tol, ϵ_orth, μ_orth, maxiter_orth)
    return DOAlgorithm(ρ, params)
end

function init(prob::MatrixDEProblem, alg::DOAlgorithm{DirectTimeMarching}, dt)
    t0, tf = prob.tspan
    u = deepcopy(prob.u0)
    @assert tf > t0 "Integration in reverse time direction is not supported"
    # number of steps
    n = floor(Int,(tf-t0)/dt) + 1 
    # compute more sensible dt # rest will be handled via interpolation/save_at
    dt = (tf-t0)/(n-1)
    # initialize solution object
    sol = DLRSolution(Vector{typeof(prob.u0)}(undef, n), collect(range(t0, tf, length=n)))
    sol.Y[1] = deepcopy(prob.u0) # add initial point to solution object
    # initialize cache
    cache = alg_cache(prob, alg, u, dt)
    return DLRIntegrator(u, t0, dt, sol, alg, cache, 0)
end

function alg_cache(prob::MatrixDEProblem, alg::DOAlgorithm{DirectTimeMarching}, u, dt)
    @unpack pinv_tol, ϵ_orth, μ_orth, maxiter_orth, DO_rhs, DO_alg, DO_kwargs = alg.alg_params
    n, r = size(u.U)
    m = size(u.Z,1)
    X = vcat(u.U,u.Z)
    if isnothing(DO_rhs)
        f_DO = function (dX,X,(dY,Y,pinv_tol),t)
                    n, m = size(Y)
                    U = @view X[1:n,:]
                    Z = @view Z[n+1:n+m,:]
                    Y .= U*Z'
                    dY .= prob.f(Y,t)
                    dX[1:n,:] .= (I - U*U')*dY*Z*pinv(Z'*Z, atol = pinv_tol)
                    dX[n+1:n+m,:] .= dY'*U  
               end
        dY = zeros(n, m)
        Y = similar(dY)
        prob = ODEProblem(f_DO, X, prob.tspan, (dY, Y, pinv_tol))
    else
        f_DO = function (dX,X,(n,m),t)
                    U = @view X[1:n,:]
                    Z = @view X[n+1:n+m,:]
                    dU = @view dX[1:n,:]
                    dZ = @view dX[n+1:n+m,:]
                    DO_rhs(dU, dZ, U, Z, t)
                end
        prob = ODEProblem(f_DO, X, prob.tspan, (n,m))
    end
    DOIntegrator = init(prob, DO_alg; save_everystep = false, DO_kwargs...)
    return DirectTimeMarching_Cache(DOIntegrator, X, n, m, ϵ_orth, μ_orth, maxiter_orth)
end

function step!(integrator::DLRIntegrator, alg::DOAlgorithm{DirectTimeMarching}, dt)
    @unpack u, t, iter, cache = integrator
    direct_time_marching_do_step!(u, cache, dt)
    integrator.t += dt
    integrator.iter += 1
end

function direct_time_marching_do_step!(u, cache, dt)
    @unpack DOIntegrator, X, m, n, ϵ_orth, μ_orth, maxiter_orth = cache
    X[1:n,:] .= u.U
    X[n+1:n+m, :] .= u.Z
    set_u!(DOIntegrator, X)
    step!(DOIntegrator, dt, true)
    u.U .= DOIntegrator.u[1:n,:]
    u.Z .= DOIntegrator.u[n+1:m+n,:]
    gd_orthonormalization!(u.U, u.Z; ϵ = ϵ_orth, μ = μ_orth, maxiter = maxiter_orth)
end

# utilities
# gradient descent algorithm to orthonormalize basis matrix U of factorization U*Z 
function gd_orthonormalization!(U, Z; ϵ = 1e-8, μ = 0.1, maxiter = 100)
    K = U'*U
    A = Matrix{eltype(K)}(I, size(K)) # its only a rank x rank matrix, so rather cheap allocation
    Ainv = Matrix{eltype(K)}(I, size(K))
    dA = similar(A)
    iter = 0
    while norm(A'*K*A - I)^2 > ϵ && iter < maxiter
        dA .= - K*A*(A'*K*A - I)
        A .+= μ*dA
        Ainv .-= μ*Ainv*dA*Ainv
        iter += 1
    end
    if iter == maxiter
        @warn "Orthonormalization did not converge. 
               Iterations exceeded maxiters = $maxiter. 
               Residual: $(norm(A'*K*A - I)^2)"
    end
    U .= U*A
    Z .= Z*Ainv'
end

# gradient descent algorithm to orthonormalize basis matrix
function gd_orthonormalization!(U; ϵ = 1e-8, μ = 0.1, maxiter = 100)
    K = U'*U
    A = Matrix{eltype(K)}(I, size(K))
    Ainv = Matrix{eltype(K)}(I, size(K))
    dA = similar(A)
    iter = 0
    while norm(A'*K*A - I)^2 > ϵ && iter < maxiter
        dA .= - K*A*(A'*K*A - I)
        A .+= μ*dA
        Ainv .-= μ*Ainv*dA*Ainv
        iter += 1
    end
    if iter == maxiter
        @warn "Orthonormalization did not converge. 
               Iterations exceeded maxiters = $maxiter. 
               Residual: $(norm(A'*K*A - I)^2)"
    end
    U .= U*A
end

# gradient descent update for truncated svd
function gd_truncated_svd!(U,Z,C,Ψ_U,Ψ_Z; μ=0.1, ϵ=1e-8, maxiter = 100, pinv_tol = 1e-8)
    iter = 0
    while iter < maxiter
        dU = -(I-U*U')*Ψ_U*(Ψ_Z'*Z*pinv(C, atol = pinv_tol))  
        dZ = Z - Ψ_Z*(Ψ_U'*U)
        if norm(dU)^2 + norm(dZ)^2 < ϵ
            break
        end
        U .-= μ*dU
        Z .-= μ*dZ    
        gd_orthonormalization!(U, Z)
        iter += 1
    end
    if iter == maxiter
        @warn "Orthonormalization did not converge. 
               Iterations exceeded maxiters = $maxiter. 
               Residual: $(norm(dU)^2+norm(dZ)^2)"
    end
end

function gd_truncated_svd!(U,Z,C,Ψ; μ=0.1, ϵ=1e-8, maxiter = 100, pinv_tol = 1e-8)
    iter = 0
    while iter < maxiter
        dU = -(I-U*U')*Ψ'*Z*pinv(C, atol = pinv_tol)
        dZ = Z - Ψ'*U
        if norm(dU)^2 + norm(dZ)^2 < ϵ
            break
        end
        U .-= μ*dU
        Z .-= μ*dZ    
        gd_orthonormalization!(U, Z)
        C .= Z'*Z
        iter += 1
    end
    if iter == maxiter
        @warn "Orthonormalization did not converge. 
               Iterations exceeded maxiters = $maxiter. 
               Residual: $(norm(dU)^2+norm(dZ)^2)"
    end
end


# opnorm(N)*dt > σ_min => update rank

 
struct GradientDescent
    maxiter::Int
    μ::Float64
    atol::Float64
    rtol::Float64
    function GradientDescent(;maxiter = 100, μ = 1.0, atol = 1e-8, rtol = 1e-8)
        return new(maxiter, μ, atol, rtol)
    end
end

struct QR end

struct SVD end

function orthonormalize!(LRA::TwoFactorApproximation, alg::GradientDescent)
    @unpack μ, atol, rtol, maxiter = alg 
    K = A.U'*A.U
    r = size(K)
    A = Matrix{eltype(K)}(I, r, r)
    Ainv = Matrix{eltype(K)}(I, r, r)
    dA = similar(A)
    iter = 0
    ϵ = norm(A'*K*A - I)
    while ϵ > atol && ϵ > r*rtol && iter < maxiter
        dA .= - K*A*(A'*K*A - I)
        A .+= μ*dA
        Ainv .-= μ*Ainv*dA*Ainv
        iter += 1
    end
    if iter == maxiter
        @warn "Gradient flow orthonormalization did not converge. 
               Iterations exceeded maxiters = $maxiter. 
               Primal residual: $(norm(A'*K*A - I)^2)"
    end
    LRA.U .= LRA.U*A
    LRA.Z .= LRA.Z*Ainv'
end

function orthonormalize!(LRA::TwoFactorApproximation, ::QR)
    Q, R = qr(LRA.U)
    LRA.U .= Matrix(Q) # ToDo: store Q in terms of householder reflections
    LRA.Z .= LRA.Z*R'
end

function orthonormalize!(LRA::TwoFactorApproximation, ::SVD)
    U, S, V = svd(LRA.U)
    LRA.U .= U 
    LRA.Z .= LRA.Z*V*S
end

# ToDo: add Gram-Schmidt orthonormalization

# gradient descent update for truncated svd
# This is work in progress and will be tailored to TwoFactorApproximation input types etc. 
#=
function gd_truncated_svd!(U, Z, C, Ψ_U, Ψ_Z; μ=0.1, ϵ=1e-8, maxiter = 100, pinv_tol = 1e-8)
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
               Primal residual: $(norm(dU)^2+norm(dZ)^2)"
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
               Primal residual: $(norm(dU)^2+norm(dZ)^2)"
    end
end
=#
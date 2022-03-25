# rank adaptation
function normal_component(U, Z, C, dY; tol = 1e-8)
    # U = basis
    # Z = coefficients
    # C = covariance matrix, Z'*Z for example (assumed to be of very small dimension)
    # dY = dynamics, f(UZ') for example
    return (I - U*U')*dY*(I-Z*pinv(C, atol = tol)*Z')
end

function normal_component(LRA::TwoFactorRepresentation, C, dY; tol = 1e-8)
    # C = covariance matrix, Z'*Z for example (assumed to be of very small dimension)
    # dY = dynamics, f(UZ') for example
    return normal_component(LRA.U, LRA.Z, C, dY; tol = tol)
end

function normal_component(LRA::TwoFactorRepresentation, dY; tol = 1e-8)
    # C = covariance matrix, Z'*Z for example (assumed to be of very small dimension)
    # dY = dynamics, f(UZ') for example
    return normal_component(LRA.U, LRA.Z, LRA.Z'*LRA.Z, dY; tol = tol)
end

# gradient descent update for truncated svd
# This is work in progress and will be tailored to TwoFactorRepresentation input types etc. 
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
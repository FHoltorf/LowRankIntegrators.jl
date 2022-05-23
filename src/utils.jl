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

function gd_truncated_svd!(Φ::TwoFactorRepresentation, Ψ::TwoFactorRepresentation, alg::GradientDescent, orth_alg)
    #UZ => lower rank approximation
    #Ψ => larger rank matrix to be approximated
    @unpack μ, rtol, atol, maxiter = alg
    iter = 0
    C = Symmetric(Φ.Z', Φ.Z)
    UtU = Ψ.U'*Φ.U
    ZtZ = Ψ.Z'*Φ.Z
    k1 = ZtZ / C 
    k2 = Ψ.Z*UtU
    k3 = Φ.U*UtU
    k4 = k3*k1
    k5 = Ψ.U*k1 

    dZ = Φ.Z - k2
    dU = k4 - k5 
    while (iter < maxiter && 
          norm(dU)^2 + norm(dZ)^2 > atol^2 && 
          (norm(dU)^2 + norm(dZ)^2)/(norm(U)^2 + norm(Z)^2) > rtol^2)

        Φ.U .-= μ*dU
        Ψ.Z .-= μ*dZ
        orthonormalize!(Φ, orth_alg)
        iter += 1

        C = Symmetric(Φ.Z', Φ.Z)
        mul!(UtU, Ψ.U',Φ.U)
        mul!(ZtZ, Ψ.Z',Φ.Z)
        k1 = ZtZ / C 
        mul!(k2,Ψ.Z,UtU)
        mul!(k3,Φ.U,UtU)
        mul!(k4,k3,k1)
        mul!(k5,Ψ.U,k1)
        dZ = Φ.Z - k2
        dU = k4 - k5
    end
    if iter == maxiter
        @warn "Orthonormalization did not converge. 
               Iterations exceeded maxiters = $maxiter. 
               Residual: $(norm(dU)^2+norm(dZ)^2)"
    end    
end

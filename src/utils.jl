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

struct SecondMomentMatching end

function orthonormalize!(LRA::TwoFactorApproximation, alg::GradientDescent)
    @unpack μ, atol, rtol, maxiter = alg 
    K = LRA.U'*LRA.U
    r = rank(LRA)
    A = Matrix{eltype(K)}(I, r, r)
    Ainv = Matrix{eltype(K)}(I, r, r)
    dA = similar(A)
    iter = 0
    ϵ = norm(A'*K*A - I)^2
    while ϵ > atol && ϵ > r*rtol && iter < maxiter
        dA .= - K*A*(A'*K*A - I)
        A .+= μ*dA
        Ainv .-= μ*Ainv*dA*Ainv
        iter += 1
        ϵ = norm(A'*K*A - I)^2
    end
    if iter == maxiter
        @warn "Gradient flow orthonormalization did not converge. 
               Iterations exceeded maxiters = $maxiter. 
               Primal residual: $(norm(A'*K*A - I)^2)"
    end
    LRA.U .= LRA.U*A
    LRA.Z .= LRA.Z*Ainv'
end

function orthonormalize!(U, alg::GradientDescent)
    @unpack μ, atol, rtol, maxiter = alg 
    r = size(U,2)
    K = U'*U
    A = Matrix{eltype(K)}(I, r, r)
    dA = similar(A)
    iter = 0
    ϵ = norm(A'*K*A - I)^2
    while ϵ > atol && ϵ > r*rtol && iter < maxiter
        dA .= - K*A*(A'*K*A - I)
        A .+= μ*dA
        iter += 1
        ϵ = norm(A'*K*A - I)^2
    end
    if iter == maxiter
        @warn "Gradient flow orthonormalization did not converge. 
               Iterations exceeded maxiters = $maxiter. 
               Primal residual: $(norm(A'*K*A - I)^2)"
    end
    U .= U*A
end

function orthonormalize!(U, ::SecondMomentMatching)
    P = Symmetric(U'*U)
    Λ, V = eigen(P)
    U .= U*V*Diagonal(1 ./ sqrt.(Λ)) * V'
end
function orthonormalize!(LRA::TwoFactorApproximation, ::QR)
    Q, R = qr(LRA.U)
    LRA.U .= Matrix(Q) # ToDo: store Q in terms of householder reflections
    LRA.Z .= LRA.Z*R'
end

function orthonormalize!(LRA::SVDLikeApproximation, ::QR)
    Q, RU = qr(LRA.U)
    P, RV = qr(LRA.V)
    LRA.U .= Matrix(Q) 
    LRA.V .= Matrix(P) 
    LRA.S .= RU*S*RV'
end

function orthonormalize!(U, ::QR)
    Q, _ = qr(U)
    U .= Matrix(Q)
end

function orthonormalize!(LRA::TwoFactorApproximation, ::SVD)
    U, S, V = svd(LRA.U)
    LRA.U .= U 
    LRA.Z .= LRA.Z*V*Diagonal(S)
end

function orthonormalize!(LRA::SVDLikeApproximation, ::SVD)
    U, SU, VU = svd(LRA.U)
    V, SV, VV = svd(LRA.V)
    LRA.U .= U
    LRA.S .= Diagonal(SU)*VU*LRA.S*VV'*Diagonal(SV)
    LRA.V .= V
end

# ToDo: add Gram-Schmidt orthonormalization
# ToDo: add minium second order matching orthonormalization:  https://link.springer.com/content/pdf/10.1007/s00211-021-01178-8.pdf

# rank adaptation
function normal_component(U, Z, C, dY; tol = 1e-8)
    # U = basis
    # Z = coefficients
    # C = covariance matrix, Z'*Z for example (assumed to be of very small dimension)
    # dY = dynamics, f(UZ') for example
    return (I - U*U')*dY*(I-Z*pinv(C, atol = tol)*Z')
end

function normal_component(LRA::TwoFactorApproximation, C, dY; tol = 1e-8)
    # C = covariance matrix, Z'*Z for example (assumed to be of very small dimension)
    # dY = dynamics, f(UZ') for example
    return normal_component(LRA.U, LRA.Z, C, dY; tol = tol)
end

function normal_component(LRA::TwoFactorApproximation, dY; tol = 1e-8)
    # C = covariance matrix, Z'*Z for example (assumed to be of very small dimension)
    # dY = dynamics, f(UZ') for example
    return normal_component(LRA.U, LRA.Z, LRA.Z'*LRA.Z, dY; tol = tol)
end

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
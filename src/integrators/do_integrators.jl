# implements basic capabilities for DO integrators
abstract type AbstractRetraction end
struct DirectTimeMarching <: AbstractRetraction end 
struct TruncatedSVD <: AbstractRetraction end
struct AlternatingProjection{oType} <: AbstractRetraction 
    order::oType
end

struct Primal end
struct Dual end
struct PrimalDual end 

mutable struct DOAlgorithm{ρType} <: AbstractDLRAlgorithm
    ρ::ρType # retraction
    alg_params # parameters
end

#=
Direct time marching
    U_{n+1} = U_{n} + dt * L_Z*(L_U'*U_{n})
    Z_{n+1} = Z_{n} + dt * ((I-U_{n}*U_{n}')*L_U)(L_Z'*Z*(Z'*Z)^-1)
    orthogonalize(U_{n+1}, Z_{n+1})
    repeat
=#

struct DirectTimeMarching_Params
    F
    orth_alg
end

struct DirectTimeMarching_Cache
    F
    Y
    LU 
    LZ
    C  
end

function DirectTimeMarching(F; orthonormalization = QRFact())
    return DOAlgorithm(DirectTimeMarching(), DirectTimeMarching_Params(F, orthonormalization))
end

function init(prob::MatrixDEProblem, alg::DOAlgorithm{<:AbstractRetraction}, dt)
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
    return DLRIntegrator(u, t0, dt, sol, alg, cache, typeof(prob), 0)
end

function alg_cache(prob::MatrixDEProblem, alg::DOAlgorithm{DirectTimeMarching}, u, dt)
    @unpack alg_params = alg
    F = (u,t,dt) -> alg_params.F(prob.f, u, t, dt)
    L = F(u, prob.tspan[1] ,dt)
    rL = rank(L)
    r = rank(u)
    LU = zeros(rL,r)
    LZ = zeros(rL,r)
    Y = deepcopy(u)
    C = zeros(r, r)
    return DirectTimeMarching_Cache(F, Y, LU, LZ, C)
end

function step!(integrator::DLRIntegrator, alg::DOAlgorithm{DirectTimeMarching}, dt)
    @unpack u, t, iter, cache = integrator
    direct_time_marching_do_step!(u, cache, t, dt)
    orthonormalize!(u, alg.alg_params.orth_alg)
    cache.Y.Z .= u.Z
    cache.Y.U .= u.U
    integrator.t += dt
    integrator.iter += 1
end

function direct_time_marching_do_step!(u, cache, t, dt)
    @unpack Y, LU, LZ, C, F = cache
    L = F(Y,t,dt)

    # Z update
    mul!(LU, L.U', Y.U) 
    mul!(u.Z, L.Z, LU, dt, 1)
    
    # U update
    #((I-U*U')*L_U)(L_Z'*Z*(Z'*Z)^-1)
    mul!(L.U, Y.U, LU', -1, 1) #(I-U*U')*L_U
    mul!(LZ, L.Z', Y.Z) # L_Z'*Z
    mul!(C, Y.Z', Y.Z) # covariance computation 
    LU .= LZ / Symmetric(C) # revisit
    mul!(u.U, L.U, LU, dt, 1)
end

#=
Alternating projection
    Primal step:
        LRA_{n+1} = LRA_{n} + dt*F(LRA_{n},t)
        solve min \|LRA{n+1} - U_{n} Z_{n+1}'\| => Z_{n+1}' = U_{n}'*LRA_{n+1} => Z_{n+1} = LRA_{n+1}'*U{n}
        solve min \|LRA{n+1} - U_{n+1} Z_{n+1}'\| 
            => procrustes problem: Q, _, P = svd(LRA_{n+1}*Z_{n+1}), U_{n+1}= Q*P'
    Dual step:
        LRA_{n+1} = LRA_{n} + dt*F(LRA_{n},t)
        solve min \|LRA{n+1} - U_{n+1} Z_{n}'\| 
            => procrustes problem: Q, _, P = svd(LRA_{n+1}*Z_{n}), U_{n+1}= Q*P'
        solve min \|LRA{n+1} - U_{n+1} Z_{n+1}'\| => Z_{n+1} = U_{n+1}'*LRA_{n+1}
=#

struct AlternatingProjection_Params
    F
    orth_alg
end

struct AlternatingProjection_Cache
    F
    UtU
    UQ
    C
    orth_alg
end

function AlternatingProjection(order, F; orthonormalization = QRFact())
    return DOAlgorithm(AlternatingProjection(order), AlternatingProjection_Params(F, orthonormalization))
end

function alg_cache(prob::MatrixDEProblem, alg::DOAlgorithm{AlternatingProjection{T}}, u, dt) where T
    @unpack alg_params = alg
    F = (u,t,dt) -> alg_params.F(prob.f, u, t, dt)
    Y = u + F(u, prob.tspan[1] ,dt)
    rY = rank(Y)
    r = rank(u)
    UtU = zeros(rY, r)
    C = zeros(rY, r)
    UQ = zeros(size(Y,1), r)
    return AlternatingProjection_Cache(F, UtU, UQ, C, alg_params.orth_alg)
end

function step!(integrator::DLRIntegrator, alg::DOAlgorithm{AlternatingProjection{Primal}}, dt)
    @unpack u, t, iter, cache = integrator
    primal_alternating_projection_do_step!(u, cache, t, dt)
    integrator.t += dt
    integrator.iter += 1
end

function step!(integrator::DLRIntegrator, alg::DOAlgorithm{AlternatingProjection{Dual}}, dt)
    @unpack u, t, iter, cache = integrator
    dual_alternating_projection_do_step!(u, cache, t, dt)
    integrator.t += dt
    integrator.iter += 1
end

function step!(integrator::DLRIntegrator, alg::DOAlgorithm{AlternatingProjection{PrimalDual}}, dt)
    @unpack u, t, iter, cache = integrator
    primal_alternating_projection_do_step!(u, cache, t, dt/2)
    dual_alternating_projection_do_step!(u, cache, t + dt/2, dt/2)
    integrator.t += dt
    integrator.iter += 1
end

function primal_alternating_projection_do_step!(u, cache, t, dt)
    @unpack F, UtU, UQ, C, orth_alg = cache
    Y = u + dt*F(u,t,dt) # investigate using the integrator interface of DifferentialEquations.jl
    orthonormalize!(Y, orth_alg)

    # coefficient update 
    mul!(UtU, Y.U', u.U)
    mul!(u.Z, Y.Z, UtU)
    
    # basis update (procrustes problem)
    mul!(C, Y.Z', u.Z)
    Q, _, P = svd!(C)
    mul!(UQ, Y.U, Q) 
    mul!(u.U, UQ, P')
end

function dual_alternating_projection_do_step!(u, cache, t, dt)
    @unpack F, UtU, UQ, C, orth_alg = cache
    Y = u + dt*F(u,t,dt) # investigate using the integrator interface of DifferentialEquations.jl
    orthonormalize!(Y, orth_alg)

    # basis update (procrustes problem)
    mul!(C, Y.Z', u.Z)
    Q, _, P = svd!(C)
    mul!(UQ, Y.U, Q) 
    mul!(u.U, UQ, P')

    # coefficient update 
    mul!(UtU, Y.U', u.U)
    mul!(u.Z, Y.Z, UtU)
end

#=
SVD => (svd, tsvd, gradient flow)
    LRA_{n+1} = LRA_{n} + dt*F(LRA_{n},t)
    LRA_{n+1} = truncated_svd(LRA_{n+1}, rmax = rank(LRA_{n+1}))
    or 
    LRA_{n+1} = truncated_svd(LRA_{n+1}, tol = #tolerance, rmax = #max rank)
=#

struct TruncatedSVD_Params
    F
    SVD_alg
    orth_alg
    tol
    rmin 
    rmax 
end

struct TruncatedSVD_Cache
    F
end

function TruncatedSVD(F; SVD_alg = SVDFact(), orthonormalization = QRFact(), 
                         rmin::Int=1, rmax::Int=Int(1e8), tol = sqrt(eps(Float64)))
    return DOAlgorithm(TruncatedSVD(), TruncatedSVD_Params(F, SVD_alg, orthonormalization, tol, rmin, rmax))
end

function alg_cache(prob::MatrixDEProblem, alg::DOAlgorithm{TruncatedSVD}, u, dt)
    @unpack alg_params = alg
    F = (u,t,dt) -> alg_params.F(prob.f, u, t, dt)
    return TruncatedSVD_Cache(F)
end

function step!(integrator::DLRIntegrator, alg::DOAlgorithm{TruncatedSVD}, dt)
    @unpack u, t, iter, cache = integrator
    truncated_svd_do_step!(u, cache, t, dt, alg.alg_params)
    integrator.t += dt
    integrator.iter += 1
end

function truncated_svd_do_step!(u, cache, t, dt, params)
    @unpack F = cache
    @unpack tol, rmin, rmax, SVD_alg, orth_alg = params
    Y = u + dt*F(u, t, dt)
    retract!(Y, u, SVD_alg, tol, rmin, rmax, orth_alg)
end

function retract!(Y, u, alg::T, tol, rmin, rmax, orth_alg) where T <: Union{SVDFact, TSVD}
    Y = round(Y, alg, tol=tol, rmin = rmin, rmax = rmax, alg_orthonormalize = orth_alg)
    u.Z = Y.Z
    u.U = Y.U
end

function retract!(Y, u, alg::GradientDescent, tol, rmin, rmax, orth_alg)
    gd_truncated_svd!(u, Y, alg, orth_alg)
end
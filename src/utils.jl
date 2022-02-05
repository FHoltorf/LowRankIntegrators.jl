full(LRA::SVDLikeApproximation) = LRA.U*LRA.S*LRA.V'
full(LRA::LowRankApproximation) = LRA.U*LRA.Z'

mutable struct MatrixDataIntegrator{yType, uType, lType, rType}
    Δy::yType
    u::uType
    left_factor::lType
    right_factor::rType 
    sign::Int
end

function step!(integrator::MatrixDataIntegrator, dt, ::Bool=true)
    step!(integrator)
end

function step!(integrator::MatrixDataIntegrator)
    @unpack Δy, u, left_factor, right_factor, sign = integrator
    u .+= sign*left_factor'*Δy*right_factor
end

function set_u!(integrator::MatrixDataIntegrator, unew)
    integrator.u .= unew
end

function update_sol!(integrator::AbstractDLRIntegrator, dt)
    if integrator.iter <= length(integrator.sol.Y) - 1
        integrator.sol.Y[integrator.iter + 1] = deepcopy(integrator.u)
        integrator.sol.t[integrator.iter + 1] = integrator.t
    else
        push!(integrator.sol.Y, deepcopy(integrator.u))
        push!(integrator.sol.t, integrator.t)
    end
end
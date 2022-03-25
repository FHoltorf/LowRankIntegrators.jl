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


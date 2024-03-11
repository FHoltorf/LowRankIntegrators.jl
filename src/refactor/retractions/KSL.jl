struct KSLRetraction <: ExtendedLowRankRetraction end
struct LSKRetraction <: ExtendedLowRankRetraction end

@concrete mutable struct KSLCache <: LowRankRetractionCache
    K0
    L0
    dK
    dK_
    dL
    dL_
    dS
    dS_
    M
    N
end

function initialize_cache(X, dX, R::KSLRetraction)
    r = rank(X)
    r_ = rank(dX)
    n,m = size(X)
    K0 = zeros(n, r)
    L0 = zeros(m, r)
    dK = similar(K0)
    dK_ = zeros(n, r_) 
    dL = similar(L0)
    dL_ = zeros(m, r_) 
    dS = similar(X.S)
    M = zeros(r, r_) 
    N = similar(M)
    dS_ = similar(M)
    KSLCache(K0, L0, dK, dK_, dL, dL_, dS, dS_, M, N)
end
initialize_cache(X, dX, ::LSKRetraction) = initialize_cache(X, dX, KSLRetraction())

# retraction 
function K_step!(X::SVDLikeRepresentation, dX)
    K = X.U*X.S
    dK = Matrix(dX*X.V)
    K .+= dK
    QRK = qr!(K) 
    X.U .= Matrix(QRK.Q) 
    X.S .= QRK.R
end
function K_step!(cache::EulerCache, dX)
    @unpack X, retraction_cache = cache
    @unpack K0, dK, dK_, N = retraction_cache
    
    mul!(K0, X.U, X.S)
    mul!(dK_, dX.U, dX.S)
    mul!(N, X.V', dX.V)
    mul!(dK, dK_, N')

    K0 .+= dK
    QRK = qr!(K0) 
    X.U .= Matrix(QRK.Q) 
    X.S .= QRK.R
end

function S_step_KSL!(cache::EulerCache, dX)
    @unpack X, retraction_cache = cache
    @unpack dS, dS_, M, N = retraction_cache
    
    mul!(M, X.U', dX.U)
    mul!(dS_, M, dX.S)
    mul!(dS, dS_, N')

    X.S .-= dS
end
function S_step_LSK!(cache::EulerCache, dX)
    @unpack X, retraction_cache = cache
    @unpack dS, dS_, M, N = retraction_cache
    
    mul!(N, X.V', dX.V)
    mul!(dS_, M, dX.S)
    mul!(dS, dS_, N')

    X.S .-= dS
end
function S_step!(X::SVDLikeRepresentation, dX)
    dS = Matrix(X.U'*dX*X.V) 
    X.S .-= dS
end

function L_step!(cache::EulerCache, dX)
    @unpack X, retraction_cache = cache
    @unpack L0, dL, dL_, M = retraction_cache
    mul!(L0, X.V, X.S')
    mul!(dL_, dX.V, dX.S')
    mul!(M, X.U', dX.U)
    mul!(dL, dL_, M')
    L0 .+= dL
    QRL = qr!(L0) 
    X.V .= Matrix(QRL.Q) 
    X.S .= QRL.R'
end
function L_step!(X::SVDLikeRepresentation, dX)
    L = X.V*X.S'
    dL = Matrix(dX'*X.U)
    L .+= dL
    QRL = qr!(L) 
    X.V .= Matrix(QRL.Q) 
    X.S .= QRL.R'
end

# retractions
function retract(X, dX, ::KSLRetraction)
    Xnew = deepcopy(X)
    K_step!(Xnew,dX)
    S_step!(Xnew,dX)
    L_step!(Xnew,dX)
    return Xnew
end
function retract(X, dX, ::LSKRetraction)
    Xnew = deepcopy(X)
    K_step!(Xnew,dX)
    S_step!(Xnew,dX)
    L_step!(Xnew,dX)
    return Xnew
end

function retract!(cache, dX, ::KSLRetraction)
    K_step!(cache,dX)
    S_step_KSL!(cache,dX)
    L_step!(cache,dX)
end
function retract!(cache, dX, ::LSKRetraction)
    L_step!(cache,dX)
    S_step_LSK!(cache,dX)
    K_step!(cache,dX)
end

function update_retraction_cache!(cache::KSLCache, SA)
    nothing 
    # need to implement when we consider updating of interpolation rank
    # need to update dK_, etc. in that case
end
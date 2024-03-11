struct KLSRetraction <: ExtendedLowRankRetraction end

@concrete struct KLSCache <: LowRankRetractionCache
    K0
    L0
    U1
    S1
    V1
    dK
    dK_
    dL
    dL_
    dS
    dS_
    M
    N
    M_
    N_
end

function initialize_cache(X::SVDLikeRepresentation, dX, ::KLSRetraction)
    r = rank(X)
    r_ = rank(dX)
    n, m = size(X)
    K0 = zeros(n, r)
    L0 = zeros(m, r)
    S1 = similar(X.S)
    U1 = similar(X.U)
    V1 = similar(X.V)
    dK = similar(K0)
    dK_ = zeros(n, r_)
    dL = similar(L0)
    dL_ = zeros(m, r_)
    M = similar(X.S)
    N = similar(X.S)
    M_ = zeros(r, r_)
    N_ = similar(M_)
    dS = similar(S1)
    dS_ = similar(M_)
    KLSCache(K0, L0, U1, S1, V1, dK, dK_, dL, dL_, dS, dS_, M, N, M_, N_)
end

# needs to be adapted if rank adaptation is considered 
function retract(X, dX, ::KLSRetraction)
    #K-step
    K = X.U*X.S
    K .+= Matrix(dX*X.V)
    QRK = qr!(K) 
    U1 = Matrix(QRK.Q) 
    M = U1'*X.U
   
    #L-step
    L = X.V*X.S'
    L .+= Matrix(dX'*X.U)
    QRL = qr!(L) 
    V1 = Matrix(QRL.Q) 
    N = V1'*X.V

    #S-step
    S1 = M*X.S*N'
    S1 += Matrix(U1'*dX*V1)
    return SVDLikeRepresentation(U1, S1, V1)
end

function retract!(cache, dX, ::KLSRetraction)
    @unpack X, retraction_cache = cache
    @unpack K0, dK, dK_ = retraction_cache
    @unpack L0, dL, dL_ = retraction_cache
    @unpack dS, dS, dS_ = retraction_cache
    @unpack M, N, M_, N_ = retraction_cache
    @unpack U1, V1, S1 = retraction_cache

    #K-step
    mul!(K0, X.U, X.S)
    mul!(dK_, dX.U, dX.S)
    mul!(N_, X.V', dX.V)
    mul!(dK, dK_, N_')
    K0 .+= dK
    QRK = qr!(K0) 
    U1 .= Matrix(QRK.Q) 
    mul!(M, U1', X.U)
   
    #L-step 
    mul!(L0, X.V, X.S')
    mul!(dL_, dX.V, dX.S')
    mul!(M_, X.U', dX.U)
    mul!(dL, dL_, M_')
    L0 .+= dL
    QRL = qr!(L0) 
    V1 .= Matrix(QRL.Q) 
    mul!(N, V1',X.V)
    
    #S-step
    mul!(S1, X.S, N')
    mul!(X.S, M, S1)
    mul!(dS_, M_, dX.S)
    mul!(dS, dS_, N_')

    # Update
    X.S .+= dS
    X.V .= V1
    X.U .= U1
end

function update_retraction_cache!(cache::KLSCache, SA)
    nothing 
    # need to implement when we consider updating of interpolation rank
    # need to update dK_, etc. in that case
end
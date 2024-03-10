@concrete struct ButcherTableau 
    a
    b 
    c
    s
end

struct PRK{RType <: ExtendedLowRankRetraction, T <: ButcherTableau}
    R::RType
    tableau::T
end

@concrete struct PRKCache
    X
    ks # low rank factorizations => s times
    ηs # intermediate points => (s-1) times (the first one is X)
    slope_range # n x 2*s*r
    slope_corange # m x 2*s*r
end
function initialize_cache(prob, PRK, SA)
    X = deepcopy(prob.X0)
    s = PRK.tableau.s
    n, m = size(X)
    r = rank(X)
    slope_range = zeros(n, 2*s*r)
    slope_corange = zeros(m, 2*s*r)
    ks = [deepcopy(X) for i in 1:s]
    ηs = [deepcopy(X) for i in 1:s]
    return PRKCache(X, ks, ηs, slope_range, slope_corange)
end
# pseudocode
function retracted_step!(cache::PRKCache, model::AbstractLowRankModel, t, h, prk::PRK, SA)
    @unpack R, tableau = PRK
    @unpack a, b, c, s = tableau
    @unpack X, ks, ηs, slope_range, slope_corange = cache

    r = rank(X)
    ηs[1] = X
    ks[1] =  evaluate_tangent_model(X, model, t + c[1]*h, SA) 
    @views slope_range[:,1:2r] .= ks[1].U # maybe use block arrays instead
    @views slope_corange[:,1:2r] .= ks[1].V # maybe use block arrays instead

    for i in 2:s
        # compute intermediate slope dη
        slope_core = BlockDiagonal([(h*a[i][j]) * ks[j].S for j in 1:i-1])
        @views dη = SVDLikeRepresentation(slope_range[:,1:2*(i-1)*r], 
                                          slope_core, 
                                          slope_corange[:,1:2*(i-1)*r])

        # compute intermediate step
        # make this cached in the long term (should be straightforward)
        ηs[i] = retract(X, dη, R)
        
        # compute new slope 
        # make this cached in the long term (slightly tricky because the rank may not be known apriori)
        ks[i] = evaluate_tangent_model(X, model, t + c[i]*h, SA) 
        @views slope_range[:,2(i-1)r+1:2*i*r] .= ks[i].U
        @views slope_corange[:,2(i-1)r+1:2*i*r] .= ks[i].V
    end
    # final step 
    slope_core = BlockDiagonal([(h*b[j]) * ks[j].S for j in 1:s])
    dX = SVDLikeRepresentation(slope_range, slope_core, slope_corange)
    Xnew = retract(X, dX, R)
    
    # update
    copyto!(X.U, Xnew.U)
    copyto!(X.S, Xnew.S)
    copyto!(X.V, Xnew.V)
end


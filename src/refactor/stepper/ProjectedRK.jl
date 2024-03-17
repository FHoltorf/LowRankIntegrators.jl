@concrete struct ButcherTableau 
    a
    b 
    c
    s
end

function ButcherTableau(a::Vector{Vector{T}},b::Vector{A},c::Vector{B}) where {T <: Number, A <: Number, B <: Number}
    s = length(a) + 1
    @assert sum(b) ≈ 1 "the step sizes must add to one"
    @assert length(c) == length(b) == s "length of a, b, and c is inconsistent for $s stage RK method"
    @assert c[end] <= 1 "c[end] > 1"
    return ButcherTableau(a,b,c,s)
end

struct ProjectedRK{RType <: ExtendedLowRankRetraction, T <: ButcherTableau} <: LowRankStepper
    R::RType
    tableau::T
end

@concrete struct PRKCache <: LowRankStepperCache
    X
    ks # low rank factorizations => s times
    ηs # intermediate points => (s-1) times (the first one is X)
    nz_stageweights
    stage_ranges 
    slope_range # n x 2*s*r
    slope_corange # m x 2*s*r
end
state(cache::PRKCache) = cache.X

function initialize_cache(prob, PRK::ProjectedRK, SA)
    X = deepcopy(prob.X0)
    s = PRK.tableau.s
    n, m = size(X)
    r = rank(X)
    slope_range = zeros(eltype(X), n, 2*s*r)
    slope_corange = zeros(eltype(X), m, 2*s*r)
    nz_stageweights = [[i for (i, a) in enumerate(stage) if a != 0.0] for stage in PRK.tableau.a]
    stage_ranges = [reduce(vcat, (l-1)*2r+1:l*2r for l in stages) for stages in nz_stageweights]
    ks = [deepcopy(X) for i in 1:s]
    ηs = [deepcopy(X) for i in 1:s]
    return PRKCache(X, ks, ηs, nz_stageweights, stage_ranges, slope_range, slope_corange)
end
# pseudocode
function retracted_step!(cache::PRKCache, model::AbstractLowRankModel, t, h, PRK::ProjectedRK, SA)
    @unpack R, tableau = PRK
    @unpack a, b, c, s = tableau
    @unpack X, ks, ηs, slope_range, slope_corange, stage_ranges, nz_stageweights = cache

    r = rank(X)
    ηs[1] = X
    ks[1] = evaluate_tangent_model(X, model, t + c[1]*h, SA) 
    @views slope_range[:,1:2r] .= ks[1].U # maybe use block arrays instead
    @views slope_corange[:,1:2r] .= ks[1].V # maybe use block arrays instead

    for i in 2:s
        # compute intermediate slope dη
        slope_core = BlockDiagonal([(h*a[i-1][j]) * ks[j].S for j in nz_stageweights[i-1]])
        #println(stage_ranges[i-1])
        #println(i)
        dη = SVDLikeRepresentation(slope_range[:,stage_ranges[i-1]], 
                                   slope_core, 
                                   slope_corange[:,stage_ranges[i-1]])

        # compute intermediate step
        # make this cached in the long term (should be straightforward)
        ηs[i] = retract(X, dη, R)
        
        # compute new slope 
        # make this cached in the long term (slightly tricky because the rank may not be known apriori)
        ks[i] = evaluate_tangent_model(ηs[i], model, t + c[i]*h, SA) 
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


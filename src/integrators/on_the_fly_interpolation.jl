rank_DEIM(::AbstractDLRAlgorithm_Cache) = -1
interpolation_indices(::AbstractDLRAlgorithm_Cache) = (Int[], Int[])

struct OnTheFlyInterpolation_Cache
    params::SparseInterpolation
    u_prev::SVDLikeRepresentation
    Π::SparseFunctionInterpolator
    Π_K::SparseFunctionInterpolator
    Π_L::SparseFunctionInterpolator
    Π_S::SparseFunctionInterpolator
end

function select_idcs(F_rows::AbstractMatrix, F_cols::AbstractMatrix, selection_alg::IndexSelectionAlgorithm)
    range = svd(F_cols)
    corange = svd(F_rows)
    S = max.(range.S, corange.S)
    row_idcs = index_selection(range.U, S, selection_alg)
    col_idcs = index_selection(Matrix(corange.V), S, selection_alg)
    return range.U, corange.V, row_idcs, col_idcs
end

function select_idcs(F::SVDLikeRepresentation, selection_alg::IndexSelectionAlgorithm)
    F_svd = svd(F)
    row_idcs = index_selection(F_svd.U, diag(F_svd.S), selection_alg)
    col_idcs = index_selection(F_svd.V, diag(F_svd.S), selection_alg)
    UF = F_svd.U[:, 1:length(row_idcs)]
    VF = F_trunc.V[:, 1:length(col_idcs)]
    return UF, VF, row_idcs, col_idcs
end

function sparse_selection(interpolation_cache, u, t, dt)
    @unpack params = interpolation_cache
    if params.update_scheme == :avg_flow
        @unpack u_prev = interpolation_cache
        @unpack selection_alg = params
        return select_idcs(u - u_prev, selection_alg)
    elseif params.update_scheme == :last_iterate
        @unpack Π_K, Π_L = interpolation_cache
        @unpack selection_alg = params
        eval!(Π_L, u, (), t+dt)
        eval!(Π_K, u, (), t+dt)
        UF, VF, row_idcs, col_idcs = select_idcs(Π_L.cache.rows, Π_K.cache.cols, selection_alg)
        return UF, VF, row_idcs, col_idcs
    else 
        error("Update scheme $(params.update_scheme) not supported")
    end
    
    # eval index selection
    
end

function compute_weights(range, corange, row_idcs, col_idcs, u, t, dt, interpolation_cache)
    @unpack params = interpolation_cache
    if size(range, 2) != length(row_idcs)
        @unpack Π = interpolation_cache
        cols = Matrix{eltype(Π.cache.cols)}(undef, Π.F.n_rows, length(col_idcs))
        eval_cols!(cols, Π, u, (), t+dt, col_idcs)
        UF = svd(cols).U
        @views range_weights = UF/UF[row_idcs,:] 
    else
        @views range_weights = range/range[row_idcs,:]
    end

    if size(corange, 2) != length(col_idcs)
        @unpack Π = interpolation_cache
        rows = Matrix{eltype(Π.cache.cols)}(undef, length(row_idcs), Π.F.n_cols)
        eval_rows!(rows, Π, u, (), t+dt, row_idcs)
        VF = svd(rows).V
        @views corange_weights = VF/VF[col_idcs,:]
    else
        @views corange_weights = corange/corange[col_idcs,:]
    end
    return range_weights, corange_weights
end

function update_interpolation!(interpolation_cache, u, t, dt)
    @unpack Π, Π_L, Π_K, Π_S, params = interpolation_cache 
    @unpack selection_alg = params

    # eval range/corange spans
    UF, VF, row_idcs, col_idcs = sparse_selection(interpolation_cache, u, t, dt)
    range_weights, corange_weights = compute_weights(UF, VF, row_idcs, col_idcs, u, t, dt, interpolation_cache)

    # new interpolators
    range = DEIMInterpolator(row_idcs, range_weights)
    projected_range = DEIMInterpolator(row_idcs, u.U'*range_weights)
    corange = DEIMInterpolator(col_idcs, corange_weights)
    projected_corange = DEIMInterpolator(col_idcs, u.V'*corange_weights)

    # update function interpolators
    update_interpolator!(Π_L, projected_range)
    update_interpolator!(Π_K, projected_corange')
    update_interpolator!(Π_S, SparseMatrixInterpolator(projected_range, projected_corange))
    update_interpolator!(Π, SparseMatrixInterpolator(range, corange))
end

# without rank adaptation
#=
function update_interpolation!(interpolation_cache, u, t)
    @unpack params, Π, Π_K, Π_L, Π_S = interpolation_cache
    @unpack selection_alg, tol, rmin, rmax = params

    eval!(Π_L, u, (), t) # rows
    eval!(Π_K, u, (), t) # columns
    VF = truncated_svd(Π_L.cache.rows, tol=tol, rmin=rmin, rmax=rmax).V # corange from rows
    UF = truncated_svd(Π_K.cache.cols, tol=tol, rmin=rmin, rmax=rmax).U # range from cols

    # find interpolation indices
    row_idcs = index_selection(range_svd.U, selection_alg)
    col_idcs = index_selection(corange_svd.V, selection_alg)
    
    # find interpolation weights
    @views range_weights = UF/UF[row_idcs,:]
    @views corange_weights = VF/VF[col_idcs,:]
    
    # new interpolators
    range = DEIMInterpolator(row_idcs, range_weights)
    projected_range = DEIMInterpolator(row_idcs, u.U'*range_weights)
    corange = DEIMInterpolator(col_idcs, corange_weights)
    projected_corange = DEIMInterpolator(col_idcs, u.V'*corange_weights)

    # update function interpolators
    update_interpolator!(Π_L, projected_range)
    update_interpolator!(Π_K, projected_corange')
    update_interpolator!(Π_S, SparseMatrixInterpolator(projected_range, projected_corange))
    update_interpolator!(Π, SparseMatrixInterpolator(range, corange))
end

# alternative but max rank limited by 2*rank(u)
# ... not great
function update_interpolation!(interpolation_cache, u_old, u, dt)
    @unpack params, Π, Π_K, Π_L, Π_S = interpolation_cache
    @unpack selection_alg, tol, rmin, rmax = params

    dF = u - u_old
    dF_svd = svd(dF)
    #println(typeof(dF_svd.S))
    cutoff = findfirst(s -> s < tol/dt, diag(dF_svd.S) )
    if isnothing(cutoff)
        # compute the full function and take derivative 
        cutoff= size(dF_svd.S,1)
    else
        println(dF_svd.S[cutoff])
        cutoff = max(rmin, min(cutoff, rmax))
    end

    if cutoff != rank(Π_K)
        println("new DEIM rank: $cutoff")
    end

    # find interpolation indices
    @views row_idcs = index_selection(dF_svd.U[:,1:cutoff], selection_alg)
    @views col_idcs = index_selection(dF_svd.V[:,1:cutoff], selection_alg)
    
    # find interpolation weights
    @views range_weights = dF_svd.U/dF_svd.U[row_idcs,:]
    @views corange_weights = dF_svd.V/dF_svd.V[col_idcs,:]
    
    # new interpolators
    range = DEIMInterpolator(row_idcs, range_weights)
    projected_range = DEIMInterpolator(row_idcs, u.U'*range_weights)
    corange = DEIMInterpolator(col_idcs, corange_weights)
    projected_corange = DEIMInterpolator(col_idcs, u.V'*corange_weights)

    # update function interpolators
    update_interpolator!(Π_L, projected_range)
    update_interpolator!(Π_K, projected_corange')
    update_interpolator!(Π_S, SparseMatrixInterpolator(projected_range, projected_corange))
    update_interpolator!(Π, SparseMatrixInterpolator(range, corange))
end
=#
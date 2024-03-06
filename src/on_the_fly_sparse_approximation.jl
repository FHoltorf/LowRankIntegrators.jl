

function select_idcs(F_rows::AbstractMatrix, F_cols::AbstractMatrix, selection_alg::IndexSelectionAlgorithm)
    range = svd(F_cols)
    corange = svd(F_rows)
    row_idcs = index_selection(range.U, range.S, selection_alg)
    col_idcs = index_selection(Matrix(corange.V), corange.S, selection_alg)
    return range.U, corange.V, row_idcs, col_idcs
end

function select_idcs(F::SVDLikeRepresentation, selection_alg::IndexSelectionAlgorithm)
    F_svd = svd(F)
    row_idcs = index_selection(F_svd.U, diag(F_svd.S), selection_alg)
    col_idcs = index_selection(F_svd.V, diag(F_svd.S), selection_alg)
    UF = F_svd.U[:, 1:length(row_idcs)]
    VF = F_svd.V[:, 1:length(col_idcs)]
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

    elseif params.update_scheme = :k_means
        @unpack Π_K, Π_L = interpolation_cache
        @unpack selection_alg, cluster_cache = params
        @unpack row_centers, col_centers = cluster_cache
        @unpack row_means, col_means = cluster_cache
        @unpack projected_cols, projected_rows = cluster_cache
        @unpack n_clusters = cluster_cache
        
        mul!(projected_cols, u.S, u.V')
        mul!(projected_rows, u.S', u.U)
        kmeans!(projected_cols, col_means)
        kmeans!(projected_rows, row_means)
        
        for i in 1:n_clusters
            @views row_centers[i] = argmin(i -> norm(projected_rows[:,i] - row_means[i]), 
                                                      axes(projected_rows,2))
            @views col_centers[i] = argmin(i -> norm(projected_cols[:,i] - col_means[i]),
                                                      axes(projected_cols,2))
        end
        
        Π_L.F.rows!(Π_L.cache.rows, x, p, t, row_centers)
        Π_L.F.cols!(Π_L.cache.cols, x, p, t, col_centers)
        
        UF, VF, row_idcs, col_idcs = select_idcs(Π_L.cache.rows, Π_K.cache.cols, selection_alg)
        return  UF, VF, row_idcs, col_idcs
    else 
        error("Update scheme $(params.update_scheme) not supported")
    end
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

using LowRankIntegrators, LinearAlgebra

@testset "DLRA + on-the-fly DEIM for Burgers' equation" begin
    # Model parameters
    N = 1024 # spatial discretization
    S = 256 # number of samples 
    σ_t = 0.01 # 0.01 # noise level in forcing
    σ_x = 0.005 # noise level in initial condition
    ν = 0.0025 #0.0025 # viscosity 
    d = 4 # dimension of uncertainty 

    # the rest is computed
    x_range = range(0+1/(N+1), 1-1/(N+1), length=N) # spatial grid
    Δx = step(x_range) # step size
    ξ = [randn(d) for i in 1:S-1] # random samples 
    pushfirst!(ξ, zeros(d)) # add mean 
    ξ_mat = reduce(hcat, ξ)

    # for initial condition uncertainty:
    gauss_kernel(x,z,σ) = exp(-(x-z)^2/(2*σ^2))
    M = [gauss_kernel(x,z,σ_x) for x in x_range, z in x_range]
    Λ, Ψ = eigen(M, sortby=-)
    x0_mean = @. 0.5*sin(2π*x_range) * (exp(cos(2π*x_range)) - 1.5)
    X0_noise = SVDLikeRepresentation(Ψ[:,1:d], diagm(sqrt.(Λ[1:d])), ξ_mat')
    X0 = TwoFactorRepresentation(x0_mean, ones(S)) + X0_noise

    @inline function D²(xleft, xmid, xright, Δx)
        return (xleft + xright - 2*xmid)/Δx^2
    end

    @inline function D(xleft, xmid, Δx)
        return (xmid - xleft)/Δx
    end

    function burgers_col!(dx, x, p, t, col)
        Δx, ν, ξ, σ_t, d, n = p
        n = size(x,1)
        x_bc = x_left(t, ξ[col], σ_t, d)
        dx[1] = ν*D²(x_bc, x[1], x[2], Δx) - 
                x[1]*D(x_bc, x[1], Δx)
        for i in 2:n-1
            dx[i] = ν*D²(x[i-1], x[i], x[i+1], Δx) - 
                    x[i]*D(x[i-1], x[i], Δx)
        end
        dx[n] = ν*D²(x[n-1], x[n], 0, Δx) - 
                x[n]*D(x[n-1], x[n], Δx)
    end

    function burgers_cols!(dx, x::TwoFactorRepresentation, p, t, cols)
        core_cache, col_cache, params = p
        # cannot use threads ... or otherwise too much allocation
        @views for (i,col) in enumerate(cols)
            mul!(col_cache, x.U, x.Z[col,:])
            burgers_col!(dx[:,i], col_cache, params, t, col) #col_cache, params, t, col)
        end
    end

    function burgers_cols!(dx, x::SVDLikeRepresentation, p, t, cols)
        core_cache, col_cache, params = p
        mul!(core_cache, x.U, x.S)
        @views for (i,col) in enumerate(cols)
            mul!(col_cache, core_cache, x.V[col,:])
            @views burgers_col!(dx[:,i], col_cache, params, t, col)
        end
    end

    function burgers_row!(dx, x, p, t, row)
        Δx, ν, ξ, σ_t, d, n = p 
        if row == 1
            for j in eachindex(dx)
                x_bc = x_left(t, ξ[j], σ_t, d)
                dx[j] = ν*D²(x_bc, x[1,j], x[2,j], Δx) - x[1,j]*D(x_bc, x[1,j], Δx)
            end
        elseif row == n
            for j in eachindex(dx)
                dx[j] = ν*D²(x[1,j], x[2,j], 0, Δx) - x[2,j]*D(x[1,j], x[2,j], Δx)
            end
        else
            for j in eachindex(dx)
                dx[j] = ν*D²(x[1,j], x[2,j], x[3,j], Δx) - x[2,j]*D(x[1,j], x[2,j], Δx)
            end
        end
    end

    function burgers_rows!(dx, x::TwoFactorRepresentation, p, t, rows)
        row_cache, col_cache, params = p
        n = params[end]
        @views for (i,row) in enumerate(rows)
            if row == 1
                mul!(row_cache[1:2,:], x.U[1:2,:], x.Z')
            elseif row == n
                mul!(row_cache[1:2,:], x.U[n-1:n,:], x.Z')
            else
                mul!(row_cache, x.U[row-1:row+1,:], x.Z')
            end
            burgers_row!(dx[i,:], row_cache, params, t, row)
        end
    end

    function burgers_rows!(dx, x::SVDLikeRepresentation, p, t, rows)
        row_cache, col_cache, params = p
        n = params[end]
        mul!(col_cache, x.S, x.V')
        @views for (i,row) in enumerate(rows)
            if row == 1
                mul!(row_cache[1:2,:], x.U[1:2,:], col_cache)
            elseif row == n
                mul!(row_cache[1:2,:], x.U[n-1:n,:], col_cache)
            else
                mul!(row_cache, x.U[row-1:row+1,:], col_cache)
            end
            burgers_row!(dx[i,:], row_cache, params, t, row)
        end
    end

    function burgers_elements!(dx, x::SVDLikeRepresentation, p, t, rows, cols)
        col_cache, row_cache, params = p
        n = params[end]
        @views for  (j, col) in enumerate(cols)
            mul!(col_cache, x.S, x.V[col,:])
            for (i, row) in enumerate(rows)
                if row == 1
                    mul!(row_cache[1:2,:], x.U[1:2,:], col_cache)
                elseif row == n
                    mul!(row_cache[1:2,:], x.U[n-1:n,:], col_cache)
                else
                    mul!(row_cache, x.U[row-1:row+1,:], col_cache)
                end
                dx[i,j] = burgers_rhs(row_cache, params, t, row, col)
            end
        end
    end

    function burgers_elements!(dx, x::TwoFactorRepresentation, p, t, rows, cols)
        col_cache, row_cache, neighbor_cache, params = p
        @views for (i, row) in enumerate(rows), (j, col) in enumerate(cols)
            if row == 1
                mul!(row_cache[1:2,:], x.U[1:2,:], x.Z[col,:])
            elseif row == n
                mul!(row_cache[1:2,:], x.U[n-1:n,:], x.Z[col,:])
            else
                mul!(row_cache, x.U[row-1:row+1,:], x.Z[col,:])
            end
            dx[i,j] = burgers_rhs(row_cache, params, t, row, col)
        end
    end

    function burgers_rhs(x, p, t, row, col)
        Δx, ν, ξ, σ_t, d, n = p
        if row == 1
            x_bc = x_left(t, ξ[col], σ_t, d)
            return ν*D²(x_bc, x[1], x[2], Δx) - x[1]*D(x_bc, x[1], Δx)
        elseif row == n
            return ν*D²(x[1], x[2], 0, Δx) - x[2]*D(x[1], x[2], Δx)
        else
            return ν*D²(x[1], x[2], x[3], Δx) - x[2]*D(x[1], x[2], Δx)
        end
    end

    x_left(t, ξ, σ_t, d) = -0.4*sin(2π*t) + σ_t * sum(1/i^2*ξ[i]*sin(i*π*t) for i in 1:d)

    # solve parameters
    r = 5
    r_deim = 15
    params = (Δx, ν, ξ, σ_t, d, N)
    rows_cache = (zeros(3,S),zeros(r,S), params)
    cols_cache = (zeros(N,r), zeros(N), params)
    elements_cache = (zeros(r), zeros(3), params) 

    rows! = (dx, x, p, t, rows) -> burgers_rows!(dx, x, rows_cache, t, rows)
    cols! = (dx, x, p, t, cols) -> burgers_cols!(dx, x, cols_cache, t, cols)
    elements! = (dx, x, p, t, rows, cols) -> burgers_elements!(dx, x, elements_cache, t, rows, cols)
    burgers_components = ComponentFunction(rows!,cols!,elements!, N, S)

    X0_lr = truncated_svd(Matrix(X0),r)
    dX0 = zeros(N, S)
    @views for k in axes(dX0, 2)
        burgers_components.cols!(dX0[:,k], X0_lr, (), 0.0, [k])
    end
    
    dX0_lr = truncated_svd(dX0, r_deim)
    # first test projections 
    idx_selections = [DEIM(), QDEIM(), LDEIM()]
    init_range, init_corange = dX0_lr.U, dX0_lr.V
    U, V = X0_lr.U, X0_lr.V
    dX_test = similar(dX0)
    for alg in idx_selections
        row_idcs = index_selection(init_range, alg)
        col_idcs = index_selection(init_corange, alg)

        # interpolators
        Π_corange = DEIMInterpolator(col_idcs, init_corange/init_corange[col_idcs,:])
        Π_range = DEIMInterpolator(row_idcs, init_range/init_range[row_idcs,:])
        Π = SparseFunctionInterpolator(burgers_components, SparseMatrixInterpolator(Π_range, Π_corange))

        # core approximation
        dX_test = zeros(N,S)
        Π(dX_test, X0_lr, (), 0.0)
        @test norm(dX_test - dX0_lr)/norm(Matrix(dX0_lr)) < 1e-8

        # low rank integration subproblem rhs
        Π_K_corange = DEIMInterpolator(col_idcs,V'*Π.interpolator.corange.weights)'
        Π_K = SparseFunctionInterpolator(burgers_components, Π_K_corange)
        
        # K step 
        dK_test = zeros(N,r)
        K0 = X0_lr.U*X0_lr.S
        Π_K(dK_test, TwoFactorRepresentation(K0, X0_lr.V), (), 0.0)
        @test norm(dK_test - dX0_lr*V)/norm(Matrix(dX0_lr*V)) < 1e-8

        # L step
        Π_L_range = DEIMInterpolator(row_idcs, U'*Π.interpolator.range.weights)
        Π_L = SparseFunctionInterpolator(burgers_components, Π_L_range)
        dL_test = zeros(r,S)
        L0 = X0_lr.V*X0_lr.S'
        Π_L(dL_test, TwoFactorRepresentation(X0_lr.U,L0), (), 0.0)
        @test norm(dL_test - U'*dX0_lr)/norm(Matrix(U'*dX0_lr)) < 1e-8

        # S step
        Π_S_mat = SparseMatrixInterpolator(row_idcs, col_idcs, 
                                            U'*Π.interpolator.range.weights, 
                                            V'*Π.interpolator.corange.weights)
        Π_S = SparseFunctionInterpolator(burgers_components, Π_S_mat)
        dS_test = zeros(r,r)
        S0 = X0_lr.S
        Π_S(dS_test, X0_lr, (), 0.0)
        @test norm(dS_test - U'*dX0_lr*V)/norm(Matrix(U'*X0_lr*V)) < 1e-8
    end

    r_deim = 5
    interpolation = SparseInterpolation(DEIM(rmax=r_deim, rmin=r_deim, tol = 1.0, elasticity=1e-1), dX0_lr.U[:,1:r_deim], dX0_lr.V[:,1:r_deim])
    alg_deim = UnconventionalAlgorithm(interpolation) 

    deim_prob = MatrixDEProblem(burgers_components, X0_lr, (0.0,1.0))
    deim_sol = LowRankIntegrators.solve(deim_prob, alg_deim, 1e-3, save_increment=10)
end
# a few lil tests
@testset "Sparse Interpolation Tests" begin
    function rows!(dx,x,p,t,idcs)
        @views dx .= x[idcs, :]
    end
    function cols!(dx,x,p,t,idcs)
        @views dx .= x[:, idcs]
    end
    function elements!(dx,x,p,t,idcsA, idcsB)
        @views dx .= x[idcsA, idcsB]
    end

    F = ComponentFunction(rows!, cols!, elements!, 5, 8)
    A = Matrix(reshape(1.0:5*8, (5, 8)))

    row_idcs = [1, 5, 2]
    range_weights = zeros(eltype(A), 5, 3)
    range_weights[row_idcs,1:3] .= Matrix{eltype(A)}(I,3,3)

    col_idcs = [1, 8, 3, 6]
    corange_weights = zeros(eltype(A),8, 4)
    corange_weights[col_idcs,1:4] .= Matrix{eltype(A)}(I,4,4)

    Π_range = DEIMInterpolator(row_idcs,range_weights)
    Π_corange = DEIMInterpolator(col_idcs,corange_weights)
    Π_mat = SparseMatrixInterpolator(Π_range, Π_corange)

    @test all(Π_range(A)[row_idcs,:] .== A[row_idcs,:])
    @test all(Π_corange'(A)[:,col_idcs] .== A[:,col_idcs])

    F_range_int = SparseFunctionInterpolator(F, Π_range, output = eltype(A))
    F_corange_int = SparseFunctionInterpolator(F, Π_corange', output = eltype(A))
    F_int = SparseFunctionInterpolator(F, Π_mat, output = eltype(A))

    eval!(F_range_int, A, (), 0.0)
    @test all(F_range_int.cache.rows .== A[row_idcs,:])
    eval!(F_corange_int, A, (), 0.0)
    @test all(F_corange_int.cache.cols .== A[:,col_idcs])
    eval!(F_int, A, (), 0.0)
    @test all(F_int.cache.elements .== A[row_idcs,col_idcs])

    dA = similar(A)
    F_range_int(dA, A, (), 0.0)
    @test all(dA[row_idcs, :] .== A[row_idcs, :])
    F_corange_int(dA, A, (), 0.0)
    @test all(dA[:, col_idcs] .== A[:, col_idcs])
    F_int(dA, A, (), 0.0)
    @test all(dA[row_idcs, col_idcs] .== A[row_idcs, col_idcs])
end

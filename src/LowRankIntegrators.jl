module LowRankIntegrators
       using UnPack, LowRankArithmetic, ProgressMeter, ConcreteStructs, 
             LinearAlgebra, BlockDiagonals, DocStringExtensions

       # basic utils
       include("refactor/primitives.jl")
       include("refactor/model_evaluation.jl")
       include("refactor/utils.jl")
       # sparse approximation utils
       include("refactor/sparse_approximation.jl")
       include("refactor/defaults.jl")
       # steppers
       include("refactor/stepper/ProjectedEuler.jl")
       include("refactor/stepper/ProjectedRK.jl")
       include("refactor/stepper/KLSStepper.jl")
       include("refactor/stepper/KSLStepper.jl")
       include("refactor/stepper/BUG.jl")
       include("refactor/stepper/ProjectorSplitting.jl")
       include("refactor/stepper/RankAdaptiveBUG.jl")
       include("refactor/stepper/SparseInterpolationRK.jl")
       # retractions
       include("refactor/retractions/default.jl")
       include("refactor/retractions/KLS.jl")
       include("refactor/retractions/KSL.jl")
end
# using Reexport, UnPack, DocStringExtensions, LinearAlgebra, ProgressMeter, Clustering, ConcreteStructs
# import DifferentialEquations: step!, set_u!, init

# @reexport using DifferentialEquations, LowRankArithmetic

# include("utils.jl")
# export orthonormalize!, GradientDescent, QR, SVD, SecondMomentMatching, normal_component

# include("sparse_interpolation.jl")
# export SparseFunctionInterpolator, SparseMatrixInterpolator, 
#        DEIMInterpolator, AdjointDEIMInterpolator, 
#        DEIM, QDEIM, LDEIM,
#        ComponentFunction, SparseInterpolation,
#        update_interpolator!, eval!,
#        index_selection

# include("primitives.jl")
# export MatrixDEProblem, MatrixDataProblem, MatrixHybridProblem
#        DLRIntegrator, DLRSolution, 
#        solve

# include("integrators.jl")
# export ProjectorSplitting, PrimalLieTrotter, DualLieTrotter, Strang,
#        UnconventionalAlgorithm,
#        RankAdaptiveUnconventionalAlgorithm,
#        DOAlgorithm, DirectTimeMarching,
#        RankAdaptiveUnconventionalAlgorithm,
#        GreedyIntegrator,
#        step!, init 
# end

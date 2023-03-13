module LowRankIntegrators
using Reexport, UnPack, DocStringExtensions, LinearAlgebra
import DifferentialEquations: step!, set_u!, init

@reexport using DifferentialEquations, LowRankArithmetic

include("utils.jl")
export orthonormalize!, GradientDescent, QR, SVD, SecondMomentMatching, normal_component

include("sparse_interpolation.jl")
export SparseFunctionInterpolator, SparseMatrixInterpolator, 
       DEIMInterpolator, AdjointDEIMInterpolator, 
       DEIM, QDEIM, LDEIM,
       ComponentFunction, SparseInterpolation,
       update_interpolator!, eval!,
       index_selection

include("primitives.jl")
export MatrixDEProblem, MatrixDataProblem, MatrixHybridProblem
       DLRIntegrator, DLRSolution, 
       solve

include("integrators.jl")
export ProjectorSplitting, PrimalLieTrotter, DualLieTrotter, Strang,
       UnconventionalAlgorithm,
       RankAdaptiveUnconventionalAlgorithm,
       DOAlgorithm, DirectTimeMarching,
       RankAdaptiveUnconventionalAlgorithm,
       GreedyIntegrator,
       step!, init 
end

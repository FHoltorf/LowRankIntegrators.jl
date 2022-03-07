module LowRankIntegrators

using LinearAlgebra, DifferentialEquations, UnPack
import DifferentialEquations: step!, set_u!, init

include("LowRankArithmetic.jl")
export SVDLikeApproximation, TwoFactorApproximation, 
       truncated_svd,
       Matrix, rank, size

include("utils.jl")
export orthonormalize!, GradientDescent, QR, SVD, SecondMomentMatching, normal_component

include("primitives.jl")
export MatrixDEProblem, MatrixDataProblem, 
       DLRIntegrator, DLRSolution, 
       solve

include("integrators.jl")
export PrimalLieTrotterProjectorSplitting,
       DualLieTrotterProjectorSplitting, 
       StrangProjectorSplitting, 
       UnconventionalAlgorithm,
       RankAdaptiveUnconventionalAlgorithm,
       DOAlgorithm, DirectTimeMarching,
       RankAdaptiveUnconventionalAlgorithm,
       GreedyIntegrator,
       step!, init 
end

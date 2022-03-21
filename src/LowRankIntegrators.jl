module LowRankIntegrators
using Reexport, UnPack
import DifferentialEquations: step!, set_u!, init

@reexport using LinearAlgebra, DifferentialEquations, LowRankArithmetic

include("utils.jl")
export orthonormalize!, GradientDescent, QR, SVD, SecondMomentMatching, normal_component

include("primitives.jl")
export MatrixDEProblem, MatrixDataProblem, MatrixHybridProblem
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

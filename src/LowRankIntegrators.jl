module LowRankIntegrators

using LinearAlgebra, DifferentialEquations, UnPack
import DifferentialEquations: step!, set_u!, init

include("primitives.jl")
export MatrixDEProblem, MatrixDataProblem, 
       DLRIntegrator, DLRSolution, 
       solve

include("low_rank_algebra.jl")
export full,
       SVDLikeApproximation, LowRankApproximation, 
       elprod, elpow

include("integrators.jl")
export PrimalLieTrotterProjectorSplitting,
       DualLieTrotterProjectorSplitting, 
       StrangProjectorSplitting, 
       UnconventionalAlgorithm,
       DOAlgorithm, DirectTimeMarching,
       RankAdaptiveUnconventionalAlgorithm,
       step!, init 
end

module LowRankIntegrators

using LinearAlgebra, DifferentialEquations, UnPack
import DifferentialEquations: step!, set_u!, init

include("primitives.jl")
export MatrixDEProblem, MatrixDataProblem, 
       SVDLikeApproximation, LowRankApproximation, 
       DLRIntegrator, DLRSolution, 
       solve

include("utils.jl")
export full

include("integrators.jl")
export PrimalLieTrotterProjectorSplitting,
       DualLieTrotterProjectorSplitting, 
       StrangProjectorSplitting, 
       UnconventionalAlgorithm,
       step!, init 
end

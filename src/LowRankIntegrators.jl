module LowRankIntegrators

using LinearAlgebra, DifferentialEquations, UnPack
import DifferentialEquations: step!, set_u!, init

include("primitives.jl")
export MatrixDEProblem, MatrixDataProblem, LowRankApproximation, DLRIntegrator, DLRSolution, solve

include("integrators.jl")
export PrimalLieTrotterProjectorSplitting, DualLieTrotterProjectorSplitting, StrangProjectorSplitting, step!, init 

include("utils.jl")
export full

end

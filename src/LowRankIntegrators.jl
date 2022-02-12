module LowRankIntegrators

using LinearAlgebra, DifferentialEquations, UnPack
import DifferentialEquations: step!, set_u!, init

include("low_rank_arithmetic.jl")
export SVDLikeApproximation, TwoFactorApproximation, 
       Matrix, rank, size, *, +, -,
       elprod, elpow, add_to_cols, add_to_rows, add_scalar,
       multiply_cols, multiply_rows

include("utils.jl")
export orthonormalize!, GradientDescent, QR, SVD

include("primitives.jl")
export MatrixDEProblem, MatrixDataProblem, 
       DLRIntegrator, DLRSolution, 
       solve

include("integrators.jl")
export PrimalLieTrotterProjectorSplitting,
       DualLieTrotterProjectorSplitting, 
       StrangProjectorSplitting, 
       UnconventionalAlgorithm,
       DOAlgorithm, DirectTimeMarching,
       RankAdaptiveUnconventionalAlgorithm,
       step!, init 
end

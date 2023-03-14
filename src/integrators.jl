# basic catch all cases:
rank_DEIM(::AbstractDLRAlgorithm_Cache) = -1
interpolation_indices(::AbstractDLRAlgorithm_Cache) = (Int[], Int[])

struct OnTheFlyInterpolation_Cache
    params::SparseInterpolation
    u_prev::SVDLikeRepresentation
    Π::SparseFunctionInterpolator
    Π_K::SparseFunctionInterpolator
    Π_L::SparseFunctionInterpolator
    Π_S::SparseFunctionInterpolator
end

include("integrators/projector_splitting.jl")
include("integrators/unconventional.jl")
include("integrators/data_integrator.jl")
include("integrators/rank_adaptive_unconventional.jl")
include("integrators/greedy_integrator.jl")


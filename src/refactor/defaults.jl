
# oracles to be dispatched!
# for factored models
F(m::AbstractLowRankModel, X, t) = error("Please implement dispatch for F($(typeof(m)), X, t)")
F!(m::AbstractLowRankModel, dX, X, t) = error("Please implement dispatch for F!($(typeof(m)), dX, X, t)")
# for sparse approximations
# inplace
rows!(m::SparseLowRankModel{true}, drows, X, t, rows) = error("Please implement dispatch for rows!($(typeof(m)), drows, X, t, rows)")
columns!(m::SparseLowRankModel{true}, dcols, X, t, cols) = error("Please implement dispatch for columns!($(typeof(m)), dcols, X, t, cols)")
elements!(m::SparseLowRankModel{true}, dels, X, t, rows, cols) = error("Please implement dispatch for elements!($(typeof(m)), dels, X, t, rows, cols)")
# outofplace
rows(m::SparseLowRankModel{false}, X, t, rows) = error("Please implement dispatch for rows($(typeof(m)), X, t, rows)")
columns(m::SparseLowRankModel{false}, X, t, cols) = error("Please implement dispatch for columns($(typeof(m)), X, t, cols)")
elements(m::SparseLowRankModel{false}, X, t, rows, cols) = error("Please implement dispatch for elements($(typeof(m)), X, t, rows, cols)")

"""
    postprocessing will be applied after every step of the integrator by alterting the cache. 
    The result will be written to the solution AFTER postprocessing.
"""
postprocess_step!(cache, model, t) = nothing

update_sparse_approximation!(SA, model, cache, t, R) = nothing
update_cache!(::LowRankStepperCache, SA::SparseApproximation) = nothing
state(cache::LowRankStepperCache) = error("Please implement dispatch for state($(typeof(cache)))") 






not_found_alt(x, alt) = isnothing(x) ? alt : x
function rank_by_tol(S, r_min, r_max, atol, rtol)
    # truncation based on absolute tolerance
    r_atol = not_found_alt(findfirst(x -> x <= atol, S), length(S))

    # truncatio based on relative tolerance
    σtot = sum(S)
    r_rtol = not_found_alt(findfirst(x -> x/σtot <= rtol, S), length(S))
    
    r_new = max(r_atol, r_rtol)

    # clamp r_new at [r_min, r_max] 
    max(min(r_new, r_max), r_min)
end
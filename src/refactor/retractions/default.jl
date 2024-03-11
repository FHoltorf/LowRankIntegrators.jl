initialize_cache(X, ::LowRankRetraction) = missing  
retract(X,V,R::LowRankRetraction) = error("Please implement dispatch for `retract(X,V,$(typeof(R)))`")
retract!(cache,V,R::LowRankRetraction) = error("Please implement dispatch for `retract!($(typeof(cache)),V,$(typeof(R)))`")
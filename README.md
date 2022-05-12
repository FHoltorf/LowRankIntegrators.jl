# LowRankIntegrators.jl &emsp; <img align = center src = "docs/assets/lowrankintegrators_logo.png" alt = "logo" width = 150/>

[![Build Status](https://github.com/FHoltorf/LowRankIntegrators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/FHoltorf/LowRankIntegrators.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/FHoltorf/LowRankIntegrators.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/FHoltorf/LowRankIntegrators.jl)

This is a package for low-rank integration in Julia. Currently, it implements Lie-Trotter and Strang splitting based algorithms as proposed in [1] as well as the "unconventional integrator" proposed in [2]. Future work will include a rank-adaptive version of the unconventional integrator as described in [3]. The selection of algorithms was made to support only those that are robust to the presence of small singular values [4]. 

## Acknowledgements
This work is supported by NSF Award PHY-2028125 "SWQU: Composable Next Generation Software Framework for Space Weather Data Assimilation and Uncertainty Quantification".

## References
[1] Lubich, Christian, and Ivan V. Oseledets. "A projector-splitting integrator for dynamical low-rank approximation." BIT Numerical Mathematics 54.1 (2014): 171-188.

[2] Ceruti, Gianluca, and Christian Lubich. "An unconventional robust integrator for dynamical low-rank approximation." BIT Numerical Mathematics (2021): 1-22.

[3] Ceruti, Gianluca, Jonas Kusch, and Christian Lubich. "A rank-adaptive robust integrator for dynamical low-rank approximation." arXiv preprint arXiv:2104.05247 (2021).

[4] Kieri, Emil, Christian Lubich, and Hanna Walach. "Discretized dynamical low-rank approximation in the presence of small singular values." SIAM Journal on Numerical Analysis 54.2 (2016): 1020-1038.
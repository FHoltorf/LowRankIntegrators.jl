# LowRankIntegrators.jl &emsp; <img align = center src = "docs/assets/lowrankintegrators_logo.png" alt = "logo" width = 150/>

[![Build Status](https://github.com/FHoltorf/LowRankIntegrators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/FHoltorf/LowRankIntegrators.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/FHoltorf/LowRankIntegrators.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/FHoltorf/LowRankIntegrators.jl)

LowRankIntegrators.jl is a package for dynamical low rank approximation (DLRA) in Julia. DLRA can help you approximate the solution to (otherwise intractably) large matrix-valued ODEs. 

## Concept
Given a matrix-valued ODE, 

$$
    \frac{dX}{dt}(t) = F(X(t),t), \ X(0) = X_0, \text{ for all } t \in [t_0, t_f]
$$

with $X(t) \in \mathbb{R}^{n\times m}$ for all $t \in [t_0,t_f]$, DLRA seeks to identify a rank $r \ll \min(n,m)$ approximation $Y(t)$ to the true solution $X(t)$. This reduces memory requirements and under appropriate structural assumptions on the flow map $F$ can also speed up integration substantially. 

Conceptually, DLRA propagates a rank $r$ approximation of the intial condition (usually $Y(0) = \Pi_{\mathcal{M}_r} X_0$, i.e., projection of $X(0)$ onto the manifold of rank $r$ matrices) forward in time according to the Dirac-Frenkel time-varying variational principle:

$$
    \frac{dY}{dt}(t) \in \mathcal{T}_{\mathcal{M}_r}(Y(t)) \text{ s.t. } \left\| \frac{dY}{dt}(t) - F(Y(t),t) \right\|_F \to \min
$$

Here, $\mathcal{T}_{\mathcal{M}_r}(Y(t))$ refers to the tangent space of the manifold of real $n\times m$ matrices of rank $r$ at the point $Y(t)$. 

## Example applications
While seemingly abstract at first, the solution of exceedingly large matrix-valued ODEs is quite a common problem. In the following we briefly discuss two general applications. 

### Time series data compression
Given stream of data as described by a function $A:[t_0,t_f] \to \mathbb{R}^{n\times m}$ mapping time point to a data matrix (think of a stream of images forming a movie) we may consider the problem of compressing this data stream. This can be cast as solving a matrix valued ODE 

$$
\frac{dX}{dt}(t) = \frac{dA}{dt}(t), \ X(0)= A(0), \text{ for all } t\in [t_0, t_f].
$$

Thus, DLRA can be used to propagate a compression of this data forward in time which can be substantially cheaper than compressing the data at every instant of time with other methods.  

### Uncertainty quantification
Given a parametric $n$-dimensional ODE,
$$
 \frac{dx}{dt}(p;t) = f(x(p;t),p, t), \ x(p;0) = x_0(p), \text{ for all } t\in[t_0, t_f]
$$
one often wishes to understand the parametric dependence of its solution. Arguably the simplest approach to this problem is sampling, where the ODE is simply evaluated for $m$ parameter values $p_1, \dots, p_m$. In many cases, however, it may be exceedingly expensive to evaluate the above ODE $m$ times, in particular when the dimension $n$ of the state is large. In those cases, DLRA may provide a means to recover tractability of the sampling procedure at the cost of approximation. To see this, note that the solution of the sampling procedure may be arranged in a $n\times m$ matrix

$$
X(t) := \begin{bmatrix}x(p_1;t) & \cdots & x(p_m;t) \end{bmatrix}
$$

whose dynamics are governed by

$$
F(X(t),t) := \begin{bmatrix}f(x(p_1;t),p_1,t) & \cdots & f(x(p_m;t), p_m, t) \end{bmatrix}.
$$

Note further that applying DLRA to this problem is equivalent to a quite intuitive function expansion strategy which forms the basis of other uncertainty quantification methods such as polynomial chaos expansion. In particular, the use of DLRA can be viewed as the discrete analog of applying the following expansion Ansatz to the parametric solution $x(p;t)$:

$$
    x(p;t) \approx \sum_{i=1}^r u_i(t) z_i(p;t)
$$

where the expansion modes $u_i(t)$ are chosen in a certain sense optimally. 

## Primitives
LowRankIntegrators.jl relies on a handful of primitives to enable the non-intrusive use of DLRA. These are described below.

### MatrixDEProblem
Given a matrix differential equation

$$
    \frac{dX}{dt}(t) = F(X(t),t), \ X(0) = X_0, \text{ for all } t \in [t_0, t_f]
$$

to be approximately solved via DLRA, the problem shall be set up as `MatrixDEProblem(F,Y0,tspan)` where

* `F` is the right-hand-side of the matrix ODE. `F` must accept two arguments, the first one being the (matrix-valued) state and the second time (or appropriate independent variable). 
* `Y0` is a low rank approximation of the initial condition. The rank of `Y0` determines the rank of the approximation unless a rank-adaptive integrator is used.
* `tspan` is a tuple holding the initial and final time for integration.

### MatrixDataProblem
If a data stream (or discrete sequence of data snapshots) is to be compressed via DLRA, then additional structure can be exploited. In this case, the problem shall be defined as `MatrixDataProblem(A,Y0,tspan)` where

* `A` is a function that describes the data stream, i.e., `A(t)` returns the data snapshot at time `t`. 
* `Y0` is a low rank approximation of the initial data point `A(0)`. The rank of `Y0` determines the rank of the approximation unless a rank-adaptive integrator is used.
* `tspan` is a tuple holding the initial and final time for integration.

If the data stream is not available continuously, but instead in form of discrete (time-ordered) snapshots, the problem shall be defined as `MatrixDataProblem(A,Y0,tspan)` where

* `A` is a (time-ordered) vector of data matrix snapshots.
* `Y0` is a low rank approximation of the initial data point `A(0)`. The rank of `Y0` determines the rank of the approximation unless a rank-adaptive integrator is used.


### LowRankArithmetic.jl
A key ingredient that allows LowRankIntegrators.jl to implement DLRA with minimal intrusion is [LowRankArithmetic.jl](https://github.com/FHoltorf/LowRankArithmetic.jl). Specifically, [LowRankArithmetic.jl](https://github.com/FHoltorf/LowRankArithmetic.jl) facilitates the propagation of low rank factorizations through finite compositions of a [wide range](https://github.com/FHoltorf/LowRankArithmetic.jl#readme) of arithmetic operations. This critically allows to take advantage of the low rank structure of the approximate solution $Y(t)$ when evaluating the dynamics $F(Y(t),t)$ and projections thereof without requiring a custom implementation of $F$. When $F$ is *not* a finite composition of the operations supported by LowRankArithmetic.jl and no custom implementation of $F$ and projections thereof is provided by the user, the DLRA routines in this package are not expected to speed up the integration, however, substantial memory savings may still be achieved.

### Integration routines
While the concept of DLRA is quite intuitive, it is no easy task to realize it algorithmically and only a handful of integration algorithms have been proposed. LowRankIntegrators.jl currently implement the Lie-Trotter and Strang projector splitting algorithms proposed in [1] as well as the "unconventional integrator" proposed in [2] and its rank-adaptive counterpart [3]. This selection of algorithms was made to support those that are robust to the presence of small singular values in the approximation [4]. 

Within LowRankIntegrators.jl the different integrators are specified by simple objects, allowing the specification of integrator specific options:

* Projector Splitting - `ProjectorSplitting(order)` where the optional argument `order` refers to the concrete type of the projector splitting algorithm. It can take values `PrimalLieTrotter()`, `DualLieTrotter()`, and `String()`. If no argument is specified, `order` defaults to `PrimalLieTrotter()`

* Unconventional Algorithm - `UnconventionalAlgorithm()`

* Rank adaptive unconventional Algorithm - `RankAdaptiveUnconventionalAlgorithm()`

### solve 
In order to finally solve a `MatrixDEProblem` or a `MatrixDataProblem`, the solve statement `solve(problem, alg, dt)` shall be called. Here

* `problem` refers to a properly defined `MatrixDEProblem` or `MatrixDataProblem`. 
* `alg` refers to one of the supported DLRA integrators.
* `dt` refers to the integration step sizes. If `problem` is a `MatrixDataProblem` where the data is specified as discrete snapshots `dt` defaults to 1 so that the integrator steps through all provided snapshots in order.


## Future Work
Future work will include integration routines for the dynamically orthogonal field equations. 

## Acknowledgements
This work is supported by NSF Award PHY-2028125 "SWQU: Composable Next Generation Software Framework for Space Weather Data Assimilation and Uncertainty Quantification".

## References
[1] Lubich, Christian, and Ivan V. Oseledets. "A projector-splitting integrator for dynamical low-rank approximation." BIT Numerical Mathematics 54.1 (2014): 171-188.

[2] Ceruti, Gianluca, and Christian Lubich. "An unconventional robust integrator for dynamical low-rank approximation." BIT Numerical Mathematics (2021): 1-22.

[3] Ceruti, Gianluca, Jonas Kusch, and Christian Lubich. "A rank-adaptive robust integrator for dynamical low-rank approximation." arXiv preprint arXiv:2104.05247 (2021).

[4] Kieri, Emil, Christian Lubich, and Hanna Walach. "Discretized dynamical low-rank approximation in the presence of small singular values." SIAM Journal on Numerical Analysis 54.2 (2016): 1020-1038.
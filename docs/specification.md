# Specification (v0)

Idealised synthetic experiments only. Nothing here is an estimate for Belauri, Nepal.

## Equation
dc/dt + div(u c) = div(K grad c) + sum_j q_j s_j(t) f_j(x,y)/H - lambda c

x,y [m]; t [s]; c [ug/m^3]; u [m/s]; K [m^2/s]; lambda [1/s]; H [m] (fixed effective mixing depth);
q_j [ug/s]; f_j [1/m^2] with sum_cells f_j * cell_area = 1 (normalised on the grid, so integrated emission is
resolution-invariant); s_j(t) dimensionless piecewise-constant profile (default 1).
Sign convention: n is the outward face normal; a face is *inflow* if u.n < 0, *outflow* if u.n > 0, tangential if u.n = 0.

`Times.duration` must be an integer multiple of `output_interval`. In periodic mode `background` must be 0.

## Defaults (demonstration values, not Belauri estimates)
10 km x 10 km, 50x50 cells; H=500 m; u=(1.0,0.3) m/s; K=100 m^2/s; lambda=1/21600 1/s; 14400 s;
dt=60 s; output every 300 s; sensors every 600 s; three Gaussian sources sigma=500 m at (2500,2500),
(4000,6000), (6500,3500) with q=(1e5,1.5e5,8e4) ug/s; zero IC and zero inflow background.

## Boundaries (per face)
open: inflow faces prescribe c_b; outflow faces have zero diffusive gradient and export by advection;
tangential faces zero diffusive flux. periodic: uniform wind. closed: no flux (zero wind only).

## Numerical diffusion of the baseline scheme (known model error)
The discretisation (FiPy exponential convection + diffusion, backward Euler) adds diffusion to K. For a Fourier mode
the effective diffusivity satisfies, with cell Peclet number Pe = |u| dx / K,

    K_eff / K - 1  ~=  (Pe/2) coth(Pe/2) - 1   +   u^2 dt / (2 K)
                       (space, exponential)        (backward-Euler time term)

(upwind: space term Pe/2 instead).  The figures are for wind along a grid axis (not re-measured for oblique wind). Measured by Fourier-mode projection (periodic, lambda = 0,
K = 100, u = 1, m = 1 mode; `scripts/verification.py`, study `numerical_diffusion`; asserted to 3% in the tests):

| scheme       | cells | dt (s) | measured K_eff/K - 1 |
|--------------|-------|--------|----------------------|
| exponential  | 50    | 10     | +0.361               |
| exponential  | 50    | 60     | +0.606               |
| exponential  | 25    | 10     | +1.11                |
| exponential  | 100   | 10     | +0.131               |
| upwind       | 50    | 10     | +1.046               |
| upwind       | 50    | 60     | +1.288               |

So at the default 50x50, dt = 60 s the effective diffusivity is about 1.6 K (not +31%: the space term alone is
+0.31, the time term adds +0.30). Plumes are therefore smoother than the stated K; this is a modelling error of the
baseline, not a bug, and `K` should be read as the *nominal* diffusivity.

## Inverse problem
Unknowns: the nonnegative amplitudes q_j only. Wind, K, lambda, H, source geometry, initial state and boundary
forcing are KNOWN in matched experiments. Linear model y = b + A theta + eps with b the zero-emission response.
`FitResult.effective_rank` (# whitened scaled singular values > 1) and `q_std` (linear-Gaussian posterior std, ignores
the q >= 0 bound) report how well the data constrain each q_j; `rank` is only a numerical rank.
Good concentration reconstruction does not imply correct source attribution; see identifiability diagnostics.

## Error metric
MSE_w = sum_t sum_cells w dA dt_q (c_hat-c_true)^2 / sum w dA dt_q, with trapezoid time weights dt_q over
[t_exclude, T]; default w=1, t_exclude=0.

The full design, discretisation and task map are in `implementation-plan.md`; `architecture.md` lists modules.

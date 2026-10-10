# Specification (v0)

Idealised synthetic experiments only. Nothing here is an estimate for Belauri, Nepal.

## Equation
dc/dt + div(u c) = div(K grad c) + sum_j q_j s_j(t) f_j(x,y)/H - lambda c

x,y [m]; t [s]; c [ug/m^3]; u [m/s]; K [m^2/s]; lambda [1/s]; H [m] (fixed effective mixing depth);
q_j [ug/s]; f_j [1/m^2] with sum_cells f_j * cell_area = 1 (normalised on the grid, so integrated emission is
resolution-invariant); s_j(t) dimensionless piecewise-constant profile (default 1).
Sign convention: n is the outward face normal; a face is *inflow* if u.n < 0, *outflow* if u.n > 0, tangential if u.n = 0.

## Defaults (demonstration values, not Belauri estimates)
10 km x 10 km, 50x50 cells; H=500 m; u=(1.0,0.3) m/s; K=100 m^2/s; lambda=1/21600 1/s; 14400 s;
dt=60 s; output every 300 s; sensors every 600 s; three Gaussian sources sigma=500 m at (2500,2500),
(4000,6000), (6500,3500) with q=(1e5,1.5e5,8e4) ug/s; zero IC and zero inflow background.

## Boundaries (per face)
open: inflow faces prescribe c_b; outflow faces have zero diffusive gradient and export by advection;
tangential faces zero diffusive flux. periodic: uniform wind. closed: no flux (zero wind only).

## Inverse problem
Unknowns: the nonnegative amplitudes q_j only. Wind, K, lambda, H, source geometry, initial state and boundary
forcing are KNOWN in matched experiments. Linear model y = b + A theta + eps with b the zero-emission response.
Good concentration reconstruction does not imply correct source attribution; see identifiability diagnostics.

## Error metric
MSE_w = sum_t sum_cells w dA dt_q (c_hat-c_true)^2 / sum w dA dt_q, with trapezoid time weights dt_q over
[t_exclude, T]; default w=1, t_exclude=0.

The full design, discretisation and task map are in `implementation-plan.md`; `architecture.md` lists modules.

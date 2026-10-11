# phase3-scheme

`scheme_check.csv`, produced by `python scripts/scheme_check.py` (about 90 s): exponential vs upwind vs vanleer
convection (`SolverOptions.convection`). Long format: study, scheme, nx, dx_m, dt_s, courant, metric, value.

* `gauss_tied` (dt = 0.06 dx) / `gauss_fixed` (dt = 7.5 s): periodic 2-D Gaussian vs the exact solution, nx = 20/40/80;
  `rel_L2`, `Keff_over_K` (1 + excess second-moment growth / 2KT), `min_value`, `order_L2` (between successive levels).
* `courant`: dx = 250 m, u = 1 m/s, Courant 0.25 to 4.
* `open_steady`: open boundaries, uniform wind, steady sources, nx = 25 vs a cell-averaged exponential nx = 100 reference
  (the reference itself carries numerical diffusion, so this is a consistency check, not an error against truth).

vanleer is explicit in the convection; `simulate` sub-steps the whole equation so that the summed axis Courant number is
<= `VANLEER_MAX_COURANT` (0.5). For Courant > 0.5 the vanleer results therefore use a smaller effective dt than the
other schemes at the same nominal dt. vanleer is nonlinear (flux limiter): the cached-response (superposition) route
refuses it.

# Winds: interface audit, prescribed toy winds (EXP-018), ERA5 feasibility

Scope: what wind forcing the forward model accepts, the prescribed winds built in `src/pravah/winds.py`, the definitions used by
`scripts/prescribed_winds.py`, and a written (not executed) ERA5 feasibility note.

**Every wind in this repo is PRESCRIBED.** A 2-D incompressible stream function is not a realistic near-surface wind (real winds near
the ground are divergent, vary with height and are turbulent). The DEM does not determine the wind: nothing in the code reads terrain to
make a wind, and the toy "ridge" is a Gaussian bump in psi placed by hand. All EXP-018 outputs carry the stamp
`EXP-018 prescribed wind, not terrain-derived`.

## 1. Audit of the existing wind interface

| Item | Where | Behaviour |
|---|---|---|
| `config.WindSegment(t_start, u, v, psi)` | `config.py` | wind holds from `t_start` to the next segment; first segment starts at 0, times strictly increase (`Transport`) |
| `psi` | `config.py`, `forward.py` docstring | cell-CORNER stream function, shape (ny+1, nx+1), m^2/s; `u = dpsi/dy`, `v = -dpsi/dx`; face fluxes `u_f[j,i] = (psi[j+1,i]-psi[j,i])/dy`, `v_f[j,i] = -(psi[j,i+1]-psi[j,i])/dx`; discrete divergence cancels identically. With `psi`, `(u, v)` are display values only |
| uniform / eddy / reversal factories | `geo/scenarios.py` (`uniform_wind`, `eddy_streamfunction`, `eddy_wind`, `build_wind`) | meteorological bearing helpers; reversal switches at `duration/2`; eddy = uniform + Gaussian bump in psi |
| step handling | `forward.step_times`, `_wind_at` | wind-segment starts are time breakpoints; the segment for a step is chosen at the step midpoint |
| open boundary | `forward._boundary_arrays` | per boundary face, classified by the sign of `u.n` (values below `UN_TOL = 1e-12` m/s snapped to 0): outflow = upwind export `a_out = un/h`; inflow = Dirichlet face value (background `c_b`, plus inflow bases) with diffusive coefficient `a_in = 2K/h^2`. Here `c_b = 0` and no bases: air entering is clean, so mass that leaves an outer domain is gone for good |
| psi restrictions | `forward.simulate` | psi needs `boundary.mode == "open"`; exponential scheme needs `K > 0` |

Supported: steady uniform; uniform varying in time (piecewise constant); spatially varying through psi; spatiotemporal as piecewise-in-time psi.
NOT supported: continuous-in-time interpolation of the wind (a step change per segment; a smooth change needs many short segments, and
each new segment builds its own FiPy objects); divergent (non-solenoidal) wind, including vertical motion and mass-continuity
corrections; an externally supplied raster or time series without conversion; any coupling of wind to terrain.

External raster or series -> segments: `winds.segments_from_series(domain, times, psi=... | u=..., v=...)`.
* Interpolation in time: none. Sample `k` holds from `times[k]` to `times[k+1]` (sample-and-hold; supply the series at the step you want).
* Space: given psi at cell corners it is used as is. Given cell-centre `u, v` (nt, ny, nx) they are averaged to faces (edge faces copy the
  nearest cell) and psi is the least-squares solution of `dpsi/dy = u_f`, `-dpsi/dx = v_f` (sparse lsqr), gauge psi[0,0] = 0. This keeps
  the divergence-free part of the field and silently drops the divergent part, which for real winds is not small.
* The regridding of a lat-lon field onto the model grid (and the lat-lon -> metres conversion) is NOT done here.
* Tested: `tests/test_winds.py::test_series_round_trip` (psi in -> segments out unchanged; u, v from a psi field -> psi gradients agree to 5 % of the speed away from the edges).

## 2. Prescribed builders (`src/pravah/winds.py`, pure functions returning `tuple[WindSegment, ...]`)

`steady_uniform(domain, u, v)`; `delayed_reversal(domain, u, v, t_r)` (u -> -u at `t_r`); `ridge_deflection(domain, u, v, centre, sigma, amplitude, t_r=None)`
(`psi = u*y - v*x + amplitude*exp(-r^2/2 sigma^2)`, peak deflection speed `amplitude/(sigma sqrt(e))`; with `t_r` the whole field is reversed,
`psi -> -psi`); `recirculation_eddy(...)` (same function with a strong bump: closed streamlines where the swirl speed exceeds the background).
Both bump builders reuse `geo.scenarios.eddy_streamfunction`. `face_divergence(domain, psi)` is exactly the per-cell divergence of the face
fluxes (tests: < 1e-12 for every builder).

## 3. EXP-018 toy study: set-up and definitions

Geometry (`configs/prescribed_winds.json`): study box 9 km x 9 km at the centre of a square outer domain; outer domain side = size factor x 27 km
for factors 1, 1.5, 2 (the 1x domain already holds the box plus 9 km on each side; a literal "1x the box" would leave no room for an upstream
source). The box and both sources keep the same position relative to the box for every size; the grid spacing is fixed (750 m, so the box is
12 cells). External source: Gaussian, a pulse on [1800 s, 5400 s], 12 km west of the box centre; local source inside the box, constant.
Wind 2 m/s eastward. K = 100 m^2/s, lam = 1/6 h, H = 500 m (the repository defaults; numerical diffusion is not removed). Output every 300 s for 12 h.

Per case two solves (external source only, local source only); the total is their sum by linearity (`test_linearity_total_is_sum`).

* Study mean: area-weighted mean over the box cells.
* Arrival time: first linearly interpolated up-crossing of the external-only study mean through (a) `arrival_fraction` (0.1) of its own run peak,
  (b) the absolute threshold 0.05 ug/m^3. `return_arrival_frac_s` is the second up-crossing of (a); NaN when the series never rises again.
* Mass: emitted, removed (decay), net boundary loss (outflow minus inflow terms of the forward budget, which includes the diffusive loss at inflow
  faces), remaining. Closure = |remaining - (emitted - removed - lost)| / emitted. Stop rule: > 1e-8. The forward budget's per-step residual is checked too.
* Per-side boundary loss: `H dA (un/h if un>0; 2K/h^2 if un<0) . c` on each side's boundary cells, at the output frames (end-of-step c, wind of the
  preceding interval); its time integral agrees with the budget total to within about 1e-2 of the emitted mass (largest value 7e-3 in the pilot; reported as `*_side_vs_budget_rel`;
  this is a post-processing cross-check, not an exact identity, since the budget sums sub-steps).
* **Returned mass fraction** (external-only run). `m(t) = H dA sum(c w)` is the external mass in the box (`w` = cell fraction inside the box). The
  box has no source, so the net inward exchange over `[t_k, t_k+1]` is `N_k = dm + lam H dA int(sum c w) dt`. The exit time `t_x` is the first time `m` has
  fallen below 95 % of its running maximum after that maximum exceeded 0.1 % of the emitted mass. Returned = `sum_{k: t_k >= t_x} max(N_k, 0) / emitted`.
  So it is the mass that flows into the box again after the plume has entered and started to leave. A plume that first misses the box and is then
  steered into it counts as a first entry, not a return. This is an Eulerian net-flux definition (mass is not tagged), so diffusive in-and-out
  exchange at the box edge is netted. Tested on two synthetic humps (`test_returned_mass_two_humps`) and on a steady versus a reversing run
  (`test_steady_has_no_return_and_reversal_has`).
* Local versus external contribution: `ext_exposure`, `loc_exposure`, `total_exposure` (time integrals of the study mean, ug s/m^3) and
  `ext_share_of_exposure`.
* Sensitivity to outer-domain size: each metric relative to the largest size of the same (wind, t_r), `sensitivity_outer_domain.csv` and `summary.md`.
* Checks: closure < 1e-8; the most negative concentration relative to the maximum > -1e-9 (the exponential scheme gives an M-matrix, so no round-off
  negatives are expected).

Run: `python scripts/prescribed_winds.py --config configs/prescribed_winds.json --profile smoke|pilot|production`. Outputs in
`results/phase3-winds-toy[-smoke|-pilot]/`: `cases_raw.csv`, `timeseries_raw.csv`, `boundary_flux_raw.csv` (raw, one row per case / time / side),
`sensitivity_outer_domain.csv`, `checks.json`, `provenance.json`, `summary.md`, three figures.

## 4. ERA5 feasibility note (written only; nothing downloaded)

Verified on the Copernicus pages on 2026-10-11 (URLs below):
* Dataset "ERA5 hourly data on single levels from 1940 to present" (`reanalysis-era5-single-levels`) and "... on pressure levels" (`reanalysis-era5-pressure-levels`):
  hourly, regridded to a regular 0.25 deg lat-lon grid (ensemble mean/spread 0.5 deg), GRIB or netCDF. Native resolution is 31 km (T639 / N320), archived
  grid 0.28125 deg (ECMWF ERA5 documentation).
* Variable names in the CDS: `10m_u_component_of_wind` (paramId 165) is listed in the ECMWF parameter table; the v component, the 100 m components
  (`100m_u_component_of_wind`, `100m_v_component_of_wind`) and the 37 pressure levels were NOT confirmed from the pages I could read (recalled, check the download form).
* Access: free CDS account, personal access token in `$HOME/.cdsapirc` (`url: https://cds.climate.copernicus.eu/api`, `key: <token>`), `pip install cdsapi`,
  `cdsapi.Client().retrieve(dataset, request, target)` (CDS "How to API" page). The dataset pages state the licence as CC-BY and give the DOI
  10.24381/cds.adbb2d47 for single levels; ECMWF requires citing the CDS catalogue entry and a clear attribution to the Copernicus programme. Whether the
  download form still asks for an explicit licence acceptance click was not verified. Request-size limits were not verified.
* Bounded extract (illustrative, not run): dataset `reanalysis-era5-single-levels`, `product_type: reanalysis`, `variable: [10m_u_component_of_wind, 10m_v_component_of_wind]`
  (add the 100 m pair if wanted), one year/month, all days, `time: 00:00..23:00`, `area: [N, W, S, E]` as a box of a few degrees around the region
  (a 4 deg x 4 deg box is 17 x 17 points per hour, 289 points x 744 hours x 4 bytes is about 0.9 MB per variable-month uncompressed), `data_format: netcdf`.
* Limitation: 0.25 deg is about 28 km in latitude (and less in longitude at this latitude). The mountain-induced flow near Belauri (valley winds, slope
  flows, channelling on scales of hundreds of metres to a few km) is not resolved; ERA5 gives a synoptic / large-scale wind that a terrain-aware
  downscaling would have to refine. It is divergent (3-D) and varies with height; reducing it to a 2-D solenoidal psi loses that.
* ERA5 would enter only through `winds.segments_from_series`, after a separate step that interpolates the lat-lon field to the model grid in metres.
  That step, the unit handling (m/s) and the choice of level are not implemented.

Sources: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview ;
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download ;
https://cds.climate.copernicus.eu/how-to-api ;
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation ;
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels .

## 5. EXP-021: converged re-run and corrected interpretation (prescribed wind, not terrain-derived)

Run in production as RUN-0034 (`scripts/prescribed_winds_converged.py`, `configs/prescribed_winds_converged.json`; results in
`results/phase3-winds-converged/`, finding F-031). Reading of the results after independent review: only the three reversal cases depend on domain size
(the outflow face has no diffusive flux, so everything else is identical at every size); the half-width needed for <1% exposure is set by the return
window (~UT/2 for T = 12 h), not by U/lambda; returned mass is good to ~10% (dx dependence); the channel_step effect (-43%) comes from where the plume
centreline sits relative to the box, not from a general channel property. Independent review of EXP-018
found four problems; this section states how EXP-021 addresses each and how the EXP-018 results must be read.

1. **Domain truncation.** The 2x reference was not converged (mass left east before the reversal). EXP-021 keeps the study box and source fixed in
   physical space and varies only the outer half-width (declared set; y half-width capped at `y_half_cap_m`, x is the long direction). The largest
   declared half-width must be >= 2 x U x duration (the plume advection distance; checked at start). After the declared set, the exposure of the study box
   is compared between the two largest domains for EVERY scenario; if any differs by >= 1 % the half-width is enlarged (x1.5, rounded) up to a declared cap and the result
   is reported (`checks.json`: `converged`, `enlarged_half_widths_m`).
2. **Source too close to the boundary.** The source must be >= 6 cells and >= 6 km from every boundary of the SMALLEST domain (hard check). The 1x-truncation
   effect in EXP-018 was confounded by the ~2 % loss through the west Dirichlet face; per-side losses are reported for every case (`leave_<side>_frac`).
   Numerical diffusion is separated from truncation by running the base scenario at dx, dx/2 and dx/2 with dt/2 on the smallest domain (`numerical_check_raw.csv`).
3. **Returned mass.** The EXP-018 Eulerian net-flux definition is kept (`returned_eul`). It starts counting when the box mass has fallen 5 % below its maximum, so the
   tail of the first passage still entering the box is counted as "returned", and decay shrinks late returns. The second definition uses the box-mass series only: first passage
   ends at `t_exit` (mass < `exit_frac` = 2 % of its first-passage peak P1); `ret_residence_ratio` = integral after `t_exit` of max(m - `floor_frac` P1, 0) over the
   first-passage integral of m, and `ret_mass_frac` = ratio x mass that entered in the first passage / emitted (exact for equal transit times; the floor clips about
   9 % of a 0.5-height hump in the synthetic test). `ret_eul_after_exit` is the EXP-018 sum restarted at `t_exit`. If the plume has not exited by the end of the run, all are NaN and `not_yet_exited` is true
   (not 0); `end_inside` marks a return cut off by the end of the run; `tail_inside_at_tr` / `exited_before_tr` / `box_frac_at_tr` state whether the plume had left before each reversal. Reversal times
   are meant to follow the exit time of the steady case (exit time is reported per scenario).
4. **Ridge interpretation.** In EXP-018 `psi = u y + A exp(-r^2/2 sigma^2)` is a localised vortex: psi returns to `u y` far away, so there is no net deflection by
   construction, and the "ridge" and the "eddy" are one function with different amplitude (the "ridge" is a sub-critical vortex: a bend that returns, not a deflection around an obstacle).
   Conclusions drawn in F-024/F-027 about "ridge deflection" apply to a sub-critical vortex only.

New builder `winds.deflected_channel(domain, u, x0, width, offset, kind)`: `psi = u (y - h(x))`, wind `u_x = u`, `v = u h'(x)`, divergence free, streamlines `y = const + h(x)`.
`kind="step"`: smooth step (tanh) with NET downstream displacement `offset` (the flow is turned and stays displaced); peak turning angle `atan(offset/(2 width))`. `kind="bump"`: Gaussian,
returns to the original y (zero net), peak angle `atan(offset/(width sqrt(e)))`. It is prescribed: no DEM, no blocking or pressure dynamics. Tests: `tests/test_winds_channel.py`.

Scenarios: `steady`, `reversal` (u -> -u at t_r), `channel_step`, `channel_bump` (between source and box), `vortex` (upstream of the box, centre >= 2 sigma outside it). External source only
(no local solve; the quantity is external exposure of the study box). Outputs in `results/phase3-winds-converged[-smoke|-pilot]/`: `cases_raw.csv`, `timeseries_raw.csv`, `numerical_check_raw.csv`
(raw), `convergence.csv`, `vs_reference.csv`, `checks.json`, `provenance.json`, `summary.md`, three figures stamped `EXP-021 prescribed wind, not terrain-derived`.
Stop rule: closure > 1e-8 or negativity > 1e-9 exits non-zero after writing the files.

# Belauri forward simulations (Stage E)

**Real geography; synthetic emissions and meteorology.** Code: `src/pravah/geo/scenarios.py` (scenario builder, crop,
winds, numerical-diffusion estimate), `src/pravah/belauri.py` (study runner, figures), configs `configs/belauri_forward.json`
(production) and `configs/belauri_forward_smoke.json` (plumbing check), notebook `notebooks/05_belauri_forward.ipynb`.
Run: `python -m pravah benchmark --config configs/belauri_forward.json --out results/belauri_forward`
(the config key `"experiment": "belauri_forward"` dispatches to `belauri.run_belauri_forward`).

## What is real and what is synthetic
| Real (from `data/belauri/manifest.json`, UTM 32644) | Synthetic / nominal |
|---|---|
| Municipality polygon as per-cell area fraction `study_fraction` (122.96 km2) | All emission rates (micrograms per second) |
| Road, built-up, facility geometry (`w_road`, `w_builtup`, `w_facility`, clipped to the polygon) | External onset profile ((0,0),(7200,1)) |
| Position of the external patch (centre (10609, 14416) m local, sigma 1000 m; ~2.9 km from the polygon, outside it; **not** asserted international) | All winds (prescribed; the stream-function demo is not derived from the DEM) |
| Domain 66 x 65 cells of 500 m (33.0 x 32.5 km), also 250 m | K = 100 m2/s, lambda = 1/21600 1/s, H = 500 m (nominal) |

OSM completeness is unknown: an absent road or building is a mapping gap, not a zero source. The 5 % of a local layer's
mass that sits in boundary cells outside the polygon is inside the "local" tag (origin-tag tolerance 6 %, documented in
`geo.scenarios.LOCAL_ORIGIN_TOL`; the manifest gives 0.949 / 0.977 / 0.968 for road / built-up / facility at 500 m).

## Inputs that change numbers, and plotting-only layers
| Input | Effect |
|---|---|
| wind (direction, segments, stream function), K, lambda, emission amplitudes, which source layer carries the emission, outer-domain size (through the open boundary), mesh, time step | change concentrations (measured in `what_changes.csv`) |
| **DEM, DEM valid fraction, hillshade, slope, elevation profiles** | **plotting-only: no solver input reads them.** `plotting_only_check` replaces them by constants, shows the `Scenario` is identical and the concentration change is exactly 0 |

If a later stage lets terrain alter source geometry or constraints, that layer moves to the first table.

## Domain and time choices (justified)
* Wind 1.5 m/s; T = 28800 s (8 h): travel u T = 43 km exceeds the 33 km domain, so plumes can cross it and leave through the open
  boundary; the external patch is 2.9 km from the polygon, so it reaches the polygon edge in ~0.5 h and the centroid in ~1-2 h
  after the 7200 s onset.
* Physical diffusion length sqrt(2 K T) = 2.4 km (nominal K), small against the 33 km domain.
* Removal lifetime 1/lambda = 6 h: comparable with T, so removal matters but does not erase the signal.
* Courant number u dt / dx <= 0.3: dt = 100 s at 500 m, 50 s at 250 m (backward Euler is stable for any dt; the bound is for accuracy).
  The reported `max_courant` is the speed Courant number (hypot of the largest face speeds for a stream function).
* Output every 900 s; study metrics exclude t < 7200 s (the external onset time).
* Outer domain: standard = the prepared 8 km buffer; small = 2 km buffer (`crop_region`, whole cells, absolute positions unchanged).
* Wind names use the meteorological convention (direction the air comes FROM): `from_west_southwest` 247.5 deg (toward the
  municipality from the border side), `from_north`, `from_east`, `reversal` (WSW, then ENE at T/2), `channelled_demo`
  (mean 1.2 m/s plus a Gaussian eddy of amplitude 1500 m2/s, sigma 4 km; the mean speed keeps the peak face speed near 1.5 m/s).

## Synthetic emission amplitudes
road 1.5e6, built-up 1.0e6, external patch 5.0e5 micrograms per second, chosen so the study-region peak of the combined field is of order
0.1-0.4 microgram/m3 at the default wind (one-source peaks at 1e6 ug/s in the study region, WSW: road 0.082, built-up 0.109,
external 0.317). Not measured emissions; rescale freely (the model is linear, `what_changes` verifies exact doubling).

## Numerical diffusion versus the physical K
The scheme adds `K_eff/K - 1 = (Pe/2) coth(Pe/2) - 1 + u2 dt/(2K)`, Pe = u dx / K (`numerical_diffusion_estimate`; reproduces the
documented 0.606 / 0.361 of `docs/specification.md` within 3 %; exact for wind along a grid axis only). At u = 1.5 m/s:

| mesh | Pe | K_eff/K - 1 | K_eff (m2/s) |
|---|---|---|---|
| 500 m, dt 100 s | 7.5 | 3.88 | ~488 |
| 500 m, dt 50 s | 7.5 | 3.32 | ~432 |
| 250 m, dt 50 s | 3.75 | 1.53 | ~253 |

So the plume spreading is set mainly by the discretisation, not by the nominal K = 100; K is reported as a nominal input, and
the spreading length is quoted for both. Finer meshes reduce, not remove, this.

## Outputs of the run (all NaN -> null; no seeds, runs are deterministic)
`config.json`, `provenance.json` (manifest sha256, resolution, config), `timings.json`, `runs.csv` (per solve: runtime, budget
residual, negative count, Courant), `wind_summary.csv`, `series/<wind>.csv` (regional mean, study peak, external fraction per
time; fraction blank where undefined), `fields/<wind>.nc` (local, external, combined), `arrival_times.csv` (five points: nearest
and farthest polygon cell from the patch, nearest and farthest facility cell, centroid; arrival = first exceedance of 0.005
microgram/m3 by the external tag), `consistency.json` (direct combined solve vs local + external, budget residual), `outer_domain*.csv`,
`mesh_sensitivity*.csv`, `what_changes.{csv,json}` (with the explicit plotting-only list), `figures/*.png` (every figure carries the label).

## Limitations
Synthetic winds and emissions; no terrain dynamics (no katabatic or channelled flow is computed); nominal K, lambda, H; the
numerical diffusion above; 250 m comparison only for one wind; arrival times are resolved to the 900 s output interval;
the municipality outline in figures is the 0.5 contour of the 250 m area fraction.

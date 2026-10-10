# Architecture

Everything here is synthetic except the Belauri geometry (study polygon, roads, buildings, facilities, outline, DEM context).
**Real geography; synthetic emissions and meteorology.**

## Modules (`src/pravah/`)

Core (v0, extended in phase 2; imports never need the GIS libraries):

| Module | Role |
|---|---|
| `config` | Frozen dataclasses and JSON loading. `Scenario` holds everything except the amplitudes. Phase 2 adds: `GaussianSource.group` (`local` / `external`), `RasterSource` (gridded shape, serialised by sha256), `InflowBasis` (in `Boundary.bases`), `StudyRegion` / `StudyMask`, optional stream-function wind in `WindSegment`, `Scenario.param_table()`, `FieldResult` |
| `sources` | `cell_centres`, `source_fields` (every source normalised so sum f dA = 1), `study_weights`, origin-tag checks |
| `forward` | The FiPy solver (`simulate`) and the replaceable-solver interface: `ForwardSolver` protocol, `check_field_result`, `BackendInfo(name, version, settings, capabilities)`, `FipyForward(convection, tolerance)`, the default instance `FIPY`, `get_forward(spec, kwargs)`, `forward_info` |
| `observations` | `sample`, `observe` (sensors, noise) |
| `inverse` | `Responses` (unit responses per **parameter**, with kinds), `ResponseCache`, `cache_key`, `fit_sources`, `solve_linear`, `reconstruct`; every route takes `forward=` |
| `attribution` | Origin tags (`contributions`), `external_fraction` / `origin_fractions`, component errors, `emission_error`, `identifiability_report` |
| `metrics` | Field RMSE / MSE with weights |
| `design` | Candidates with categories, `Constraints`, `DesignCase` (cached arrays and named `QuadForm` objectives), greedy / spread / random layouts, role guards (design / test / validation), held-out blocks |
| `experiments` | Benchmark, alpha-selection and mismatch runners with separate truth and inference solvers |
| `io`, `plotting`, `cli` | Output writers; figures; `python -m pravah ...` |

Phase 2 studies:

| Module | Role |
|---|---|
| `transboundary` | External-versus-local synthetic study (named scenarios, domains A / B, recovery, prior and scale sensitivity, boundary-versus-plume equivalence, confounded, no-arrival, wind-mismatch cases) |
| `indoor` | One-way-coupled indoor box (exact piecewise-linear integration); not part of the outdoor inverse problem |
| `geo/` (optional extra `[geo]`) | `fetch` (bounded, cached network access), `prepare` (offline: raw to processed arrays and `data/belauri/manifest.json`, boundary verification, `Region`), `regionio` (CRS, grids, polygon-cell fractions, rasterisation, DEM aggregation, candidate lattice), `scenarios` (Belauri scenario builders, winds, cropping, numerical-diffusion estimate). GIS libraries are imported lazily |
| `belauri` | Forward simulations on the prepared region (winds, outer-domain size, mesh, what-changes table) |
| `geodesign` | Candidate sites (CSV + GeoJSON), ensembles, alpha frozen on validation, four layout methods, two settings, held-out evaluation |
| `geodesign_robust` | Re-uses the fixed layouts and alphas of a `geodesign` run; mismatch conditions; numerical refinement |

Research Panel is used only from `scripts/`, never imported by the package.

## Parameter vector and bases

`theta` = emission amplitudes (ug/s, one per source, in source order) followed by inflow-basis amplitudes (ug/m^3, one per
basis). `scenario.param_table()` gives `Param(name, kind, units, group)` with `kind` in `emission` / `inflow`. The two kinds are
never compared by raw magnitude: errors are reported per kind, and the inversion works in scaled units (`inverse.SCALE_BY_KIND`:
emission 1e5 ug/s, inflow 0.05 ug/m^3; with alpha > 0 the scale is a prior width). `q` of `simulate` and `ForwardSolver` is `theta`;
with no bases it is the v0 emission vector and results are bit-identical.

An `InflowBasis(name, sides, profile)` sets the face concentration `a * profile(t)` on the listed sides only where `u.n < 0` under the
current wind (faces with `|u.n| < 1e-12` m/s are tangential). The static `Boundary.background` stays a known input.

## Data flow

```
Scenario (sources, bases, wind, study region) + theta
   -> forward solver (FIPY | FipyForward(...) | module:callable)   -> FieldResult (xarray; diagnostics["backend"])
   -> observe                                                      -> Observations (readings only, no tags)
   -> ResponseCache (n_params + 1 solves; key includes BackendInfo) -> Responses (base, g_j)
   -> fit_sources / solve_linear                                   -> FitResult (theta_hat, q_std, effective rank)
   -> reconstruct                                                  -> FieldResult
   -> metrics, attribution.contributions / external_fraction      -> numbers per tag and per kind
design: Responses -> DesignCase (arrays + QuadForms) -> layouts (greedy on DESIGN cases only) -> evaluate on TEST cases
```

Solver roles: the **truth** solver generates every synthetic truth (matched, refined-grid, perturbed-wind, extra-source), the
**inference** solver builds responses, cases, fits and reconstructions. `get_forward` resolves a spec once (`None` / `"fipy"` to
`FIPY`; `"module:callable"` is a factory called once with the kwargs). The cached route needs `capabilities["linear"]`, declared
explicitly and never assumed; the cache key always contains the full `BackendInfo`. See `solver-interchangeability.md`.

Origin tags (`attribution.TAGS`: `local`, `external_sources`, `boundary`, `background`, `initial`) come from linear superposition of
the unit responses and sum to the total field. They are used to evaluate synthetic truths; `fit_sources` receives only
`Observations`.

## Study polygon, simulation domain, outer boundary

```
+--------------------------------------------------------------+  <- outer inflow boundary: the domain edge. Only faces
|  simulation domain (the PDE is solved on all of it)          |     where u.n < 0 carry the boundary input (InflowBasis /
|                                                              |     background); elsewhere the flow leaves.
|     external source (explicit, outside the polygon)          |
|            *                                                 |
|            ~~~~> wind                                        |
|          +-----------------------------+                     |
|          |  study polygon / StudyMask  |  local = emitted    |
|          |  (reporting mask + weights; |  inside it; its     |
|          |   NOT a wall)               |  metrics use cell   |
|          +-----------------------------+  area fractions     |
|   buffer on every side (Belauri: 8 km, padded to 500 m)      |
+--------------------------------------------------------------+
```

The study polygon (a `StudyRegion` rectangle, or a `StudyMask` of exact polygon-cell area fractions) only decides which emissions are
*local* and which cells carry weight in the metrics. Plumes cross it freely and may leave and re-enter; a local plume that returns
after a wind reversal stays local. The simulation domain is the study region plus a buffer chosen from travel time (`u T`) and
diffusion length; the outer inflow boundary is its edge, and its distance from the study region changes results when the wind
reverses (`belauri-forward.md`, `transboundary.md`). *International transboundary* is decided only by verified national (ADM0)
polygons, never by the polygon edge, and nothing in the current results asserts it. *Indoor* is a separate box that reads the outdoor
field at a building and never feeds back.

Coordinates: `Domain` is local metres with origin 0. Geography keeps a separate `GeoRef(epsg, x0, y0)` (UTM 44N, EPSG:32644 for Belauri)
in the prepared manifest; arrays are `[y, x]`, y increasing northward.

## What stays the same

* The equation, the backward-Euler/FiPy discretisation, the open / periodic / closed boundary modes and the default `simulate` are
  unchanged; the default outputs are bit-identical to v0 (`results/solver-regression`: 21 files, 0 differ).
* `simulate`, the `ForwardSolver` signature, `forward=` keywords and the v0 configs, notebooks 01-03 and frozen v0 results.
* The leakage rules: design, validation and test ensembles use disjoint seeds; layouts and alpha are chosen without test data;
  tags never reach the fit.
* The DEM, hillshade and slope are plotting context only; no solver input reads them.

User docs: `../README.md`, `extending.md` (worked examples, including phase 2), `validation.md` (v0 record) and
`phase2-validation.md` (phase-2 record and requirement table). Scripts: `scripts/verification.py`, `verify_notebooks.py`,
`capture_regression.py`, `compare_runs.py`, `geo_report.py`, `geo_relief_profile.py`, `transboundary_tables.py`,
`register_research.py`. See `implementation-plan.md` for the discretisation and conventions.

# Phase 2 validation record

This is **synthetic verification plus a geography acquisition check**, not empirical validation. No measurement of a real
atmosphere was used, no real emissions or meteorology were used, and nothing here identifies any real emitter or says how
good the model is at predicting real concentrations. Real geography enters only as the study polygon, road / building /
facility geometry, outline and DEM context. Everything that moves air or pollutant is synthetic: **Real geography; synthetic
emissions and meteorology.**

Every number below is read from a file under `results/` or `data/belauri/`, from `docs/*.md` that cite such files, from the
`.research` records (read only), or from the tests. Terms are used as in `docs/phase2-plan.md`: *local* = emitted inside the study
polygon; *external* = outside it (explicit source or outer-inflow-boundary input); *international transboundary* = crossing a
verified national boundary (never asserted in this repository, see the border section); *indoor* = inside a building (separate
one-way-coupled box). The study polygon is not the simulation domain and not the outer inflow boundary.

Companion documents: `docs/solver-interchangeability.md`, `docs/transboundary.md`, `docs/geography.md`,
`docs/belauri-forward.md`, `docs/geodesign.md`, v0 record `docs/validation.md`.

## (a) Reproduce

All commands run from the repository root with `.venv/bin/python`. Use a fresh `--out`; non-empty directories are refused.
Times are those recorded in the cited run files (one measurement on the machine in `docs/validation.md`, not a benchmark).

| What | Command | Recorded time | Result files |
|---|---|---|---|
| Solver regression, before / after | `scripts/capture_regression.py --out results/solver-regression/before` (then after the change), `scripts/compare_runs.py before after` | 1m47s before, 1m45s after (EXP-008 runs) | `results/solver-regression/{before,after,comparison.json}` |
| External vs local study (production) | `python -m pravah benchmark --config configs/transboundary.json --out results/<name>` | 27.8 min (`results/transboundary-rerun`) | `results/transboundary-rerun/`, tables in `docs/transboundary.md` via `scripts/transboundary_tables.py` |
| External vs local study (smoke) | `python -m pravah benchmark --config configs/transboundary_smoke.json --out <empty>` | about 65 s (`docs/transboundary.md`) | plumbing only |
| Geography acquire (network) | `python -m pravah geo fetch --config configs/belauri_geo.json` | 52 min with the Overpass mirror fallback (EXP-009 RUN-0016); two earlier attempts (10 and 7 min) were cancelled | `data/belauri/raw/` (git-ignored) |
| Geography prepare (offline) | `python -m pravah geo prepare --config configs/belauri_geo.json` | 14 s (RUN-0022) | `data/belauri/{manifest.json,processed/}` |
| Geography report | `python scripts/geo_report.py --manifest data/belauri/manifest.json --out results/geography` | 5 s (RUN-0023) | `results/geography/` |
| Relief strip | `python scripts/geo_relief_profile.py` (bounded public tile reads, network) | 8 s (RUN-0021) | `results/geography/relief_profile.{json,png}` |
| Belauri forward | `python -m pravah benchmark --config configs/belauri_forward.json --out results/<name>` | 285 s (`results/belauri-forward/timings.json`) | `results/belauri-forward/` |
| Geographic design | `python -m pravah benchmark --config configs/geodesign.json --out results/<name>` | 708 s, peak 354 MB (`results/geodesign/timings.json`) | `results/geodesign/` |
| Geographic design robustness | `python -m pravah benchmark --config configs/geodesign_robust.json --out results/<name>` (`robust.layouts_from` = the geodesign run) | see `ROBUSTNESS_RESULTS` below | slot |
| Notebooks 04-06 | `.venv/bin/python scripts/verify_notebooks.py notebooks/0N_*.ipynb` | see `FINAL_NOTEBOOK_RESULT` below | slot |
| Tests, types | `.venv/bin/pytest -q`, `.venv/bin/pyright` | see `FINAL_TEST_RESULT` below | slot |

## (b) Tolerances and what each test class establishes

Tolerances are the ones written in the tests; none was added here. "Interface test" means it shows a call reaches a solver,
not that a physical result is right.

| Test file | What it establishes | Tolerances as asserted |
|---|---|---|
| `tests/test_solver_routing.py` | Every workflow (simulate, response bases, direct fit, case building, benchmark / alpha-selection / mismatch truth and inference roles, CLI via a `module:callable`) reaches the injected solver; FiPy is not called when another solver is given (sentinel); a +1 ug/m^3 marker solver detects hidden FiPy calls (interface test, not physics); `FipyForward(convection="upwind")` changes results and cache keys; a solver that does not declare `linear` is refused by the response cache | marker offsets compared with `atol` 1e-12 (fields) and 1e-9 (readings) |
| `tests/test_forward_interface.py` | `check_field_result` rejects broken results; an alternative backend recovers with its own responses; cache keys and disk caches never mix backends; contract violations are rejected | see file |
| `tests/test_contracts.py`, `tests/test_config.py` | `RasterSource` normalised like a Gaussian and serialised by hash; loader verifies hashes and shapes; `StudyRegion` weights are exact cell overlaps; origin tags must match geometry; `param_table` orders emissions then inflow with units; cache key depends on raster content and group; config round trip and validation | see files |
| `tests/test_bases.py` | Inflow bases act only on inflow faces and flip with wind reversal; mass budget closes; superposition and linearity; uniform basis equals scalar background; stream-function uniform wind equals the legacy path; constant solution invariant under a divergence-free non-uniform wind; tangential faces (|u.n| < 1e-12) are neither inflow nor outflow | round-off identities 1e-12 (relative to the field maximum); superposition 1e-10; budget residual 1e-9 (docstring of the file) |
| `tests/test_attribution.py` | Parameter bookkeeping by kind; tags sum to the total and to a direct simulate (with background, IC and wind reversal); `external_fraction` named quantity and its undefined cases; component errors never mix ug/s with ug/m^3; noise-free recovery; identifiability flags (no-arrival column is small, rank / effective rank / confounded) | tag sums 1e-10 (relative to the field maximum), responses vs direct 1e-6, noise-free recovery 1e-4 (1e-3 in one case) |
| `tests/test_transboundary.py` | Scenario set, no double counting of basis and explicit source, seed guards, `fit_sources` cannot see tags, alpha selected on validation only, studies run end to end on the smoke profile, equivalence study uses matching study-relative sites, no-arrival ratio < 1e-3 | see file |
| `tests/test_indoor.py` | Indoor box: exact piecewise-linear integration against the analytic exponential, steady state, `a` -> 0 and `k_dep` -> 0 limits, an independent ODE integration (DOP853), one-way coupling, indoor-only ambiguity (cosine 1) | analytic cases 1e-12 relative; ODE comparison 1e-10; limits 1e-8 |
| `tests/test_geo.py` (synthetic GIS) | Projection round trip, polygon-cell fractions sum to the polygon area, line / point rasterisation, DEM aggregation | round trip < 1e-6 m; area relative error <= 1e-9 |
| `tests/test_geo_real.py` | Integrity of the prepared Belauri data: every raw and processed file re-hashed against `data/belauri/manifest.json`, mask area equals polygon area, normalised sources sum to 1, round trip | area 1e-9 relative; round trip < 1e-6 m |
| `tests/test_belauri.py` | Tag sum equals a direct combined solve on the Belauri grid; budget residual; mask and crop consistency; numerical-diffusion estimate reproduces the documented 0.606 / 0.361 excess; plotting-only layers change nothing | 1e-10 (fields), 1e-9 (budget), 3 % (diffusion estimate) |
| `tests/test_design_objectives.py`, `tests/test_design.py` | Component objectives as exact quadratic forms; role guards; constraints and categories; greedy first pick equals the exhaustive argmin; attribution greedy against exhaustive search on a tiny fixture (the gap is printed, not asserted small) | see files |
| `tests/test_geodesign.py` | Ensembles pairwise disjoint (`check_ensembles`); declared draw distributions; candidate IDs stable, hypothetical sites excluded by default; constraints respected in every layout; alpha and layouts independent of the test ensemble; the inverse never receives truth tags; greedy layouts are prefixes of one trace; deterministic reruns | see file |
| `tests/test_geodesign_robust.py` | Layouts and alphas re-used verbatim (selection functions patched to raise); matched rows equal the main run's rows; each condition changes one thing; flags equal their matched partners under mismatch (flags cannot detect a wrong model); a config that differs from the main run is refused | see file |

What the tests do **not** establish: agreement with any real data; correctness of any alternative to FiPy (none exists); that the
Belauri polygon is an official boundary.

## (c) Measured results

### c1. Solver regression (EXP-008, F-009)

`results/solver-regression/comparison.json`: **21 files compared, 0 differ** (`n_files` 21, `n_differences` 0). The files are the
reference field (`concentration.nc`, `budget.nc`), `metrics.json`, the inverse metrics (linear and direct), the smoke benchmark
(raw metrics, summaries, paired tables, `layouts.json`, design scores), the smoke alpha selection (`alpha_selected.json`,
`validation_raw.csv`, `layouts.json`) and the smoke mismatch tables. Volatile keys (times, paths, git state, `backend` additions)
are excluded. Two independent captures of the old code also compared 21 files, 0 differ (the baseline is deterministic).
**Negative control:** the same simulate job with `FipyForward(convection="upwind")` is flagged: max concentration 0.1608 ->
0.1517 ug/m^3 (the 0.1615 / 0.1476 in `docs/extending.md` example 9 is a different config, `smoke.json`). Caveat: routing is proven with FiPy-derived stand-ins (recorder, +1 marker, upwind), not an independent
solver. Old on-disk response caches are invalidated by the new key format.

### c2. External vs local study (EXP-011 rerun, F-012)

Reference rather than duplicate: the generated tables (recovery, prior sensitivity, equal-budget layout comparison, scale
sensitivity, boundary-versus-plume equivalence, outer-domain size, confounded, no-arrival, wind mismatch, omitted source) are in
`docs/transboundary.md` between `PRODUCTION_RESULTS_BEGIN/END`, produced from `results/transboundary-rerun` (27.8 min; 4
validation + 6 test cases, 3 noise reps; alpha chosen on validation only). Headline points taken from those tables: with 8
interior sensors at 2 % noise the local / external amplitude relative errors are 0.0105 / 0.00595; whole-side boundary bases
reduce the study-region RMSE / peak from 0.172 (local sources only) to 0.118 (uniform basis) and 0.114 (four side bases) when
the basis onset matches the plume's, and to 0.135 / 0.133 when it does not, against 0.00925 with the plume explicit; the
wind-mismatch case has total RMSE / peak 0.486 and an external fraction of 0.212 against a true 0.397 **with no identifiability
flag**; with no arrival the alpha = 0 estimate of E1 is 7.34e10 ug/s (true 2e5) and flagged `rank_deficient`,
`unidentifiable:E1`. Confounded example (constructed, 5 m/s, 3.8 km upwind): at 2 % noise the flag set is `confounded` only and
the split is recovered (amplitude errors 0.05 / 0.0297), at 30 % noise errors are 0.315 / 0.207.
The two different representations (explicit external source vs a boundary basis) are not interchangeable: the bases removed
about 22-34 % of the error caused by the omitted plume, not all of it.

### c3. Geography verification (EXP-009; `docs/geography.md`, `data/belauri/manifest.json`, `results/geography/`)

| Check | Result |
|---|---|
| Study polygon | geoBoundaries gbOpen NPL ADM3 `Belauri` (shapeID 92635248B27542858320702; source Open Data Nepal / MoFAGA, data year 2019) |
| `boundary_status` | `verified` (manifest and `geography_report.json`) |
| Name / shapeID, validity | match / valid |
| Area | 122.96 km^2 against 123 expected (+/- 10 %) |
| Bounding-box deviation | 0.000 deg (tolerance 0.03) |
| OSM administrative relation R5927493 | IoU 0.9384; Hausdorff 238 m; area 122.96 vs 122.89 km^2 |
| Nominatim locator | matches the OSM relation at 0.0 km |
| **ADM2 coverage (informational)** | **fails: 71.7 % of the polygon lies inside ADM2 `Kanchanpur`**. The ADM2 layer is the 2006 vintage and coarse (36 vertices), the ADM3 the 2019 one; the cause given in `docs/geography.md` is an inference, not verified. The parent-district requirement is met instead by the OSM relation (IoU >= 0.9), the hierarchy naming Kanchanpur and no other district overlapping > 1 % (measured 0.00 %). The check was extended with this documented evidence rather than loosening the 90 % threshold. |
| CRS | WGS84 (lon, lat), projected EPSG:32644 (UTM 44N); round-trip error 0.0 m |
| Grids | 500 m: 66 x 65 cells, 584 polygon cells; 250 m: 132 x 130, 2147 polygon cells; mask area equals polygon area to relative error ~1e-15 |
| DEM | Copernicus GLO-30 **DSM** (surface, not bare earth), 0 of 1,614,830 pixels nodata; native pixel about 27.1 x 30.7 m, aggregated to 500 / 250 m |
| Elevation (500 m cells) | inside polygon 167.7-204.4 m (range 36.7 m); whole domain 161.9-241.3 m (range 79.4 m); mean slope in the polygon 0.21 deg |
| Relief strip (`relief_profile.json`) | median 177.9 m within 8 km of the centroid; terrain north of the centroid first exceeds that by 50 / 100 / 200 / 300 / 500 m at 28.5 / 31.5 / 34.5 / 35.5 / 37.0 km; highest strip point 1946.7 m at 77.5 km north. So the first substantial relief begins roughly 20 km beyond the municipality's northern edge, outside the simulated domain |
| Border | the polygon touches the Nepal ADM0 boundary (distance 0); centroid-to-border 3.14 km (WSW); 37 % of the 72.5 km perimeter within 50 m of it. ADM0 line and ADM3 polygon share a data lineage, so their coincidence is not independent evidence. The India outline is coarse (2,082 vertices against 66,082), so the external patch (2.92 km from the polygon edge) lies in no ADM0 polygon and is labelled `external_outside_home`, **not** international |
| OSM | 1,025 km road inside the polygon (2,207 km in the domain, 8,206 segments); 50,510 buildings; 1,020 residential landuse polygons; 98 facilities (60 inside). All four layers come from the `maps.mail.ru` Overpass **mirror** (main endpoint unavailable); completeness unknown |
| Files | every raw and processed file is sha256-recorded in the manifest and re-checked by `tests/test_geo_real.py` |

### c4. Belauri forward (EXP-012, F-011; `results/belauri-forward/`, 285 s, clean commit)

Synthetic emissions road 1.5e6, built-up 1.0e6, external patch 5.0e5 ug/s, external patch off until 7200 s, then on (step profile); 500 m, dt 100 s, T = 8 h,
K = 100 m^2/s, lambda = 1/21600 1/s, H = 500 m (all nominal). Study-region (municipality mask) values:

| Wind (prescribed, synthetic) | Mean (ug/m^3) | Peak (ug/m^3) | External fraction |
|---|---|---|---|
| from_west_southwest | 0.1245 | 0.301 | 0.2207 |
| from_north | 0.1098 | 0.261 | 6.95e-05 |
| from_east | 0.1101 | 0.282 | 2.01e-04 |
| reversal (WSW, then ENE at T/2) | 0.1457 | 0.395 | 0.0507 |
| channelled_demo (stream function; not terrain physics) | 0.1524 | 0.370 | 0.1940 |

Conservation (`consistency.json`): tags equal the direct combined solve to <= 3.2e-15 relative, budget residual <= 1.9e-14, no
negative cells. Arrival of the external tag (first exceedance of 0.005 ug/m^3) occurs only under WSW and reversal winds: 789 s after
onset at the nearest polygon cell, 2020 s at the nearest facility cell, 2627 s at the centroid (the farthest polygon and facility
cells are not reached in 8 h).

| Outer domain (2 km buffer vs 8 km) | Mean change | External fraction (small / standard) |
|---|---|---|
| steady winds (WSW, north, east, channelled) | abs(change) < 1e-4 % | agree to 5 digits |
| reversal | **-10.5 %** (study RMSE of total / peak 0.0693) | 0.0565 / 0.0507 |

| Mesh | dt (s) | Courant | Mean change vs 250 m | Peak change vs 250 m | K_eff / K | K_eff (m^2/s) |
|---|---|---|---|---|---|---|
| 500 m | 100 | 0.30 | +1.03 % | -3.26 % | 4.88 | 488 |
| 500 m | 50 | 0.15 | +1.07 % | -3.26 % | 4.32 | 432 |
| 250 m (reference) | 50 | 0.30 | 0 | 0 | 2.53 | 253 |

K_eff / K = 1 + the Fourier estimate `numerical_diffusion_estimate` (exact along a grid axis only; an approximation for oblique wind).
Halving dt at 500 m changes the mean by 0.04 percentage points. Doubling the physical K changes the mean by +0.15 %, halving it by
-0.02 %, because the scheme's numerical spreading is several times K. The DEM, hillshade, slope and profiles are plotting-only: replacing
them by constants leaves the `Scenario` identical and the concentration change exactly 0 (`plotting_only_check`).

### c5. Geographic design (EXP-013, F-020; `results/geodesign/`, 708 s, peak 354 MB)

Setup: 6 design (seed 101), 6 validation (202), 6 test (303) scenarios; noise 5 % and 15 % of a documented signal scale (test:
sigma 0.0158 and 0.0475 ug/m^3, from `noise_levels.csv`), 3 noise reps; 54 interior candidates (1.5 km lattice in the polygon),
47 domestic reference candidates, 41 across-the-border hypothetical candidates (`candidates.csv`) **excluded by default** (101 available);
budgets 1, 2, 3, 5, 8; alpha = 0.01 selected on **validation** cases for both noise levels and frozen before the test ensemble
existed (`alpha_selected.json`). No budget was infeasible (`infeasible_budgets.json`). Tables below: test ensemble, selected
alpha, standard regime, `summary.csv`.

Mean total regional RMSE (ug/m^3) with its **sd** (a standard deviation of 18 rows = 6 cases x 3 reps; random: 360 rows; not a
confidence interval; rows are not independent). 5 % noise, `interior_only`:

| Method | 1 sensor | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| greedy_total | 0.0223 (sd 0.0246) | 0.0093 (sd 0.0031) | 0.00525 (sd 0.00235) | 0.00353 (sd 0.00180) | 0.00272 (sd 0.00132) |
| greedy_attribution | 0.0105 (sd 0.0054) | 0.0065 (sd 0.0029) | 0.00542 (sd 0.00176) | 0.00370 (sd 0.00170) | 0.00327 (sd 0.00189) |
| spread | 0.0148 (sd 0.0081) | 0.0117 (sd 0.0092) | 0.00580 (sd 0.00363) | 0.00512 (sd 0.00217) | 0.00323 (sd 0.00159) |
| random (mean over 20 layouts) | 0.0250 (sd 0.0255) | 0.0135 (sd 0.0151) | 0.0103 (sd 0.0104) | 0.00666 (sd 0.00777) | 0.00328 (sd 0.00320) |

Attribution error (dimensionless, `0.5 (MSE_local/s_local + MSE_external/s_external)`, scales from design truths only), 5 %,
`interior_only`: greedy_attribution 0.132 / 0.087 / 0.072 / 0.048 / 0.042; greedy_total 0.285 / 0.107 / 0.071 / 0.045 / 0.035;
spread 0.192 / 0.169 / 0.076 / 0.049 / 0.038; random 0.283 / 0.172 / 0.132 / 0.086 / 0.039.

Paired differences, 5 % noise, `interior_only` (negative = first method lower; mean and sd of the difference over 18 pairs;
cases won out of 6):

| Comparison (total RMSE) | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| greedy_total minus spread | +0.0075 (sd 0.0247), 3/6 | -0.0024 (0.0093), 2/6 | -0.0006 (0.0039), 4/6 | -0.0016 (0.0032), 5/6 | -0.0005 (0.0018), 5/6 |
| greedy_total minus random mean | -0.0027 (0.0168), 5/6 | -0.0042 (0.0074), 4/6 | -0.0051 (0.0077), 5/6 | -0.0031 (0.0044), 6/6 | -0.0006 (0.0015), 4/6 |
| greedy_attribution minus greedy_total | -0.0118 (0.0247), 5/6 | -0.0028 (0.0040), 5/6 | +0.0002 (0.0025), 3/6 | +0.0002 (0.0021), 3/6 | +0.0006 (0.0017), 1/6 |

Spatial-block held-out error (`heldout_block_rmse`, ug/m^3; a true hold-out only in the `block_holdout` regime where layouts keep
2 km away from the held-out east third; 5 % noise, interior only): greedy_total 0.0191 / 0.0062 / 0.0043 / 0.0037 / 0.0029;
greedy_attribution 0.0098 / 0.0067 / 0.0044 / 0.0038 / 0.0031; spread 0.0099 / 0.0094 / 0.0082 / 0.0044 / 0.0030; random
0.0184 / 0.0093 / 0.0088 / 0.0048 / 0.0031.

15 % noise, `interior_only`, total RMSE: greedy_total 0.0366 / 0.0174 / 0.0156 / 0.0108 / 0.0083; greedy_attribution 0.0625 / 0.0298 /
0.0212 / 0.0086 / 0.0079; spread 0.0285 / 0.0214 / 0.0158 / 0.0116 / 0.0082; random 0.0422 / 0.0227 / 0.0182 / 0.0128 / 0.0083.

Reference sites: in the `with_reference` setting (at most 2 domestic reference sites, counted in the budget) greedy_total chose none,
greedy_attribution chose one at budget 5 and two at budget 8, and spread with reference sites has a higher total RMSE than
interior-only spread at budgets 3 and 5 (0.0083 vs 0.0058 and 0.0070 vs 0.0051). The paired with_reference-minus-interior_only tables are in
`paired.csv`.

**What these tables support, honestly.**

* Greedy layouts are **heuristics, not optimal**: they are chosen on 6 design scenarios, and a tiny exhaustive-search fixture in the
  tests checks greedy against the true minimum (v0 showed greedy design MSE at 3.7x the optimum on one fixture, `README.md`).
* Differences between methods are **modest and mixed**. Design-optimised layouts help mainly at 1-3 sensors, mostly by avoiding poor
  coverage (random is worst there); beyond 5 sensors differences are within the spread of the rows. greedy_total did **not**
  beat spread at 1 sensor (mean +0.0075, 3 of 6 cases) and its 2-sensor advantage is small.
* greedy_attribution has the lowest attribution error at 1-2 sensors, but at 15 % noise it is the **worst** method at 1-3 sensors on
  total RMSE; so "not worse on total error" holds at 5 % noise only. Selection on 6 design scenarios did not transfer reliably to
  6 test scenarios at small budgets.
* **Reference sites are not a clear win at equal budget.**
* The model is **numerically diffusive**: at the design mesh K_eff / K is about 2.4 at 0.8 m/s and about 9.4 at 2.5 m/s
  (`docs/geodesign.md`), and the Courant number reaches 0.49 at the highest design wind (above the 0.3 guideline of the forward
  study). These matched experiments share that diffusion between truth and inference, so they are optimistic.
* Matched, known-wind, 6 + 6 + 6 scenarios, 3 noise draws: descriptive only. Mapped accessibility is not permission, power,
  security or connectivity.

### Robustness and refinement (F2b)

<!-- ROBUST_BEGIN -->
Run `geodesign-robust` (fixed design-stage layouts re-evaluated, never reselected); interior-only, 5% noise, selected alpha, mean total RMSE (ug/m3) of `greedy_total`.

| condition | 1 sensors | 2 sensors | 3 sensors | 5 sensors | 8 sensors |
|---|---|---|---|---|---|
| boundary_time_profile | 0.0221 | 0.0092 | 0.0055 | 0.0043 | 0.0036 |
| hidden_wind_error | 0.0436 | 0.0385 | 0.0442 | 0.0462 | 0.0312 |
| matched | 0.0223 | 0.0093 | 0.0052 | 0.0035 | 0.0027 |
| omitted_external | 0.0233 | 0.0108 | 0.0071 | 0.0048 | 0.0041 |
| omitted_facility | 0.0303 | 0.0170 | 0.0144 | 0.0156 | 0.0145 |
| refined_truth | 0.0245 | 0.0110 | 0.0082 | 0.0081 | 0.0066 |
| unseen_wind | 0.0446 | 0.0418 | 0.0412 | 0.0413 | 0.0294 |
| unseen_wind_declared | 0.0174 | 0.0128 | 0.0047 | 0.0033 | 0.0028 |

**Identifiability flags versus error (all methods and budgets)**: fraction of rows whose flags equal the matched rows, and the median error ratio to matched.

| condition | frac_flags_equal_matched | rmse_ratio |
|---|---|---|
| boundary_time_profile | 1.000 | 1.019 |
| hidden_wind_error | 1.000 | 3.504 |
| matched | 1.000 | 1.000 |
| omitted_external | 1.000 | 1.130 |
| omitted_facility | 1.000 | 1.679 |
| refined_truth | 1.000 | 1.213 |
| unseen_wind | 1.000 | 3.557 |
| unseen_wind_declared | 0.4065 | 0.8465 |

**Do method rankings survive the condition?**

| condition | fraction of method contrasts keeping their matched sign |
|---|---|
| boundary_time_profile | 0.9708 |
| hidden_wind_error | 0.7208 |
| omitted_external | 0.9292 |
| omitted_facility | 0.9083 |
| refined_truth | 0.9125 |
| unseen_wind | 0.7208 |
| unseen_wind_declared | 0.7417 |

**Numerical diffusion at the design mesh (K = 100 m2/s is the nominal physical value; not tuned)**

| mesh | speed_label | speed_m_s | peclet | courant | K_eff_over_K |
|---|---|---|---|---|---|
| design | design_min | 0.894 | 4.47 | 0.179 | 2.69 |
| design | design_mean | 1.7 | 8.48 | 0.339 | 5.68 |
| design | design_max | 2.45 | 12.2 | 0.49 | 9.12 |
| design | validation_min | 0.952 | 4.76 | 0.19 | 2.87 |
| design | validation_mean | 1.53 | 7.64 | 0.305 | 4.99 |
| design | validation_max | 1.99 | 9.94 | 0.398 | 6.95 |
| design | test_min | 0.812 | 4.06 | 0.162 | 2.43 |
| design | test_mean | 1.81 | 9.05 | 0.362 | 6.17 |
| design | test_max | 2.41 | 12.1 | 0.482 | 8.94 |
| design | reference_0.8 | 0.8 | 4 | 0.16 | 2.39 |
| design | reference_1.5 | 1.5 | 7.5 | 0.3 | 4.88 |
| design | reference_2.5 | 2.5 | 12.5 | 0.5 | 9.38 |
| design | reference_3.5 | 3.5 | 17.5 | 0.7 | 14.9 |
| design | unseen_wind_max | 3.41 | 17.1 | 0.682 | 14.3 |
| refined | design_min | 0.894 | 2.23 | 0.179 | 1.58 |
| refined | design_mean | 1.7 | 4.24 | 0.339 | 2.9 |
| refined | design_max | 2.45 | 6.12 | 0.49 | 4.57 |
| refined | validation_min | 0.952 | 2.38 | 0.19 | 1.66 |
| refined | validation_mean | 1.53 | 3.82 | 0.305 | 2.58 |
| refined | validation_max | 1.99 | 4.97 | 0.398 | 3.51 |
| refined | test_min | 0.812 | 2.03 | 0.162 | 1.49 |
| refined | test_mean | 1.81 | 4.53 | 0.362 | 3.13 |
| refined | test_max | 2.41 | 6.03 | 0.482 | 4.48 |
| refined | reference_0.8 | 0.8 | 2 | 0.16 | 1.47 |
| refined | reference_1.5 | 1.5 | 3.75 | 0.3 | 2.53 |
| refined | reference_2.5 | 2.5 | 6.25 | 0.5 | 4.7 |
| refined | reference_3.5 | 3.5 | 8.75 | 0.7 | 7.44 |
| refined | unseen_wind_max | 3.41 | 8.53 | 0.682 | 7.17 |

**250 m re-selection versus 500 m layouts (3 design cases, 5% noise)**

| method | budget | jaccard | n_shared_sites | mean_nearest_site_distance_m |
|---|---|---|---|---|
| greedy_total | 3 | 0.2000 | 1 | 1500 |
| greedy_total | 5 | 0.1111 | 1 | 1412 |
| greedy_attribution | 3 | 0.00e+00 | 0 | 3004 |
| greedy_attribution | 5 | 0.00e+00 | 0 | 2588 |
<!-- ROBUST_END -->

## (d) Requirement / status table

Status: **done** = implemented, tested and a result file exists; **partial** = part of the requirement is met, the caveat says what
is not; **blocked** = could not be done. Findings F-009, F-011, F-012, F-013 were `preliminary` when this section was first written (see REVIEW STATUS for their independent verdicts and replacements) (independent
review status is in the slot below). "Pending" means a result is expected from the coordinator and nothing is claimed.

| Requirement | Status | Evidence | Caveat |
|---|---|---|---|
| **A. Solver interchangeability** | | | |
| A1 CLI `--solver module:callable`, `--truth-solver`, `--inference-solver`, `--solver-kwargs`, config `backends` | done | `docs/solver-interchangeability.md`; `tests/test_solver_routing.py`; `docs/extending.md` examples 9-10 | no independent non-FiPy solver exists |
| A2 Truth vs inference roles | done | routing tests (benchmark, alpha selection, mismatch truth and inference) | tested with FiPy-derived stand-ins |
| A3 Cache keys include solver identity | done | `inverse.cache_key` uses `BackendInfo`; `docs/extending.md` example 10 shows different keys | old on-disk caches are re-computed |
| A4 Linearity is an explicit capability | done | cached route refuses an undeclared backend (test; example 10) | declaring `linear` is the author's claim; not tested by the framework |
| A5 Regression: FiPy results unchanged | done | `results/solver-regression/comparison.json`: 21 files, 0 differ; negative control flagged | one machine; other platforms not tested |
| **B. External vs local** | | | |
| B1 Two representations (explicit external source; boundary basis) | done | `config.py` (`GaussianSource.group`, `InflowBasis`); `docs/transboundary.md` | whole-side bases only; they remove 22-34 % of the omitted-plume error (c2) |
| B2 Origin tags and attribution | done | `attribution.contributions` / `external_fraction`; `tests/test_attribution.py` | concentration share for one wind history; not a property of "the place" |
| B3 Boundary bases, wind reversal | done | `tests/test_bases.py` | |
| B4 No-arrival case | done | `docs/transboundary.md` no-arrival table | estimate is noise or the prior, not a recovery |
| B5 Confounded case | done | `special/diagnostics.json` and table in `docs/transboundary.md` | constructed example; flag is pairwise |
| B6 Prior sensitivity (and scale as prior width) | done | prior and scale-sensitivity tables | strong prior can lower error under a wrong model by shrinking to zero; not better inference |
| B7 Domain-size study | done | outer-domain table (A 5 km vs B 10 km buffer) | truncation mixed with a moved forcing for `mixed` |
| B8 Wind mismatch / omitted source | done | wind-mismatch and omitted-source tables | no identifiability flag detects a wrong wind |
| B9 Indoor box | done | `indoor.py`; `tests/test_indoor.py` | demonstration building parameters; separate from the outdoor objective |
| B10 Notebook 04 | done | `notebooks/04_transboundary_pollution.ipynb`; smoke-sized, about one minute (`docs/transboundary.md`) | final clean-kernel result: see `FINAL_NOTEBOOK_RESULT` |
| **C. Geography** | | | |
| C1 Verified Belauri boundary | **partial** | `boundary_status = verified` in the manifest; ADM3 polygon + OSM relation cross-check (IoU 0.9384) | **authoritative Survey Department of Nepal boundary NOT obtained**; ADM2 coverage check fails (71.7 %) and is informational |
| C2 DEM | done | Copernicus GLO-30 DSM, `results/geography` | DSM, no vertical validation; relief >20 km north of the municipality is outside the domain; DEM is context only |
| C3 Features (roads, buildings, facilities) | done | manifest counts; `data/belauri/processed` | OSM completeness unknown; all layers via an Overpass mirror |
| C4 CRS and projection | done | EPSG:32644, round trip 0.0 m, tests | |
| C5 Provenance manifest | done | `data/belauri/manifest.json` (URLs, release / commit, retrieved UTC, licences, sha256, CRS, assumptions); tests re-hash | Nepal / India border position not independently verified |
| C6 International border classification | partial | border distances, origin classes in the manifest | India outline coarse; the external patch is `external_outside_home`, not asserted international |
| C7 Notebooks run offline after prepare | done | notebooks 05-06 read `data/belauri/processed` only | needs the prepared data (tests skip without it) |
| C8 Belauri forward and notebook 05 | done | `results/belauri-forward`; `notebooks/05_belauri_forward.ipynb`; c4 | synthetic winds and emissions; K_eff / K 2.5-4.9 |
| C9 Terrain-resolved wind | **blocked / not done** | none | terrain dynamics are **unresolved**: relief is a few tens of metres inside the domain and begins >20 km north of it; no terrain-following flow, slope winds, blocking or valley channelling; needs WRF / CFD / measured winds |
| **D. Deployment (geographic design)** | | | |
| D1 Candidate sets as GeoJSON and CSV | done | `results/geodesign/candidates.{csv,geojson}` with manifest sha256 | mapped accessibility is not permission, power, security or connectivity |
| D2 Four methods (random, spread, greedy-total, greedy-attribution) | done | `results/geodesign/summary.csv`, c5 | greedy is a heuristic |
| D3 Two settings (interior only; with <= 2 reference sites) | done | `summary.csv`; hypothetical setting exists but is off by default | reference sites not a clear win |
| D4 Tiny exhaustive fixture | done | `tests/test_geodesign.py`, `tests/test_design_objectives.py` | small fixture only |
| D5 Held-out random points and spatial blocks | done | `heldout_*` columns, `infeasible_budgets.json` (none infeasible) | block hold-out is real only in the `block_holdout` regime |
| D6 Mismatch / robustness of the fixed layouts | partial | code `geodesign_robust.py`, tests, smoke profile; production results in the `ROBUSTNESS_RESULTS` slot | production numbers pending from the coordinator when written |
| D7 Refinement (250 m) | partial | `numerical_diffusion.*`, `refined_truth` condition and optional 250 m re-selection in code and tests | production results pending (slot) |
| D8 Notebook 06 | done | `notebooks/06_belauri_sensor_design.ipynb` (about 2.5 min for both smoke profiles, `docs/geodesign.md`) | final clean-kernel result: `FINAL_NOTEBOOK_RESULT` |
| D9 Exports with provenance | done | `layouts.{csv,geojson}` with objective, alpha, seeds, config hash and the provisional label | not field-validated |
| **Cross-cutting** | | | |
| Empirical validation against measurements | **not done** | none | no measurement data of any kind was used |
| Independent finding reviews | partial | 5 supported, 5 needs_work (replaced by F-016..F-021, preliminary) | see REVIEW STATUS above |

**Independent review status (Research Panel, reviewer agent `scientific-reviewer`, Opus; not the author of the findings).**
Recorded verdicts: *supported* F-002 (high), F-006 (medium), F-007 (medium), F-009 (high), F-012 (medium); *needs_work* F-001, F-008, F-011,
F-014, F-015. Each `needs_work` finding was replaced by a corrected, still **preliminary** finding: F-016 (forward scheme; observed spatial order
about 1.4 in the periodic test), F-017 (v0 mismatch; per-case counts mixed), F-018 (Belauri forward; the plume also arrives under the
reversal and channelled winds), F-019 (boundary bases; values in the right order), F-020 (geographic design; ranking depends on noise level and on
the design noise used for selection; reference-site use depends on it too), F-021 (robustness; flags saturated by construction, ranking metric
unresolved against noise). F-005 (failure-kind) and F-016-F-021 are `preliminary`: no independent verdict has been recorded for them. Decisions:
D-002 stands (v0 baseline); D-003 records this phase's baseline. The reviewer found **no blocker**, no leakage of the test ensemble into alpha,
layouts or attribution scales, and no mismatch between the doc tables and the result CSVs.

Final run on the committed tree (coordinator, 2026-10-10, macOS, Python 3.12.7): `.venv/bin/pytest -q` -> **447 passed**, 15 warnings in 783.5 s
(13 min 3 s); `.venv/bin/pyright` -> **0 errors, 0 warnings, 0 informations**; legacy FiPy regression (`scripts/compare_runs.py` before/after the
solver refactor and after every later physics/inverse/design change) -> **21 files compared, 0 differ**, re-run independently by the reviewer.

Clean-kernel execution (`scripts/verify_notebooks.py`, all six): 01_transport PASS 21.5 s; 02_inverse PASS 20.6 s; 03_sensor_design PASS 59.2 s;
04_transboundary_pollution PASS 62.5 s; 05_belauri_forward PASS 188.8 s; 06_belauri_sensor_design PASS 179.3 s. The stored notebooks are no longer
output-free in the working tree because they were executed and saved from the VS Code Jupyter extension; those saved versions were not committed.

## (e) Known limitations and what this does not claim

* **Nothing here is empirical.** The truth and the fitted model share their equations and grid family, so recovery numbers are
  optimistic bounds; only the wind-mismatch, omitted-source and (in robustness) perturbed-truth cases probe model error.
* **No identification of any real emitter.** The external patch and external source are synthetic spatial priors on real geometry. The
  patch is not asserted to be on the Indian side (`external_outside_home`); no statement about any real border emitter is made.
* **Boundary status.** The polygon is verified against one independent open source (OSM relation) and a name / area / bbox check;
  it is not an official survey product, and the ADM2 coverage check fails (71.7 %), explained but not resolved.
* **Terrain is not in the dynamics.** DEM, hillshade and slope are context only (proved to change no number). No terrain-resolved wind
  or mixing depth; the DEM is a surface model.
* **Numerical diffusion.** The first-order scheme's effective diffusivity is 2.5-4.9 x K at the Belauri meshes and about 2.4-9.4 x K at
  the design mesh for the drawn winds; K is a nominal input and the physical K is not tuned.
* **Prescribed uniform winds** (and one synthetic stream-function demo); K, lambda, H nominal; one emission set per study.
* **Small ensembles.** 4 + 6 cases (external / local), 6 + 6 + 6 (design); sd is a standard deviation, never a confidence interval;
  win counts are descriptive; no significance tests.
* **OSM and sensors.** OSM completeness is unknown (absent roads are mapping gaps, not zero sources) and the layers come from a
  mirror that may lag the main database. Mapped accessibility is not permission, power, security, connectivity or safety.
* **Known profiles and scales.** The inversion is given the true temporal profiles and wind history; scale choices act as prior
  widths when alpha > 0.
* **Not tested:** bit-for-bit reproduction on other machines or library versions.


## Wrap-up log: what works, what does not, what was deliberately left

**Works (executed, verified, recorded).**
* Solver interchangeability: one `ForwardSolver`/`BackendInfo` interface used by simulate, direct fit, response bases, greedy case building,
  benchmark/mismatch (separate truth and inference solvers), alpha selection and the CLI; backend identity is part of every cache key; a replacement
  solver must declare `linear`, `boundary_bases`, `wind_field` (enforced). FiPy outputs are bit-identical before and after (21 files, 0 differ; the
  comparison detects a real alternate FiPy configuration).
* External vs local study (synthetic): inflow-concentration bases (µg/m³) and explicit external sources (µg/s) as separate parameter kinds, origin tags by
  linear superposition, concentration-derived external fraction, identifiability diagnostics, no-arrival and confounded cases, outer-domain study,
  literal indoor box (exact ODE solution, separate from the outdoor problem), notebook 04.
* Real Belauri geography: bounded fetch with cache and fallback endpoint, manifest with sha256/licences, boundary `verified` with the evidence and caveats
  listed in `docs/geography.md`, DEM/relief, OSM features, UTM 32644, 46 geo tests including real-data integrity tests, notebook 05.
* Belauri forward simulations and geographic sensor design with held-out evaluation, mismatch/robustness and a refinement study, notebook 06.
* Tests, types, notebooks as recorded above.

**Does not work / not established (stated, not hidden).**
* No terrain-coupled flow: relief begins about 20 km north of the polygon and the model is flat 2-D with prescribed synthetic winds. No empirical
  validation, no real emissions, no identification of real emitters.
* The authoritative survey-department boundary was not obtained: geoBoundaries ADM3 (government-derived) cross-checked against OpenStreetMap; the Indian
  outline in geoBoundaries is coarse, so country classification near the border is uncertain.
* Numerical diffusion dominates the nominal K at the Belauri meshes (K_eff/K about 2.4-9.4 at the design mesh over the drawn winds, higher at the unseen
  winds). Sensor-design results describe this numerically diffusive model; specific sites and the "differences beyond 5 sensors are within noise"
  statement are mesh-dependent.
* Layout rankings at small budgets are not resolved with 6 test scenarios: they depend on the noise level, and under unknown wind errors method
  contrasts are close to chance.

**Known issues from the final independent review that were NOT fixed (judged not worth another hour of reruns; each is a limitation of the shipped results).**
1. *Test noise level.* The sigma told to the estimator on the test ensemble is fraction x the test ensemble's own signal scale (0.0158 versus 0.0154 for
   design, 2.6% higher). A mild independence leak. Fix if needed: take sigma from the validation or design scale and rerun `geodesign` and `geodesign_robust`
   (about 12 + 45 minutes of compute).
2. *Identifiability flags are saturated.* `weakly_constrained` fires on every row because the emission scale (1e5 µg/s) is small next to the true
   emissions (0.75-2.25e6 µg/s), and "flags equal matched" holds by construction (flags depend only on the sensitivity matrix and sigma). They carry no
   information in the geodesign runs. A data-based check (weighted-residual chi-square per degree of freedom) is the proposed replacement.
3. *Ranking-stability metric.* It counts contrasts whose matched difference is exactly zero as "kept" (30 of 240 rows per condition) and has no noise
   baseline; excluding those rows lowers hidden/unseen-wind agreement from 0.72 to 0.68. Read it as "rankings were never resolved", not "wind error flips them".
4. *Selection stability under refinement.* The 250 m re-selection used 3 fine design cases against 6, with no 500 m resampling baseline, so Jaccard 0-0.2
   cannot be separated from ordinary greedy instability.
5. *Held-out sites.* The "random" held-out points are only guaranteed more than 1 m from a candidate (measured minimum 220 m, median 614 m); the block regime
   holds out sensor locations, but the greedy objective still integrates the block area on design cases. The "per added sensor" rows for `random` compare
   non-nested layouts.
6. *Geography wording.* The OSM relation and the Nominatim hierarchy are the same source (OpenStreetMap), so the independence is between geoBoundaries and
   OpenStreetMap, not between two open sources; the "no other district overlaps" test is weak because 28% of the polygon lies outside every ADM2 polygon;
   14 of the candidate sites lie inside the coarse Indian outline but are not flagged per candidate.
7. *Provenance.* The two geodesign production runs record `git_dirty` with a 12-character diff hash but the patch itself is not archived; I checked that
   only files outside the executed code path changed between the run revision and HEAD.
8. *Unexecuted work.* A worker's attempt to fix items 1-7 (including a data-based chi-square check, candidate flags, a patch archive and two offline analysis
   scripts) is preserved in the git stash named `G2-unshipped` and was not merged or rerun.

**One evidence-motivated next experiment.** Add a data-based model check (chi-square per degree of freedom of the weighted residual) to the robustness study
and re-run it with the test noise level fixed from the validation scale: the robustness results say the dominant risk to any sensor design here is the
meteorology the estimator is told, and nothing the pipeline currently reports detects a wrong wind.

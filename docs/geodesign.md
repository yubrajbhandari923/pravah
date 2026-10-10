# Geographic sensor design (Stage F2a)

Real geography; synthetic emissions and meteorology.  Everything below is a **provisional simulation design, not
field-validated**.  Code: `src/pravah/geodesign.py`; configs: `configs/geodesign.json` (production) and
`configs/geodesign_smoke.json` (plumbing check); tests: `tests/test_geodesign.py`.

## Terms
- **Candidate**: a possible sensor position with a stable ID, category and flags.  **Layout**: a set of candidates (a
  sensor budget = its size).  **Setting**: which candidates may be used.  **Regime**: `standard`, or `block_holdout`
  (layouts built with an exclusion radius around the held-out block).
- **Total error**: study-area x time weighted RMSE of the reconstructed concentration (ug/m3).  **Component errors**:
  the same for the local (road + built-up) and external (patch + inflow + background) parts of the field, from the tags
  of the truth responses.  **Attribution objective**: `0.5 (MSE_local / s_local + MSE_external / s_external)`, the scales
  `s_*` being the mean-square true component fields of the DESIGN cases only (dimensionless; `design.attribution_scales`).
- **Greedy** layouts add, one site at a time, the site that most lowers the design objective.  They are heuristics and are
  never called optimal (a tiny exhaustive-search fixture in the tests checks greedy against the true minimum).

## Setup
Inference model on the Belauri 500 m grid (66 x 65): theta = [road (local, ug/s), builtup (local, ug/s), external_patch
(external, ug/s), bg (uniform inflow basis on all four sides, ug/m3)], patch and bg onset `(0,0),(7200,1)` s.  T = 28800 s,
dt = 100 s, output every 900 s, sensors read every 1800 s from 1800 s, `t_exclude` = 7200 s, K = 100, lam = 1/21600,
H = 500 (nominal).  Metrics use the municipality area-fraction mask x trapezoid time weights.  Matched experiments: the
estimator is given the case's declared wind; all other forcing is known; truth = a direct forward solve of the truth
scenario (the linear response sum agrees to round-off, checked per case); truth components use the same cached responses.

Ensembles (config `ensembles`): design (scenario seed 101, noise seeds 1000+), validation (202, 2000+), test (303, 3000+),
6 cases each; scenario seeds, noise-seed ranges, (theta, wind) draws and case names are pairwise disjoint
(`check_ensembles` raises).  A case draws road, builtup amplitudes x U(0.5, 1.5) of the config bases, the patch x U(0, 2)
(some cases nearly local-only, some external-dominated), bg U(0, 0.05) ug/m3, wind direction = one of 8 compass sectors
+-15 deg, speed U(0.8, 2.5) m/s; every third case (index % 3 == 1) has a second wind segment (independent draw) at a random
multiple of 3600 s in [T/3, 2T/3].  The speed range gives u dt/dx up to 0.5, above the 0.3 guideline of the forward study
(`max_courant` is recorded per case in `cases.csv`).  Noise sigma = fraction x signal scale (per ensemble, the median over
cases of the peak noiseless reading at the candidate sites), fractions 0.05 and 0.15; `noise_levels.csv` lists sigma in
ug/m3 and as percent of the scale.  Noise draws are common random numbers across noise fractions and layouts; the test
ensemble has its own seeds (fresh noise).  The inverse sees only noisy readings, the declared model inputs and the options.

## Candidates and assumptions
- `interior`: regular lattice (1500 m, origin = the prepared grid's south-west corner, IDs `int_c<row><col>` from the
  lattice indices, so stable under cropping) clipped to the municipality polygon.
- `reference_domestic`: lattice (3000 m, IDs `refd_...`) outside the polygon, inside the simulation domain (500 m inside
  the sensor hull) and inside the Nepal ADM0 polygon.  `reference_hypothetical` (`refh_...`): inside the domain but outside
  Nepal ADM0 (across the national border): flagged `hypothetical`, **excluded by default** (`available_default` false) and
  usable only in the labelled `hypothetical_sensitivity` setting (`settings.hypothetical: true`).
- Polygons come from the raw geoBoundaries files named in `data/belauri/manifest.json`; every raw file is sha256-verified
  against the manifest (no network).  Columns: lon/lat, local x/y, UTM x/y, `in_polygon`, `in_nepal_adm0`, distance to the
  nearest mapped road and facility (m), `on_or_near_road` (<= 100 m), `assumption_flags`.  Outputs `candidates.csv` and
  `candidates.geojson` (WGS84) carry the manifest sha256 and the config hash.
- Assumptions, repeated in the outputs: **mapped accessibility is not permission, power, security or connectivity**; **OSM gaps
  are not absence of feasible sites** (OSM completeness is unknown); international access is hypothetical.

## Settings and methods
Settings: (A) `interior_only`; (B) `with_reference`: the same total budget with at most 2 `reference_domestic` sites
(counted in the budget, `Constraints.max_per_category`); (C, off by default) `hypothetical_sensitivity`: also at most 2
`reference_hypothetical` sites.  Budgets 1, 2, 3, 5, 8.  Methods, each its own single-objective experiment: `random` (20
seeded feasible layouts per budget), `spread` (farthest point), `greedy_total` (objective `total`), `greedy_attribution`
(objective `attribution`).  Greedy selection uses DESIGN cases only, at the design sigma and the frozen alpha of the noise
fraction, and one trace to the largest feasible budget is truncated to every budget (the layout of budget b is the
b-prefix of the trace; tested).  Role guards of `design.py` apply everywhere.

## Alpha (ridge) selection
alpha (scaled units: q / q_scale, emission 1e5 ug/s, inflow 0.05 ug/m3, prior mean 0) is chosen per noise fraction on the
VALIDATION ensemble from the pre-declared grid {0, 1e-4, ..., 1e3} by the mean total regional RMSE over alpha-independent
layouts (random + spread, every setting, standard regime).  The choice is frozen in `alpha_selected.json` (which holds
nothing from the test ensemble) before the test ensemble is built.  Test results are also given for alpha = 0 and for a
strong prior (alpha = 1000) (`alpha_label`).

## Evaluation (independent test ensemble, fresh noise)
Per (layout, case, noise rep) row in `test_raw.csv`: total, local and external RMSE; attribution error; external-fraction error
(blank where the true or estimated fraction is undefined, i.e. total ~ 0); emission relative L2 error (`source_rel_err`)
and bg absolute error (`inflow_abs_err_bg`, ug/m3) separately (plus signed per-parameter errors); rank, effective rank,
condition and identifiability flags (`attribution.identifiability_report`, selected-alpha rows only); held-out RMSE.
`summary.csv` has means and sds (sd labelled as sd, never a CI; cases and reps are not independent).  `paired.csv`: paired
(case, rep) differences a - b (negative = a better; layouts of a method averaged first) for greedy_total vs spread and vs
the random mean, greedy_attribution vs greedy_total (and vs spread) on BOTH total RMSE and attribution error,
with_reference vs interior_only at equal budget, and consecutive-budget improvement (`per_added_sensor`); every row has the
number of cases won (`a_wins_cases` of `n_cases`).

## Held-out design
(i) 20 deterministic random points inside the polygon (not candidates); (ii) a spatial block: a fixed 1000 m lattice inside the
polygon cut into thirds by `design.heldout_blocks`, the east third being held out.  In the `block_holdout` regime every
layout respects `Constraints.exclusion_radius_m` (2000 m) around the block points.  Budgets that become impossible are not
silently dropped: `infeasible_budgets.json` lists them with reasons (also in provenance).  `heldout_block_rmse` is a
true hold-out only in the `block_holdout` regime.

## How to run
```
# smoke (about 40 s, cropped 2 km buffer, T = 7200 s, 2+2+2 cases; plumbing only)
.venv/bin/python -m pravah benchmark --config configs/geodesign_smoke.json --out <empty dir>
# production (about 25 min estimated; 18 cases x ~8 solves of ~4-5 s, then fits)
.venv/bin/python -m pravah benchmark --config configs/geodesign.json --out results/geodesign
```
Outputs: `config.json`, `provenance.json` (manifest sha256, seeds, alpha_selected sha256, git dirty hash), `timings.json` (stage
times, peak memory), `candidates.{csv,geojson}`, `noise_levels.csv`, `cases.csv`, `alpha_selected.json`, `validation_raw.csv`,
`layouts.{csv,geojson}` (objective, alpha, seeds, config hash and the provisional label per layout), `design_scores.csv`,
`infeasible_budgets.json`, `test_raw.csv`, `summary.csv`, `paired.csv`, and `figures/` (selected sites over the hillshade,
error versus count with +-1 sd, attribution versus total, held-out block map).  Non-empty directories are refused.

## Limitations
Matched, idealised experiments (known wind, K, lam, H, source shapes, onsets); robustness to unseen winds, boundary time
profile, omitted sources and refined-grid truth is F2b.  Only 6 + 6 + 6 cases and 3 noise reps: differences between methods
are descriptive.  Uniform winds only; the DEM and hillshade are plotting context.  alpha is chosen with random and spread
layouts, so it is not tuned to the greedy layouts.  Greedy sites depend on the design cases drawn; layouts are heuristics.
Distances to roads and facilities use OSM, which is incomplete.  Reference sites are only as realistic as the ADM0 outline.

GEODESIGN_RESULTS: <to be filled by coordinator>

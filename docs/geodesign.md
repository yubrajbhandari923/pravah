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


# Stage F2b: robustness of the fixed layouts, numerical refinement, exports

Real geography; synthetic emissions and meteorology.  Provisional simulation design, not field-validated.  Code:
`src/pravah/geodesign_robust.py`; configs: `configs/geodesign_robust.json` (production) and
`configs/geodesign_robust_smoke.json` (plumbing check); tests: `tests/test_geodesign_robust.py`; notebook:
`notebooks/06_belauri_sensor_design.ipynb`.

## Robustness
The layouts and frozen alphas of a finished `geodesign` run (`layouts.csv`, `alpha_selected.json`, the candidate IDs; they were
selected on DESIGN scenarios only) are **re-used verbatim**: nothing is re-selected, no alpha is re-chosen (a test patches every
selection function to raise).  The TEST ensemble is regenerated deterministically (same scenario seeds, noise seeds and sigma =
fraction x the MATCHED signal scale), so the matched rows are the main run's selected-alpha rows (checked against the main
`test_raw.csv` to round-off and recorded in `provenance.json`; the run is refused otherwise).  The robustness config must describe
the same study as the run's `config.json` (everything except name / notes / experiment / robust is compared).  The DESIGN ensemble
is regenerated only for the attribution scales and is checked against `design_scores.csv`.  Each condition changes one thing, with
the same noise arrays, so every row has a matched partner (paired on setting, noise fraction, layout, case, rep):

| condition | truth | what the estimator is told |
|---|---|---|
| `matched` | the case's draw | its wind, all else known (baseline) |
| `unseen_wind` | wind segments replaced: direction = nearest design sector centre +- 22.5 deg (random sign; the design draws are within +- 15 deg), speed U(2.6, 3.5) m/s (design U(0.8, 2.5)); segment times kept | the NOMINAL wind of the same case, i.e. the draw before the replacement |
| `unseen_wind_declared` | the same unseen wind | the unseen wind (a pure distribution shift with the wind known; one extra response set per case) |
| `hidden_wind_error` | declared wind with speed x (1 + 0.2), rotated 15 deg (`experiments.perturb_wind`) | the declared (unperturbed) wind |
| `boundary_time_profile` | the bg inflow profile `[[0,0],[14400,0.5],[21600,1]]` instead of the assumed onset 7200 s | the assumed profile |
| `omitted_facility` | an extra LOCAL source: the OSM facility raster `w_facility`, 5e5 ug/s | the model without it |
| `omitted_external` | an extra EXTERNAL Gaussian (sigma 1000 m, 5e5 ug/s, same onset as the patch) at 12 % / 88 % of the domain | the model without it |
| `refined_truth` | the same fields on a 2x finer grid (250 m, dt 50 s): raster weights replicated onto the fine cells, so only the numerics differ | the 500 m model |

Truth parameters are matched to the inference parameters by kind block inside `design.build_case` (emissions first, then inflow; the
extra source is appended after the inference sources so the blocks line up; the extra source's origin tag is checked).  Truth
components (local / external) come from the truth itself: `total` and `local` by two direct solves, `external = total - local`
(exact because background and initial field are zero).  For `refined_truth` the fine total, local and external fields are
block-averaged (`experiments.block_average`) to the 500 m grid for the regional error and the local / external components; sensors
read the FINE field.  This is the inverse-crime contrast (same scheme on a finer mesh, so a LOWER bound on the effect of model
error); the 500 m inference scenario is handed to `build_case` as the truth scenario only for the parameter table / source-signature
check (`truth_field` carries the fine solution).

Outputs (`out_dir`): `test_raw.csv` (every row, column `condition`), `summary.csv` (means and sds per condition, setting, regime,
noise fraction, method, budget), `paired_rows.csv` (condition minus matched per (layout, case, rep) for total / local / external
RMSE, attribution error, external-fraction error, emission relative error, bg inflow error and the per-parameter absolute errors,
emission in ug/s and inflow in ug/m3 never mixed), `paired_mismatch.csv` (layouts of a method averaged first; mean, sd, pairs,
`n_cases_worse` of `n_cases`, fraction of pairs worse; worsening in every draw is NOT required), `contrasts.csv` (the method
contrasts of `paired.csv`: greedy_total vs spread / random mean, greedy_attribution vs greedy_total, with_reference vs
interior_only, consecutive budgets, recomputed under each condition) and `ranking_changes.csv` (does the sign agree with matched?),
`flags_vs_error.csv`, `cases.csv` (truth and nominal wind, Courant number of the truth, unmodelled emission share), `timings.json`,
`provenance.json`, `figures/`.

**Identifiability flags cannot detect a wrong model.**  `attribution.identifiability_report` is a function of the sensor
operator and sigma only; it never sees the data.  So for every mismatch condition that keeps the inference model (all but
`unseen_wind_declared`) the flags of a row equal those of its matched partner (`flags_same`, `frac_flags_equal_matched` = 1) while the
error changes (`flags_vs_error.csv` lists the flagged fraction next to `rmse_ratio_to_matched`).  A clean flag set therefore says the
data would have constrained the MODEL's parameters, not that the model is right.

Methods compared in every table: random (mean over its layouts), spread, greedy_total, greedy_attribution, and with_reference
vs interior_only at the same budget.  Only the standard regime and the selected-alpha rows are evaluated by default
(`robust.regimes`, `robust.alpha_labels`).

## Numerical refinement
*Numerical diffusion.*  `numerical_diffusion.{csv,json}` give K_eff/K = 1 + `geo.scenarios.numerical_diffusion_estimate` (exponential
convection + backward Euler, along-axis Fourier estimate) for the speeds actually drawn in the three ensembles, reference speeds and
the unseen-wind maximum, at the design mesh (500 m, 100 s) and the refined mesh (250 m, 50 s).  At 500 m / 100 s the excess is large
(for example K_eff/K about 2.4 at 0.8 m/s, about 9.4 at 2.5 m/s and about 14.9 at 3.5 m/s) because the cell Peclet number is 4-18: the
numerical spreading exceeds the physical K = 100 m2/s for almost every wind of the study.  The physical K is **not tuned**; the
numerical spreading is reported separately and is part of truth and inference alike in the matched experiments (which is why the
matched results are optimistic and the refined-truth contrast, with about half the excess, is informative).  For oblique winds the
number is only a scalar-speed approximation.

*Layouts selected on the 500 m problem against 250 m truth* is the `refined_truth` condition above.

*Re-selection on the 250 m model* (`robust.refinement.enabled`, expensive): design cases are built on the 250 m grid (real 250 m
rasters, fine inference model; `refinement.n_design_fine` <= 6, the first design draws and noise seeds of the main run), the design
sigma and frozen alpha of the main run are used, and `greedy_total` / `greedy_attribution` layouts are selected for the budget subset
`refinement.budgets` in `refinement.settings` (the candidate pool is the same 500 m pool).  Outputs: `refinement_layouts_250m.csv`,
`refinement_overlap.csv` (Jaccard index of the site sets, shared sites, mean and maximum nearest-site distance in metres, symmetrised
over both directions), `refinement_test_raw.csv` and `refinement_gap.csv` (250 m-selected minus 500 m-selected layouts on the SAME
fine test truth, fine inference model, sigma of the matched test ensemble; negative = the 250 m layouts do better; cases won),
`refinement_info.json` (cost).  Its `attr_err` uses the fine design cases' attribution scales and is not comparable with the 500 m
tables.  Cost: each 250 m response set is about 8 solves of ~30 s at production size (measured: one fine case, i.e. its response set,
direct truth and base parts, takes about 4 min; the smoke profile uses the cropped domain and about 25 s per case).

## Exports and how to reproduce
```
# smoke (about 40 s + about 2 min under load; cropped domain; plumbing only)
.venv/bin/python -m pravah benchmark --config configs/geodesign_smoke.json --out <empty dir A>
# edit "robust.layouts_from" in a copy of configs/geodesign_robust_smoke.json to <dir A>, then
.venv/bin/python -m pravah benchmark --config <that copy> --out <empty dir B>

# production
.venv/bin/python -m pravah benchmark --config configs/geodesign.json --out results/geodesign
.venv/bin/python -m pravah benchmark --config configs/geodesign_robust.json --out results/geodesign-robust   # robust.layouts_from = "results/geodesign"
```
Cost (measured at production size on a loaded laptop core; the production `geodesign` run took 12 min): the robustness study without
the 250 m re-selection takes about 22 min (design regeneration 3.7 min, matched test cases free after that, `unseen_wind_declared`
3.6 min = 6 cases x 8 solves, `refined_truth` 6 min = 6 cases x 2 solves of ~30 s, each of the other four mismatch conditions 1 min,
evaluation of all rows about 4 s per condition); the 250 m re-selection adds about 4 min per fine design case and 4 min per fine
test case (3 + 3 cases: about 25 min).  Non-empty output directories are refused.  Set `robust.refinement.enabled` to `false` to skip the expensive 250 m part,
`robust.conditions` to run a subset (`matched` is always kept: it is the pairing baseline).  Every directory carries
`config.json`, `provenance.json` (config hash, seeds, sha256 of the layouts / alpha files used, the check against the main run)
and `timings.json`.  The notebook `notebooks/06_belauri_sensor_design.ipynb` runs both smoke profiles in about 2.5 minutes and, if
present, shows the saved production tables and figures.

## Limitations (F2b)
- Each condition is a single, hand-set perturbation (one wind error, one profile, one facility cluster, one extra Gaussian,
  one refinement factor); other sizes, shapes or locations may behave differently.  Amplitudes of the omitted sources are
  chosen, not measured.
- The unseen winds reach u dt/dx = 0.7 at 3.5 m/s (the design winds: up to 0.5; forward-study guideline 0.3); `cases.csv` has the
  Courant number of every truth.  Those truths carry more numerical spreading than the nominal ones.
- `refined_truth` replicates the 500 m raster weights onto the fine cells: it isolates the numerics, not finer geography.
  Re-selection on the 250 m model does use the real 250 m rasters, so its layouts differ from the 500 m ones in two ways.
- Only 6 test cases and 3 noise repetitions (3 + 3 fine cases): differences are descriptive; sds are never confidence intervals.
- The 250 m re-selection uses at most 6 design cases; greedy layouts are heuristics and can change with the cases drawn.
- Still synthetic: uniform winds, no terrain-driven flow, no instrument bias, no source-shape uncertainty beyond the above.

GEODESIGN_ROBUST_RESULTS: <to be filled by coordinator>

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
(`max_courant` is recorded per case in `cases.csv`).  Noise sigma = fraction x signal scale, fractions 0.05 and 0.15, where the scale is the median over the
**validation** cases of the peak noiseless reading at the candidate sites, used for **all three ensembles**.  Why: the estimator is told
sigma, so sigma must be a fixed measurement-noise level that does not depend on the truths of the test ensemble (an earlier
version scaled the test sigma with the test ensemble's own signal and so leaked a test-truth property into the estimator).
Noise realisations still differ per ensemble (own seeds).  `noise_levels.csv` records each ensemble's own signal scale (for
transparency), the common sigma in ug/m3 and sigma as percent of that ensemble's own scale.  Noise draws are common random numbers across noise fractions and layouts; the test
ensemble has its own seeds (fresh noise).  The inverse sees only noisy readings, the declared model inputs and the options.

## Candidates and assumptions
- `interior`: regular lattice (1500 m, origin = the prepared grid's south-west corner, IDs `int_c<row><col>` from the
  lattice indices, so stable under cropping) clipped to the municipality polygon.
- `reference_domestic`: lattice (3000 m, IDs `refd_...`) outside the polygon, inside the simulation domain (500 m inside
  the sensor hull) and inside the Nepal ADM0 polygon.  `reference_hypothetical` (`refh_...`): inside the domain but outside
  Nepal ADM0 (across the national border): flagged `hypothetical`, **excluded by default** (`available_default` false) and
  usable only in the labelled `hypothetical_sensitivity` setting (`settings.hypothetical: true`).
- Polygons come from the raw geoBoundaries files named in `data/belauri/manifest.json`; every raw file is sha256-verified
  against the manifest (no network).  Columns: lon/lat, local x/y, UTM x/y, `in_polygon`, `in_nepal_adm0`,
  `inside_coarse_ind_outline` (inside the coarse geoBoundaries IND ADM0 outline, 2014 data, read from the sha256-verified raw
  file named in the manifest), `near_border_km` (distance to the Nepal ADM0 boundary line), distance to the
  nearest mapped road and facility (m), `on_or_near_road` (<= 100 m), `assumption_flags`.  The two border flags only inform a
  reader, nothing is excluded by them.  The two geoBoundaries outlines are not consistent with each other (coarse Indian outline,
  Nepal line of a different lineage, see `docs/geography.md`): in the production candidate table 38 of 142 candidates are
  flagged `inside_coarse_ind_outline` (interior 8 of 54, reference_domestic 6 of 47, reference_hypothetical 24 of 41; the
  last number shows that the hypothetical sites outside Nepal ADM0 are not all inside the Indian outline either); counts are
  computed by `tests/test_geodesign.py::test_production_candidate_flag_counts`.  A flagged interior or domestic site is
  therefore one whose country is uncertain at this data quality, and the largest `near_border_km` is 17.3 km.  Outputs `candidates.csv` and
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
condition and identifiability flags (`attribution.identifiability_report`, selected-alpha rows only); held-out RMSE; and
`chi2_per_dof` = weighted data residual squared, ||(b + A theta_hat - y) / sigma||^2, over (fitted sensors x times - number of
parameters) (NaN when there are no degrees of freedom, e.g. one sensor with 16 readings has 12; the ridge prior rows are not
counted).  **Flags depend only on the sensitivity matrix and sigma**; they never see the data, so they are saturated here
(`weakly_constrained` fires on every row (16,560 of 16,560 selected-alpha rows of the previous production run) because the prior scale q_scale = 1e5 ug/s is small next to the true emissions of
0.75-2.25e6 ug/s) and "flags equal matched" in the robustness tables is true by construction.  `chi2_per_dof` is the data-based
check: it responds to model error and is expected to be about 1 when the model and sigma are right (it is a diagnostic of the
fit, not of the layout; for a regularised or bound-constrained fit the effective degrees of freedom are smaller, so values
slightly above 1 are normal).  `summary.csv` has its mean and median.
`summary.csv` has means and sds (sd labelled as sd, never a CI; cases and reps are not independent).  `paired.csv`: paired
(case, rep) differences a - b (negative = a better; layouts of a method averaged first; a column `nested` is filled for
the consecutive-budget rows: False when the layouts of the two budgets are not nested, which is always the case for `random`,
whose layouts are independent draws per budget, so its `per_added_sensor` compares different layouts, not an added sensor; for
spread and greedy it is True) for greedy_total vs spread and vs
the random mean, greedy_attribution vs greedy_total (and vs spread) on BOTH total RMSE and attribution error,
with_reference vs interior_only at equal budget, and consecutive-budget improvement (`per_added_sensor`); every row has the
number of cases won (`a_wins_cases` of `n_cases`).

## Held-out design
(i) 20 deterministic random points inside the polygon, each at least `heldout.min_distance_m` (default 500 m) from EVERY candidate
site of every category, so they are genuinely held out (the earlier rule only required more than 1 m; the nearest was 220 m;
`ValueError` if the points cannot be placed); (ii) a spatial block: a fixed 1000 m lattice inside the
polygon cut into thirds by `design.heldout_blocks`, the east third being held out.  In the `block_holdout` regime every
layout respects `Constraints.exclusion_radius_m` (2000 m) around the block points.  Budgets that become impossible are not
silently dropped: `infeasible_budgets.json` lists them with reasons (also in provenance).  `heldout_block_rmse` is a
true hold-out only in the `block_holdout` regime.

## How to run
```
# smoke (about 40 s, cropped 2 km buffer, T = 7200 s, 2+2+2 cases; plumbing only)
.venv/bin/python -m pravah benchmark --config configs/geodesign_smoke.json --out <empty dir>
# production (measured 11.8 min on the laptop: 18 cases x ~8 solves of ~4-5 s, then fits)
.venv/bin/python -m pravah benchmark --config configs/geodesign.json --out results/geodesign
```
Outputs: `config.json`, `provenance.json` (manifest sha256, seeds, alpha_selected sha256, git dirty hash; when the git tree is dirty also `git_diff.patch`
(`git diff HEAD`, capped at 5 MB, `git_diff_patch_truncated` says if it was cut) and `git_status.txt` (which lists untracked files the
patch does not contain), with their sha256 in `extra.git_archive`), `timings.json` (stage
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

**Numerical diffusion.**  The scheme (exponential convection + backward Euler) at the 500 m / 100 s design mesh adds numerical
spreading: K_eff/K is 2.4 at 0.8 m/s and 9.4 at 2.5 m/s over the drawn design speeds (2.4-9.4), and up to 14.3-14.9 at the unseen-wind
speeds (3.4-3.5 m/s) (`numerical_diffusion.csv` of the robustness run; along-axis estimate, a scalar-speed approximation for
oblique winds; the physical K = 100 m2/s is not tuned).  The Courant number u dt/dx reaches 0.49 for the design winds and 0.68 for
the unseen winds, above the 0.3 guideline of the forward study.  Consequence: **the specific selected sites, and the result that
methods are within noise beyond about 5 sensors, depend on this mesh**; truth and inference share the numerics in the matched
experiments, so matched errors are optimistic, and the 250 m re-selection comparison shows the selected sites move when the mesh
changes (the resampling baseline in `docs/phase2-validation.md` says how much they move anyway when only the design cases are
resampled).

<!-- GEODESIGN_BEGIN -->
Run `geodesign`; alpha selected on validation cases only: {'0.05': 0.01, '0.15': 0.01} (frozen before the test ensemble); interior-only setting, standard regime, selected alpha; means over 6 test scenarios x 3 noise draws; budgets in sensors.

**total regional RMSE (ug/m3), noise 5% of the signal scale**

| method | 1 sensors | 2 sensors | 3 sensors | 5 sensors | 8 sensors |
|---|---|---|---|---|---|
| greedy_attribution | 0.0105 | 0.0065 | 0.0054 | 0.0037 | 0.0033 |
| greedy_total | 0.0223 | 0.0093 | 0.0052 | 0.0035 | 0.0027 |
| random | 0.0250 | 0.0135 | 0.0103 | 0.0067 | 0.0033 |
| spread | 0.0148 | 0.0117 | 0.0058 | 0.0051 | 0.0032 |

**attribution error (dimensionless), noise 5% of the signal scale**

| method | 1 sensors | 2 sensors | 3 sensors | 5 sensors | 8 sensors |
|---|---|---|---|---|---|
| greedy_attribution | 0.1321 | 0.0871 | 0.0719 | 0.0485 | 0.0415 |
| greedy_total | 0.2853 | 0.1067 | 0.0714 | 0.0452 | 0.0350 |
| random | 0.2826 | 0.1724 | 0.1322 | 0.0857 | 0.0389 |
| spread | 0.1924 | 0.1689 | 0.0760 | 0.0487 | 0.0379 |

**total regional RMSE (ug/m3), noise 15% of the signal scale**

| method | 1 sensors | 2 sensors | 3 sensors | 5 sensors | 8 sensors |
|---|---|---|---|---|---|
| greedy_attribution | 0.0625 | 0.0298 | 0.0212 | 0.0086 | 0.0079 |
| greedy_total | 0.0366 | 0.0174 | 0.0156 | 0.0108 | 0.0083 |
| random | 0.0422 | 0.0227 | 0.0182 | 0.0128 | 0.0083 |
| spread | 0.0285 | 0.0214 | 0.0158 | 0.0116 | 0.0082 |

**attribution error (dimensionless), noise 15% of the signal scale**

| method | 1 sensors | 2 sensors | 3 sensors | 5 sensors | 8 sensors |
|---|---|---|---|---|---|
| greedy_attribution | 0.5526 | 0.3480 | 0.2694 | 0.1035 | 0.0877 |
| greedy_total | 0.4358 | 0.2265 | 0.1895 | 0.1333 | 0.1060 |
| random | 0.4390 | 0.2820 | 0.2281 | 0.1563 | 0.0962 |
| spread | 0.3801 | 0.3106 | 0.2110 | 0.1302 | 0.1075 |

**Paired total-RMSE comparisons, noise 5% (A is the first-named method)**

| comparison | budget | mean (A minus B), total RMSE | sd | cases where A is lower | n_cases |
|---|---|---|---|---|---|
| greedy_total vs spread | 1 | 0.0075 | 0.0247 | 3 | 6 |
| greedy_total vs spread | 2 | -0.0024 | 0.0093 | 2 | 6 |
| greedy_total vs spread | 3 | -5.54e-04 | 0.0039 | 4 | 6 |
| greedy_total vs spread | 5 | -0.0016 | 0.0032 | 5 | 6 |
| greedy_total vs spread | 8 | -5.14e-04 | 0.0018 | 5 | 6 |
| greedy_total vs random_mean | 1 | -0.0027 | 0.0168 | 5 | 6 |
| greedy_total vs random_mean | 2 | -0.0042 | 0.0074 | 4 | 6 |
| greedy_total vs random_mean | 3 | -0.0051 | 0.0077 | 5 | 6 |
| greedy_total vs random_mean | 5 | -0.0031 | 0.0044 | 6 | 6 |
| greedy_total vs random_mean | 8 | -5.60e-04 | 0.0015 | 4 | 6 |
| greedy_attribution vs greedy_total | 1 | -0.0118 | 0.0247 | 5 | 6 |
| greedy_attribution vs greedy_total | 2 | -0.0028 | 0.0040 | 5 | 6 |
| greedy_attribution vs greedy_total | 3 | 1.75e-04 | 0.0025 | 3 | 6 |
| greedy_attribution vs greedy_total | 5 | 1.72e-04 | 0.0021 | 3 | 6 |
| greedy_attribution vs greedy_total | 8 | 5.52e-04 | 0.0017 | 1 | 6 |

**Paired total-RMSE comparisons, noise 15% (A is the first-named method)**

| comparison | budget | mean (A minus B), total RMSE | sd | cases where A is lower | n_cases |
|---|---|---|---|---|---|
| greedy_total vs spread | 1 | 0.0082 | 0.0225 | 2 | 6 |
| greedy_total vs spread | 2 | -0.0040 | 0.0099 | 5 | 6 |
| greedy_total vs spread | 3 | -2.49e-04 | 0.0064 | 4 | 6 |
| greedy_total vs spread | 5 | -8.52e-04 | 0.0062 | 3 | 6 |
| greedy_total vs spread | 8 | 1.27e-04 | 0.0049 | 3 | 6 |
| greedy_total vs random_mean | 1 | -0.0055 | 0.0195 | 5 | 6 |
| greedy_total vs random_mean | 2 | -0.0053 | 0.0096 | 5 | 6 |
| greedy_total vs random_mean | 3 | -0.0026 | 0.0118 | 3 | 6 |
| greedy_total vs random_mean | 5 | -0.0020 | 0.0059 | 3 | 6 |
| greedy_total vs random_mean | 8 | 6.85e-05 | 0.0036 | 3 | 6 |
| greedy_attribution vs greedy_total | 1 | 0.0259 | 0.0433 | 2 | 6 |
| greedy_attribution vs greedy_total | 2 | 0.0124 | 0.0284 | 3 | 6 |
| greedy_attribution vs greedy_total | 3 | 0.0056 | 0.0188 | 2 | 6 |
| greedy_attribution vs greedy_total | 5 | -0.0022 | 0.0055 | 3 | 6 |
| greedy_attribution vs greedy_total | 8 | -4.18e-04 | 0.0033 | 4 | 6 |
<!-- GEODESIGN_END -->


# Stage F2b: robustness of the fixed layouts, numerical refinement, exports

Real geography; synthetic emissions and meteorology.  Provisional simulation design, not field-validated.  Code:
`src/pravah/geodesign_robust.py`; configs: `configs/geodesign_robust.json` (production) and
`configs/geodesign_robust_smoke.json` (plumbing check); tests: `tests/test_geodesign_robust.py`; notebook:
`notebooks/06_belauri_sensor_design.ipynb`.

## Robustness
The layouts and frozen alphas of a finished `geodesign` run (`layouts.csv`, `alpha_selected.json`, the candidate IDs; they were
selected on DESIGN scenarios only) are **re-used verbatim**: nothing is re-selected, no alpha is re-chosen (a test patches every
selection function to raise).  The TEST ensemble is regenerated deterministically (same scenario seeds, noise seeds and sigma =
fraction x the main run's VALIDATION signal scale, read from `alpha_selected.json`), so the matched rows are the main run's selected-alpha rows (checked against the main
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
`flags_vs_error.csv`, `chi2_by_condition.csv` (median, quartiles and mean of `chi2_per_dof` per condition, noise fraction, method and
budget, the matched median and the ratio), `cases.csv` (truth and nominal wind, Courant number of the truth, unmodelled emission share), `timings.json`,
`provenance.json`, `figures/`.

**Identifiability flags cannot detect a wrong model, and here they are saturated.**  `attribution.identifiability_report` is a
function of the sensor operator and sigma only; it never sees the data.  So for every mismatch condition that keeps the inference
model (all but `unseen_wind_declared`) the flags of a row equal those of its matched partner (`flags_same`,
`frac_flags_equal_matched` = 1) **by construction**, not as a finding, while the error changes (`flags_vs_error.csv` lists the flagged
fraction next to `rmse_ratio_to_matched`).  The flags are also saturated: `weakly_constrained` fires on every row, because the
prior scale q_scale = 1e5 ug/s is small relative to the true emissions (0.75-2.25e6 ug/s), so a flagged fraction of 1 carries no
information.  A clean flag set would say the data would have constrained the MODEL's parameters, not that the model is right.
The data-based check is `chi2_per_dof` (per row in `test_raw.csv`, `paired_rows.csv` as condition minus matched, medians in
`chi2_by_condition.csv` and `summary.csv`): about 1 when model and sigma are right, larger when the data disagree with the
model.  A smoke-size illustration of the matched distribution (about 1) and its shift under the wind conditions is in `docs/phase2-validation.md`
(F2b, "Reading the tables"); the production distribution is read from `chi2_by_condition.csv` after the rerun.

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
Offline analyses of the saved results (no solver; they write only their own new files):
```
.venv/bin/python scripts/ranking_stability.py       # results/geodesign-robust/test_raw.csv -> ranking_stability.{csv,md}
.venv/bin/python scripts/selection_stability.py     # builds the 6 design cases once (~3.5 min) -> selection_stability.{csv,md}, selection_stability_layouts.csv
```
`ranking_stability.py` restates "do method rankings survive the condition?" on an explicit subset (interior-only, 5 % noise, standard
regime, selected alpha), excludes contrasts whose matched difference is exactly zero and adds a noise baseline (sign agreement of the
matched contrast between disjoint halves of the noise repetitions).  `selection_stability.py` is the resampling baseline for greedy
site selection (full 6 design cases against random 3-case subsets, budgets 3 and 5) against which the 250 m re-selection comparison is read.

Cost (measured at production size on a loaded laptop core; the production `geodesign` run took 11.8 min and the full robustness run, with
the 250 m re-selection, 44.8 min of compute; the study without the re-selection takes about 21 min (design regeneration 3.7 min, matched test cases free after that, `unseen_wind_declared`
3.6 min = 6 cases x 8 solves, `refined_truth` 6 min = 6 cases x 2 solves of ~30 s, each of the other four mismatch conditions 1 min,
evaluation of all rows about 4 s per condition); the 250 m re-selection adds about 4 min per fine design case and 4 min per fine
test case (3 + 3 cases: measured 23.5 min).  Non-empty output directories are refused.  Set `robust.refinement.enabled` to `false` to skip the expensive 250 m part,
`robust.conditions` to run a subset (`matched` is always kept: it is the pairing baseline).  Every directory carries
`config.json`, `provenance.json` (config hash, seeds, sha256 of the layouts / alpha files used, the check against the main run)
and `timings.json`.  The notebook `notebooks/06_belauri_sensor_design.ipynb` runs both smoke profiles in about 2.5 minutes and, if
present, shows the saved production tables and figures.

## Limitations (F2b)
- Each condition is a single, hand-set perturbation (one wind error, one profile, one facility cluster, one extra Gaussian,
  one refinement factor); other sizes, shapes or locations may behave differently.  Amplitudes of the omitted sources are
  chosen, not measured.
- Numerical diffusion is large at the design mesh (K_eff/K 2.4-9.4 over the design winds, up to 14-15 for the unseen winds) and the
  Courant number reaches 0.49 (design) and 0.68 (unseen winds); see the limitations of F2a.  Selected sites and the "within noise
  beyond 5 sensors" result depend on the mesh.
- The unseen winds reach u dt/dx = 0.7 at 3.5 m/s (the design winds: up to 0.5; forward-study guideline 0.3); `cases.csv` has the
  Courant number of every truth.  Those truths carry more numerical spreading than the nominal ones.
- `refined_truth` replicates the 500 m raster weights onto the fine cells: it isolates the numerics, not finer geography.
  Re-selection on the 250 m model does use the real 250 m rasters, so its layouts differ from the 500 m ones in two ways.
- Only 6 test cases and 3 noise repetitions (3 + 3 fine cases): differences are descriptive; sds are never confidence intervals.
- The 250 m re-selection uses at most 6 design cases; greedy layouts are heuristics and can change with the cases drawn.
- Still synthetic: uniform winds, no terrain-driven flow, no instrument bias, no source-shape uncertainty beyond the above.

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

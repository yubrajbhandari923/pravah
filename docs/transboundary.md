# External vs local pollution (synthetic) and the indoor box

Everything here is **synthetic**. All numbers in `configs/transboundary*.json` are demonstration values, not estimates for any place.
Code: `src/pravah/transboundary.py` (scenarios, studies), `attribution.py` (tags, fractions, identifiability flags),
`indoor.py` (indoor box). Notebook: `notebooks/04_transboundary_pollution.ipynb`.

## Terms (used literally)

* **Local**: emitted inside the study region.
* **External**: originating outside the study region, either an explicit source placed outside it or an input through the outer
  inflow boundary of the simulation domain (a *boundary basis*, standing for distant sources we do not resolve).
* **International transboundary**: crossing a national border. **Not used** for the synthetic domain: no verified national
  boundary exists there. The study region is only a square used for reporting.
* **Indoor**: the air in a building that exchanges with outdoors (section "Indoor box").

## Model structure

* **Three regions.** The *study region* (a reporting mask and weight, never a PDE wall), the *simulation domain* (where the
  advection-diffusion-decay equation is solved; the study region plus a buffer on every side), and the *outer inflow boundary*
  (the domain edge; only inflow faces carry a boundary input). Config coordinates are study-relative metres (origin = south-west
  corner of the study region), so domains A and B hold the same physical problem.
* **Parameter vector** `theta` = emission amplitudes (units **ug/s**, one per explicit source) followed by boundary-basis
  amplitudes (units **ug/m^3**, one per basis). The two kinds are never compared by raw magnitude; errors are reported per kind
  (`attribution.emission_error`), and the optimiser works in scaled units (emission 1e5 ug/s, basis 0.05 ug/m^3; documented scaling only).
* **Boundary basis vs explicit source**: never both for the same plume. `transboundary.check_no_double_count` refuses a
  basis flagged (`represents`) to stand for a region that contains an explicit source, and refuses explicit sources outside the domain.
* **Tags by linear superposition** (`attribution.contributions`): `local`, `external_sources`, `boundary`, `background`,
  `initial`; their sum equals the total field to round-off. A tag is the **origin** of the mass, not the last boundary crossing:
  a local plume that leaves the study region and returns after a wind reversal stays local. Tags are used only to evaluate;
  `fit_sources` receives only `Observations`.
* **Concentration-derived external fraction** `F_ext = sum_w(external_sources + boundary + background) / sum_w(total)`,
  with `sum_w` over study cells (area-fraction weights) times trapezoid time weights over the output times in `[t_exclude, T]`.
  It is a **concentration** quantity; an inward mass-flux fraction is a different quantity and is never substituted.
  Returned as `None` (JSON null) when the time-mean study-weighted total is below `eps_rel = 1e-3` times the peak of the
  study-weighted mean total over the window, or the total is zero. The threshold is relative, so it flags a vanishing
  *relative* total, not an absolutely tiny one. `fraction_timeseries` gives the instantaneous version with the same rule.

## Default synthetic setup (`configs/transboundary.json`)

| item | value |
|---|---|
| study region | 10 km x 10 km square |
| domain A / B | buffer 5 km (20 km, 80x80 cells) / buffer 10 km (30 km, 120x120 cells), dx = 250 m |
| physics | K = 100 m^2/s, H = 500 m, lam = 1/21600 s^-1 (4.63e-5), T = 21600 s, output 600 s, dt 60 s, t_exclude 3600 s |
| wind | 1.0 m/s towards 22.5 deg (counter-clockwise from east) |
| local L1 / L2 | (3000, 3500) m, 1.0e5 ug/s, sigma 500 m / (7000, 6000) m, 0.7e5 ug/s, sigma 500 m |
| external E1 | (-4000, 4000) m, 2.0e5 ug/s, sigma 1000 m, step onset 0 -> 1 at 3600 s |
| confounded E1c | 3800 m upwind of L1 on the wind line, 2.0e5 ug/s, sigma 300 m; wind 5 m/s towards 0 deg |
| boundary bases | uniform 0.03 ug/m^3, west 0.03, south 0.02, east 0.01, north 0.01 (step onset at 7200 s); `mixed` uses uniform at 0.02 |
| scenarios | `local_only`, `external_sources`, `boundary_only`, `mixed`, `sector`, `confounded`, `no_arrival` (wind towards 180 deg) |
| ensemble | wind speed 0.6-1.6 m/s, direction +-35 deg, amplitude factor 0.5-1.5; every second case has a wind switch |
| seeds | design 101, validation 202, test 303 (noise seeds 20200 / 30300 / 40400 / 50500, disjoint ranges checked) |
| noise | 2 % and 10 % of the signal scale = median over cases of the peak noiseless reading of the `interior` layout, reported in ug/m^3 and % |
| layouts | `interior` (6), `interior8` (8 interior), `interior+upwind` (6 interior + 2 reference sites at x = -4500 m) |

`configs/transboundary_smoke.json` extends this: A is 30x30 cells at 500 m (buffer 2.5 km), B 40x40, T = 7200 s, dt = 120 s,
t_exclude = 1800 s, E1 moved to x = -2000 m with onset 1800 s, tiny ensembles. It is a plumbing check, not for conclusions.

## Identifiability diagnostics

`attribution.identifiability_report(fit)` works on the whitened, `q_scale`-scaled sensitivity matrix (columns A_j q_scale_j / sigma).
It reports singular values, **rank** (numerical: singular values above `max(shape) * eps * s_max`), **effective rank** (number of
singular values > 1, i.e. directions constrained better than one scale at the stated noise), **condition** (`s_max / s_min`, inf if rank
deficient), per-parameter `q_std` (linear-Gaussian, ignores the `q >= 0` bound), `column_norm` and column cosines. Flags:

* `rank_deficient`: rank < n, or effective rank < n, or condition > 1e6;
* `unidentifiable:<name>`: column norm < 1 or `q_std / q_scale` > 10;
* `confounded:<a>,<b>`: |cos| of two columns > 0.99 (columns with negligible norm skipped).

`FitResult.success` is optimiser convergence only and says nothing about identifiability. The flags describe the *assumed*
model, so they cannot detect a wrong wind or a wrong source geometry (the notebook shows a wind-mismatch fit with no flag).
Cases built on purpose: **confounded** (E1c collinear upwind of L1; total field fitted well, split poorly), **no arrival**
(verified with the model, `verify_no_arrival`; the E1 column is numerically zero, so with no prior its estimate is arbitrary and
with a prior it equals the prior, neither is a recovery), wind mismatch, prior sensitivity (alpha 0 = no prior, weak, moderate,
strong), omitted external source.

## Regularisation history audit

* The ridge strength `alpha` (scaled units, prior mean 0) in the v0 benchmark: the **v0 headline benchmark once used alpha = 1,
  chosen after viewing test results**. That evidence is **relabelled exploratory** (`results/benchmark`,
  `results/benchmark-unregularized`, `results/mismatch`; test seeds 22 / 5000; superseded).
* A **validation-frozen alpha now exists**: `configs/alpha_selection.json` -> `results/alpha-selection/alpha_selected.json`
  (alpha = 0.1, lowest mean validation regional RMSE on its alpha grid), used by `results/*-final`.
* **Here** (`transboundary.run_transboundary`): alpha is selected **on validation cases only, per noise level**, written to
  `recovery/alpha_selected.json` **before** the test ensemble is built, and frozen. The test ensemble is never used for any
  choice; validation and test have different seeds, no shared (theta, wind) draw and disjoint noise-seed ranges
  (`check_ensembles`). Each test row is reported for three roles: `selected`, `none` (alpha 0) and `strong`. The notebook uses
  fixed illustrative alphas (0, 1e-2, 1, 1e5) and tunes nothing.

## Commands

```bash
# notebook: bounded profile (smoke sizes), runs top to bottom in about one minute on this machine (see below)
.venv/bin/python scripts/verify_notebooks.py notebooks/04_transboundary_pollution.ipynb
# smoke profile (all four studies on tiny grids; use a fresh --out directory)
.venv/bin/python -m pravah benchmark --config configs/transboundary_smoke.json --out results/transboundary-smoke
# production profile (fine grids, 6 h, ensembles; long)
.venv/bin/python -m pravah benchmark --config configs/transboundary.json --out results/transboundary
.venv/bin/pytest -q tests/test_transboundary.py tests/test_indoor.py
```

Output tree of a run: `scenarios/param_table.csv`, `fields/` (summary, arrival times, fraction time series, wind sweep),
`recovery/` (alpha selection, validation and test rows, layouts, paired layout comparisons), `special/`, `domain_size/`,
`figures/`, `provenance.json`, `timings.json`. The notebook's bounded settings: domain A 30x30 at 500 m, T = 7200 s, one
cached set of unit responses; its "confounded" section overrides this locally to T = 14400 s, dt = 60 s and wind 8 m/s so
that the two columns become collinear (the same override the test suite uses).

## Indoor box (separate from the outdoor inverse problem)

`dc_in/dt = P a c_out(t) - (a + k_dep) c_in + q_in(t) / V`, with `P` dimensionless in [0, 1], `a` and `k_dep` in 1/s,
`V` in m^3, `q_in` in ug/s (piecewise constant), concentrations in ug/m^3. Integrated exactly per interval for a
piecewise-linear `c_out` (Taylor series for the small-argument phi functions, no division by zero).

* **One-way coupling.** The box reads the outdoor field at the building (`outdoor_at_building`, via `observations.sample`);
  indoor exhaust is never added to any outdoor source.
* Checks (tests and notebook): constant forcing equals `c_ss (1 - exp(-lam t))`, `c_ss = (P a C + q/V)/(a + k_dep)`;
  `lam = 0` gives accumulation `q t / V`.
* Estimate ONE indoor amplitude with known building parameters (`fit_indoor_emission`, nonnegative, closed form) from paired
  indoor / outdoor readings (`paired_observations`).
* Ignoring indoor emission: `misattribution_demo` shows the naive "outdoor equivalent" (reading / steady indoor-outdoor ratio)
  overstating the true outdoor value by the indoor contribution.
* Indoor-only ambiguity (`indoor_only_ambiguity`): with a constant outdoor level the indoor columns for the outdoor amplitude and
  `q_in` are proportional (cosine 1, smallest singular value ~0); a paired outdoor reading removes it.
* This is **not** part of the municipality-wide outdoor sensor-placement objective, and the building parameters here are
  demonstration values.

## RESULTS (production run EXP-010)

Production run EXP-010 / RUN-0017 (`results/transboundary`, 26.5 min wall, domains A 20 km / B 30 km, validation 4 + test 6 cases, 3 noise reps; alpha selected on VALIDATION cases only: {'0.02': 1.0, '0.1': 1.0}; sd is a standard deviation, not a confidence interval).

**Recovery on the test ensemble (selected alpha; means over cases x reps)**

| layout | noise (% of signal scale) | total RMSE | local RMSE | external RMSE | |F_ext error| | local amp. rel. err | external amp. rel. err |
|---|---|---|---|---|---|---|---|
| interior | 2% | 2.64e-03 | 2.59e-03 | 1.91e-04 | 0.015 | 0.147 | 8.10e-03 |
| interior | 10% | 4.21e-03 | 3.95e-03 | 9.43e-04 | 0.026 | 0.189 | 0.039 |
| interior+upwind | 2% | 2.64e-03 | 2.62e-03 | 1.12e-04 | 0.015 | 0.140 | 4.31e-03 |
| interior+upwind | 10% | 4.72e-03 | 4.63e-03 | 5.48e-04 | 0.026 | 0.237 | 0.021 |
| interior8 | 2% | 2.94e-04 | 2.19e-04 | 1.68e-04 | 1.62e-03 | 0.011 | 5.95e-03 |
| interior8 | 10% | 1.47e-03 | 1.11e-03 | 8.17e-04 | 7.94e-03 | 0.053 | 0.028 |

**Prior sensitivity (test ensemble)**

| layout | noise | prior | total RMSE | local amp. rel. err | external amp. rel. err |
|---|---|---|---|---|---|
| interior | 2% | none (alpha=0) | 3.31e-03 | 0.193 | 8.20e-03 |
| interior | 2% | strong (alpha=1e5) | 0.037 | 0.806 | 0.916 |
| interior | 10% | none (alpha=0) | 0.013 | 0.735 | 0.041 |
| interior | 10% | strong (alpha=1e5) | 0.045 | 0.990 | 0.996 |
| interior+upwind | 2% | none (alpha=0) | 2.62e-03 | 0.137 | 4.33e-03 |
| interior+upwind | 2% | strong (alpha=1e5) | 0.036 | 0.807 | 0.917 |
| interior+upwind | 10% | none (alpha=0) | 6.59e-03 | 0.306 | 0.021 |
| interior+upwind | 10% | strong (alpha=1e5) | 0.045 | 0.990 | 0.996 |
| interior8 | 2% | none (alpha=0) | 2.94e-04 | 0.010 | 6.02e-03 |
| interior8 | 2% | strong (alpha=1e5) | 0.033 | 0.697 | 0.893 |
| interior8 | 10% | none (alpha=0) | 1.47e-03 | 0.052 | 0.030 |
| interior8 | 10% | strong (alpha=1e5) | 0.045 | 0.979 | 0.994 |

**Equal-budget comparison, 8 sensors (paired by case and noise draw; selected alpha)**

| noise | metric (interior8 minus interior+upwind) | mean diff | sd of diff | pairs where interior+upwind is lower |
|---|---|---|---|---|
| 2% | total RMSE | 2.34e-03 | 3.68e-03 | 4/18 |
| 2% | local RMSE | 2.40e-03 | 3.67e-03 | 1/18 |
| 2% | external RMSE | -5.54e-05 | 9.82e-05 | 10/18 |
| 2% | external-fraction error | 0.013 | 0.023 | 6/18 |
| 10% | total RMSE | 3.26e-03 | 3.70e-03 | 3/18 |
| 10% | local RMSE | 3.53e-03 | 3.61e-03 | 1/18 |
| 10% | external RMSE | -2.69e-04 | 4.83e-04 | 10/18 |
| 10% | external-fraction error | 0.018 | 0.026 | 5/18 |

**Boundary input versus a localised plume (E1 explicit in A is the truth; fitted models omit E1)**

| model (study-region field vs explicit-E1 truth) | RMSE / peak |
|---|---|
| A_E1_explicit_true_theta | 0.009 |
| A_local_only | 0.257 |
| A_sector_bases | 0.250 |
| A_uniform_basis | 0.250 |

**Outer-domain size (same physical problem; open boundary 5 km vs 10 km from the study region)**

| scenario | A vs B RMSE / peak | external fraction A | external fraction B |
|---|---|---|---|
| external_sources | 0.009 | 0.429 | 0.424 |
| mixed | 0.033 | 0.484 | 0.435 |

**Special cases.** Confounded: column cosine between L1 and E1c = 0.991, flags ['rank_deficient', 'confounded:L1,E1c'], effective rank 2, condition number 14.9. No-arrival: external-to-local study-mean ratio 4.09e-06 (verified by the forward model; threshold 0.001). Details: `results/transboundary/special/`.

## Reproduced with the smoke profile (not production numbers)

I ran `configs/transboundary_smoke.json` into a scratch directory (one run, 1 noise repetition, noise 2 %, frozen alpha = 1.0
from its own 3-point validation grid). Boundary-basis equivalence (truth: explicit E1 outside the study region; model has no
E1; error = total-field study-region RMSE relative to the peak of the true total over the study block):

| model without E1 | rmse_total / peak |
|---|---|
| reference: E1 explicit, true amplitudes (no fit) | 0.0052 |
| local sources only | 0.227 |
| local sources + uniform boundary basis | 0.302 |
| local sources + four sector bases | 0.301 |

So at smoke scale the boundary basis did **not** mimic the localised plume: it was no better than leaving E1 out (and
worse in the uniform case). One noise draw, tiny grid: it shows the effect exists, not how large it is in general.

## What this does NOT establish

* No real data. The truth and the fitted model share the same equations and grid family (an inverse crime in the strict
  sense), so recovery results are an optimistic bound; only the wind-mismatch and omitted-source cases probe model error.
* No identification of any real foreign or distant emitter, and no statement about any real border or place.
* A boundary basis cannot stand in for a localised external plume (smoke equivalence result above). It represents unresolved
  distant sources only.
* The external fraction is a concentration share for one wind history; it is not a property of "the place".
* Optimiser success and a good total-field fit are not evidence of identifiability or of a correct origin split.
* The research-profile ensembles are small (4 validation, 6 test cases); any statistics from them are indicative.
* Whether reruns on other machines match to the last digit was not tested.

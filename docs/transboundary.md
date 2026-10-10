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
  corner of the study region), so domains A and B hold the same sources and study region.  They are NOT the same physical
  problem in every scenario: the open boundary is 5 km (A) vs 10 km (B) from the study region.  Without boundary bases the
  A-B difference is pure domain truncation; for `mixed` the basis sits at the domain edge, so the difference mixes truncation
  with a moved forcing.
* **Parameter vector** `theta` = emission amplitudes (units **ug/s**, one per explicit source) followed by boundary-basis
  amplitudes (units **ug/m^3**, one per basis). The two kinds are never compared by raw magnitude; errors are reported per kind
  (`attribution.emission_error`), and the optimiser works in scaled units (emission 1e5 ug/s, basis 0.05 ug/m^3).
* **The scales are a prior width when alpha > 0.** `SCALE_BY_KIND` is only a numerical scaling at alpha = 0, but with alpha > 0
  the ridge penalty is `alpha (a/0.05)^2` for an inflow amplitude a and `alpha (q/1e5)^2` for an emission q, so the scale IS the
  width of a zero-mean prior, and the emission scale 1e5 ug/s equals the true L1 amplitude.  `effective_rank` (and the
  `weakly_constrained` flag) also depend on the scales.  `recovery/scale_sensitivity.csv` reruns the interior8 fits (test ensemble
  only, both noise levels, alpha as selected at the default scales, not reselected) with the inflow scale and then the emission
  scale multiplied by 0.1 and 10.
* **Known temporal profiles.** The inversion is given the TRUE temporal profiles: wind history, E1's step onset (0 -> 1 at 3600 s),
  and the basis step profile (onset 7200 s in the scenarios).  Only amplitudes are estimated.  In the equivalence study the basis
  is run at E1's true onset AND at a different onset (7200 s vs 3600 s in the production config) to measure the sensitivity to
  that assumption.
* **Tangential boundary faces.** A face with |u.n| < 1e-12 m/s is exactly tangential, neither inflow nor outflow (`forward.UN_TOL`;
  without it sin(pi) = 1.2e-16 made the south side an inflow side with the full 2K/h Dirichlet term), and wind components built
  from a direction are rounded to 1e-12.  The inflow coefficient jumps from 0 to |u.n| + 2K/h as u.n changes sign: that is a
  property of the Dirichlet-inflow treatment, not a bug, so a basis on a nearly tangential side has a discontinuous effect.
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

* `rank_deficient`: rank < n or condition > 1e6 (singular or nearly singular unregularised problem);
* `weakly_constrained`: effective rank < n (some direction is constrained worse than one `q_scale` at the stated noise; depends on
  the scales and the noise level, it is not a singularity);
* `unidentifiable:<name>`: column norm < 1 or `q_std / q_scale` > 10;
* `confounded:<a>,<b>`: |cos| of two columns > 0.99 (columns with negligible norm skipped).  This is PAIRWISE: a column in
  the span of two or more others, but close to none of them, is not flagged.

`FitResult.success` is optimiser convergence only and says nothing about identifiability. The flags describe the *assumed*
model, so they cannot detect a wrong wind or a wrong source geometry (the notebook shows a wind-mismatch fit with no flag).
Cases built on purpose: **confounded** (E1c on the wind line upwind of L1, a constructed example; diagnostics are stored PER noise
level in `special/diagnostics.json` under `confounded.by_noise_fraction`, 2 % and 30 %.  At 2 % the only flag is `confounded`
(effective rank 3) and the split is recovered; at 30 % the amplitude errors are large (numbers in the generated results table
from `confounded.csv`).  The flag alone does not mean the split failed, and |cos| is close to the 0.99 threshold at the production
geometry (5 m/s, 3.8 km upwind, dedicated layout)), **no arrival** (verified with the model, `verify_no_arrival`; the E1 column
is very small, NOT numerically zero: at alpha = 0 the estimate is noise amplified through a near-null direction (production
numbers in the generated table), with a prior it falls to the prior; neither is a recovery), wind mismatch (production numbers
in the generated table: no identifiability flag despite a wrong F_ext), prior sensitivity (alpha 0 = no prior, weak, moderate,
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
# regenerate the RESULTS section between the PRODUCTION_RESULTS markers below from results/transboundary (offline)
.venv/bin/python scripts/transboundary_tables.py
.venv/bin/pytest -q tests/test_transboundary.py tests/test_indoor.py
```

Output tree of a run: `scenarios/param_table.csv`, `fields/` (summary, arrival times, fraction time series, wind sweep),
`recovery/` (alpha selection, validation and test rows, layouts, paired layout comparisons), `special/`, `domain_size/`,
`figures/`, `provenance.json` (git revision, dirty flag, short hash of `git diff HEAD` and the dirty-file list, so a run from a dirty
tree is identifiable), `timings.json`. The notebook's bounded settings: domain A 30x30 at 500 m, T = 7200 s, one
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
  overstating the true outdoor value.  The overstatement is `from_indoor / F` with `F = P a / (a + k_dep)` (the indoor reading
  due to indoor emission, divided by the steady indoor/outdoor ratio), plus a lag error from the box's response time; it is not
  simply "the indoor contribution" itself.
* Indoor-only ambiguity (`indoor_only_ambiguity`): with a constant outdoor level the indoor columns for the outdoor amplitude and
  `q_in` are proportional (cosine 1, smallest singular value ~0); a paired outdoor reading removes it.
* This is **not** part of the municipality-wide outdoor sensor-placement objective, and the building parameters here are
  demonstration values.

## RESULTS (production run)

Generated by `scripts/transboundary_tables.py` from `results/transboundary/*.csv|json`; do not edit by hand.

<!-- PRODUCTION_RESULTS_BEGIN -->
Results directory `transboundary-rerun`; 27.8 min wall; validation 4 + test 6 cases, 3 noise reps; alpha selected on VALIDATION cases only: {'0.02': 1.0, '0.1': 1.0}; code revision b27027627c, dirty = True, diff hash 0ce77fd2a947. sd is a standard deviation across rows, not a confidence interval. Cases x repetitions are not independent samples.

**Recovery on the test ensemble (selected alpha; means over cases x reps; RMSE in ug/m3)**

| layout | noise (% of signal scale) | selected alpha | n rows | total RMSE | local RMSE | external RMSE | abs F_ext error | local amp. rel. err | external amp. rel. err | uniform inflow abs. err (ug/m3) |
|---|---|---|---|---|---|---|---|---|---|---|
| interior | 2% | 1 | 18 | 0.00264 | 0.00259 | 0.000191 | 0.0154 | 0.147 | 0.0081 | 0.000547 |
| interior | 10% | 1 | 18 | 0.00421 | 0.00395 | 0.000943 | 0.0261 | 0.189 | 0.0387 | 0.0027 |
| interior+upwind | 2% | 1 | 18 | 0.00264 | 0.00262 | 0.000112 | 0.015 | 0.14 | 0.00431 | 0.000193 |
| interior+upwind | 10% | 1 | 18 | 0.00472 | 0.00463 | 0.000548 | 0.026 | 0.237 | 0.0207 | 0.000962 |
| interior8 | 2% | 1 | 18 | 0.000294 | 0.000219 | 0.000168 | 0.00162 | 0.0105 | 0.00595 | 0.000496 |
| interior8 | 10% | 1 | 18 | 0.00147 | 0.00111 | 0.000817 | 0.00794 | 0.0532 | 0.0281 | 0.00244 |

**Prior sensitivity (test ensemble; alpha acts in scaled units, see 'Scales' in the text)**

| layout | noise | prior | total RMSE | local amp. rel. err | external amp. rel. err |
|---|---|---|---|---|---|
| interior | 2% | none (alpha=0) | 0.00331 | 0.193 | 0.0082 |
| interior | 2% | selected (alpha=1) | 0.00264 | 0.147 | 0.0081 |
| interior | 2% | strong (alpha=1e+05) | 0.0369 | 0.806 | 0.916 |
| interior | 10% | none (alpha=0) | 0.0127 | 0.735 | 0.0413 |
| interior | 10% | selected (alpha=1) | 0.00421 | 0.189 | 0.0387 |
| interior | 10% | strong (alpha=1e+05) | 0.0451 | 0.99 | 0.996 |
| interior+upwind | 2% | none (alpha=0) | 0.00262 | 0.137 | 0.00433 |
| interior+upwind | 2% | selected (alpha=1) | 0.00264 | 0.14 | 0.00431 |
| interior+upwind | 2% | strong (alpha=1e+05) | 0.0365 | 0.807 | 0.917 |
| interior+upwind | 10% | none (alpha=0) | 0.00659 | 0.306 | 0.0213 |
| interior+upwind | 10% | selected (alpha=1) | 0.00472 | 0.237 | 0.0207 |
| interior+upwind | 10% | strong (alpha=1e+05) | 0.045 | 0.99 | 0.996 |
| interior8 | 2% | none (alpha=0) | 0.000294 | 0.0105 | 0.00602 |
| interior8 | 2% | selected (alpha=1) | 0.000294 | 0.0105 | 0.00595 |
| interior8 | 2% | strong (alpha=1e+05) | 0.0329 | 0.697 | 0.893 |
| interior8 | 10% | none (alpha=0) | 0.00147 | 0.0524 | 0.0301 |
| interior8 | 10% | selected (alpha=1) | 0.00147 | 0.0532 | 0.0281 |
| interior8 | 10% | strong (alpha=1e+05) | 0.0447 | 0.979 | 0.994 |

**Equal-budget comparison (8 vs 8 sensors; interior8 versus interior+upwind; selected alpha)**

Each noise level has 18 pairs = 6 cases x 3 noise reps: the pairs are not independent, so the per-case count (cases averaged over reps) is the more honest tally.  Difference = interior+upwind minus interior8 (positive = the interior8 layout is lower on that metric).  The noise is not literally shared between the 6- and 8-sensor layouts (the normals are drawn per sensor and reading) nor across the two noise levels (the same standard normals are reused, scaled by sigma).

| noise | metric (interior+upwind minus interior8) | mean diff | sd of diff | pairs where interior8 is lower | cases where interior8 is lower |
|---|---|---|---|---|---|
| 2% | total RMSE | 0.00234 | 0.00368 | 14/18 | 6/6 |
| 2% | local RMSE | 0.0024 | 0.00367 | 17/18 | 6/6 |
| 2% | external RMSE | -5.54e-05 | 9.82e-05 | 8/18 | 1/6 |
| 2% | abs F_ext error | 0.0134 | 0.0226 | 12/18 | 4/6 |
| 2% | inflow amplitude abs. error (ug/m3) | -0.000303 | 0.000438 | 5/18 | 1/6 |
| 10% | total RMSE | 0.00326 | 0.0037 | 15/18 | 6/6 |
| 10% | local RMSE | 0.00353 | 0.00361 | 17/18 | 6/6 |
| 10% | external RMSE | -0.000269 | 0.000483 | 8/18 | 1/6 |
| 10% | abs F_ext error | 0.018 | 0.0258 | 13/18 | 4/6 |
| 10% | inflow amplitude abs. error (ug/m3) | -0.00148 | 0.00223 | 5/18 | 1/6 |

**Scale sensitivity (interior8, test ensemble, alpha as selected at the baseline scales, not reselected)**

Multipliers apply to the default per-kind scales (emission 1e5 ug/s, inflow 0.05 ug/m3).  With alpha > 0 the scale is the prior width, and the emission scale 1e5 equals the true L1 amplitude.  effective_rank (and so the weakly_constrained flag) depends on the scales by construction.

| noise | variant | alpha | total RMSE | local amp. rel. err | external amp. rel. err | inflow abs. err (ug/m3) | mean effective rank | rows weakly_constrained |
|---|---|---|---|---|---|---|---|---|
| 2% | baseline | 1 | 0.000294 | 0.0105 | 0.00595 | 0.000496 | 4 | 0/18 |
| 2% | inflow_x0.1 | 1 | 0.000295 | 0.0104 | 0.0058 | 0.000488 | 4 | 0/18 |
| 2% | inflow_x10 | 1 | 0.000294 | 0.0105 | 0.00595 | 0.000497 | 4 | 0/18 |
| 2% | emission_x0.1 | 1 | 0.000562 | 0.0174 | 0.0218 | 0.00104 | 4 | 0/18 |
| 2% | emission_x10 | 1 | 0.000294 | 0.0105 | 0.00602 | 0.000496 | 4 | 0/18 |
| 10% | baseline | 1 | 0.00147 | 0.0532 | 0.0281 | 0.00244 | 4 | 0/18 |
| 10% | inflow_x0.1 | 1 | 0.00181 | 0.0502 | 0.0384 | 0.00446 | 4 | 0/18 |
| 10% | inflow_x10 | 1 | 0.00147 | 0.0532 | 0.0282 | 0.00246 | 4 | 0/18 |
| 10% | emission_x0.1 | 1 | 0.00737 | 0.181 | 0.343 | 0.0151 | 3.83 | 3/18 |
| 10% | emission_x10 | 1 | 0.00147 | 0.0524 | 0.03 | 0.00246 | 4 | 0/18 |

**Boundary input versus a localised plume (truth: E1 explicit in domain B; fitted models in domain A omit E1)**

Truth amplitudes L1 = 1e+05, L2 = 7e+04 ug/s.  RMSE / peak is the total-field RMSE over the study region relative to the peak of the true total.  The basis temporal profile is a known input: the two onsets measure the sensitivity to it.

| model | basis onset | RMSE / peak | mean L1 / L2 estimate (ug/s) | flags (union over reps) | mean effective rank |
|---|---|---|---|---|---|
| A_E1_explicit_true_theta | - | 0.00925 | - | - | - |
| A_local_only | no basis | 0.172 | 1.27e+05 / 1.03e+05 | none | 2 |
| A_uniform_basis | 3600 s (matches E1 onset) | 0.118 | 1.03e+05 / 9.38e+04 | none | 3 |
| A_sector_bases | 3600 s (matches E1 onset) | 0.114 | 1.02e+05 / 9.34e+04 | rank_deficient, unidentifiable:east, unidentifiable:north, weakly_constrained | 4 |
| A_uniform_basis | 7200 s (differs from E1 onset) | 0.135 | 1.11e+05 / 1.02e+05 | none | 3 |
| A_sector_bases | 7200 s (differs from E1 onset) | 0.133 | 1.11e+05 / 1.02e+05 | rank_deficient, unidentifiable:east, unidentifiable:north, weakly_constrained | 4 |

**Outer-domain size (open boundary 5 km (A) vs 10 km (B) from the study region)**

| scenario | A vs B RMSE / peak | external fraction A | external fraction B | note |
|---|---|---|---|---|
| external_sources | 0.00925 | 0.429 | 0.424 | same sources and study region; the open boundary is 5 km (A) vs 10 km (B) from it: pure domain truncation |
| mixed | 0.0334 | 0.484 | 0.435 | same sources and study region, but the boundary basis sits at the domain edge (5 km vs 10 km away): truncation mixed with a moved forcing |

**Special cases**

Confounded (alpha = 0; diagnostics of the first noise draw of each noise level; amplitude errors are means over the repetitions):

| noise | abs cos(L1, E1c) | flags | effective rank | condition | total RMSE / peak | local amp. rel. err | external amp. rel. err | abs F_ext error |
|---|---|---|---|---|---|---|---|---|
| 2% | 0.991 | confounded:L1,E1c | 3 | 14.9 | 0.00379 | 0.05 | 0.0297 | 0.0169 |
| 30% | 0.991 | weakly_constrained, confounded:L1,E1c | 2 | 14.9 | 0.0267 | 0.315 | 0.207 | 0.109 |

In words: at 2% noise the flags are [confounded:L1,E1c] (effective rank 3) and the mean amplitude errors are 0.05 (local) and 0.0297 (external); at 30% noise the flags are [weakly_constrained, confounded:L1,E1c] (effective rank 2) and the mean amplitude errors are 0.315 (local) and 0.207 (external).  The confounded flag alone does not mean the split failed: compare the amplitude errors across noise levels.  abs cos is close to the 0.99 threshold at the production geometry (5 m/s, 3.8 km upwind, dedicated layout); this is a constructed example, and the flag is pairwise.

No arrival: external-to-local study-mean ratio 4.09e-06 (verified by the forward model; threshold 0.001).  The E1 column is SMALL, not numerically zero; at alpha = 0 the estimate is determined by noise amplified through a near-null direction.  True E1 = 2e+05 ug/s (mean).

| prior | mean E1 estimate (ug/s) | total RMSE / peak | F_ext true | F_ext est | condition | flags |
|---|---|---|---|---|---|---|
| none (alpha=0) | 7.34e+10 | 1.47 | 4.41e-06 | 0.393 | 2.3e+09 | rank_deficient, unidentifiable:E1, weakly_constrained |
| selected (alpha=1) | 0.00291 | 0.000813 | 4.41e-06 | 6.43e-14 | 2.3e+09 | rank_deficient, unidentifiable:E1, weakly_constrained |
| strong (alpha=1e+05) | 8.4e-05 | 0.13 | 4.41e-06 | 3.58e-15 | 2.3e+09 | rank_deficient, unidentifiable:E1, weakly_constrained |

Wind mismatch (truth wind rotated / scaled, inference declares the nominal wind):

| case | prior | total RMSE / peak | F_ext true | F_ext est | local amp. rel. err | flags (union) |
|---|---|---|---|---|---|---|
| wind_matched | none (alpha=0) | 0.00248 | 0.484 | 0.484 | 0.0122 | none |
| wind_matched | selected (alpha=1) | 0.00248 | 0.484 | 0.484 | 0.0122 | none |
| wind_matched | strong (alpha=1e+05) | 0.252 | 0.484 | 0.479 | 0.818 | none |
| wind_mismatched | none (alpha=0) | 0.486 | 0.397 | 0.212 | 1.55 | none |
| wind_mismatched | selected (alpha=1) | 0.485 | 0.397 | 0.212 | 1.55 | none |
| wind_mismatched | strong (alpha=1e+05) | 0.263 | 0.397 | 0.339 | 0.725 | none |

Omitted external source (the model lacks the source that is in the truth):

| case | prior | total RMSE / peak | F_ext true | F_ext est |
|---|---|---|---|---|
| omitted_external_sources_by_local_only | none | 0.189 | 0.429 | 0 |
| omitted_external_sources_by_local_only | selected | 0.189 | 0.429 | 0 |
| omitted_external_sources_by_local_only | strong | 0.257 | 0.429 | 0 |
| omitted_mixed_by_boundary_only | none | 0.146 | 0.484 | 0.318 |
| omitted_mixed_by_boundary_only | selected | 0.146 | 0.484 | 0.319 |
| omitted_mixed_by_boundary_only | strong | 0.264 | 0.484 | 0.319 |
<!-- PRODUCTION_RESULTS_END -->

## Reproduced with the smoke profile (not production numbers)

`configs/transboundary_smoke.json` into a scratch directory (one run, 1 noise repetition, noise 2 %, frozen alpha = 1.0 from its
own 3-point validation grid, about 65 s wall).  Boundary-basis equivalence: the truth is E1 explicit in domain B; the models in
domain A have no E1; sensors sit at the same study-relative sites in both domains (an earlier version of the code sampled
the truth at domain-A coordinates, which read the wrong place and returned L1 = 0; fixed).  Error = total-field study-region
RMSE relative to the peak of the true total over the study block; the basis onset equals E1's true onset (1800 s):

| model without E1 | RMSE / peak | L1 / L2 estimate (truth 1.0e5 / 0.7e5) |
|---|---|---|
| reference: E1 explicit in A, true amplitudes (no fit) | 0.0052 | - |
| local sources only | 0.097 | 1.12e5 / 6.75e4 |
| local sources + uniform boundary basis | 0.080 | 1.09e5 / 6.81e4 |
| local sources + four whole-side bases | 0.079 | 1.09e5 / 6.81e4 |

With the basis onset at 3600 s instead of E1's 1800 s the two basis models give 0.084 and 0.083.  So at smoke scale the whole-side
bases removed only part of the error that omitting E1 causes (0.097 -> 0.08), they did not reproduce the explicit plume (0.005),
and the result depends on the assumed basis timing.  One noise draw, tiny grid: it shows that the effect exists, not how large
it is in general.

## What this does NOT establish

* No real data. The truth and the fitted model share the same equations and grid family (an inverse crime in the strict
  sense), so recovery results are an optimistic bound; only the wind-mismatch and omitted-source cases probe model error.
* No identification of any real foreign or distant emitter, and no statement about any real border or place.
* Whole-side boundary bases (one uniform basis, or four side bases) did not reproduce a localised external plume in the smoke
  equivalence study above; that is all that was tested (no other basis shapes).  Such a basis represents unresolved distant
  sources only.
* The external fraction is a concentration share for one wind history; it is not a property of "the place".
* Optimiser success and a good total-field fit are not evidence of identifiability or of a correct origin split.
* The research-profile ensembles are small (4 validation, 6 test cases); any statistics from them are indicative.
* Whether reruns on other machines match to the last digit was not tested.

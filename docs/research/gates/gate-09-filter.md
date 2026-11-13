# Gate-09: independent audit of E09 (causal sequential filtering)

**Date:** 2026-11-13 · **Reviewer:** independent gate agent (no edits to src/, scripts/, state/, tables).
**Object:** `src/pravah/filter.py`, `tests/test_filter.py`, `scripts/e09_filter.py` (run at abe3956, integrated in 9ac2606),
tables `reports/quarto/tables/e09-local-*`, design note `docs/research/filter-design.md` (§11 deviations 1–3).
**Check scripts and outputs:** `docs/research/gates/gate-09/`.
- `check_pilot.py` → `pilot_rows.csv`, analysed by `report_pilot.py` → `report_pilot.out`. This is the 32² pilot,
  regenerated locally.
- `check_sweep.py` → `sweep_rows.csv`, `check_sweep.out`. These use the cached E07 64² fields.
- `check_floor.py` → `floor_rows.csv`, `check_floor.out`. This is the model-error control.

The 64² fields of job 204894 (dd2c51c) were copied read-only from `<hpc-host>:.../e07-sweep-dd2c51c/fields`, 312 files,
6.9 GB. Four sha256 sums matched the cluster (`fields_sha256_checked.txt`). Every cache name was recomputed locally by
`load_scenario`, so the config matches the run's. The copy was deleted afterwards. All work was local: ≤ 6 processes,
OMP_NUM_THREADS=1, no cluster jobs.

## Verdict: PASS WITH CONDITIONS

The filter code is correct, and the E09 numbers reproduce exactly. There is no causal leakage. The conditions concern
interpretation:

- **kf_aug's gain over map24 is memory, not prior and not filtering.** The moment-matched prior contributes about 1–2 %
  of the gain.
- **The dopt/source_near reversal holds at 64², sweep-wide and strongly.** E07's "source_near beats dopt / the gain is
  location knowledge, not D-optimisation" is therefore a property of the 24 h-window estimator. It is not a property of
  the designs.

## What I verified (no issue found)

| Check | Evidence |
|---|---|
| KF algebra (`predict`, `update`) | Read line by line. A P Aᵀ via `f(f(P).T)` is correct for symmetric P. The gain uses a Cholesky solve. P⁻ − K H P⁻ is symmetrised. Innovation, whitened innovation and NIS are standard. Absent rows are dropped, and partially missing rows are refused. 12/12 tests pass (`tests/test_filter.py`) |
| Causal ordering in `run` | The h-step forecast for hour h is stored before `update` reads Y[h]. Pending multi-horizon means are advanced by the same transition. Independent confirmation: my explicit-loop MAP uses only rows ≤ t by construction, and it equals `kf_aug`'s forecasts (next row). So kf_aug uses no data after t |
| E09 table numbers reproduced by independent code (explicit per-issue-time loop, not `map_series`/`estimate_series`) | All 192 (scenario, k, design, net, arm) rows. `map24` 1 h forecast RMSE: max diff 4e-13. `kf_aug`: 3e-15. `map_full`: 7e-15. `source_rmse`: ≤ 2e-12. `cov95_active`: 2e-16. Paired table, white, k = 8, dopt, kf_aug vs map24: median +0.74 [0.62, 0.85] (committed 0.7406 [0.617, 0.847]) |
| E07 64² map24 rows reproduced from the cached fields (E07 code path, committed sites, sweep's own noise draws) | 1440/1440 white-arm rows (random ×10, dopt, source_near; k = 4/8/16; 40 scenarios): max diff **0.0** |
| Growing-window cumulative algebra at 64² | Cross-checked against an explicit solve at three issue times (assert < 1e-8) |
| AR(1) augmentation | e_h = ρ e_{h−1} + w with Q_e = (1−ρ²)σ². The stationary initial covariance is σ²I, with R = 1e-10σ². On the white arm, kf_aug_ar1's NIS/k = 4.58 equals the analytic (1+ρ²)/(1−ρ²) = 4.56 for this misspecification. This is a consistency check, not a bug |
| CRPS, coverage | Closed-form Gaussian CRPS. Coverage uses central intervals. Tested against numerics |
| Prior moments | ā = 4.5/ln 10 = 1.954 and s² = 1.555 are the exact mean and variance of LogUniform(0.5, 5) |

## Issues, ranked

### 1. (Major, interpretation) kf_aug's gain over map24 is almost entirely memory length; the prior adds ≈ nothing
The 2×2 factorial (window 24 h / all past; prior N(0, I) / moment-matched), on the same designs and noise draws, gives
the following median relative 1 h forecast-RMSE reductions vs E07's map24. Over 12 scenarios, white arm, k = 8:

| design | prior only (24 h, mm) | memory only (all past, N(0, I)) | both (= kf_aug) |
|---|---|---|---|
| dopt | +0.02 [−0.06, 0.24] | +0.74 [0.61, 0.84] | +0.74 [0.62, 0.85] |
| random | +0.15 [−0.17, 0.47] | +0.68 [0.58, 0.77] | +0.70 [0.57, 0.83] |
| source_near | −0.05 [−0.09, −0.01] | +0.58 [0.50, 0.61] | +0.58 [0.50, 0.61] |

- The share of the log-RMSE gain due to memory at the N(0, I) prior is median 0.99 (IQR 0.96–1.00) across all 96 white
  (scenario, k, design) cells.
- At 64² over 1440 rows, the median reduction vs map24 is:

  | design | prior only | memory only | both |
  |---|---|---|---|
  | random | +0.29 | +0.935 | +0.936 |
  | dopt | +0.17 | +0.943 | +0.942 |
  | source_near | +0.01 | +0.851 | +0.851 |

Is the moment-matched prior an oracle? It uses the generating distribution, not the realised strengths. Designs already
see the same distribution (`DesignInput.prior_loguniform`). It is a "known population prior", legitimate in a synthetic
benchmark but not available in reality, and it should be labelled as such. It matters only for short windows, where
E07's N(0, I), centred at 0 while the true mean is 1.95, shrinks toward zero. With unlimited memory it is irrelevant.
**Fix:** state that "kf_aug ≡ growing-window batch MAP, and its gain over map24 is the window length (≥ 98 % of the
log-RMSE gain), not the prior or sequential filtering". The E09 journal's "gains reflect memory length and prior" should
drop "and prior".

### 2. (Major, changes E07's wording) The source_near > dopt reversal is estimator-dependent and reverses at 64²
These are 64² results on the committed E07 sweep designs and noise draws (white, R = 5). The paired relative difference
is (dopt − source_near)/dopt, so > 0 means source_near is better. CIs are distribution-free order-statistic intervals.

| estimator | all 120 pairs | source_near wins | k = 4 | k = 8 | k = 16 | layouts (20) favouring source_near |
|---|---|---|---|---|---|---|
| map24 (E07 frozen) | +0.410 [0.328, 0.467] | 115/120 | +0.45 (37/40) | +0.43 (39/40) | +0.38 (39/40) | 20/20 |
| 24 h, mm prior | +0.314 [0.279, 0.379] | 113/120 | +0.32 (35/40) | +0.31 (39/40) | +0.32 (39/40) | 20/20 |
| growing, N(0, I) | **−0.247 [−0.375, −0.176]** | 17/120 | −0.50 (3/40) | −0.22 (5/40) | −0.15 (9/40) | 1/20 |
| growing, mm (= kf_aug) | **−0.236 [−0.375, −0.175]** | 16/120 | −0.50 (3/40) | −0.23 (4/40) | −0.15 (9/40) | 1/20 |

dopt vs the mean of 10 random networks, (random − dopt)/random:

| estimator | median | wins | CI | layouts |
|---|---|---|---|---|
| map24 | +0.710 | 117/120 | [0.661, 0.763] | 20/20 |
| growing N(0, I) | +0.735 | 120/120 | [0.690, 0.782] | 20/20 |
| growing mm | +0.733 | 120/120 | [0.675, 0.782] | 20/20 |

Growing-window k = 8 per cell (wind × K): source_near wins only 0–2 of 10 seeds in every cell.

Median RMSE, growing N(0, I), dopt / source_near / random: 0.0032 / 0.0044 / 0.0114. Under map24 they are
0.0495 / 0.0270 / 0.1796.

**Model-error control (`check_floor.py`).** The growing-window errors are small, so I checked them against the VanLeer
superposition floor:
- The forecast RMSE with the true strengths is median 0.00057 and max 0.0021.
- dopt's growing-window RMSE is a median 6.2× the floor; source_near's is 8.2×.
- Re-scored on a linear truth with the same noise draws, the reversal holds: median −0.330, source_near wins 17/120,
  1/20 layouts.

The reversal is not a floor artefact.

**Mechanism (inference).** It is consistent with gate-07. dopt's loss is per-window rank deficiency in held-out
southerly blocks. A growing window carries about 1000 h of training-wind information about the constant strengths, so
those windows are no longer deficient. dopt's better-conditioned sites then win.

**Fix (E07 journal, chapter 07, decision 2026-11-13):**
- "source_near beats dopt" and "the gain is attributable to knowing source locations, not to D-optimisation" must be
  qualified as **"under a 24 h estimation window"**.
- With an estimator that uses all past data, D-optimal beats source_near in 103–104/120 pairs and 19/20 layouts.
- The confirmatory dopt-vs-random result is estimator-robust: 117/120 → 120/120.

### 3. (Moderate, scope) "Unlimited memory" is only an advantage because strengths are exactly constant for 60 days
Both estimators sit at corners:
- map24 throws away information about static sources.
- The growing window assumes perfect stationarity over ≈ 1000 h, and its RMSE (≈ 0.003) is only 6–8× the model-error
  floor.

Neither is a realistic operating point. The placement ranking flips between them, so **placement conclusions are
conditional on the assumed source persistence**. This is E10's subject (source dynamics).
**Fix:** E10/E11 should treat memory length (or the source random-walk variance) as a declared factor and report
placement orderings at at least two settings.

### 4. (Moderate, validation wording) The kf_aug ≡ map_full identity does not validate the transition operator
After deviation 1, B_h = F_h − A_h F_{h−1} and x_0 = F_0 a. Then x_h = F_h a holds for **any** A_h, so the identity at
4e-13 checks the KF recursion only. A_h affects only `kf_field`. It is validated by the step-operator test (≤ 1e-8 vs
`simulate`) and by `recon_max_abs`: ≤ 1.1e-3 µg m⁻³, ≈ 1e-4 of the peak, attributed to the FiPy LU tolerance.
**Fix:** reword design note §1 item 2 and the journal ("validation of the filter recursion; the operator is validated by
recon_max_abs").

### 5. (Minor, unrecorded falsified prediction) P3 for kf_field failed
P3 predicted NIS > 1 and under-coverage for kf_field. The observed values are:
- NIS/k = 1.02 (white) and 1.02 (ar1).
- 95 % coverage 0.99, but 50 % coverage only 0.19.

The forecast errors are peaked with heavy tails relative to the Gaussian. The filter is mis-shaped, not overconfident.
**Fix:** record P3 as not supported, with these numbers. Also record that NIS is insensitive to error autocorrelation:
kf_aug on AR(1) data has NIS/k 0.96 while 95 % coverage is 0.48. Lag-1 innovation autocorrelation (0.80) is the
diagnostic that detects it. Calibration claims should cite coverage or lag-1, not NIS alone.

### 6. (Minor) kf_aug_ar1 is not a free upgrade
On white data it is worse than map24 for source_near (white k = 8: −0.12, 1/12 wins) and much worse than kf_aug. Use it
only when AR(1) noise is the declared model. The journal's AR(1) statements (0.945 coverage; 12/12 vs map24) are
correct.

## Check 4: NEES / calibration on real scenarios
These are source-estimate NEES/K and 95 % forecast coverage on active cells (sd ≥ 0.01 σ). Medians are over rows.

| setting | estimator | cov95_active | NEES/K (median; 10–90 %) |
|---|---|---|---|
| 32² white | map24 N(0, I) | 0.942 | 1.01 (0.83–2.37) |
| 32² white | growing N(0, I) | 0.957 | 0.98 (0.58–1.44) |
| 32² white | growing mm (kf_aug) | 0.957 | 0.95 (0.57–1.27) |
| 32² AR(1) | growing mm (kf_aug) | 0.525 | 6.9 (3.8–12.0): overconfident, as expected |
| 64² white | map24 N(0, I) | 0.930 | 1.27 (0.94–2.86) |
| 64² white | growing N(0, I) | 0.947 | 1.01 (0.62–1.50) |
| 64² white | growing mm | 0.950 | 0.99 (0.60–1.44) |

- Under white noise, the growing-window MAP is calibrated at 64², despite the VanLeer superposition error.
- map24 with N(0, I) is mildly overconfident in a tail of windows (90th percentile NEES/K 2.9). This fits a prior
  centred at 0 for strengths whose mean is 1.95.
- NEES is averaged over issue times of a single realised a per scenario, so values are serially correlated. Read them as
  descriptive.

## Baseline for E10/E11
Use the **growing-window static MAP on the K cached unit responses** (≡ kf_aug, Q = 0) as the reference estimator. This
is the design note's §3.1 option (i). Use prior **N(0, I)** for continuity with E07: it is numerically indistinguishable
from the moment-matched prior at full memory and needs no generating-distribution knowledge. If the moment-matched prior
is used, label it "known population prior".

Keep map24 as the short-memory reference. Report placement comparisons under both, because the design ranking flips
(issue 2). Do not use kf_field as a baseline: it is 1.6–13× worse than map24.

## Defensible wording

**E09 (exploratory, 32², upwind, 12 scenarios).**
> In a perfect-model setting with constant source strengths, the correctly specified augmented Kalman filter is
> algebraically the batch MAP over all past observations, matching to 4e-13. Its 1 h held-out forecast RMSE was a median
> 58–74 % lower than E07's 24 h-window MAP (10–12/12 scenarios per design). A 2×2 control shows this gain comes from the
> estimation window: ≥ 98 % of the log-RMSE gain persists with E07's N(0, I) prior. It is not due to the prior or to
> sequential filtering. A field-only filter that does not estimate the strengths was 1.6–13× worse than the 24 h MAP.

**E07 qualifier (64², cached sweep, white noise; post hoc, exploratory).**
> The advantage of near-source placement over D-optimal placement holds for the 24 h-window estimator (115/120 pairs,
> 20/20 layouts). It reverses when the estimator uses all past data: D-optimal is better in 103/120 pairs and 19/20
> layouts (source_near's forecast RMSE a median 25 % above dopt's). Both placements beat random under either estimator. Which placement is
> better therefore depends on how much past data the estimator can use, i.e. on how persistent the sources are.

**Not warranted.**
- "Filtering beats batch inversion."
- "The moment-matched prior improves forecasts."
- "Knowing source locations matters more than D-optimisation" without the 24 h-window qualifier.
- Anything about time-varying sources, model error beyond A2 superposition, or real data.

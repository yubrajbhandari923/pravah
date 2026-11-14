# E14 reliability diagnostics and uncertainty calibration: design note (pre-pilot)

Status: written 2026-11-13 on branch `experiment/e14-reliability`, **before any E14 diagnostic or label was computed**,
and committed alone. Later changes go into §11 (Deviations) with their reason. Items marked **[LEAD REVIEW]** are
choices the plan and protocol leave open. Plan-03 lists the artifact as `calibration-design.md` and the code as
`src/pravah/calibration.py`; the lead named them `reliability-design.md` and `src/pravah/experiments/reliability.py`.

Inputs: plan-03 key `e14-uncertainty-calibration`; `preregistered-protocol.md` §10.2 item 3 (50/80/95 % coverage,
CRPS, reliability); E12 (`wind-mismatch-design.md`, `state/experiments/e12.md`: under wind error the sensor fit barely
changes while source/forecast error explodes; lag-1 innovation autocorrelation 0.13–0.89; 95 % strength coverage
0.00–0.54); gate-09 (NIS is insensitive to autocorrelated error; lag-1 detects it); gate-10/11 (coverage rewards width;
report CRPS beside coverage; cluster = (K, seed)).

## 1. Question and hypotheses

Can an estimator tell, **from the data it has, without the truth**, that its own map/forecast is about to be
unreliable? And how wrong is its nominal uncertainty when the model is misspecified?

All hypotheses are **exploratory** (32² pilot, 6 clusters; no confirmatory claim).
- **H1 (miscalibration).** With the true model (oracle wind, nominal noise) field-forecast intervals are roughly
  calibrated; under wind error the nominal intervals under-cover and CRPS rises (nominal variance underestimates error).
- **H2 (detectability).** Truth-free diagnostics that look at the *temporal/predictive* behaviour of the residuals
  (lag-1 innovation autocorrelation, sensor-level skill vs persistence, leave-one-sensor-out error) warn of
  "forecast worse than persistence" with AUC > 0.5 on held-out time; the in-sample sensor fit and the nominal posterior
  spread do not (AUC near 0.5: "deceptive fit").
- **H3 (limits).** Detection is family-dependent: weak, slowly acting errors (speed × 0.8) give weak signals, and some
  failures are missed by every diagnostic (overconfident false negatives exist). NIS misses autocorrelated error.

## 2. Data: reuse of the E12 32² pilot pipeline (no new physics)

- Scenarios, truth, readings, designs and the perturbed MODEL-wind responses are exactly E12's pilot
  (`scripts/e12_wind_mismatch.py` called unmodified: `field_names`, `run_field`, `_find`, `WIND = mixed`,
  K ∈ {2, 4}, seeds {0, 1, 2} → **6 clusters (K, seed)**, upwind 32², split days (1–4 train, 5 validation, 6–11 test)).
- Mismatch arms (13): E12's 9 wind specs (oracle, dir +10/+30°, speed ×0.8/×0.6, lag 2/6 h, AR(1) direction sd 15/30°),
  plus 3 simple non-wind mismatches using the oracle wind:
  - `noise_x2`: readings carry noise with sd 2σ; the estimator assumes σ (same white draws, scaled);
  - `noise_ar1`: AR(1) noise ρ = 0.8 (observations.noise_field `ar1`), estimator assumes white;
  - `omit1`: the estimator's source basis omits the last source (truth unchanged).
  (A time-varying source strength is NOT included: superposition of constant-source responses with a(t) is not the
  physical response; that belongs to E13.)
- Designs: random (5 networks), dopt, source_near (from oracle training responses, as E12). k ∈ {2, 4, 8}.
  Estimators: `map24` and `map_full` (e09 `map_series`, prior N(0, I) in a_ref units). Noise draws R = 3 (E07 draws);
  each draw is one "deployment" (a deployment sees one noise realisation).

## 3. Time split (held-out time; nothing tuned on the evaluation blocks)

Issue times are E12's (test segment, 6 h warm-up). They are cut into consecutive 24 h **blocks** b = 0, 1, …; a partial
last block is dropped. For block b with first issue hour t_b:
- **diagnostics** use only data at hours ≤ t_b (the trailing 24 h window ending at t_b, plus the estimator's own window);
  they are computed *before* any forecast of the block is issued → an early warning;
- **labels** use the truth over the block's issue times (t_b … t_b + 23) and their 1 h targets.
Blocks 0–1 are **validation** (thresholds chosen there); blocks ≥ 2 are **evaluation** (AUC and all reported rates).

## 4. Pre-registered diagnostics (truth-free; larger = more alarm)

All on whitened quantities z = (y − c_bg)/σ, A = a_ref G/σ; û_t, C_t = MAP and posterior covariance at hour t with the
estimator's window. Trailing window W = hours t_b − 24 … t_b − 1 (pairs t → t + 1, t + 1 ≤ t_b).
1. `innov_ac1`: lag-1 autocorrelation of the 1 h innovations ν_{t+1} = z_{t+1} − A_{t+1} û_t, centred per site, pooled
   over sites (E12's definition).
2. `nis`: mean over t of ν_{t+1}ᵀ S_{t+1}⁻¹ ν_{t+1} / k, S = I + A C_t Aᵀ (expected 1 under the model).
3. `resid_xcorr`: mean off-diagonal correlation of the sites' innovation series (residual spatial structure; k ≥ 2).
4. `loso`: leave-one-sensor-out: for each site s, re-estimate at t_b from the other sites (same window), then
   mean over W and s of (z_s − A_s û₋ₛ)² / (1 + A_s C₋ₛ A_sᵀ).
5. `spread`: nominal posterior spread of the next-hour field forecast at t_b: mean cell sd / mean |forecast anomaly|.
6. `drift`: source-estimate instability between windows, (û_{t_b} − û_{t_b−24})ᵀ (C_{t_b} + C_{t_b−24})⁻¹ (…) / K.
7. `sensor_skill`: model-vs-persistence disagreement at the sensors: RMS ν_{t+1} / RMS (z_{t+1} − z_t) over W
   (> 1: at the sensors the model forecast is worse than persistence of the last reading).
8. `sensor_fit` (negative control): in-sample RMS(z_t − A_t û_t) over W — the "fit looks fine" statistic.

A test (`tests/test_reliability.py`) runs the diagnostic path with every truth array set to NaN and requires identical,
finite diagnostics.

## 5. Labels (need the hidden truth; per unit = cluster × arm × design/net × k × estimator × draw × block)

- **L1 `worse_than_persistence`:** block 1 h field-forecast RMSE > block RMSE of persistence (c(t) for c(t+1), truth).
- **L2 `coverage_failure`:** empirical coverage of the nominal 90 % central field-forecast interval over *active* cells
  (nominal sd ≥ 0.01 σ) in the block < 0.70.
  [LEAD REVIEW] thresholds 0.90/0.70 are a judgment call; 50/90/95 % coverages are reported continuously regardless.

## 6. Calibration endpoints (protocol §10.2 item 3; truth needed)

Per arm × estimator × k × design class, evaluation blocks pooled: nominal 50/90/95 % field-forecast coverage on active
cells, Gaussian CRPS (µg m⁻³), PIT histogram (10 bins), and 95 % coverage of the source strengths. Coverage is always
reported beside CRPS (gate-10/11: width is rewarded by coverage alone). Field intervals are **parametric only**
(forecast = F_m û, var = diag(F_m C F_mᵀ)); there is no model-error term — that is the point being tested.

## 7. Analysis

- **AUC** (Mann–Whitney; ties = ½) per diagnostic × label on evaluation blocks, pooled over all units; and per family
  (units from oracle + that family only) to show family-dependent detectability. Direction fixed in §4 (no flipping).
- **Uncertainty:** cluster = (K, seed). Per-cluster AUC (median, min, max over the 6 clusters) and a cluster-bootstrap
  95 % CI (2000 resamples of clusters, seed 4000). With 6 clusters the CI is coarse; it is reported, not interpreted
  as confirmatory.
- **Thresholds:** for each diagnostic, the Youden-optimal threshold on validation blocks; applied unchanged to
  evaluation blocks → TPR, FPR, precision, and **selective risk**: median model/persistence RMSE ratio on units
  that would NOT be flagged vs all units.
- **Honest negatives:** list the false negatives with the largest forecast degradation (overconfident misses), and
  report per-family AUC for speed × 0.8 even if near 0.5.
- No combined classifier is fitted in the pilot (6 clusters is too few to train and test one honestly).

## 8. Outputs

`reports/quarto/tables/e14-local-{units,auc,auc-family,thresholds,calibration,pit,misses,meta}.*`,
`reports/quarto/figures/e14-local-roc.png`, `e14-local-calibration.png`.
Code: `src/pravah/experiments/reliability.py` (diagnostics, labels, AUC, calibration; no I/O),
`scripts/e14_reliability.py` (driver; calls the E12 pipeline for cached fields).

## 9. Limits known in advance

32² upwind, 6 clusters, one wind regime family (mixed), one noise level; the persistence baseline uses the truth field
(as E12); diagnostics are evaluated on the same trajectories that E12 already studied (E12 reported innovation lag-1 as
a candidate warning *after* seeing data, so H2 for `innov_ac1` is partly post hoc — say so). Units within a cluster are
strongly dependent; only the cluster-level statistics carry uncertainty.

## 10. 64² plan (deferred by user, 2026-11-13)

Written after the pilot; not run. E14 adds no PDE solves: it consumes the E12 64² fields (E12 design §10).
- **Prerequisite:** the E12 64² stage 1 (5 specs + oracle, 20 clusters; ≈ 1 270 CPU-h ≈ 22 h on 64 CPU, reusing the
  E07 sweep cache). Without it E14 64² cannot run; E14 must not trigger those solves itself.
- **Code change needed first (small):** `scripts/e14_reliability.py` hard-codes `SCALE = "pilot"`, seeds 0–2, k 2/4/8.
  Add `--sweep` → SCALE "sweep", seeds 0–9, k 4/8/16, specs `U.SWEEP_SPECS`, output prefix `e14-sweep-`; blocks and
  validation split unchanged (test days 43–60 → ≈ 17 blocks; validation = blocks 0–1).
- **Cost estimate (extrapolated from the pilot, not measured):** pilot = 6 clusters × 13 arms in ≈ 5 CPU-min (≈ 50 CPU-s
  per cluster). Per cluster at 64²: × 4 (grid) × 3.5 (blocks) × 2 (k up to 16 → LOSO) × 0.7 (9 vs 13 arms) ≈ × 20 →
  ≈ 1 000 CPU-s; 20 clusters ≈ 6 CPU-h, memory ≈ 1.3 GB per worker (E12 stage-2 figure). One job: 16 CPU, 32 GB, 2 h.
- **Command (cluster, after `squeue` shows the cluster idle of other users):**
  `sbatch -J pv-rel -c 16 --mem=32G -t 02:00:00 --export=ALL,PRAVAH_HPC_ROOT=$R -o $R/logs/%x-%j.out scripts/hpc/run.sbatch
  $R/envs/main/bin/python scripts/e14_reliability.py --sweep --fields <E12 sweep out> --workers 16`
  from a `git archive` snapshot of the committed revision at `$R/code/snapshots/<rev>`, env `$R/envs/main`.

## 11. Deviations

1. 2026-11-14, **after seeing pilot results (post hoc):** added a *within-configuration* AUC (pairs compared only
   inside the same k × design class × estimator stratum, i.e. what a deployer already knows), reported beside the
   pre-registered pooled AUC. Reason: the pooled AUC can be earned by recognising weak configurations (k = 2, map24)
   rather than mismatch; `spread` is data-independent up to |forecast| and looked suspicious. Both are in
   `e14-local-auc.csv` (`scope`). The family table reports both (`auc`, `auc_within`).
2. Provenance: the pilot units table was produced from the working tree whose src/scripts content is commit 23c3235
   (committed while the run was in flight); `e14-local-meta.json` therefore records git_rev 33efe67 with git_dirty
   true. The report step was re-run after deviation 1 (code in the later commit).
3. The ROC curves in `e14-local-roc.png` are pooled; the legend gives the within-config AUC (stated in the legend).

### Lead decisions (2026-11-14)
- Both AUCs are reported. The within-config AUC is the more useful one, because a deployer knows the configuration, and
  is labelled **post hoc**. The pre-registered pooled AUC is always shown beside it.
- The L2 thresholds are accepted as written (90 % nominal; failure below 0.70).
- No combined classifier at pilot scale. Revisit at 64², pre-declared.
- The 64² run is deferred with the E12 64² fields (user decision 2026-11-13).

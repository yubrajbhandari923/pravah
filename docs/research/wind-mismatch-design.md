# E12 meteorological uncertainty and source–transport compensation: design note (pre-pilot)

Status: written 2026-11-13 on branch `experiment/e12-wind-mismatch`, **before any E12 computation**, and committed alone.
Later changes go into §12 (Deviations) with their reason. Items marked **[LEAD REVIEW]** are choices the plan and
protocol leave open. Synthetic 2-D passive tracer, known source locations, uniform hourly winds. Plan-03 lists the
artifact as `wind-uncertainty-design.md`; the lead named this file `wind-mismatch-design.md`, which replaces it.

Inputs: plan-03 key `e12-wind-mismatch`; `preregistered-protocol.md` §5, §8–§12; E07 (`cross-objective-design.md`,
gate-07: dopt fails through per-window blindness in the unseen southerly regime); E09/gate-09 (the placement ordering
depends on estimator memory: source_near wins under map24, dopt under the growing-window MAP); E10/gate-10-11 (random-walk
filter, cluster = (K, seed)); E11 (wind-robust forecast design).

## 1. Question and hypotheses

How fast do source, field and forecast skill degrade when the **inverse/forecast model's wind** differs from the
**truth's wind**? Do the estimators absorb the transport error by distorting the source strengths (source–transport
compensation), so that the sensor fit looks good while the field forecast is poor? Does longer estimator memory help or
hurt under the wrong wind? Are some placements more robust?

Hypotheses (stated before data; all **exploratory**, see §9):
- **H1 (degradation).** For each perturbation family, held-out 1 h forecast RMSE grows with the perturbation magnitude;
  the oracle (true wind) is best in the median.
- **H2 (compensation).** Under a direction or speed bias the relative increase of the in-sample sensor residual is
  smaller than the relative increase of the source RMSE and of the held-out field/forecast RMSE (data fit is not enough).
  The posterior 95 % intervals of the strengths under-cover (incorrect confidence).
- **H3 (memory).** A systematic bias does not average out: the growing-window MAP's advantage over map24 (E09: +0.58 to
  +0.74 with the true wind) shrinks under bias. Under zero-mean, low-frequency random direction error it shrinks less.
  Two-sided: longer memory may also *help* (more wind diversity constrains the strengths better), so the direction is
  not assumed.
- **H4 (placement robustness).** source_near degrades less (smaller RMSE ratio to its own oracle) than dopt, because
  near-source readings are dominated by the local footprint, which depends weakly on the wind direction. Random is the
  floor. The ordering may differ between memories (gate-09).

## 2. Truth / inverse forcing separation

- **Truth:** `benchmarks.generate_scenario` unchanged, driven by the scenario's own winds (`sc.truth.winds`). E12 never
  modifies the truth, its winds, its noise or its seeds. The noisy readings are the E07 readings (same draws).
- **Inverse/forecast model:** unit-source responses F_m (K, hours, grid) computed with the *model* wind
  w_m = perturb(w_true, spec), by the same solver, spin-up rule (replaying the first model-wind hours) and source basis.
  The estimator sees only F_m sampled at its sites and the readings; the forecast is F_m·â. The truth field enters only
  the scorer.
- **Oracle branch:** spec magnitude 0 ⇒ w_m ≡ w_true and F_m is the E07 response (bit-identical; cache shared).
- **Designs:** the E07 designs are built from **oracle** training responses (as committed in E07), so the same networks
  are scored under every perturbation (isolates the estimation-time effect). A realistic variant `dopt_m` is designed from
  the *model* wind's training responses (the designer is also misinformed); random and source_near do not use the wind.
- Code: `src/pravah/uncertainty.py` (perturbations, model responses); tests in `tests/test_wind_mismatch.py`:
  the truth (field, winds, readings) is bit-identical with and without E12; every non-oracle response solve is driven by
  the perturbed wind, never by `sc.truth.winds`; the oracle equals the E07 response.

## 3. Perturbation families (physical units; pre-declared magnitudes)

Winds are hourly uniform (u_x, u_y) with the protocol convention u_x = −s sin θ, u_y = −s cos θ (θ = direction the
wind blows from). Each family perturbs the whole calendar (spin-up, train, validation, test) of the model wind.

| id | family | definition | modest | severe |
|---|---|---|---|---|
| dir | direction bias | θ_m = θ + Δ (veer, clockwise) | Δ = 10° | Δ = 30° |
| spd | speed bias | s_m = f·s | f = 0.8 | f = 0.6 |
| lag | timing shift | w_m(h) = w(h − L) (model late; first L hours hold w(0)) | L = 2 h | L = 6 h |
| ar | low-frequency random direction error | θ_m = θ + e_h, e AR(1), τ = 12 h, marginal sd σ_θ | σ_θ = 15° | σ_θ = 30° |

Rationale: 10° and ±20 % equal the per-block jitter already in the benchmark generator (sd 10°, ×U(0.8, 1.2)), i.e.
"typical" reanalysis-scale errors; 30° and ×0.6 are stress cases. Speed is only *under*-estimated: at 64² VanLeer with
dt 30 s (A2), the held-out southerly reaches 4.8 m/s (Courant 0.92), so ×1.25 would exceed Courant 1. The AR error uses
one seed stream per scenario (SeedSequence([7000, replicate])) with the standard normal draws shared across magnitudes,
so σ_θ = 30° is the σ_θ = 15° path scaled by 2 (seed-consistent). Only positive Δ is run (one sign; [LEAD REVIEW]).
Reported per spec: realised mean |θ_m − θ| and mean s_m/s over the test hours (`wind_direction_error_deg`).

Not included (scope, stated so it is not mistaken for coverage): spatially varying wind error, diffusivity/mixing-depth
error, perturbations that differ between train and test (an error *regime change*). Boundary sensitivity (plan step 2)
is not run in the pilot.

## 4. Estimators (memory factor)

All linear-Gaussian MAP with prior N(0, a_ref² I), white noise, scalar σ (E07), on the model responses:
- `map24` — the frozen E07 24 h window;
- `map72` — intermediate memory (E10 family);
- `map_full` — growing window over all past hours (≡ E09 kf_aug with q = 0, gate-09).

The E10 random-walk filter needs the field-state linear hour operator (upwind only, E09/E10) and is **not** run in the
pilot (cost and time); `map72` gives a third memory point. [LEAD REVIEW] whether to add it at 32² later.

## 5. Placements

E07 pilot designs at k ∈ {2, 4, 8}: random (5 networks, averaged), dopt (greedy + swap), source_near, plus `dopt_m`
(§2). The E11 robust design (fc_w24_rob) is not in the pilot (needs the 8-direction prior solves); it is listed for the
64² job only if the E11 64² per-direction prior blocks exist by then.

## 6. Endpoints (all on the held-out test segment, protocol §10 issue times, 6 h warm-up)

Held-out, truth-scored:
- `forecast_rmse_h1` (the protocol primary metric, applied here as E12's main metric; it does not redefine the
  protocol endpoint, §8), `forecast_rmse_h6`;
- `field_rmse` = analysis RMSE at the issue time (post-update field);
- `source_rmse` (a_ref units), signed per-source relative bias, 95 % posterior-interval coverage of a_k;
In-sample / observable (what a user would see):
- `sensor_fit` = whitened RMS residual at the chosen sites at the issue hour, using the current estimate (≈ 1 if the model
  is right);
- `innov_rms` and `innov_ac1` = RMS and lag-1 autocorrelation (per-sensor centred) of the whitened one-step-ahead
  innovations y_{t+1} − F_m(t+1)·â_t at the sites (the plan's innovation_autocorrelation).

Derived per scenario: degradation ratio D = metric(spec)/metric(oracle) for the same design, estimator and k;
compensation gap C = log D(source_rmse) − log D(sensor_fit) (> 0 = fit hides source error); memory gain
M = (map24 − map_full)/map24 on forecast_rmse_h1; placement robustness log D(source_near) − log D(dopt).

## 7. Statistics

Cluster = (K, seed) (gate-10/11 condition 2). The pilot runs only the `mixed` training wind (veering shares layout,
strengths, noise and test winds with mixed, so it adds no independent cluster): 6 clusters = 6 scenarios. Per cluster the
three noise draws and five random networks are averaged first. Summaries: median over clusters, distribution-free
order-statistic CI (at n = 6 this is (min, max) with 96.9 % coverage), sign test. No pooling over k. With n = 6, a
6/6 sign result has p = 0.031 and nothing is Holm-significant; the pilot is descriptive.

## 8. Counterexamples (pre-declared search)

Reported explicitly: (a) cells where a perturbation *lowers* forecast RMSE vs oracle; (b) "deceptive fit" cases with
sensor_fit ratio < 1.1 but forecast ratio > 1.5; (c) cases where map_full is worse than map24 under perturbation but
better under the oracle (memory reversal).

## 9. Pre-declared vs exploratory

Pre-declared here: families, magnitudes, estimators, designs, endpoints, ratios, cluster definition, counterexample
rules, the 64² subset in §10. Everything in E12 is **exploratory** (protocol §8: wind-error studies cannot redefine the
endpoint; no confirmatory test). Anything added after seeing pilot results is labelled post hoc in §12.

## 10. 64² plan (one job, ≤ 64 CPU; not submitted by this branch)

- World: E07 sweep (A2: VanLeer, dt 30 s, protocol calendar 1440 h, spin-up 48 h), training wind `mixed`, K ∈ {2, 4},
  seeds 0–9 → 20 scenarios / 20 clusters; k ∈ {4, 8, 16} (E07 sweep budgets).
- **Reuse:** truth fields and oracle unit responses from the E07 sweep cache (same content hashes,
  `e07_cross_objective.scenario_fields`), read-only via `--fields`. Noise readings are regenerated deterministically.
- **New solves:** one unit-response solve per (scenario, source, perturbation): 60 per perturbation
  (10 seeds × (2 + 4) sources). Measured E07 cost: 4.4 CPU-h (15.8 k CPU-s) per 64² unit solve (resp_cpu_s mean 47 339 s per scenario
  for K̄ = 3; e07-sweep-compute.csv). Pre-declared 64² subset: dir 10°, dir 30°, spd 0.6, lag 6 h, ar 30° (5 specs) →
  300 solves ≈ 1 315 CPU-h ≈ 20.6 h on 64 workers (perfect packing; 300/64 = 4.7 rounds × 4.4 h ≈ 22 h wall), + stage 2
  (E07 stage 2: 230 CPU-s per scenario; × 6 wind variants × 20 scenarios ≈ 8 CPU-h). Request 64 CPU, 128 GB,
  30 h. If the time limit binds, the pre-declared reduction is to drop `ar 30°` first, then `dir 10°`.
- Memory per stage-2 task (K = 4): 7 field sets × 4 × 1440 × 64² × 8 B ≈ 1.3 GB → cap stage-2 workers at 32.
- Plan step 4's `cpus: 16, time: 4 h` is ≈ 20× too small for new VanLeer responses; this note supersedes it.

## 11. Leakage and validity checks

Truth bit-identity test; perturbed-response provenance test; oracle = E07 response; estimators read no row after the
issue time (reuse of `forecast_design.estimate`); a superposition check |c_bg + Σ a_k F_oracle − truth| (E07) per scenario.

## 12. Deviations

1. 2026-11-13 (before the pilot finished; no results used): unit typo in §10 fixed (4.4 CPU-h per solve, not 15.8; the
   totals were already computed with 4.4). A `--dry-run` check of the 64² task list shows 290, not 300, perturbed solves
   and 58, not 60, oracle responses (two source positions coincide between K = 2 and K = 4 scenarios of a seed and share a
   content hash), so the estimate is ≈ 1 270 CPU-h. On the cluster the dry run must show truth 0 and src 0 to prove E07
   cache reuse (otherwise ≈ 78 extra solves, ≈ 340 CPU-h).
2. 2026-11-13, **post hoc** (after seeing the pilot): the oracle forecast RMSE is near zero (perfect model), so ratios to
   the oracle (§6 D) are huge and not comparable across designs. Added model-free baselines on the same issue times
   (persistence c(t) → c(t+h); background only) and the contrast `vs_persistence_h1` = (persistence − model)/persistence
   per cluster. Reported beside, not instead of, the pre-declared ratios. No other change to metrics or rules.

### Lead decisions (2026-11-13, after the 32² pilot, before any 64² computation)
- The primary comparison stays as declared (perturbed vs oracle). The persistence comparison is reported as a
  **post-hoc secondary** baseline. It is clearly the more interpretable one under a near-perfect oracle, and that is
  stated wherever it is used.
- Not added now: the E10 random-walk filter arm, negative direction bias, and speed overestimation. Plan-3 E13/E14 may
  revisit them.
- **The 64² run is deferred by the user** ("option 3", 2026-11-13). The documented plan (≈ 1,270 CPU-h, about 22 h on
  64 CPUs; the commands are in the agent handoff and in §9) stays here for later.
  The pre-declared reduction order applies if time binds.

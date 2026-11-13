# E10 joint source-strength and field sequential inference: design note (pre-implementation)

Status: written 2026-11-13 on branch `experiment/e10-joint`, **before any E10 code or computation**. Committed alone;
later changes go in §10 (Deviations) with their reason. Synthetic 2-D passive tracer only. Everything in E10 is
**exploratory** (protocol §10.2 multiplicity; no confirmatory test is attached to E10).

Inputs: plan-02 `e10` ("Joint source-strength and field sequential inference"); `preregistered-protocol.md` §4 (diurnal
variant g_k(t) = 1 + 0.5 sin(2π(t − t_k)/24)), §7, §9–§12; E09 design note and gate-09 (kf_aug ≡ all-past-data MAP;
its gain over map24 is memory; the dopt/source_near ordering flips between 24 h and unlimited memory; E10/E11 must
declare memory length or the source random-walk variance as a factor); decision A2 (64² VanLeer is non-linear, no exact
KF transition matrix).

## 1. Question and what is already known

Plan question: can sequential observations update unknown strengths and the field together without spurious certainty?

Known before E10 (gate-09): with **constant** strengths, the correctly specified augmented filter (q = 0) is the
growing-window MAP. So E10's content is not "filter vs batch" but:

- (a) **admissibility**: unconstrained Gaussian estimates of a_k ≥ 0 can be negative; how often, and does an explicit
  constraint change source or forecast error?
- (b) **time-varying strengths**: with a source random walk of variance q per hour the augmented filter has finite
  memory. q is the declared memory factor (q = 0 ≡ all past data; large q ≈ short window). Compare against the
  growing-window MAP (`map_full`, primary reference, prior N(0, I) in a_ref units) and `map24` (E07 frozen, short-memory
  reference), plus a 72 h window (`map72`) to separate "window" from "random walk" memory.
- (c) EnKF with the real step: optional (§8).

## 2. State, dynamics, prior

Hours h = 0..T−1 after spin-up (E09 §2 conventions). State z_h = [x_h; a_h], x_h ∈ R^N the anomaly field
(c − c_bg, row-major), a_h ∈ R^K the strengths in a_ref units **emitted during hour h → h+1**.

- x_h = A_h x_{h−1} + B_h a_{h−1}, a_h = a_{h−1} + w_h, w_h ~ N(0, q I_K) (independent of the field; no field process
  noise: perfect transport model at 32² upwind).
- A_h: exact linear hour operator (`filter.step_operator`/`hour_operator`, upwind only). B_h = F_h − A_h F_{h−1} from
  the cached unit responses F (E09 deviation 1), so q = 0 reproduces the MAP response model exactly.
- Prior at hour 0: a_0 ~ N(0, I) (gate-09 recommendation, continuity with E07; no population knowledge),
  x_0 = F_0 a_0, so P_0 = [[F_0 F_0ᵀ, F_0], [F_0ᵀ, I]], m_0 = 0.
- Observation: y_h = H x_h + e, e ~ N(0, σ² I), σ = protocol white σ (fixed per scenario from the constant-strength
  training truth and reused for every truth variant, so noise draws are identical across variants).
- The covariance recursion depends on (design, k, q) only, not on the data, so all truth variants and noise draws run
  as paired data columns of one filter run.

q grid (a_ref² per hour): **{0, 1e-4, 1e-3, 1e-2, 1e-1}**. q = 0 is run as a filter for one design per scenario
(consistency check vs map_full) and otherwise taken from map_full. Rough memory: for per-hour information I about a
source the random-walk filter's memory is ~1/√(qI) hours.

Data-driven q choice (`kf_qsel`): per data column, the q minimising the mean squared 1 h-ahead innovation at the
sensors over the **validation** hours (no truth, no test data). Reported alongside the fixed-q arms.

## 3. Admissibility (one declared method)

**Truncated-Gaussian posterior mode, applied at output (non-recursive).** At each issue time, given the Gaussian
marginal N(m_a, C_a) of the strengths, a⁺ = argmin (a − m_a)ᵀ C_a⁻¹ (a − m_a) s.t. a ≥ 0 (a K-dim bounded least
squares; a⁺ = m_a when m_a ≥ 0). For the filter the field is conditioned on it: x⁺ = m_x + C_xa C_a⁻¹ (a⁺ − m_a), and the
constrained 1 h forecast is A_{t+1} x⁺ + B_{t+1} a⁺. For the MAPs the same mode equals the bound-constrained MAP (NNLS
with the Gaussian prior), and the forecast is F_{t+1} a⁺. The recursion itself stays the exact Gaussian one (the
constraint is not fed back), so the Gaussian calibration endpoints refer to the unconstrained posterior; this is a
declared caveat, not a claim that the Gaussian update preserves positivity. Recorded: fraction of (issue time, source)
estimates with m_a < 0, and the source/forecast error with and without the constraint (estimator suffix `_nn`).

## 4. Truth variants

All share locations, winds, noise draws and σ; the strengths per scenario a_k come from the benchmark generator.

| variant | a_k(h) | status |
|---|---|---|
| `const` | a_k (benchmark truth field, E09/E07 identical) | protocol base |
| `diurnal` | a_k (1 + 0.5 sin(2π(h − t_k)/24)), t_k ~ U(0, 24) per source, seeded stream | protocol §4 secondary variant |
| `logrw` | a_k exp(z_k,h), z AR(1) with lag-1 φ = exp(−1/72), stationary sd 0.5 | exploratory (E10-only) |

Time-varying truths are generated by the same linear recursion as the filter (x_0 = F_0 a(0), x_h = A_h x_{h−1} +
B_h a(h−1)): a perfect-model setting at 32² upwind. Check: the recursion with constant a reproduces the solver truth to
the E09 `recon_max_abs` level. The truth-matched random-walk variance is ≈ (0.13 ā)² ≈ 0.07 (diurnal) and ≈ 0.03
(logrw) a_ref² h⁻¹ for ā ≈ 2; the q grid brackets both.

## 5. World, designs, replication

E09/E07 pilot world (`scripts/e09_filter.py::make_cfg`): 32², upwind, dt 300 s, split days (1–4, 5, 6–11), 264 h;
winds {veering, mixed}; K ∈ {2, 4}; seeds {0, 1, 2} → 12 scenarios. Designs from training hours only (E07 code):
`random` (net 0), `dopt`, `source_near`; k ∈ {4, 8}. White noise, R = 3 draws per scenario (AR(1) not run: E09 already
characterised it, and cost). Issue times = E07 test issue times (warm-up 6 h).

## 6. Endpoints (protocol §10, all exploratory here)

1. **1 h pre-assimilation forecast RMSE** (full grid, test issue times; primary per protocol).
2. **Source RMSE** vs the current truth a(t) at issue time t (protocol §10.2 item 1, extended to a trajectory).
3. Analysis RMSE; 6 h forecast RMSE.
4. Calibration: 50/80/95 % coverage of the 1 h forecast at active cells (sd ≥ 0.01 σ, E09), CRPS; source 95 %
   coverage and NEES/K vs a(t); NIS/k and lag-1 innovation autocorrelation (gate-09: cite coverage or lag-1, not NIS).
5. Admissibility: fraction of negative source means; `_nn` vs unconstrained differences.

## 7. Hypotheses and statistics

- **H1 (const):** map_full (q = 0) is best; forecast RMSE is non-decreasing in q; small q (≤ 1e-4) is within 10 %.
- **H2 (diurnal, logrw):** some q > 0 beats both map_full and map24 on 1 h forecast and source RMSE; the best q is near
  the truth-matched variance (1e-2 to 1e-1).
- **H3 (calibration):** under time-varying truth map_full is overconfident (95 % coverage < 0.9, lag-1 > 0.3); the
  near-matched q is calibrated within ±0.05.
- **H4 (placement, links E07/E11):** the dopt − source_near ordering depends on memory: dopt better at q = 0 (gate-09),
  moving toward the map24 ordering (source_near better) as q grows.
- **H5 (admissibility):** negative means occur mainly for short memory (map24, large q) and weak designs (random, k = 4);
  the constraint changes forecast RMSE by < 1 % in median and lowers source RMSE where it binds.

Statistics: unit = scenario (mean over the 3 noise draws), paired across estimators/designs (same truth and noise).
For each (variant, k, design) the paired relative reduction (ref − est)/ref with ref ∈ {map_full, map24}; median with
the **distribution-free order-statistic 95 % CI for the median** (n = 12: (x₍₃₎, x₍₁₀₎), coverage 96.1 %); wins/losses.
Design contrasts (dopt vs source_near, dopt vs random) per estimator in the same way. No multiplicity adjustment; nothing
is confirmatory.

## 8. Not done / scope limits

- EnKF with the real (VanLeer) step and DAPPER cross-check: optional per the lead brief. Decided at implementation
  time by budget; if skipped it is stated in the handoff, not implied.
- Background confounding (plan step 1): not tested in this pilot (background known); stated as open.
- 64²: no cluster jobs. The exact KF cannot use VanLeer (A2); the projected path is in the handoff.

## 9. Software checks (tests before runs)

- q = 0 filter ≡ growing-window MAP (independent dense batch posterior).
- q > 0 filter ≡ batch posterior of the random-walk model over (a_0, w_1..w_t) (independent dense linear algebra).
- Truncated mode: equals m_a when feasible; KKT conditions; agrees with an independent bounded optimiser.
- New `run(..., keep=)` covariance columns equal the full covariance in a toy.
- Leakage: changing Y after the issue time leaves issued forecasts unchanged.

## 10. Deviations

1. **q grid extended with q = 1 a_ref² h⁻¹** (before any 32² run; after the 16² pipeline smoke, mixed K=2 seed 0).
   The §4 "truth-matched" values used ā ≈ 2; the per-hour increment scales with each scenario's own a_k (up to 5), so
   the matched q for diurnal reaches (0.13 · 5)² ≈ 0.4. In the smoke the best fixed q and the validation-selected q sat
   at the grid edge (0.1). The grid edge would otherwise censor H2. Grid now {0, 1e-4, 1e-3, 1e-2, 1e-1, 1}.
2. Smoke observation that informs H5 (not a change): negative source means occurred (up to 19 % of estimates for
   random k = 4, short memory), so the constraint does bind in this world.
3. **MAP windows 6 h and 12 h added** (`map6`, `map12`; before any 32² result). Reason: at 64² the exact RW filter is
   unaffordable (≈ 11 s per dense predict step at N = 4096, e09-local-cost.csv, and no exact operator for VanLeer, A2),
   so the realistic 64² memory factor is a short window on the cached constant-strength unit responses. The pilot
   should show how much of the RW filter's gain such a proxy recovers. Zero extra solves.

### Lead decisions (2026-11-13, after the 32² pilot)
- **Default sequential estimator:**
  - q = 0 (growing-window MAP) when sources are declared constant;
  - `kf_qsel` (q chosen on validation innovations only) when they may vary.
  - Placement orderings from now on are reported under at least map24, map_full and one q > 0.
- **`nonneg_mode` is reported beside the Gaussian mean** for new experiments. It never hurt in 1253 affected rows.
  Earlier results are not rewritten.
- **`logrw` stays an E10-only exploratory truth variant.** Time-varying sources enter plan 3 through the E13
  source-mismatch families.
- **64² follow-up deferred.** The exact random-walk filter and EnKF are out of budget. Plan 3 decides whether the cheap
  memory proxies are worth a 64² run.
- **Audit:** gate-10/11 runs jointly before plan 3 builds on these results.

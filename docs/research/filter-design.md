# E09 causal sequential filtering of the concentration state: design note (pre-implementation)

Status: written 2026-11-13 on branch `experiment/e09-filter`, **before any E09 code or computation**. Committed alone;
every later change goes in §11 (Deviations) with its reason. Items marked **[LEAD REVIEW]** are choices the protocol
does not settle. Synthetic 2-D passive tracer only; nothing here concerns real PM2.5 or real exposure.

Inputs: plan-02 `e09-kalman` (and what `e10-joint-assimilation`, `e11-forecast-design` need from it);
`preregistered-protocol.md` (approved baseline; §7 split, §9, §10 endpoints, §11 evaluation contract); amendments A1
(a_ref = 1 g/s) and A2 (64²: split VanLeer, dt 30 s, ~1 % non-linear); E07 journal and gate-07 (the static MAP 24 h
baseline and its failure under an unseen wind direction). Literature keys as in `literature-map.md`: enkf2009,
sharma2019, cheng2022, raanes2024 (DAPPER), roy2018.

Plan artifact names differ: the plan lists `causal-filter-formulation.md`, `src/pravah/assimilation.py` and
`tests/test_kalman.py`; the lead asked for this file, `src/pravah/filter.py` and the tests in `tests/test_filter.py`.
Same content.

## 1. Question and what E09 can and cannot show

**Plan question.** Does a forecast–observe–update filter estimate the field and forecast it better than a current-only
batch inverse (E07's static MAP over a 24 h window)?

**A structural fact to state before any run (derived in §3.4).** In the benchmark's perfect-model world (known winds,
known source locations, constant unknown strengths, linear transport, no model error) the *correctly specified* filter
carries the strengths in its state, `z = [x; a]`, with no process noise. That filter is algebraically identical to a
batch MAP over **all** past observations with the same prior. So "filter vs batch" in this setting is a contrast of
**memory length and prior**, not of filtering per se. E09 therefore separates three things:

1. `kf_aug` (correctly specified, augmented) vs `map24` (E07 frozen): effect of unlimited memory + a moment-matched prior.
2. `kf_aug` vs `map_full` (growing-window batch MAP, same prior): must agree to round-off. This is a validation of the
   transition operator and the filter code, not a finding.
3. `kf_field` (field-only filter: corrects the concentration state, does not estimate strengths): the genuinely
   sequential estimator. It is deliberately misspecified (§3.3), as any field-only filter must be when the forcing
   error is persistent.

Joint source–state estimation with dynamics, positivity and background confounding is **E10**; E09 uses the augmented
filter only as the correctly specified reference for its own question. Placement for forecasting is **E11**.

## 2. Time indexing and data availability

- Hour h = 0, 1, … counts hours after spin-up. `truth.field[h]` is the state at the end of hour h (wind `winds[h]`
  acts during hour h; `sensitivity.anomaly_fields` and `benchmarks.generate_scenario` use this convention).
- Observation y_h = bilinear sample of `truth.field[h]` at the chosen sites + noise (the benchmark's readings).
- **Event order at hour h:** (a) forecast x̂_{h|h−1} from the analysis at h−1 (issued at h−1); (b) score the forecast
  against the held-out truth; (c) reveal y_h at the chosen sites only; (d) update to x̂_{h|h}; score the analysis; (e)
  issue forecasts for h+1, h+6, h+24.
- **Causality rule (protocol §5, §11):** a forecast issued at t depends only on y_0 … y_t, the prior, known winds up to
  the target hour (perfect forcing, labelled idealised as in E07, whose forecasts also use the known test winds) and
  known source locations. A frozen test perturbs a future observation and requires every earlier-issued forecast to be
  bitwise unchanged.
- Filters start at h = 0 (first post-spin-up hour) and run through train and validation into test. Earlier-split
  observations are past data at test issue times; using them is causal. No setting is tuned on any split: every filter
  setting is fixed in this note. Placement uses training hours only (E07 designs, unchanged).

## 3. State-space model

### 3.1 Transition

Work with the anomaly d = c − c_bg (c_bg known, inflow Dirichlet value 0 for the anomaly, exactly as the benchmark
solves it). One FiPy step under wind u is affine in (d, source) and, with c_bg = 0, **linear and homogeneous** in d:

    d_{n+1} = M_u^{-1} d_n + (source term),        M_u sparse (5-point: implicit diffusion, removal, upwind convection, study BC).

One hour of n_s = 3600/dt steps under the hour's wind gives

    x_h = A_h x_{h−1} + B_h a,     A_h = (M_{u_h}^{-1})^{n_s},     B_h[:, k] = a_ref · (one-hour response from zero to unit source k under u_h)

with x = vec(d) (row-major [i, j]) and a in a_ref units. **Operator extraction (black-box, uses the verified solver):**
M_u is recovered from `forward.simulate` by probing: for P ≥ 8 random fields d_p, y_p = one solver step of d_p, and
each row of M_u satisfies d_p[r] = Σ_{j ∈ stencil(r)} M_u[r, j] y_p[j]; 5 unknowns per row from P equations. The fit
residual and a fresh validation probe must agree with `simulate` to ≤ 1e-9 relative, otherwise the extraction refuses
(this is what happens for a non-linear scheme). B_h comes from separate `simulate` calls from zero with each unit
source, so the identity `truth.field[h] − c_bg = A_h(truth.field[h−1] − c_bg) + B_h a_true` is a non-trivial check of
both (recorded per scenario).

**Scheme (A2).** Local runs (16²/32²) use the protocol's upwind backward-Euler at dt 300 s, as the E07 pilot did; there
the transition is exactly linear and the KF is exact. At 64², A2's split VanLeer is TVD and explicit, its limiter
makes the step non-linear in d (superposition error ≈ 1 % over 96 h; up to 0.73 σ at a single cell-hour over the full
calendar, but ≤ 1e-4 change in any E07 RMSE, gate-07 issue 7). No exact linear operator exists. Options for 64²:

| option | transition | cost per hour | verdict |
|---|---|---|---|
| (i) augmented, no field process noise | x_h = F_h a, F from K unit-response solves (the superposition E07 already uses) | K solves per scenario; K×K algebra | **recommended for `kf_aug`/E10 baseline**; model error = A2 superposition error, already measured negligible for RMSE |
| (ii) EnKF, members propagated by the real VanLeer step | non-linear, exact solver | N_e solves (N_e ≈ 50–100) | **recommended for any filter with field process noise** (`kf_field`, E10/E11 with model error); needs localisation/inflation |
| (iii) exact KF with upwind as the linear surrogate | linear | dense N³ (N = 4096) | rejected: upwind's K_eff 350–2900 m²/s vs 50 (A2) is a gross model error |
| (iv) tangent-linear of VanLeer by Jacobian–vector products | linearised | N solves/hour | rejected: cost |

### 3.2 Observation operator and noise

- y_h − c_bg = H x_h + v_h. H (k × N) holds the bilinear interpolation weights of `observations.bilinear_sample` at the
  site coordinates (benchmark lattice sites sit at cell centres, so H is a selection, but the general weights are used).
- **White (primary):** v_h ~ N(0, σ² I_k), σ = the protocol scalar (0.05 × median training truth over candidate sites,
  `cross_objective.site_sigma`). The filter is given the true σ (the benchmark's σ is a known instrument property).
- **AR(1) (variant, protocol §5):** v_h = e_h, e_h = ρ e_{h−1} + sqrt(1 − ρ²) σ η_h, ρ = 0.8, independent sites.
  Correct handling: augment the state with e (k states), observe y = H x + e with R = 0 (plus a 1e-10 σ² jitter for
  numerical positive-definiteness). `kf_aug_ar1` does this; `kf_aug` and `kf_field` keep assuming white noise on AR(1)
  data, the honest misspecified comparison.
- **Absent sensors:** a NaN reading drops that row of H and R for that hour (no update from it).

### 3.3 Filters and process noise

Strength prior, moment-matched to the benchmark's generating LogUniform(0.5, 5) × a_ref:
ā = 4.5/ln 10 = 1.954, s² = 24.75/(2 ln 10) − ā² = 1.555 (s = 1.247), a ~ N(ā 1, s² I) in a_ref units. This is the
generating distribution, not the realised strengths. Gaussian, so negative a has non-zero prior mass (≈ 6 %); E10 owns
positivity. **[LEAD REVIEW]** E07's frozen `map24` uses N(0, I); I keep it frozen and report `map_full` with the
moment-matched prior so the prior effect can be separated from the memory effect only partly (see §6).

Initial state: x_0 = F_0 a exactly (spin-up from zero with constant a), F_0 = unit responses at h = 0. Hence
`kf_aug`: z_0 ~ N([F_0 ā; ā], [[F_0 S F_0ᵀ, F_0 S]; [S F_0ᵀ, S]]), S = s² I (rank K: the joint state is a deterministic
function of a; the KF never inverts P, so this is fine).

| filter | state | transition | process noise Q | specification |
|---|---|---|---|---|
| `open_loop` | x | x_h = A_h x_{h−1} + B_h ā | — | no observations (plan's "no update") |
| `kf_aug` | [x; a] | [A_h, B_h; 0, I] | 0 | correct (up to Gaussian prior on a LogUniform truth) |
| `kf_aug_ar1` | [x; a; e] | adds e_h = ρ e_{h−1} | diag(0, 0, (1 − ρ²) σ² I) | correct under AR(1) data |
| `kf_field` | x | x_h = A_h x_{h−1} + B_h ā + w_h | B_h S B_hᵀ + q₀² I, q₀ = 1e-3 σ | misspecified: forcing error B_h(a − ā) is constant in time, modelled as i.i.d. hourly |

`kf_field` expectation (stated before any run): because Q treats a persistent forcing error as independent each hour,
its forecast variance under-grows, so NIS > 1 and coverage below nominal; it corrects the state only where sensors see
the plume and forgets corrections as they advect out. q₀ is a numerical floor, not a tuned model-error level.

### 3.4 Filter equations and the batch identity

Predict: m⁻ = A m + u, P⁻ = A P Aᵀ + Q. Update (rows with data): S = H P⁻ Hᵀ + R, K = P⁻ Hᵀ S⁻¹ (Cholesky solve),
m = m⁻ + K(y − H m⁻), P = P⁻ − K H P⁻, then symmetrised. Joseph form is not needed: with the optimal gain it equals
P⁻ − K S Kᵀ, and the cost is the same O(N² k); PSD is tested (§4). Innovation ν = y − H m⁻ and S are kept for NIS.

Batch identity: with Q = 0, x_h = F_h a for every h, so y_h = H F_h a + v_h and the joint posterior is that of the static
linear model y = G a + v with prior N(ā, S); the KF computes it recursively (information form: the posterior precision
after hour t is S⁻¹ + Σ_{h≤t} (H F_h)ᵀ R⁻¹ (H F_h)). Forecasts x̂_{t+h|t} = F_{t+h} â_t. This equals E07's estimator
with an unlimited window and the moment-matched prior. A toy test checks the identity; each scenario records
max |kf_aug − map_full| forecast difference.

### 3.5 Cost

N = n_x n_y. Covariance recursion: two dense N×N products per hour (A P Aᵀ) + O(N² k) update. Dense A_h per distinct
wind: one sparse LU solve with N right-hand sides, then n_s-th power by repeated squaring. Measured on this machine
(single thread): 32² dense N×N product 0.035 s, sparse solve with N RHS 0.04 s; 64² 2.3 s and 0.96 s. Expected:
32² ≈ 0.1 s/h per filter (exact KF fine); 64² with dt 30 s (n_s = 120) ≈ 5 s/h per filter + ≈ 17 s per distinct wind,
i.e. hours per filter run over the 1440 h protocol calendar, and it would still be the wrong (upwind) physics. Hence
the §3.1 recommendation: option (i) for augmented/static-source filters at 64², option (ii) EnKF when field process
noise is needed. The local run measures actual per-hour cost at 16²/32² and a single-hour timing at 64².

## 4. Implementation and independent tests (`src/pravah/filter.py`, `tests/test_filter.py`)

API: `predict`, `update`, `run` (causal loop with forecast-before-update, multi-column data for paired noise draws,
multi-horizon forecasts by propagating pending means), `step_operator` (black-box probe of `simulate`),
`hour_operator`, `observation_matrix`, `gaussian_crps`, `coverage`. Tests (reference values computed independently,
by hand or by dense batch algebra, never by the module under test):

1. scalar random walk: one predict/update against hand-computed mean/variance;
2. linear-Gaussian toy with static parameter: `run` with Q = 0 equals the batch posterior (mean and covariance) and
   its forecasts;
3. causality: perturbing y at t′ leaves every forecast issued before t′ (all horizons) bitwise unchanged, and does
   change later ones (the test detects leakage, not just absence of change);
4. indexing: forecast for h uses the transition of hour h and is recorded before y_h is used (delayed-observation
   probe);
5. limits: R → ∞ gives the open-loop forecast; Q → 0 with no data gives deterministic propagation; absent (NaN)
   sensors give no update;
6. perfect-model toy calibration: NEES/NIS means within Monte Carlo bands, covariance stays PSD over a long run;
7. `step_operator` vs `simulate` on random fields (upwind, study and periodic BC) and refusal for VanLeer.

EnKF: not implemented in E09 (optional per lead); its form for 64² is §3.1 option (ii) and belongs with E10/E11.
The offline RTS smoother (plan: oracle diagnostic only) is not needed for any E09 comparison and is omitted.

## 5. Causal replay (local, exploratory)

Same pilot world as E07 (`scripts/e07_cross_objective.py` scale "pilot"): 32², upwind dt 300 s, spin-up 12 h, train
days 1–4, validation day 5, test days 6–11 (held-out winds: steady southerly, stagnation then front, diurnal
rotation), a_ref = 1 g/s, c_bg = 5.

| factor | levels |
|---|---|
| training winds | veering, mixed |
| sources K | 2, 4 |
| seeds | 0, 1, 2 |
| designs (E07 code, training hours only) | random (2 networks), dopt, source_near |
| sensors k | 4, 8 |
| noise arms | white (primary), AR(1) ρ = 0.8 |
| noise draws | R = 3 per arm, shared by every design and estimator (paired) |

12 scenarios × 8 designs × 2 arms × 3 draws. Covariances do not depend on the data, so each filter's covariance
recursion runs once per (scenario, design) and carries all 6 data columns. ≤ 6 worker processes, OMP_NUM_THREADS=1.
No test-period setting is chosen after looking at results.

## 6. Metrics (all exploratory; protocol §10 definitions where they exist)

- **Forecast RMSE, h = 1** (protocol primary definition: all cells, issue times = E07 `issue_times`, test start + 6 h
  … last test hour − 1), and **h = 6, 24** (issue times with t + h inside the test segment; n reported).
- **Analysis RMSE** at the issue times (protocol secondary 2).
- **Source RMSE**/a_ref for estimators that estimate a (`kf_aug*`, `map24`, `map_full`).
- **Calibration (protocol secondary 3)** of the one-step forecast field: coverage of central 50/80/95 % Gaussian
  intervals over all cells and issue times; mean Gaussian CRPS; reliability = coverage − nominal. For `map24` the
  forecast variance is diag(F_{t+1} N_t⁻¹ F_{t+1}ᵀ) a_ref², with N_t its posterior precision.
- **Innovation diagnostics:** mean NIS/k over all update hours (1 if calibrated; MC band from χ²_k), test-period
  only as well; lag-1 autocorrelation of whitened innovations (≈ 0 if white).
- **Runtime:** wall and CPU seconds per filter-hour; operator extraction time.

## 7. Baselines (all reported)

`background_only` (anomaly 0), `climatology` (training-period mean truth field, as E07), `persistence_oracle` (truth at
t: **not causally available**, a reference only, the protocol's literal persistence), `persistence_analysis`
(`kf_aug` analysis at t held fixed), `open_loop`, `map24` (E07 frozen estimator, `cross_objective.estimate_series`,
prior N(0, I)), `map_full` (growing window, moment-matched prior).

## 8. Comparison plan

Per scenario and k, every estimator sees the same designs and noise draws; RMSEs are averaged over the 3 draws first.
Report: medians over scenarios of each metric; paired relative differences (map24 − X)/map24 with wins/losses and a
percentile bootstrap CI over the 12 scenarios (seed 4000, 10 000 resamples), separately for k = 4, 8 and per design.
Clusters: designs and k within a scenario share truth and noise, so the scenario is the unit (gate-07 issue 6).
n = 12 is small, the calendar is short and the numerics are upwind: **everything is exploratory**; no confirmatory
claim. Specific pre-run predictions, recorded to be checked: (P1) `kf_aug` ≡ `map_full` to ≤ 1e-6 relative;
(P2) `kf_aug` ≤ `map24` forecast RMSE in most scenarios, largest gain for designs/windows with deficient 24 h rank
(dopt under the southerly block, gate-07); (P3) `kf_field` NIS > 1 and under-coverage; (P4) under AR(1) data
`kf_aug_ar1` is better calibrated than `kf_aug`.

## 9. What E10 and E11 can reuse

- E10: `predict`/`update`/`run` on any augmented state; the `kf_aug` construction is the exact linear-Gaussian baseline
  E10 extends (source dynamics, positivity, background parameter); `map_full` as its batch cross-check; the AR(1)
  augmentation.
- E11: `run` with fixed filter settings and multi-horizon pending forecasts, the causal scorer and calibration metrics;
  the posterior forecast variance from the covariance recursion is the "posterior field variance proxy" E11 compares
  with realised loss. Because the covariance recursion does not depend on the data, a design's forecast-variance
  trace can be computed before any observation, which is what a training-only forecast-placement objective needs.

## 10. Limitations

Perfect model (known winds incl. the forecast hour, known locations, constant strengths, no model error); upwind
numerics at 32² over-diffuse (A2); short pilot calendar (6 test days); 12 scenarios; Gaussian prior on a LogUniform
truth and unconstrained a; no EnKF and no smoother; σ given to the filter.

## 11. Deviations

All made after a 16² smoke run of one scenario (mixed, K = 2, seed 0) and before the 32² run; none looked at 32²
or test-period results of the declared run.

1. **Forcing B_h from the unit responses, not from direct one-hour solves.** The smoke run showed `kf_aug` vs
   `map_full` forecasts differing by up to 1.2e-3 µg m⁻³ (prediction P1 said ≤ 1e-6 relative). Diagnosis: A_h matches
   `simulate` to 1e-16, but FiPy's LU solve stops at a residual tolerance, so a one-hour solve from zero differs from
   the difference of consecutive unit-response fields by up to 6e-5 (≈ 6e-6 of the plume peak). The truth carries the
   same solver-tolerance inexactness (truth vs Σ a_k F_k: 3e-5 at 16²). The filters now use
   B_h = F_h − A_h F_{h−1}, i.e. exactly the response model of `map24`/`map_full`, so all estimators share one model
   and P1 holds to 1e-13. The truth-reconstruction check (`recon_max_abs`) still uses independent direct solves.
2. **Coverage on "active" cells.** Over all cells, coverage is dominated by cells the plume never reaches (forecast sd
   and error both ≈ 0). Coverage is reported both over all cells (`cov*_all`, as declared) and over cells whose
   forecast sd ≥ 0.01 σ (`cov*_active`, a mask that depends on the forecast only, never on the truth), with the
   active fraction.
3. `kf_aug_ar1` is also run on the white arm (it costs nothing extra: the covariance recursion is shared); the AR(1)
   arm comparison is as declared.

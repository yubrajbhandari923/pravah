# PRAVAH Phase 1 — Preregistered protocol (E00)

**Status:** APPROVED 2026-11-12 (`state/decisions/2026-11-12-protocol-approval.md`): baseline, amendable on documented evidence of infeasibility. Originally a draft for human approval. Values below resolve the "open decisions" in `methodological-decisions.md`. They become binding when a human approves them; after approval, any change is a recorded scientific amendment, never a silent edit. Nothing here reports a result.

**Scope statement.** The model is a 2-D, depth-averaged *passive tracer* on a rectangle. It is **not** PM2.5 (no chemistry, aerosol microphysics, vertical structure or real emissions). Synthetic-benchmark outcomes do not validate observed air quality, real exposure or public-health claims.

## 1. Governing model

- Domain: x in [0, L_x], y in [0, L_y], L_x = L_y = 10 km. Uniform FiPy `Grid2D`, 64 x 64 cells (production); 32 x 32 and 128 x 128 are used only for convergence checks (E01).
- PDE for concentration c(x, y, t) [ug m^-3]:
  dc/dt + u(t)·grad c = div(K grad c) − lambda c + S(x, y, t) + (background handled in §4)
- K = 50 m^2 s^-1 (isotropic, constant); lambda = 1/(12 h) = 2.31e-5 s^-1 (first-order removal/deposition, constant). Wind u(t) is prescribed, spatially uniform, hourly-piecewise-constant; speed 0.5–5 m s^-1.
- Effective mixing depth H = 500 m, fixed. Emission rates are areal; the depth-averaged source term is S = q / H.
- Units: length m, time s (reported in h), concentration ug m^-3, emission strength q in ug m^-2 s^-1 (areal flux into the column); S in ug m^-3 s^-1.

## 2. Boundary and initial conditions

- Inflow boundaries (where u·n < 0): Dirichlet c = c_bg (the background, §4). Outflow boundaries: zero diffusive gradient (Neumann, dc/dn = 0) with advective outflow. Walls are never closed.
- Initial condition: c(x, y, 0) = c_bg for every scenario. A spin-up of 48 h is run and discarded before the observation calendar starts, so initial-state error is not an unmodelled nuisance.
- Mass-balance diagnostic (E01) is evaluated with these exact boundary rules.

## 3. Time integration

- Implicit (backward-Euler) diffusion and removal; first-order upwind advection treated implicitly (FiPy `UpwindConvectionTerm`); fixed dt = 300 s. Observation/assimilation cadence is hourly (12 solver steps per hour). **Amendment A2 (2026-11-12, `state/decisions/2026-11-12-advection-scheme.md`): 64² runs use dimensionally split VanLeer advection with dt = 30 s (120 steps per hour); upwind at 64² failed the dt-halving gate and carried 5–60× numerical diffusion. Gates unchanged.**
- Numerical acceptance (E01, fixed now): observed order of convergence of the L2 error against the analytic/manufactured solution must be ≥ 0.9 on the 32/64/128 grids for the first-order scheme; relative global mass-balance residual ≤ 1e-3; dt-halving changes the primary-endpoint field by ≤ 1% relative L2. Failure of any gate stops downstream experiments (§12).

## 4. Source and background model

- Sources: K_s fixed, known-location basis functions, unknown non-negative strengths. Each basis is a normalised isotropic Gaussian footprint phi_k(x, y), sigma = 400 m, integrating to 1 over the domain, truncated at 3 sigma. Emission field q(x, y, t) = sum_k a_k g_k(t) phi_k. Normalisation means a_k has units ug s^-1 (total rate) and is comparable across sources.
- Temporal factor g_k(t): constant (=1) in the *base* scenario (source persistence = whole window). A diurnal variant, g_k(t) = 1 + 0.5 sin(2 pi (t − t_k)/24 h) with random phase t_k, is a secondary robustness variant and is not part of the primary endpoint.
- Scenarios: K_s in {2, 4, 8}; locations drawn from a fixed 12 x 12 candidate source lattice with minimum centre separation 1.5 km; a_k ~ LogUniform(0.5, 5) times a reference rate a_ref = 100 ug s^-1 (**superseded by amendment A1, 2026-11-12: a_ref = 1e6 ug s^-1 = 1 g s^-1**, `state/decisions/2026-11-12-source-scale.md`).
- Background: spatially uniform, constant c_bg = 5 ug m^-3 (known in base; unknown scalar b ~ N(5, 1^2) in the background-uncertainty variant, estimated as an extra parameter).

## 5. Observations, noise and sensors

- Sensors are fixed point samplers at grid cell centres, ideal (no drift, bias or lag) in the base case.
- Noise, sampled independently of the truth with its own seed stream:
  - **White:** y = c + e, e ~ N(0, sigma^2) with sigma = 0.05 · (median truth concentration of the training period at the sensor candidates), fixed per scenario family and recorded.
  - **Correlated:** AR(1) in time, rho = 0.8 per hourly step, same marginal sigma; plus spatially independent. Reported as a variant, not substituted post hoc.
  - **Bias/calibration:** reserved for P3 (E14); out of scope here.
- Observations after the forecast issue time are never visible to an estimator or placement algorithm.

## 6. Factors and design

| Factor | Levels |
|---|---|
| Number of sources K_s | 2, 4, 8 |
| Sensor budget N | 2, 4, 8, 16, 32 |
| Window length | snapshot (1 sample), 6 h, 24 h, 7 d |
| Noise | white (primary), AR(1) correlated (variant) |
| Wind | known (primary), held-out regimes (§8) |

- Candidate sensor set: the 16 x 16 interior lattice of cell centres (256 sites), minimum inter-sensor separation 1.25 km. Every placement algorithm sees the same candidate set and budget N.
- Full factorial is 3 x 5 x 4 = 60 cells per noise/wind condition. Primary analysis is confined to the 60 cells under white noise and known wind.
- Replication: 20 independent scenario seeds per cell, paired across estimators/placements (same truth and noise for every method). Seed streams: `truth_seed = 1000 + s`, `noise_seed = 2000 + s`, `placement_seed = 3000 + s`, s = 0..19. Random placement baseline uses 10 random networks per scenario seed.
- Compute budget: local only (no cluster before verified E01). If a full sweep exceeds a 48-core-hour local budget, the pre-declared reduction is to drop K_s = 8 x N in {2, 4} first and report the cell as NOT RUN; no other silent cell removal.

## 7. Calendar split

Simulated time covers 60 days at hourly cadence after spin-up (1440 h).

| Segment | Days | Purpose |
|---|---|---|
| Train | 1–30 | fit inverse hyperparameters (regularisation, noise scale), choose placement objective settings |
| Validation | 31–42 | model selection, early stopping, tuning |
| Test | 43–60 | single use: all headline endpoint values |

Rules: test truth is never used for tuning, placement or model selection. Test-period results are computed once per frozen configuration; re-running after a change to the configuration is an amendment, not a re-roll. Sensor networks are placed using train-period information only. The 7-day window ends at, and does not extend past, the forecast issue time.

## 8. Wind regimes

- Training/validation: three regimes (steady westerly 3 m s^-1; veering 270° → 315° over 12 h; weak/variable 0.5–1.5 m s^-1 with 6 h reversals), seeded random speed/direction perturbations about each.
- **Held-out test regimes (never used for placement, hyperparameters or model selection):** (a) steady southerly with speed 4 m s^-1; (b) 24 h stagnation (0.2–0.5 m s^-1) followed by a frontal 90° shift; (c) diurnal sea-breeze-like rotation. The test calendar uses (a)–(c) only.
- Primary endpoint uses known wind. Wind-error studies (E12) are separate and cannot redefine the endpoint.

## 9. Estimation vs. prediction — two separate questions

- **Identifiability assessment** asks whether the parameters a_k *can* be distinguished by a network: structural rank and conditioning of the sensitivity (Jacobian) matrix, singular-value spectrum, and noiseless recovery on the same network. It is a property of (network, model), not of an estimator, and is reported separately from error.
- **Prediction accuracy** asks how well the chosen estimator predicts held-out concentrations given noisy data. A network can be well-conditioned yet predict poorly (e.g. under wind mismatch), or predict well while individual a_k are unidentifiable. We never infer one from the other without reporting both.
- Identifiability metrics: rank of the weighted Jacobian at relative tolerance 1e-8 times its largest singular value, 2-norm condition number, fraction of a_k whose Cramér–Rao standard deviation ≤ 50% of the true a_k. These are descriptive and do not enter endpoint ranking.

## 10. Endpoints

### 10.1 PRIMARY endpoint (one only)

> **Held-out, pre-assimilation, one-step-ahead field-forecast RMSE.**

- Definition: at forecast issue time t_i, the estimator has used observations up to t_i only. It issues the one-step (1 h) forecast of the full concentration field at t_i + 1 h *before* the observation at t_i + 1 h is assimilated. RMSE = sqrt(mean over all grid cells and all test issue times of (c_forecast − c_truth)^2), where c_truth is the noise-free truth, in ug m^-3.
- **Declared timing window:** issue times are hourly at every hour of the test segment (days 43–60; 432 issue times), on held-out wind regimes (§8). The RMSE is averaged over all of these, with the first 6 h after the test start excluded as estimator warm-up, so the window is 426 issue times. No other window may be substituted.
- Primary comparison: paired difference in primary RMSE between a placement strategy and the random-placement baseline (and between the static-inverse estimator and the persistence/climatological baseline), per scenario seed, averaged across the 60 primary cells. 95% CI from a paired bootstrap over seeds (10,000 resamples, seed 4000). Cells are not pooled before computing paired differences.
- Baselines (reported in every comparison): persistence (c(t_i) forecast unchanged), climatology (training-period mean field), and the random placement network.

### 10.2 SECONDARY endpoints, ranked (all reported; none can rescue a failed primary)

1. **Source RMSE:** sqrt(mean_k (a_k_hat − a_k)^2) / a_ref, plus relative error per source; undefined-estimate cases counted as failures.
2. **Post-analysis field error:** RMSE of the analysis field (after assimilating the observation at the issue time) against truth, test period.
3. **Probability calibration:** coverage of nominal 50/80/95% credible intervals for field values at held-out cells, and the continuous ranked probability score; reliability slope reported.
4. **Population-weighted error (optional):** RMSE_w = sqrt(sum_j w_j (ĉ_j − c_j)^2 / sum_j w_j) with a *synthetic* smooth population density w (fixed, seeded, normalised); a real-geography weight is conditional and not part of Phase 1 unless approved. It is explicitly an exposure-weighted field error, not a health outcome.

Multiplicity: the primary is the single confirmatory test; secondaries are labelled exploratory and are reported with unadjusted CIs, so no secondary is claimed as confirmatory.

## 11. Evaluation contract (frozen once approved)

- Truth and inference APIs are separate; estimators cannot import truth objects. A leakage test (oracle sensor reading from the future must change nothing in the issued forecast) is a frozen test.
- Config, seeds, metric definitions and splits are committed before any test-period computation. Metric code is not modified after test results exist except through a documented amendment applied to all methods and re-run for all of them.
- Each run records: command, git SHA, seed(s), full configuration, machine, package versions, dataset IDs and artifact hashes.

## 12. Success, stop and negative-result rules

- **Primary hypothesis (stated before data):** an optimised network of N sensors reduces primary RMSE relative to random placement of the same N by at least 10% of the random-placement RMSE for N ≥ 8, in the paired comparison.
- **Supported** (preliminary until independent review): the 95% paired-bootstrap CI of the relative reduction lies entirely above 10% in at least 70% of the N ≥ 8 cells.
- **Contradicted:** the 95% CI of the relative reduction lies entirely below 10% in ≥ 70% of those cells (including no benefit or harm).
- **Unresolved:** anything else, including cells NOT RUN. Unresolved is reported as such.
- A negative or unresolved outcome is a valid result and is published with the same prominence; it does not trigger changes to the endpoint, thresholds, splits or windows.
- **Stop rules:** (1) any numerical gate in §3 fails → stop, fix the solver via a recorded decision, repeat E01; (2) leakage test fails → no result is reportable; (3) identifiability audit shows rank < K_s for all candidate networks at the largest budget in a scenario family → that family is reported as non-identifiable and excluded from source-RMSE claims, not tuned away; (4) three failed runs without review, or five runs without synthesis → stop per project policy.
- Outputs interpreted as deployment guidance, real exposure accuracy or public-health benefit are out of scope in all outcomes.

## 13. Known limitations

Uniform wind and constant K, a single mixing depth, known source locations, ideal sensors, and a synthetic population limit generality. The 10% threshold, 70% fraction, Gaussian width, K, lambda and noise level are judgement calls made without reference to outcome data; they have not been tuned. The Bhardwaj 2026 identifiability approach is cited from an unreviewed preprint, and the Green (2006) thesis was not read in full during the literature audit, so the identifiability diagnostics here are PRAVAH adaptations, not reproductions.

## 14. Provenance

Authored for task T-002 on experiment EXP-001; supersedes no earlier protocol. Inputs: `methodological-decisions.md` (open decisions), `phase1-proposal.md` (pre-analysis commitments), `literature-replication.md`. No computation was run to write this document. Approval, amendments and the freeze commit are recorded through Research OS.

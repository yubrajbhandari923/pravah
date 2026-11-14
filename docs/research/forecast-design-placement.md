# E11 placement for causal forecasting: design note (pre-pilot)

Status: written 2026-11-13 on branch `experiment/e11-forecast-design`, **before any E11 computation**, and committed
alone. Every later change goes into §11 (Deviations) with its reason. Items marked **[LEAD/USER REVIEW]** are choices
the protocol does not settle. This concerns a synthetic 2-D passive tracer with perfect, known winds and known source
locations. All results are **idealised (perfect forcing)**.

Inputs:
- plan-02 key "Placement optimized for causal forecasting and sequential reconstruction" (E11);
- `preregistered-protocol.md` §7 (split), §9–§12 (endpoints, contract, rules);
- E07 (`cross-objective-design.md`, `state/experiments/e07.md`);
- gate-07 (dopt's loss is per-window blindness in the unseen steady-southerly regime);
- gate-09 (the placement ordering flips with estimator memory: source_near wins with a 24 h window, dopt wins with all past
  data; report both memories);
- E09 (`filter.py`; the perfect-model augmented KF with constant sources is the growing-window batch MAP).

## 1. Question

Can a placement objective that targets the **forecast variance of the estimator actually deployed**, computed from
training-period information alone, match or beat the better of `dopt` and `source_near` under each estimator memory?
Does a wind-robust version, which averages over a prior on wind directions, remove dopt's E07 failure mode (blindness under
unseen wind directions)?

Plan hypothesis: a placement that is excellent for source estimation may not minimise multi-horizon forecast risk.

## 2. Why the objective can be computed from training information only

The estimators are linear-Gaussian, and the noise model is fixed (white, scalar σ, prior N(0, a_ref² I) on the strengths).
Their posterior covariance therefore does not depend on the data. It depends only on:
- the chosen sites S;
- the sensitivities A_h(S) (whitened, a_ref-scaled) of the hours in the estimator's window.

For the static-source model, the KF covariance recursion with Q = 0 on [field; a] gives Σ_t = (I + Σ_{h∈W(t)} A_hᵀA_h)⁻¹.
This is the same as the batch MAP (E09/gate-09).

The h-step forecast error of the field mean at hour t + h is F_{t+h}(â_t − a) with a_ref-scaled unit responses F.
Its expected squared error, averaged over the N cells, is therefore

  V(S; t, h) = tr(Q_{t+h} Σ_t(S)),   Q_s = a_ref² F_sᵀ F_s / N   (K × K, the field weight of hour s).

The expectation is over noise and the prior. In a perfect model it is the whole error.

V needs A_h and Q_s, which depend only on the winds and known source locations. It never involves readings, noise draws,
strengths or truth. The design cannot know the test winds. So V is evaluated either:
- on the **training hours** (as training-period sensitivities are used by every E07 design), or
- on a **declared prior over wind directions** (§3, robust variants).

## 3. Strategies and what each may know

All designs share the E07 pilot candidate set (`admissible_sites()`), k, the 1250 m spacing rule, σ and the source
locations, as in E07 §3.

| strategy | knows | rule |
|---|---|---|
| `random` (10 nets) | nothing | E07 |
| `coverage` | nothing | E07 (farthest point) |
| `source_near` | source locations | E07 (locations-only control) |
| `dopt` | locations, prior, σ, training G | E07 greedy Bayesian D + 1-swap |
| `fc_grow` | + training full-grid field weights Q_h | minimise J_grow, greedy + 1-swap |
| `fc_w24` | same | minimise J_w24, greedy + 1-swap |
| `fc_grow_rob` | + responses under the design wind prior (§3.1) | minimise J_grow_rob, greedy + 1-swap |
| `fc_w24_rob` | responses under the design wind prior only | minimise J_w24_rob, greedy + 1-swap |

Horizons are H = {1, 6, 24} h. T is the number of training hours, and M_tr(S) = Σ_{h<T} A_hᵀA_h(S).

- **J_grow(S)** = mean_{h∈H} mean_{t∈[0, T−h)} tr(Q_{t+h} (I + M_tr(S))⁻¹).
  - Deployment-time proxy: at any test issue time, the growing-window estimator holds at least all training information.
  - Σ does not depend on t, so horizons enter only through which training hours weight Q. J_grow is essentially a
    full-grid field-weighted A-optimality. It is E07's `fieldvar` with the full grid in place of the lattice sites as
    "field".
  - This is a derived consequence, recorded before seeing data. **With static sources the growing-window forecast
    variance is (nearly) horizon-independent.**
- **J_w24(S)** = mean_{h∈H} mean_{t∈[23, T−h)} tr(Q_{t+h} (I + Σ_{s=t−23..t} A_sᵀA_s(S))⁻¹).
  - This is a causal replay of the 24 h-window estimator over the training calendar.
  - It penalises training windows in which the network is blind.
- **J_grow_rob(S)** = mean_{d∈D} tr(Q̄_d (I + M_tr(S) + M_d(S))⁻¹), where:
  - M_d is the information in the 24 h block of design direction d;
  - Q̄_d is the mean field weight over that block.

  The scenario modelled is "after training, the wind blows from d for a day".
- **J_w24_rob(S)** = mean_{d∈D} tr(Q̄_d (I + M_d(S))⁻¹). The 24 h window then holds only that day.

### 3.1 Design wind prior (robust variants)

D = 8 compass directions, every 45°. The 24 h steady blocks run at 3 m/s (the training-regime speed) without jitter, in
the order N, NE, …, NW. This builds a 192 h design calendar per source, with the scenario's model and spin-up.

The prior is **uniform over directions**. It does not depend on the test regimes. The test's steady southerly (4 m/s,
±10° jitter) is one of eight directions, at a different speed.

**Honest framing.** The robust variants exist *because* E07 found a failure under an unseen direction. They answer a
question motivated post hoc. Their prior is generic, and is not tuned to the southerly.

**Cost.** One extra unit-source solve per source, over 192 h + spin-up: about 0.16 of a 64² protocol-calendar solve.
The block's first hours include the decay of the previous block's plume. This is kept, since it is realistic.

### 3.2 Optimisation

- Greedy forward selection with `placement.greedy`, then `placement.swap_refine`, as dopt in E07. Values are negated so
  that larger is better.
- Per-site information is precomputed as A_iᵀA_i per hour (and per window, by cumulative sums).
- Each objective evaluation solves at most T (K × K) systems.
- Correctness tests:
  - greedy at k = 1 equals the brute-force best site;
  - greedy+swap is compared with brute force on a tiny case;
  - the objective equals an explicit Kalman covariance recursion (`filter.update`, data-free);
  - designs are unchanged when all non-training information is NaN (as in E07's tests).

## 4. Estimators (gate-09: both memories)

Both use prior N(0, a_ref² I), white noise σ, known winds and known source locations. They read observations ≤ t only.
1. **map24**: the E07 frozen estimator (`cross_objective.estimate_series`, 24 h window).
2. **map_grow**: the growing-window MAP over all hours 0..t (≡ kf_aug with Q = 0, the gate-09 baseline, N(0, I) prior).

Forecast: ĉ(t+h) = c_bg + Σ_k â_k(t) F_k(t+h). This is pre-analysis: â(t) uses readings ≤ t, and the forecast target is
hour t+h. The design and the estimator use the same prior and the same σ for every strategy.

## 5. Endpoints

Issue times are E07's: test start + 6 h … test end − 1. For horizon h, only issue times with t + h ≤ test end are used.

- **Primary (protocol §10.1): held-out pre-assimilation forecast RMSE at h = 1**, over all cells and issue times.
- **Secondary (exploratory):**
  1. forecast RMSE at h = 6 and h = 24;
  2. analysis RMSE (field from â(t) at t);
  3. source RMSE / a_ref.
- **Descriptive:** each design's objective values on the training information, and the same quantities on the held-out
  test winds (the realised forecast variance, which the design never sees). Their rank agreement shows whether the
  training-only objective predicts the held-out outcome. E07 found that training D mispredicted it.

Noise: the white arm only, R = 5 paired draws per scenario (E07 noise streams). Every design reads the same noise at a
shared site and hour. AR(1) and per-site arms are dropped for cost and focus. This deliberately narrows E07.

## 6. Hypotheses (declared before any E11 result)

- **H1 (memory-matched forecast design ≥ per-memory winner).**
  - Under map24, `fc_w24_rob` is non-inferior to `source_near` and better than `dopt` at h = 1.
  - Under map_grow, `fc_grow` (and `fc_grow_rob`) is non-inferior to `dopt` at h = 1.
  - Non-inferiority margin: 5 % of the comparator's RMSE.
- **H2 (robustness helps short memory).** Under map24, `fc_w24_rob` beats `fc_w24`, because training windows lack the
  unseen directions. Under map_grow, the robust/non-robust difference is small (|median| < 5 %).
- **H3 (memory mismatch).** A design matched to one memory loses relative to the matched design when scored under the
  other memory (cross table; descriptive).
- **H4 (horizons).** Strategy orderings at h = 6 and h = 24 agree with h = 1 (expected from §3 with static sources).
- **H5 (sanity).** Every fc_* design beats the mean of 10 random networks.

## 7. Statistics

- **Unit:** a scenario (wind history, K, seed).
- Per scenario, design and estimator, the RMSE pools all valid issue times and cells and averages the R draws. Hourly
  samples are never treated as independent.
- **Contrast:** the paired relative difference ρ = (RMSE_comparator − RMSE_design) / RMSE_comparator, so ρ > 0 favours
  the design. random = mean RMSE of its 10 networks.
- **Per cell (estimator, k, h):**
  - the median of ρ over scenarios;
  - a distribution-free order-statistic 95 % CI for the median (`cross_objective.df_median_ci`, gate-07 issue 5);
  - wins/losses and a two-sided sign test.
- **Verdicts:**
  - non-inferior if the CI lower bound is > −0.05;
  - better if it is > 0;
  - worse if the CI upper bound is < 0;
  - otherwise unresolved.
- **Pilot:** n = 12 scenarios per (k, estimator), so every pilot verdict is exploratory. There is no multiplicity
  correction. The pooled-over-k summary is descriptive (the k-cells share scenarios).
- **64² (§9):** H1's two contrasts (map24 vs source_near; map_grow vs dopt, h = 1, k ≥ 8) form one Holm family.
  **[LEAD/USER REVIEW]** whether that is confirmatory or remains exploratory under the approved protocol. E07's single
  confirmatory test is unaffected either way.

## 8. Local pilot (32², exploratory)

- **World:** the E07 pilot world (upwind, dt 300 s, spin-up 12 h, split 1–4 / 5 / 6–11 days, 144 h test with all three
  held-out regimes).
- **Scenarios:** winds veering and mixed × K ∈ {2, 4} × seeds {0, 1, 2} = 12 scenarios.
- **Factors:** k ∈ {2, 4, 8}; R = 5; workers ≤ 5; OMP_NUM_THREADS = 1.
- **Outputs:** `reports/quarto/tables/e11-local-*` and one figure (`figures/e11-local-forecast-design.png`). The run
  record and journal belong to the lead.

## 9. 64² plan (stage 2 on cached E07/E08 fields)

- **Inputs:** the cached E07 64² sweep fields (job 204894, dd2c51c: truth and unit responses, veering and mixed,
  K 2 and 4, seeds 0–9, 40 scenarios), loaded with `e07_cross_objective.load_scenario`.
- **Non-robust designs** (fc_grow, fc_w24, plus the E07 baselines) and both estimators need **no new solves**.
- **Robust variants** need one design-prior solve per distinct source. Source layouts depend only on (K, seed): 20
  layouts, 60 sources. Each solve covers 192 h + 48 h spin-up of a 1488 h protocol calendar, about 0.16 of a field.
  At E07's measured ≈ 2.5 CPU-h per 64² field, that is **≈ 25 CPU-h** (≈ 25 min on 64 CPUs).
- **Stage-2 cost:** dominated by loading fields and by the J_w24 greedy over ≈ 700 training windows. Estimate ≤ 1 h on
  64 CPUs (to be measured in the pilot).
- **Optional:** the E08 cached fields (K 8) would add a K = 8 cell.

## 10. Limits

- Perfect model: known winds, known locations, constant strengths. The forecast variance is the whole error only up to
  VanLeer's superposition error at 64².
- The robust prior covers 8 steady directions at one speed. It does not cover stagnation or rotation.
- The growing window assumes 60 days of exactly constant sources. This is the gate-09 corner case, and E10 relaxes it.
- Pilot scale and n = 12 make every pilot statement indicative only.

## 11. Deviations

These were added in code before the pilot ran, and were not in the committed note:

1. **h = 1 RMSE per held-out regime** (`rmse_h1_<regime>`; issue times whose target hour falls in that regime's 48 h
   block). This is a diagnostic for H2 and the E07 southerly mechanism. It is secondary and descriptive.
2. **The §5 "objective on test winds" descriptive** is implemented as `pred_rmse_h{1,6,24}`: the data-free expected RMSE
   (§2) of each network under each estimator on the realised calendar, against the realised RMSE
   (`e11-local-predcheck.csv`).
3. **The summary also pools k = 2/4/8 (`k = all`, n = 36).** These pairs share scenarios, so the pooled summary is
   descriptive only.

### Lead decisions (2026-11-13, before any 64² E11 computation)
4. **The 64² Holm family is exploratory, not confirmatory.** The wind-robust objectives were motivated by E07's
   southerly finding (data-informed), and the gate-09 memory dependence was found after the protocol was written.
5. **The robust design-prior solves are approved** (≈ 25 CPU-h). They run as one job after the E08 sweep (one-job rule),
   as stage 2 on the cached E07 fields plus those extra solves.
6. **No memory-averaging objective is added now.** If tried later, it is labelled post hoc.

### 64² addenda A1–A5 (gate-10/11 conditions; declared before any 64² E11 computation)
These are revisions motivated by the 32² pilot and the gate-10/11 audit (`docs/research/gates/gate-10-11.md`), i.e. by
observed results. They are recorded here before the 64² run so that the 64² analysis is fixed in advance. The 64² study
stays exploratory (decision 4). Code: `scripts/e11_forecast_design.py` (v2), `forecast_design.py`, tests in
`tests/test_forecast_design.py`; 32² re-run through the v2 path: `reports/quarto/tables/e11-local-v2-*`.

- **A1. Design prior from independent per-direction blocks; co-reported out-of-sample prior.** §3.1's 8 × 24 h calendar
  (one continuous solve per source) is replaced by 8 independent 24 h steady blocks per source, each after its own
  steady spin-up (`FD.prior_block`), cached per source × direction by content hash. Any subset prior is then a
  concatenation (`FD.prior_fields`); the robust objectives average per block, so a subset prior's objective is the mean of
  the single-direction objectives (tested). Two priors are declared (`FD.PRIORS`):
  - `all8` (all 8 directions): the designs keep their names `fc_grow_rob`, `fc_w24_rob`;
  - `noSN` (drop 0° = N and 180° = S, the test's own directions): `fc_grow_rob_noSN`, `fc_w24_rob_noSN`, the
    **out-of-sample variant**, reported beside `all8` in every table. A claim about wind-robust design is made only if it
    holds for `noSN`; an `all8`-only advantage is reported as partly in-sample.

  Block spin-up: **12 h** at both scales (`SCALE[*]["rob_spinup"]`). Checked at 64² before any 64² E11 run
  (`scripts/e11_spinup_check.py`, `e11-local-v2-spinup.csv`): one source (K = 4, seed 0, source 0), all 8 directions, one
  72 h steady solve each (the block after s h of spin-up is hours [s, s + 24) of that solve). Against the 48 h block, the
  12 h block differs by ≤ 4e-16 (relative L2, full field and candidate sites); even no spin-up differs by ≤ 3e-4. The 10 km
  domain is crossed in < 1 h at 3 m/s, so a steady plume is reached within hours. Cost of the check: 0.80 CPU-h locally.
  (The 32² gate check used the 12 h config spin-up and gave the same sites as the continuous calendar in 100 % of cases.)
- **A2. Cluster-level statistics.** The veering and mixed scenarios of one (K, seed) share sources, strengths, noise
  streams and test winds. The reported unit is the **cluster (K, seed)**: per cell (estimator, k, metric, contrast) the
  ρ of a cluster is the mean of its scenarios' ρ, and the median, distribution-free 95 % CI, sign test and verdict
  (§7 rules) are computed over clusters (`FD.cluster_summary`; columns `cl_*`). Scenario-level numbers stay in the table
  as descriptive. 64²: 20 clusters (32²: 6, smallest two-sided sign p 0.031). **Exploratory Holm family:** h = 1, both
  estimators, every k, the six forecast designs (`fc_grow`, `fc_w24`, `fc_grow_rob`, `fc_w24_rob`, `fc_grow_rob_noSN`,
  `fc_w24_rob_noSN`) vs `dopt` and vs `source_near`, on the cluster sign-test p (`cl_holm_p`). This replaces §7's
  two-contrast family and is exploratory per decision 4.
- **A3. Predicted-RMSE check uses the declared strength prior.** Beside `pred_rmse_h*` (N(0, I), the objective's
  assumption), `pred_m2_rmse_h*` uses the declared prior's second moment E[aaᵀ] = s²I + ā²11ᵀ of LogUniform(0.5, 5)
  (ā = 1.954, s² = 1.555): expected MSE tr(QΣ(E[aaᵀ] + M)Σ). `pred_true_rmse_h*` uses the true a (diagnostic only; truth).
  The design objectives are **unchanged** (N(0, I), as preregistered). At 64² the check also absorbs VanLeer's ≈ 1 %
  superposition error (gate-09); that is recorded, not read as a prior effect.
- **A4. Multi-start swap refinement.** Every forecast design is 1-swap-refined from three starts (greedy, the `dopt`
  set, the `source_near` set, same k) and the best objective value is kept (ties keep the earlier start, greedy first).
  The rows record the winning start, the greedy + swap sites and value (`greedy_sites`, `greedy_refined_value`); the
  `*-multistart.csv` table counts winners. The result is still a local optimum; "local optimum of the objective" applies
  to every design comparison.
- **A5. Per-k reporting only.** No pooled-k row and no pooled verdict (supersedes deviation 3).
- **Unchanged:** worlds, scenarios, k grid (4, 8, 16 at 64²), estimators, scorer, R = 5, the four §3 objectives, §7's
  verdict thresholds and margin.
- **Cost (A1):** 60 sources × 8 directions × (12 + 24) h = 17 280 model hours (vs 60 × 240 = 14 400 for the §9
  calendar). At E07's cluster rate (≈ 11.3 s per 64² VanLeer hour, job 204894) that is ≈ 54 CPU-h, ≈ 55 min on 64 CPUs
  (480 blocks, 8 rounds); at the local rate measured in the spin-up check (≈ 5.0 s/h) ≈ 24 CPU-h. This supersedes §9's
  ≈ 25 CPU-h estimate. Stage 2 (6 forecast designs × 3 starts, 2 priors) is not yet measured at 64²; at 32² it took
  8–10 s per scenario.
- **32² re-run through the v2 path** (`e11-local-v2-*`, not a 64² result): rows whose sites are unchanged reproduce
  the committed v1 pilot to ≤ 2.2e-16 (all baselines; `e11-local-v2-reproduce.csv`). The greedy + swap sites equal v1's in
  286/288 forecast-design rows (2 `fc_grow_rob` rows move with independent blocks); the multi-start then changes many
  forecast designs (`fc_w24_rob` at k = 8: greedy start wins 0/12). Numbers: `e11-local-v2-contrasts.csv`.

### Lead sign-off (2026-11-13, before any 64² E11 computation)
- Addenda A1–A5 are accepted, including A2's exploratory Holm family of 24 contrasts (cluster level). Verdicts are per k,
  from cluster-level statistics only.
- The 64² run is queued to start after the E08 job ends (one-job rule), from snapshot 927a7af.

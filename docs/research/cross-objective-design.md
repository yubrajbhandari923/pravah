# E07 cross-objective placement ablation: design note (pre-pilot)

Status: written 2026-11-12 on branch `experiment/e07-cross-objective`, **before any E07 computation**. Committed alone
(message "E07 design note (pre-pilot)"); every later change is listed in §12 (Deviations) with its reason. Items marked
**[LEAD/USER REVIEW]** are choices the protocol does not settle. Synthetic 2-D passive tracer only; nothing here concerns
real PM2.5 or real exposure.

Inputs: plan-02 `e07-cross-objective`; `preregistered-protocol.md` (approved baseline, A1: a_ref = 1 g/s); gate-06
conditions 1–3 (`state/experiments/e06.md`); gate-05 recommendations (prior-referenced ridge, medians, constant-wind null).

## 1. Question and endpoints

**Plan question.** Which placement objective is best for which held-out inference task? Hypothesis (plan): no single
network minimises source error, field error and forecast error at once; the mismatch can be large at small budgets.

**Primary endpoint (protocol §10.1, unchanged).** Held-out, pre-assimilation, one-step-ahead field-forecast RMSE
(µg m⁻³) over all grid cells and the declared test issue times. The protocol primary fits E07: it is the "forecast" task
of the plan. In E07 the forecast comes from a *static* source estimate with known wind and known source locations
(§5), so it is a footprint-weighted source error on held-out winds, **not** an operational filter (that is E09).

**Primary comparison (protocol §10.1/§12).** Per scenario, relative reduction ρ_d = (RMSE_random − RMSE_d)/RMSE_random,
where RMSE_random is the mean over the 10 random networks. **[LEAD/USER REVIEW]** The protocol tests "an optimised
network" against random; E07 has several. I designate **`dopt` (greedy D-optimal + 1-swap, §3)** as the single
confirmatory optimised network for the 64² sweep. Every other strategy-vs-random contrast is secondary (§8).
The 32² pilot is exploratory and makes no confirmatory claim.

**Secondary endpoints (protocol §10.2 numbering, all exploratory):**
1. Source RMSE: sqrt(mean over issue times and sources of (â_k − a_k)²) / a_ref.
2. Post-analysis field RMSE: field from â(t) at the issue time t itself, vs truth at t.
3. Calibration (coverage/CRPS): **not computed in E07** (deferred to E09, where a filter issues predictive distributions).
4. Population-weighted forecast RMSE with the fixed synthetic density of §5.

**E07-specific secondary (the plan's question):** the cross-objective regret matrix and rank concordance (§8).
**Descriptive (protocol §9, never ranked as endpoints):** design-time criteria on training G (FIM_logdet = Bayesian D,
A, E, field-variance proxy, coherence), FIM_logdet over the test hours, practical rank of each 24 h estimator window,
fraction of issue times with window practical rank < K. Compute: wall and CPU core-hours, peak RSS.

## 2. Design matrix

| factor | 32² local pilot | 64² HPC sweep |
|---|---|---|
| grid / spin-up | 32², 12 h | 64², 48 h (protocol) |
| calendar (days) | train 1–4, validation 5, test 6–11 (144 h) | train 1–30, validation 31–42, test 43–60 (protocol §7) |
| sources K | 2, 4 | 2, 4, 8 |
| sensors k | 2, 4, 8 | 2, 4, 8, 16, 32 |
| training wind history | 3 (below) | 3 (same) |
| seeds (replicate) | 0, 1, 2 | 0–19 |
| noise arms | 3 (below) | 3 (same) |
| noise draws per arm | R = 5 | R = 5 |
| random networks | 10 per scenario | 10 per scenario |

A **scenario** = (training wind history, K, seed); pilot 3 × 2 × 3 = 18, sweep 3 × 3 × 20 = 180. Each scenario runs
every k, strategy, noise arm and draw, so all comparisons inside a scenario are paired.

**Training wind histories** (train + validation segments; 48 h blocks cycling through the list, `benchmarks.make_winds`):
- `constant` = [`constant_westerly`]: exactly constant 3 m/s from 270°, the gate-05 null (designs see one direction);
- `veering` = [`veering`];
- `mixed` = [`steady_westerly`, `veering`, `weak_variable`] (protocol §8 training regimes).

**Test winds are always the protocol's held-out regimes** (`steady_southerly`, `stagnation_then_front`,
`diurnal_rotation`, 48 h blocks). The pilot test segment is 144 h so that all three appear once. Designs never see them.

**Noise arms** (protocol §5): `white` (primary; scalar σ = 0.05 × median training truth over candidate sites),
`ar1` (ρ = 0.8 per hour, same scalar σ), `persite` (white, σ_i = 0.05 × median training truth at site i). Designs that use
σ are rebuilt for `persite` with the per-site vector; in `white`/`ar1` they use the scalar. Noise for draw r is
`observations.noise_field(SeedSequence([noise_seed, r]), all sites, all hours, σ, kind)`, drawn once for all candidate
sites and hours, so every design reads the same noise at a shared site and hour (paired draws).

**Seed streams** (protocol §6): truth 1000+s, noise 2000+s, placement 3000+s (random networks j = 0..9 use
`default_rng([placement, j])`; the POD prior ensemble uses `default_rng([placement, 99])`), bootstrap 4000.

## 3. Strategies and what each may know

Every strategy gets the **same** candidate set (`Scenario.admissible_sites()`: interior lattice, > 1 footprint σ from
every true source; this mask uses true locations and is shared by all designs, a documented leak to the "blind" ones),
the same k and the hard 1250 m spacing rule (`placement.Problem.feasible`). No design receives true strengths, any noise
draw, any validation/test hour, or any truth field. G3_train = per-unit-source sensitivities at all sites over the
**training hours only**, computed from the known source locations and the training winds.

| strategy | knows | rule |
|---|---|---|
| `random` (10 nets) | nothing | seeded permutation, first feasible k |
| `uniform` | nothing | `placement.uniform_design` |
| `coverage` | nothing | farthest-point sampling (`placement.coverage_design`) |
| `source_near` | source locations | round robin nearest feasible site; **locations-only control** (gate-06 F2) |
| `hotspot` | locations + strength prior | top-k feasible by prior-mean training anomaly Σ_h G3_train[i,h,:]·E[a] |
| `dopt` | locations + Gaussian prior (sd a_ref) + σ | greedy Bayesian D (`placement.dopt`) then 1-swap |
| `aopt` | same | greedy Bayesian A then 1-swap (gate-06 F5: greedy A alone is weak) |
| `fieldvar` | same | greedy minimisation of the posterior field-variance proxy tr(M_S⁻¹ Q), Q = a_ref² Σ_{h∈train, i∈sites} G_{ih}ᵀG_{ih}/(n_sites·T) (un-whitened a_ref-scaled responses at all lattice sites as the field; M_S = I + A_SᵀA_S as for D), then 1-swap |
| `diversity` | locations + σ | `placement.diversity_design` (footprint coherence) |
| `podqr` | locations + strength prior + σ | POD of a **prior-ensemble archive** (below), QR pivots (`placement.qr_select`) |

E-optimality is reported, not optimised (gate-06). The unknown-source-location branch is out of scope (protocol: known
locations).

**Strength prior used at design time.** Two priors, both declared: the estimator/D/A/fieldvar prior N(0, a_ref² I)
(protocol estimator prior, as E05/E06), and for `hotspot`/`podqr` the benchmark's generating distribution
LogUniform(0.5, 5)·a_ref (a population-level fact, not the realised strengths). E[a] = a_ref·4.5/ln 10 ≈ 1.954 a_ref.

**POD archive (gate-06 condition 1).** M = 32 strength vectors a^(m) drawn from the LogUniform prior; snapshot columns
G3_train[:, h, :] a^(m) for every training hour h and member m (n_sites × T·M), centred by the column mean. No noise-free
truth enters. **Mode-count rule (gate-06 condition 3), fixed now:** mode j is "above noise" if
max_{i∈cand} s_j |U_ij| √(T/(T·M)) / σ_i ≥ 1, i.e. the per-reading rms contribution of mode j at its best site,
accumulated over the T training hours (the same √T accumulation D-opt uses for T independent readings); r* = the length of
the **leading run** of above-noise modes. QR uses r = min(k, r*) modes and picks r sites. If k > r*, `podqr` is
**undefined** for that k (recorded, scored as a failure, §8). No fallback filling.

**1-swap refinement (gate-06 condition 3, F5).** For `dopt`, `aopt`, `fieldvar`: starting from the greedy set, repeatedly
apply the best single exchange (remove one site, add one feasible candidate) that improves the criterion by more than
1e-9 (absolute), up to 50 passes. The greedy value, refined value and number of swaps are recorded (a lower bound on the
greedy optimality gap under binding spacing).

**Exhaustive check where spacing binds.** On a contiguous 5 × 5 block of the real lattice (lattice indices 5–9 in both
directions, admissible sites only; 625 m spacing < 1250 m, so spacing binds), for k = 2, 3: exhaustive optimum of D and
of the fieldvar proxy vs greedy and greedy+swap restricted to the same block. Reported as a count of exact cases and the
gap distribution.

## 4. Data access and split integrity

- Design inputs: training-hour G3 (all sites), σ (arm), a_ref, source locations, candidate set, the strength prior and
  the placement seed. Implemented as a `DesignInput` object that holds nothing else; tests check that designs are
  unchanged when true strengths, all post-training fields and all noise are replaced by NaN.
- Estimator inputs at issue time t: noisy readings of the chosen sites in hours t−23..t (validation and test hours are
  past observations at that point), their G rows, σ, a_ref, c_bg. Never readings after t; never truth.
- Truth is used only by the scorer. Test-period values are computed once per frozen configuration (protocol §7, §11).

## 5. Fixed evaluation

- **Estimator (frozen).** Primary: MAP with prior N(0, a_ref² I) and noise σ_i, assuming independent noise in every arm
  (so `ar1` measures robustness to a misspecified noise model): â(t) = argmin Σ ((y − c_bg − G a)/σ_i)² + Σ (a_k/a_ref)²
  over the 24 h window ending at t. Secondary: the same objective with a ≥ 0 (NNLS on the stacked whitened system).
  Window 24 h is a protocol window level; it is fixed for E07 **[LEAD/USER REVIEW]**.
- **Forecast.** ĉ(t+1) = c_bg + Σ_k â_k(t) F_k(t+1), F_k = full-grid response to unit source k under the known winds
  (same solver, spin-up and loop as the truth). Truth c(t+1) = `Scenario.truth.field`.
- **Issue times.** Every test hour t with t ≥ test start + 6 h (warm-up excluded) and t+1 inside the test segment: pilot
  137 issue times; 64² 425 (protocol says 426 because it lets the last forecast fall one hour past the test segment; I keep
  every forecast inside the test segment).
- **Population density** (secondary 4): w ∝ Σ_{j=1..3} exp(−|x − c_j|²/(2·1500²)), c_j ~ U([2, 8] km)² from
  `default_rng(6000)`, normalised; fixed across scenarios.
- **Baselines (design-free, reported):** background only (â = 0), climatology (training-period mean truth field;
  an oracle reference, not a competitor), and the random networks.

## 6. Failures and non-identifiable cases (never dropped)

- A design that cannot place k feasible sites, or `podqr` with k > r*, is **undefined**: one row with its status. In
  every paired statistic it counts as the worst outcome (ρ = −∞, regret = +∞, a loss in sign tests); medians and
  bootstrap CIs are order statistics, so they stay defined. Counts of undefined cases are reported next to every median.
- An estimator exception (NNLS) is recorded the same way.
- A scenario where practical rank over the whole test segment is < K for every design at the largest k is flagged
  non-identifiable (protocol §12 stop rule 3); its forecast scores stay in, source-RMSE statements exclude it explicitly.

## 7. Statistics

- **Unit:** scenario (wind history, K, seed). Within a scenario a metric is the mean over the R = 5 noise draws (and over
  the 10 networks for `random`). Noise arms and estimators are analysed separately; `white` + MAP is primary.
- **Per cell (K, k[, wind])**: median of ρ_d over scenarios, 95 % percentile bootstrap CI over scenarios (10 000
  resamples, seed 4000), exact two-sided sign test (|ρ| < 1e-12 counted as tie and excluded, ties reported).
  Pilot cells pool the 3 winds × 3 seeds = 9 scenarios per (K, k).
- **Protocol hypothesis (64² only):** relative reduction of `dopt` vs random ≥ 10 % for k ≥ 8; supported if the CI lies
  above 10 % in ≥ 70 % of the k ≥ 8 cells, contradicted if it lies below in ≥ 70 %, otherwise unresolved (protocol §12).
- **Multiplicity:** one confirmatory contrast (`dopt` vs random, primary endpoint). The other 8 deterministic strategies
  vs random: Holm-adjusted sign-test p within each (K, k) cell, labelled secondary. Secondary endpoints and the regret
  matrix: unadjusted, exploratory.
- **design_regret.** For metric m in scenario s: regret_d,m = (m_d − min_d' m_d') / min_d' m_d', over all strategies
  with `random` as its network mean. **Cross-objective matrix:** for selector metric m_sel (each endpoint and each
  design-time criterion), take b = argmin (or argmax for D/A) over strategies in that scenario; entry [m_sel, m_eval]
  = median over scenarios of regret_b,m_eval. Diagonal = 0 for endpoints. Also the median Spearman correlation, across
  strategies within a scenario, of design-time D (training) with each held-out endpoint.
- **Rank table:** mean rank of each strategy per metric (undefined ranks last).

## 8. Outputs

- `reports/quarto/tables/e07-local-pilot.csv`: one row per (scenario, arm, estimator, k, strategy[, network]) with
  status, sites, all metrics (draw means) and timings; `e07-local-summary.csv` (cell statistics), `e07-local-regret.csv`,
  `e07-local-swapcheck.csv`, `e07-local-compute.csv`, `e07-local-meta.json` (git rev, command, config, environment).
- `reports/quarto/figures/e07-local-crossobjective.png`: regret heatmap (selector × evaluation metric) and ρ vs k.

## 9. Compute plan

- Workers: `ProcessPoolExecutor`, one task per scenario, `OMP_NUM_THREADS=1`, `if __name__ == "__main__"` guard.
- Per scenario: 1 truth solve (`generate_scenario`) + K unit-source solves over spin-up + full calendar (full-grid fields
  kept in memory for forecasting), then designs and evaluation. Record wall and CPU time per phase and peak RSS.
- **Pilot (32²)**: measured before writing this note: one FiPy hour costs ≈ 0.14 s at 32² and ≈ 0.23 s at 64² (1 core,
  upwind). Pilot ≈ (K+1) × 276 h × 0.14 s ≈ 2–3 min of solves per scenario; 18 scenarios on ≤ 16 local cores.
- **64² sweep (later; waits for B3).** 180 scenarios; solves ≈ (K+1) × 1488 h × 0.23 s ≈ 17 / 29 / 52 min for
  K = 2 / 4 / 8, ≈ 100 core-hours of solves in total; memory per worker ≈ 0.5–1 GB (K = 8 fields:
  8 × 1488 × 4096 × 8 B ≈ 0.4 GB). One SLURM job at 64 CPUs / 128 GB: 64 workers × ≤ 1.5 GB, wall ≈ 3–5 h, far under
  the 2-day limit. A per-scenario output file makes the job restartable. Final sizing uses the pilot's measured
  design/evaluation cost (k = 16/32 is not in the pilot; its greedy/swap cost will be extrapolated and checked on one
  scenario first). **Blocked on B3**: the Van Leer scheme under test in `forward.py` may change cost and fields; no 64² run
  before it reports.

## 10. Known limitations (declared now)

Perfect model (estimator uses the truth's solver and winds); constant sources; known locations; the shared exclusion
mask uses true locations; independent-noise assumption in design and estimator; 24 h window only; pilot test segment is
144 h (not 432 h) and the pilot calendar is shorter than the protocol's; σ is derived from training truth (gate-06).
Pilot n = 9 scenarios per (K, k) cell, so CIs are wide and any ordering is exploratory.

## 11. Pilot run

`OMP_NUM_THREADS=1 uv run python scripts/e07_cross_objective.py --pilot` (exactly the pilot column of §2). No design
change after seeing results; anything changed goes to §12.

## 12. Deviations

1. **Pipeline smoke before the pilot (procedural, no design change).** After this note was committed (3e7d4d2), one
   pilot scenario (veering, K = 4, seed 0) was run once to check the code end to end, before the full pilot
   (`--only veering:4:0`, output kept outside the repo). I saw its numbers. Nothing in §1–§11 was changed afterwards; the
   pilot ran from commit e3f7de0 with the same code, and that scenario is deterministic, so its pilot rows equal the smoke rows.
2. **Implementation details the note left open (fixed in e3f7de0, before the pilot):** the swap-check block is lattice
   indices lattice_n/2 − 3 … lattice_n/2 + 1 (= 5–9 for the 16 × 16 lattice, as stated); the design-time descriptor
   `D_test` is log det(I + A_Sᵀ A_S) over **all** test hours (evaluation only); `prank_test` the matching practical
   rank. The regret matrix's design-criterion selectors (D/A/fieldvar on training, D on test) choose among the nine
   deterministic strategies only (`random` has no single network); endpoint selectors include `random` (network mean).
3. **After the pilot (no effect on any pilot number):** the recorded command in `e07-local-meta.json` showed `--pilot`
   twice (string bug, corrected in the file and the driver); the sweep now submits scenarios largest-K first so the
   longest tasks start early. Pilot completed 18/18 scenarios, commit e3f7de0, 539 s wall on 12 workers.
4. **Observed design properties the note did not anticipate (reported, not changed):** `podqr` is undefined in 36 of
   the 162 pilot (scenario, noise arm, k) cases (r* = 2–7 leading modes, so k = 4/8 often exceeds it); `aopt` and `fieldvar` often pick
   the same sites as `dopt` after swap. The 64² sweep keeps the rule as written; at k = 16/32 `podqr` will mostly be
   undefined and is scored as a failure.
5. **Lead decisions after the pilot (2026-11-12, recorded before any 64² computation):**
   - `dopt` stays the single confirmatory optimised network, as committed in 3e7d4d2. Redesignating it after seeing the
     pilot would be outcome-driven. `source_near` remains the locations-only control and is reported prominently.
   - The sweep uses amendment A2: split VanLeer advection at dt = 30 s (`state/decisions/2026-11-12-advection-scheme.md`).
     The pilot used upwind at dt = 300 s on 32², so pilot and sweep fields differ in effective diffusivity.
   - No wind-robust design arm is added. Its sensitivities would need extra simulations under alternative winds, and
     robust placement is plan-3 E15's question. The pilot's train-vs-test wind-shift finding is carried to E15 as a
     hypothesis.
   - The 24 h estimator window and the protocol calendar are kept for the sweep. The pilot's shortened 4/1/6-day split
     was a pilot-only cost choice.
6. **User-requested scope reduction for compute (2026-11-12), decided before any 64² computation.** The 64² sweep is
   cut so it fits one 64-CPU SLURM job (≤ 2 days): train winds {veering, mixed} (the constant null control is covered by
   the pilot); K ∈ {2, 4} (K = 8 is E08's question and over half the cost); seeds 0–9; k ∈ {4, 8, 16}; strategies
   random, uniform, coverage, hotspot, source_near, dopt, diversity, podqr (aopt and fieldvar dropped for the sweep
   only: in the pilot they mostly picked the same sites as dopt after swap); MAP estimator only (NNLS was within 0.05 of
   MAP everywhere in the pilot). All three noise arms are kept (no simulation cost). The pilot configuration and code
   path are unchanged and stay reproducible. The confirmatory rule (dopt vs random, ≥ 10 % for k ≥ 8, CI above 10 % in
   ≥ 70 % of k ≥ 8 cells) is unchanged and now applies to the 2 winds × 2 K × k ∈ {8, 16} cells. Consequences: the Holm
   family of the forecast-RMSE sign tests per (arm, K, k) has 7 strategies instead of 9; the exhaustive greedy/swap
   check (dopt and fieldvar criteria on the 5 × 5 block, k = 2, 3) is a solver check and is kept. Projected cost:
   40 scenarios, 160 simulations of 1536 h (48 h spin-up + 1488 h), ≈ 160 × 1536 h × 7.8 s ≈ 530 core-hours unloaded
   (A2 serial rate), i.e. 3 rounds on 64 workers ≈ 10 h wall before contention; the preflight measures the loaded rate.
   Engineering change for this sweep (no effect on any number): solves run one simulation per task with on-disk
   caches, then designs/evaluation per scenario; the full 32² pilot rerun through this path reproduced
   `e07-local-pilot.csv` and `e07-local-swapcheck.csv` exactly (all columns except the wall-clock `design_s`).

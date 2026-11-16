# E08 sensor sufficiency, field complexity and measurement cost: design note (pre-pilot)

Status: written 2026-11-13 on branch `experiment/e08-sufficiency`, **before any E08 computation**. Committed alone
(message "E08 design note (pre-pilot)"); every later change goes to §13 (Deviations) with its reason. Items marked
**[LEAD/USER REVIEW]** are choices the protocol does not settle or that touch its confirmatory claim. Synthetic 2-D
passive tracer only; nothing here concerns real PM2.5, real exposure or deployment guidance.

Inputs: plan-02 `e08-sufficiency`; `preregistered-protocol.md` (approved baseline; A1 a_ref = 1 g/s; A2 VanLeer dt 30 s
at 64²); `state/experiments/e05.md` (v2 findings, gate-05 recommendations); `state/experiments/e07.md` and
`gates/gate-07-cross-objective.md`; `cross-objective-design.md` (E07 machinery reused here).

Before writing this note I looked at no E08 output. I did know the E07 results (dopt beats random; source_near beats
dopt; dopt's per-window rank loss under the held-out steady southerly) and E05 v2. The hypotheses in §9 are derived from
them and say so. One geometry-only check was run (no fields): random sequential placement on the 14 × 14 interior
lattice at 1250 m spacing jams at 34–48 sites (median 41, 200 draws), so k = 32 is placeable for every strategy in
the absence of source exclusions.

## 1. Question

**Plan question.** How do sensor count k, number of sources K, noise level, sampling cadence and observation duration
interact, and where are the breakpoints ("k needed to reach X on the held-out endpoint")?
**Plan hypothesis.** Sensor sufficiency is conditional on source-basis dimension, transport history and noise; it is
not one universal k threshold.

**Sufficiency definition (plan step 1).** For one scenario, design, noise arm and sampling schedule, the normalised
forecast error is nRMSE(k) = forecast RMSE(k) / forecast RMSE of the background-only forecast (â = 0) of the same
scenario and issue times. The **breakpoint** k*(τ) is the smallest tested k such that nRMSE(k') ≤ τ for every tested
k' ≥ k (a curve that dips below τ and rises again does not count until it stays below). If no tested k qualifies,
k*(τ) = ">32" (coded as 64 for order statistics). Thresholds τ ∈ {0.5, 0.3, 0.2, 0.1}; **τ = 0.2 is the headline**
(the forecast leaves 4 % of the plume's mean-square anomaly unexplained). τ values are judgement calls made without
E08 data; the full range is always reported.

## 2. Endpoints

**Primary (protocol §10.1, unchanged).** Held-out, pre-assimilation, one-step-ahead field-forecast RMSE (µg m⁻³) over
all grid cells and the declared test issue times, from a static source estimate with known wind and known source
locations (the E07 forecast, `cross_objective.score`). Issue times as E07: every test hour t ≥ test start + 6 h with
t + 1 inside the test segment (64²: 425; the protocol says 426, see E07 note §5).

**Secondary (protocol §10.2, exploratory):** (1) source RMSE / a_ref; (2) post-analysis field RMSE at the issue time.
Calibration and population weighting are not computed (no predictive distribution; E07 showed popw tracks forecast).

**E08 quantities (exploratory):**
- nRMSE and k*(τ) (§1), overall and per held-out test wind regime (the regime of the 48 h block containing the issue
  hour: `steady_southerly`, `stagnation_then_front`, `diurnal_rotation`).
- Identifiability descriptors (protocol §9; never ranked as endpoints): per issue time, eigenvalues of the window
  information AᵀA (whitened, a_ref-scaled, only the sampled hours of the window). Reported: median over issue times of
  the smallest eigenvalue (plan metric `smallest_singular_value` = its square root), fraction of issue times with
  practical rank (eigenvalues ≥ 1) < K, and structural rank (eigenvalues ≥ 1e-8 × largest; the "noise 0" level).
- **time_to_information:** per issue time, the shortest hourly look-back L ∈ {1, 2, 3, 6, 12, 24, 48, 96, 168} h whose
  information has practical rank K; median over issue times (">168" if never), and the fraction of issue times that
  reach it within 168 h. Computed for hourly sampling at each noise level.
- **Measurement cost:** readings per sensor n, total readings B = k × n, and sensor-hours k × W are recorded on every
  row. No monetary cost model is assumed; marginal gains are reported per doubling of k and per ×4 in readings, and as
  a Pareto frontier of nRMSE against (k, B). Rows (k × n) are reported separately from independent degrees of freedom
  (practical rank).

## 3. Factors and levels

A **scenario** = (K, seed) under the `mixed` training-wind history (protocol §8 training regimes). Every scenario runs
every k, strategy, noise arm, draw and schedule, so all comparisons inside a scenario are paired.

| factor | 32² local pilot | 64² HPC sweep | costs solves? |
|---|---|---|---|
| grid, scheme, dt, spin-up | 32², upwind, 300 s, 12 h | 64², VanLeer, 30 s (A2), 48 h | yes |
| calendar (days) | train 1–4, validation 5–7, test 8–13 | protocol: 1–30 / 31–42 / 43–60 | yes |
| sources K | 2, 4, 8 | 2, 4, 8 | yes |
| seeds | 0–3 | 0–19 (protocol) | yes |
| sensors k | 2, 4, 8, 16, 32 | 2, 4, 8, 16, 32 | no |
| strategy | 4 (§4) + 10 random networks | same | no |
| noise arm | 4 (below) | same | no |
| noise draws R | 5 | 5 | no |
| sampling schedule (W, n) | 7 (below) | same | no |

Only K × seed costs forward solves (one truth + K unit-source solves per scenario); everything else is evaluated on
the cached fields, so the cheap factors are crossed in full (no fractional design is needed).

**Training winds:** `mixed` only. Training winds only affect the designs (dopt's sensitivities); the estimator works on
held-out test winds that depend on the seed alone. The veering arm of E07 and the constant-wind training null are not
repeated. **Transport history during estimation** enters through the held-out test regimes and the window span (below).
Within a `steady_southerly` block the wind is exactly constant (one draw of speed and direction per block), so 24 h
windows inside it are the E05 constant-wind null on held-out data.

**Noise arms** (protocol §5): σ₀ = 0.05 × median training truth over the candidate sites (scalar, the protocol level).
- `w0.2`, `w1`, `w5`: white noise with σ = 0.2σ₀, σ₀, 5σ₀. The draws are common random numbers: readings =
  truth + m·σ₀·z with the same z for all m. **[LEAD/USER REVIEW]** the protocol fixes one σ; the 0.2 and 5 levels are an
  E08 factor (plan: "noise 0/low/moderate") and do not change the protocol primary. "Noise 0" is represented by the
  structural-rank descriptor, since MAP at σ → 0 is not defined by the protocol estimator.
- `ar1`: AR(1), ρ = 0.8 per hour, marginal σ₀ (protocol variant; the estimator still assumes independence, so this arm
  measures robustness to a misspecified noise model, as in E07).
- Noise for draw r = `observations.noise_field(SeedSequence([2000 + s, r]), all sites, all hours, ·, kind)`, drawn once
  for all candidate sites and hours and shared by every design (paired draws), exactly as E07.

**Sampling schedules (duration W, readings per sensor n).** A sensor samples at the calendar hours h with
h mod c = 0, cadence c = W / n (h = 0 is the first post-spin-up hour). The estimator at issue time t uses the readings
at sampled hours in (t − W, t]: exactly n per sensor.

| id | W (h) | n | cadence c (h) | role |
|---|---|---|---|---|
| S1 | 1 | 1 | 1 | snapshot |
| S2 | 6 | 6 | 1 | 6 h |
| S3 | 24 | 6 | 4 | 24 h span, 6 readings |
| **S4** | **24** | **24** | **1** | **reference: protocol/E07 24 h window, hourly** |
| S5 | 168 | 6 | 28 | 7 d span, 6 readings |
| S6 | 168 | 24 | 7 | 7 d span, 24 readings |
| S7 | 168 | 168 | 1 | 7 d, hourly |

This separates span from readings: at n = 6, W = 6 / 24 / 168 (S2, S3, S5); at n = 24, W = 24 / 168 (S4, S6); at
W = 168, n = 6 / 24 / 168 (S5–S7). With the k levels it gives exact matched-budget pairs B = k × n:
(k, n = 24) vs (4k, n = 6) at W = 24 (S4 vs S3) and at W = 168 (S6 vs S5), for k = 2, 4, 8.
Windows reach back into validation hours (past observations at the issue time; allowed by protocol §7). Sites are
fixed in place for the whole calendar; no design is re-optimised during the test period.

**Reference configuration** (the "E07 cell"): `w1`, S4, MAP. Headline sufficiency curves use it.

## 4. Strategies and what each may know

Same candidate set, spacing rule and information separation as E07 (`cross_objective.DesignInput`; the shared exclusion
mask removes sites within one footprint σ of a true source, a documented leak to the blind designs).

| strategy | knows | rule |
|---|---|---|
| `random` (10 nets) | nothing | seeded permutation, first feasible k (E07) |
| `coverage` | nothing | farthest-point sampling (`placement.coverage_design`); **the no-location-knowledge design** |
| `source_near` | source locations | round robin nearest feasible site (E07 locations-only control) |
| `dopt` | locations + prior N(0, a_ref²) + σ | greedy Bayesian D on training hours + 1-swap (E07 confirmatory network) |

`dopt` is rebuilt for each noise level with that level's σ (the design knows its instrument); `ar1` uses the `w1`
design. The other strategies do not depend on σ. hotspot, aopt, fieldvar, diversity and podqr are not run (E07 covers
them; none is needed for the sufficiency question).

## 5. Fixed evaluation

- **Estimator (frozen, E07's):** MAP with prior N(0, a_ref² I), independent noise σ, over the schedule's readings
  (`cross_objective.estimate_series` with a sampling mask: non-sampled hours contribute nothing). Primary only; NNLS is
  not run (E07: within 0.05 of MAP).
- **Forecast and truth** as E07 (unit responses F under known winds; truth is the joint solve).
- **Scoring:** exact algebraic rewrite of `cross_objective.score`: per issue time the squared forecast error is
  âᵀM_tâ − 2âᵀb_t + c_t with M_t = Σ_cells F_t F_tᵀ, b_t = Σ_cells F_t r_t, c_t = Σ_cells r_t², r_t = truth − c_bg.
  A test asserts agreement with `score` to round-off. Per-regime RMSE uses the issue times of that regime.
- **Baselines:** background only (â = 0; the nRMSE denominator, overall and per regime) and climatology (training-mean
  truth; oracle reference).

## 6. Failures and undefined cases (never dropped)

- A design that cannot place k feasible sites is **undefined** (one row with its status). `random` at k is defined only
  if all 10 networks are (E07 rule); the count of undefined networks is reported.
- Undefined counts as nRMSE = +∞ (not sufficient at that k); paired differences involving it are +∞/−∞ order
  statistics; medians and order-statistic CIs stay defined. Counts are reported next to every summary.
- Protocol stop rule 3: a scenario where practical rank over the whole test period is < K for every design at k = 32
  (`w1`) is flagged non-identifiable; source-RMSE statements exclude it explicitly; forecast scores stay in.

## 7. Data access and split integrity

Designs see only training-hour sensitivities, σ, a_ref, source locations, candidates, prior and placement seed (E07
`DesignInput`; the E07 NaN-replacement test applies). The estimator sees readings of the chosen sites at sampled hours
≤ t only. Truth is read only by the scorer and by `readings`. A test checks that changing readings after t (or at
non-sampled hours) leaves the issued estimate unchanged.

## 8. Statistics

- **Unit:** scenario (K, seed); n = 20 per K at 64², 4 per K in the pilot. Within a scenario a metric is the mean over
  R = 5 draws (and over the 10 networks for `random`).
- **Curves:** per (K, strategy, arm, schedule, k): median nRMSE with a **distribution-free CI for the median** (order
  statistics x(j), x(n+1−j), largest j with binomial coverage ≥ 95 %; n = 20 gives x(6), x(15), 95.9 %). If n is too
  small for 95 % (n ≤ 5), the interval is (min, max) and its actual coverage is printed (n = 4: 87.5 %).
- **Breakpoints:** per cell, median k*(τ) with the same order-statistic CI (k* is ordinal, ">32" = 64), plus the
  fraction of scenarios sufficient at each k with a Wilson 95 % interval. A cell's breakpoint is reported both as
  "median k*" and as "first k where the CI of median nRMSE lies entirely ≤ τ" (sufficient with confidence).
- **Paired contrasts:** per scenario difference (or ratio) between two configurations; median, distribution-free CI,
  exact two-sided sign test (|Δ| < 1e-12 ties excluded, counts reported). No percentile bootstrap of a median
  (gate-07 finding 5).
- **Dependence:** K = 2, 4, 8 scenarios with the same seed share test winds and noise seeds but not sources; contrasts
  across K are paired by seed. Pooled-over-k pair counts are descriptive only; tests are per cell or per scenario.
- **Multiplicity:** the five primary tests of H1–H5 (§9) form one Holm family (64² only). Everything else is
  exploratory, unadjusted.

## 9. Preregistered hypotheses (64² sweep; all E08 claims are exploratory under the protocol)

Each has one primary test at the reference configuration unless stated; supported if the predicted direction holds and
the Holm-adjusted p < 0.05; contradicted if the opposite direction is significant; otherwise unresolved.

- **H1 (plan; K dependence).** k*(0.2) is larger at K = 8 than at K = 2. Primary test: `dopt`, paired by seed, sign
  test on k*_K8 − k*_K2. Reported for every strategy and τ. Prediction from E07: source_near's k* is close to the
  smallest k ≥ K; random's is larger and often ">32".
- **H2 (noise; E07 SNR reasoning).** Raising σ from σ₀ to 5σ₀ hurts location-blind designs more than source_near
  (near-source SNR ≈ 50–80 at σ₀, gate-07 finding 10). Primary test: at k = 8, all K pooled by scenario (60 units),
  sign test on log[nRMSE_w5/nRMSE_w1](random) − log[nRMSE_w5/nRMSE_w1](source_near) > 0.
- **H3 (span × transport; E05 constant-wind null + gate-07 mechanism).** At matched readings n = 24, extending the span
  from 24 h (S4) to 168 h (S6) lowers dopt's forecast RMSE in `steady_southerly` issue hours, because the window then
  includes other wind directions (gate-07: dopt's deficient windows are all in the steady southerly), and the gain is
  larger there than in the other two regimes. Primary test: dopt, k = 8, 60 scenarios, sign test on
  [log RMSE_S4 − log RMSE_S6] (southerly) − [same] (other regimes) > 0. Secondary predictions: source_near's span gain
  is small (median |relative change| < 10 %); in `steady_southerly`, the window practical rank is the same for S2 (6 h)
  and S4 (24 h) (constant-wind null: elapsed time without wind change adds no rank).
- **H4 (matched budget; E05 v2).** At equal total readings, more sensors beat more readings per sensor: (4k, n = 6)
  has lower forecast RMSE than (k, n = 24). Primary test: W = 168 (S5 at 4k vs S6 at k), k = 2 → 8, `random`
  (expected network), 60 scenarios, sign test. Reported for W = 24, k = 4 → 16, k = 8 → 32 and every strategy.
- **H5 (replication of gate-07 on fresh scenarios).** source_near has lower forecast RMSE than dopt. Primary test: the
  60 scenarios **not used by E07** (K = 8 seeds 0–19; K = 2, 4 seeds 10–19), k = 8, sign test. The 40 E07 scenarios
  (K = 2, 4, seeds 0–9 under `mixed`) re-enter only as reference and are not new evidence for H5.

E05 also predicts that a 7-day window substitutes for sensors only when the wind changes; this is checked descriptively
by comparing (k, S7) with (2k, S1) per regime.

## 10. Protocol primary cells **[LEAD/USER REVIEW]**

E07's confirmatory test (dopt vs random, ≥ 10 % for N ≥ 8) left K = 8 and N = 32 NOT RUN, and used 10 seeds.
The E08 reference configuration contains exactly those cells (K = 8 × k ∈ {8, 16, 32}; K = 2, 4 × k = 32; 20 seeds;
`mixed` training wind). E08 computes the committed rule for every N ≥ 8 cell (median relative reduction with
distribution-free CI and, for continuity with E07's committed implementation, the percentile bootstrap) and reports it
as **"protocol-cell completion (secondary)"**. Whether these cells join the confirmatory population (which would change
the denominator of E07's verdict) is a lead/user decision. Nothing in E08 alters the protocol's endpoint, thresholds,
splits or E07's verdict.

## 11. Outputs

Pilot: `reports/quarto/tables/e08-local-rows.csv` (one row per scenario × arm × schedule × k × strategy[× network]:
status, sites, n, B, metrics, per-regime RMSE, identifiability descriptors), `e08-local-curves.csv` (median nRMSE and
CIs), `e08-local-breakpoints.csv` (k*(τ)), `e08-local-contrasts.csv` (H1–H5 statistics, exploratory at pilot scale),
`e08-local-compute.csv`, `e08-local-meta.json`; figure `reports/quarto/figures/e08-local-sufficiency.png`
(nRMSE vs k per K and strategy, schedules overlaid; breakpoint panel). Sweep: the same set with prefix `e08-sweep-`.

## 12. Compute plan

- Reuse the E07 two-stage driver (`scripts/e07_cross_objective.py`: stage 1 one solve per task with content-hashed
  on-disk caches; stage 2 one task per scenario). E08 registers its scales and calls the same `scenario_fields` /
  `run_field`, so field caches are interchangeable. `ProcessPoolExecutor`, `OMP_NUM_THREADS=1`, `__main__` guard.
- **Pilot:** 12 scenarios (3 K × 4 seeds), (K + 1) solves each = 68 solves of 324 h (12 h spin-up + 13 d) at 32² upwind (≈ 0.14 s/h) ≈
  1 min each; ≤ 6 workers. Measures stage-2 cost per scenario for extrapolation.
- **64² sweep, one job:** 60 scenarios. The 40 `mixed` scenarios with K = 2, 4, seeds 0–9 are identical to E07's; their
  78 distinct field files exist under `results/e07-sweep-dd2c51c/fields` (all names verified 2026-11-13 by recomputing the
  hashes locally) and are symlinked into the E08 output, not re-solved. New: K = 8 seeds 0–19 (20 × 9 = 180 solves) and
  K = 2, 4 seeds 10–19 (10 × 8 = 80 solves): **≈ 260 solves**. E07's loaded rate (64 concurrent workers) was 13 000–
  17 000 s per 1536 h solve, ≈ 4.5 h. 260 solves = 5 rounds on 64 workers (64 + 64 + 64 + 64 + 4) ≈ 23 h, ≈ 1 200
  core-hours; plus stage 2 (sized from the pilot). Request 64 CPUs, 128 G, 2-00:00:00. Memory: stage 2 at K = 8 holds
  F (8 × 1488 × 4096 × 8 B ≈ 0.4 GB) per worker. Restartable by resubmitting the same command.
- If the projected wall time from the pilot plus the E07 rate exceeds 40 h, the pre-declared reduction is seeds
  10–19 of K = 2, 4 first (reported NOT RUN), then nothing else.

## 13. Deviations

1. **Pipeline smoke before the pilot (procedural, no design change).** After this note (a846753) and the implementation
   (eae033b), one pilot scenario (K = 8, seed 0) was run once (`--only 8:0`) to check the code end to end and time
   stage 2. Only timings were read (stage 2: 8 s designs + 21 s evaluation), no metric. It is deterministic and its cache
   was reused by the pilot, so its pilot rows are the smoke rows.
2. **Implementation details the note left open (fixed in eae033b, before the pilot):** `cross_objective.window_info`
   now refuses a window that starts before hour 0, as `estimate_series` already did (before, a negative index would
   have wrapped silently; no E07 call ever reached it). `time_to_information` skips look-backs longer than the calendar
   before the first issue time; this only affects the 16² unit-test calendar (pilot first issue hour 174 ≥ 168, sweep 1014).
   For `random`, every descriptor is the mean over its 10 networks, including `srank_min` (so it can be fractional).
3. **Pilot ran as written:** commit eae033b, 12/12 scenarios, 68 solves, 1630 s wall on 6 workers; no undefined design
   at any k (k = 32 placeable for every strategy). Upwind superposition is exact (`linearity_max_abs` = 0).
4. **Observed after the pilot, no design change:** H2 as formulated (relative noise penalty, log ratio) went the
   opposite way in all 12 pilot scenarios: source_near's error is noise-limited, so 5× σ raises it by a larger
   factor, while random's error is dominated by rank deficiency and moves less, even though source_near stays far lower
   in absolute nRMSE. The preregistered test is kept unchanged for the sweep; a reversal there is reported as
   "contradicted", not reformulated.
5. **Compute plan refined (no design change):** recomputing the cache names gives 334 distinct sweep fields, of which
   78 are E07's, so **256 new solves**, exactly 4 rounds on 64 workers (≈ 18–19 h at E07's loaded rate), not 5. Stage 2
   in the pilot took ≤ 30 s per scenario on a 312 h calendar; scaled to 1488 h and 4096 cells, ≲ 5 min per scenario,
   one round of 60 scenarios. The §12 reduction rule is not triggered.

### Lead decisions (2026-11-13, before any 64² E08 computation)
6. **§10 protocol cells stay secondary.** E07's confirmatory population remains the user-approved 8 cells
   (`state/decisions/2026-11-13-e07-confirmatory-scope.md`). Adding cells after E07's verdict would change its denominator
   post hoc. E08 reports the K = 8 and k = 32 dopt-vs-random cells as "protocol-cell completion (secondary)".
7. **Approved as declared:** the 0.2× and 5× noise levels, the `mixed` training wind only, and the τ thresholds. All are
   exploratory beyond the protocol's single σ.
8. **Added exploratory arm (declared now, prompted by the E07 follow-ups):** after the sweep, stage 2 is rerun on the cached
   fields with `--exclude-radius 1200`, because in E07 excluding in-footprint sites cut dopt's gain over random from 0.71 to
   0.20. It is reported beside the main results and never replaces them.
9. The user authorised the 64² job ("get started with E08"). It runs as one 64-CPU job per the cluster rules.

### Addendum 2026-11-14 (exploratory stage-2 reruns on the cached fields; no preregistered test changes)
10. **`--exclude-radius M`** (decision 8) is implemented by `cross_objective.exclude_near_sources`, as in E07: all four strategies place only on candidates farther than M of every true source (inclusive boundary excluded); scoring, estimator and schedules are unchanged. Outputs carry the tag `excl1200`.
11. **`--estimator map_full`** (motivated by gate-09: the all-past-data MAP equals the augmented Kalman filter under a perfect model). Only the estimator changes: at every issue hour t the MAP (prior N(0, I), same whitening) uses all sampled readings from hour 0 to t instead of the W-hour window ending at t (`estimate_series(window=None)`). The schedule still decides which hours are read (cadence W/n), so S1, S4 and S7 (all hourly) become identical and the span W no longer enters the estimator; the identifiability descriptors (`prank`, `tti`) keep their nominal-W definition and are not comparable under this variant. Outputs carry the tag `mapfull`; H1-H5 are recomputed by `--report` on these rows as exploratory, never as verdicts, and never replace the reference run.
12. **Erratum to section 9, H5:** "60 scenarios not used by E07" is wrong; K = 8 seeds 0-19 plus K = 2, 4 seeds 10-19 is 40 scenarios (the code uses 40).

### Post hoc bug fix 2026-11-16 (H4/H5 undefined pairs)

H5 (and H4) contrasts were `log(RMSE_a) - log(RMSE_b)` with undefined forecasts coded as inf, so a pair with both designs
undefined gave inf - inf = NaN and a NaN median (only `excl1200` H5 k=32, one scenario). Rule now (`sufficiency.dlog`):
both undefined = tie (0); exactly one undefined = +-inf (win/loss for the defined design); medians and sign tests then run over
all pairs. Effect on the 64² tables: only `e08-sweep-excl1200-contrasts.csv` H5 k=32 fresh changed (median NaN -> 0.355,
CI 0.101-0.774; wins/losses/ties 34/5/1 and p unchanged). No other reported number changed. Post hoc; not part of the preregistered analysis.

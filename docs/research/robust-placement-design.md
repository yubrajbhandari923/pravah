# E15 nominal vs robust (scenario-distribution) sensor placement: design note (pre-pilot)

Status: written 2026-11-13 on branch `experiment/e15-robust-placement`, **before any E15 computation**, and committed
alone. Later changes go to §12 (Deviations) with their reason. Everything in E15 is **exploratory** (plan-03; the robust
objectives were motivated by E07/E11/E12 results). Synthetic 2-D passive tracer, known source locations, constant
strengths; 32² pilot only (heavy 64² compute deferred by the user, 2026-11-13; §10 is the plan for later).

Inputs: plan-03 key `e15-robust-placement`; E07 (`cross-objective-design.md`), E11 (`forecast-design-placement.md`,
addenda A1–A5), E12 (`wind-mismatch-design.md`); gates gate-07, gate-09, gate-10-11.

Lessons carried over (they shape the design):
- The placement ordering depends on the estimator memory (gate-09): report the 24 h window (`map24`) and the
  all-past-data estimator (`map_full`) separately, never pooled.
- `dopt` is blind under wind directions unseen in training (gate-07). E11's wind-robust 24 h design gained, but ~40 % of
  the gain was in-sample because the prior contained the test directions (gate-10/11). **E15's scenario ensembles exclude
  the test-period directions.**
- Under a wrong model wind, `dopt` ≈ `source_near` (E12 H4): robustness to wind error must be measured in absolute RMSE.
- Statistics: cluster = (K, seed), per-k verdicts only (E11 A2, A5).

## 1. Question and hypotheses

Can a placement optimised over a declared distribution of meteorological scenarios — by expected risk, CVaR or worst
case — outperform nominal designs on unseen winds and under wind-model error, and at what cost in nominal accuracy?

- **H1 (tail gain):** some robust design (CVaR or worst case) lowers the held-out tail loss (§6) relative to `dopt` at
  some k, under at least one memory.
- **H2 (nominal cost):** the tail gain costs nominal (oracle-wind, all-test) forecast accuracy relative to the best
  nominal design; no robust criterion is universally best (plan hypothesis: no universal winner).
- **H3 (ensemble breadth):** the broader declared ensemble (`noSN6`) helps on the unseen southerly more than the
  training-only ensemble (`train3`).
- **H4 (memory):** as in gate-09/E11, a design robust for `map24` is not robust for `map_full` and vice versa.

Negative results (no robust gain, or gain only in-sample) are reported as such.

## 2. Loss of a design under one scenario (data-free)

As in E11 §2: the estimators are linear-Gaussian MAPs with prior N(0, I) on the a_ref-scaled strengths, so the error of
the field forecast does not depend on the readings. For scenario s (a 24 h steady or veering wind block, design-prior
unit responses at the sites A_s(S) and on the full grid, field weight Q̄_s = mean over the block of
a_ref² F_hᵀF_h / N) and a strength second moment m2_s = E[aaᵀ]:

  L_s(S) = tr(Q̄_s Σ (m2_s + M) Σ) / tr(Q̄_s m2_s),   Σ = (I + M)⁻¹,

with M = M_s(S) (block information; `map24`-type memory: only the scenario day is in the window) or
M = M_tr(S) + M_s(S) (`map_full`-type memory: all training information plus the scenario day). This is the expected
squared field-forecast error (E11 A3 second-moment form), **normalised by the no-sensor loss** of the same scenario so
that slow-wind/high-concentration scenarios do not dominate worst case and CVaR (L_s ∈ (0, 1] for m2 = I; the empty
network has L_s = 1). With m2 = I, L_s = tr(Q̄_s Σ)/tr(Q̄_s), the E11 objective per block, normalised.

## 3. Scenario ensembles (declared; training/declared information only, no test directions)

Scenario = (wind block, strength prior). Wind blocks are independent 24 h blocks after a 12 h steady spin-up (E11 A1),
compass index d (FROM d·45°; 0 = N, 4 = S):

| ensemble | directions d | speeds (m/s) | veering blocks (12 h d → 12 h d+1, 3 m/s) | strength priors | n |
|---|---|---|---|---|---|
| `noSN6` (primary) | 1, 2, 3, 5, 6, 7 | 1.5, 3, 5 | 1→2, 2→3, 5→6, 6→7 | N(0, I); declared LogUniform m2 | (18 + 4) × 2 = 44 |
| `train3` | 2 (E), 6 (W), 7 (NW) | 1.5, 3, 5 | 6→7 | same | (9 + 1) × 2 = 20 |

- `train3` is purely training-derived: E, W and NW are the directions of the training regimes (steady westerly, veering
  270°→315°, weak variable 270°/90°); speeds span the training range (0.5–1.5 weak, 3 ± 20 % otherwise; 5 m/s is the
  generator's clip).
- `noSN6` is E11's pre-declared out-of-sample compass (A1 `noSN`) plus speeds and veers. It removes N and S, the test
  split's own directions. Removing exactly them is itself test-informed, and conservative *against* the robust designs
  (the steady southerly stays unseen). The diurnal-rotation test regime still sweeps all directions; no ensemble sees it.
- Strength priors: N(0, I) (the estimator's prior) and m2 = s²I + ā²11ᵀ of the benchmark's declared LogUniform(0.5, 5)
  generating distribution (`FD.loguniform_m2`; not the realised strengths).
- Speeds 1.5 and 5 m/s and veering blocks are new solves; the 3 m/s steady blocks equal E11's `FD.prior_block`.
- All E11 `all8` directions at 3 m/s are solved too, only for the E11 comparator `fc_w24_rob` (in-sample, labelled).

No test hour, test wind, truth field, realised strength or noise draw enters any ensemble or objective (§9 test).

## 4. Robust criteria (per memory, per ensemble)

Over the n scenario losses L(S) (equal weights):
- `mean`: E[L] (sample-average approximation);
- `cvar`: CVaR_α with **α = 0.8 declared** (mean of the worst 20 % of scenarios; discrete Rockafellar–Uryasev form
  with a fractional boundary weight: for m = (1 − α)n, the sum of the ⌊m⌋ largest plus (m − ⌊m⌋) × the next, over m);
- `worst`: max L.

Designs: `rob_{mean,cvar,worst}_{w24,full}_{noSN6,train3}` (12). Equal search budget for every optimised design
(E11 A4): greedy + best-improvement 1-swap from three starts (greedy, `dopt`, `source_near`), best objective kept; local
optimum only. Same candidate set, k, 1250 m spacing, σ and source locations as E07/E11. α is not tuned on any test score.

## 5. Comparators

`dopt`, `source_near`, `coverage`, `uniform`, `podqr` (where defined), `random` (E07 nets, averaged per scenario);
E11 nominal forecast designs `fc_w24`, `fc_grow`; E11 robust `fc_w24_rob_noSN`, `fc_grow_rob_noSN` (raw mean, 3 m/s,
unit strengths) and `fc_w24_rob` (`all8`; **includes the test directions — in-sample, reported but never used for a
claim**).

## 6. Evaluation (held out; truth enters only through scoring)

World: the E11/E12 32² pilot (upwind, dt 300 s, calendar 11 d: train 1–4, validation 5, test 6–11 = steady southerly,
stagnation-then-front, diurnal rotation 48 h each); K ∈ {2, 4}; seeds 0–2; k ∈ {2, 4, 8}; R = 5 paired white-noise draws;
estimators `map24` and `map_full` (growing window from hour 0; E11 `map_grow`). Training winds `veering` and `mixed`
(E11) for nominal evaluation; wind-error evaluation on `mixed` only (E12 convention).

Conditions per (scenario, design, estimator):
- **nominal**: oracle wind, h = 1 forecast RMSE over all test issue times (mean over `veering`/`mixed` per cluster);
- **held-out conditions** (mixed only): {oracle, dir +10°, dir +30°, spd × 0.6, lag 6 h, AR(1) 30°} (E12 `SWEEP_SPECS`,
  `pravah.uncertainty.perturb_winds`; the estimator and forecast use the model wind for every hour, designs are made from
  the correct training responses, as E12) × {3 test regimes} = 18 condition losses (h = 1 RMSE per regime block).

Held-out aggregates per design: `nominal`, `cond_mean`, `cond_cvar` (α = 0.8 over the 18 conditions), `cond_worst`,
`max_regret` (max over conditions of RMSE − the best non-random design's RMSE in that condition). Also h = 6 and
`source_rmse` (oracle) as secondary. Design-time (in-sample) ensemble loss values and design wall time (`compute_cost`)
are recorded.

## 7. Statistics and reporting

ρ = (L_cmp − L_design)/L_cmp per cluster (K, seed) (> 0 favours the design); median, distribution-free 95 % CI, sign test
and verdict (`FD.cluster_summary`, margin 0.05) per k and estimator; comparators `dopt`, `source_near`, `fc_w24_rob_noSN`.
n = 6 clusters (smallest two-sided sign p = 0.031); nothing is confirmatory. Pareto figure: median ρ_nominal vs median
ρ_cond_cvar vs `dopt`, per k × estimator.

Artifacts: `reports/quarto/tables/e15-local-*`, `reports/quarto/figures/e15-*.png`.

## 8. Compute (32², local)

New solves at 32² (≈ 6 CPU-s per block, E11): 18 distinct sources × (6 dirs × 3 speeds + 4 veers + 2 extra 3 m/s
directions) = 432 blocks ≈ 0.7 CPU-h; base truth + unit responses (veering, mixed) ≈ 0.6 CPU-h; E12 perturbed responses
5 specs × 18 sources ≈ 1.1 CPU-h. Total ≈ 2.5 CPU-h, ≈ 40 min on 4 local workers (`OMP_NUM_THREADS=1`, fork pools).

## 9. Leakage and validity checks (tests)

- Designs are identical when every non-training hour, the truth field and strengths are NaN (E11 test pattern).
- The ensembles contain no test direction (0, 4) and are built from prior blocks + training information only.
- L_s equals an explicit loop with matrix inverses; CVaR equals the Rockafellar–Uryasev minimum over η and lies between
  mean and max; CVaR at α = 0 is the mean; the robust greedy + multi-start reaches the brute-force optimum on a tiny case.
- Losses decrease when a site is added (m2 = I).
- The 3 m/s steady block equals `FD.prior_block`.

## 10. 64² plan (documented after the pilot; NOT submitted — heavy 64² compute deferred by the user, 2026-11-13)

- **World:** E07 sweep (A2: VanLeer, dt 30 s, protocol calendar, spin-up 48 h), winds `veering` + `mixed`, K ∈ {2, 4},
  seeds 0–9 → 20 clusters, k ∈ {4, 8, 16}, R = 5 (`SCALE["sweep"]`). Block spin-up 12 h (E11 A1 check at 64²).
- **Reuse (read-only, `--fields`, comma list, E07 cache first):** E07 sweep truth + unit responses (job 204894;
  156 fields); if the E12 64² sweep has run, its perturbed responses (290, E12 names) as the second entry.
- **New solves (dry run, `--sweep --dry-run`):** 1 032 design-prior blocks (43 distinct sources × 24 blocks) ×
  36 model hours at ≈ 11.3 s/h (E07 cluster rate, E11 A1) ≈ **117 CPU-h**. Without the E12 cache, perturbed responses:
  5 specs → 290 solves × 4.4 CPU-h ≈ 1 276 CPU-h; **pre-declared reduction `--specs dir30,lag6`** → 116 solves ≈ 510 CPU-h
  (dry run: 272 fields, 116 of them perturbed, once the E07 cache is visible; then drop dir 10°, spd 0.6, AR 30°).
- **Stage 2:** 32² measured 22–54 s design + 2–29 s evaluation per scenario (`e15-local-compute.csv`). Scaling by
  grid (4×) and calendar (≈ 5.5×) and k up to 16 gives an *unmeasured* guess of 0.5–2 CPU-h per cluster ≈ 10–40 CPU-h.
  Memory per stage-2 task (K = 4, all specs): ≈ 30 fields × 1440 × 64² × 8 B ≈ 1.4 GB → `--stage2-workers 16`.
- **Totals:** with the E12 cache ≈ 130–160 CPU-h (≈ 5–6 h on 32 CPUs); without it, reduced specs ≈ 640–670 CPU-h
  (≈ 21 h on 32 CPUs). Recommended order: E12 64² first (reuse), then E15.
- **Command** (only after `squeue` shows no other users' jobs; `$R` from untracked `settings.local.yaml`; snapshot =
  `git archive` of the committed revision to `$R/code/snapshots/<rev>`; env `$R/envs/main`):

      cd $R/code/snapshots/<rev>
      $R/envs/main/bin/python scripts/e15_robust_placement.py --sweep --dry-run --fields $R/<e07>/fields[,$R/<e12>/fields] --out $R/results/e15-sweep
      sbatch -J pv-rob -c 32 --mem=64G -t 24:00:00 --export=ALL,PRAVAH_HPC_ROOT=$R -o $R/logs/%x-%j.out \
        scripts/hpc/run.sbatch python scripts/e15_robust_placement.py --sweep \
        --fields $R/<e07>/fields[,$R/<e12>/fields] [--specs dir30,lag6] --out $R/results/e15-sweep --workers 32 --stage2-workers 16

  The dry run must show 0 E07 fields to solve. Then `--report` on the sweep rows (prefix `e15-sweep`, written to OUT).

## 11. Limits

Steady/veer 24 h blocks do not represent stagnation or rotation; the estimator prior N(0, I) is miscentred relative to
the truth (gate-10/11); upwind numerics at 32²; n = 6 clusters; every statement is indicative.

## 12. Deviations

Pilot rows: commit e217a02 (clean), `reports/quarto/tables/e15-local-rows.csv`. Changes after the pilot ran (analysis and
64² tooling only; no design, ensemble, α or endpoint changed):
1. Contrasts need n ≥ 6 clusters (the distribution-free 95 % CI is undefined at n = 5). `podqr` is undefined in some
   clusters at larger k, so those cells are absent rather than crashing the report.
2. The Pareto figure omits `random` and `coverage` (far off-scale: nominal gain vs dopt −1 to −14); they stay in the tables.
3. The sites figure built a placeholder truth with the wrong shape (fixed; figure only).
4. For 64²: `--fields` takes a comma list (E07 cache, then E12 cache) and `--specs` implements the pre-declared reduction.

# Reproducible claims (E19)

These are the claims a manuscript may make. Each one has a row in `reports/evidence-matrix.csv` (id in brackets) that
gives the table or figure, run record, commit and scheduler job. Wording follows the gate audits; where an audit
corrected a journal, the audit wins. Scope for every claim: a synthetic 2-D passive tracer with a perfect transport
model, known source locations and spatially uniform wind. **None is a claim about PM2.5, exposure or deployment.**

## How to check

- **Links resolve.** `uv run python scripts/validate_report.py` checks that every figure, table, citation, matrix path
  and run id resolves. `--self-test` runs its fixture test.
- **Numbers from committed tables.** Most numbers can be recomputed from `reports/quarto/tables/`. The gate check scripts
  do exactly that and their outputs are committed:
  - `docs/research/gates/gate-07/check_tables.py`
  - `docs/research/gates/gate-09/check_sweep.py`
  - `docs/research/gates/gate-10-11/summarise.py`
  - `docs/research/gates/gate-12-17/check_tables.py`
- **Re-running a pilot (32², local).** Use the command recorded in the experiment's `tables/eNN-local-meta.json`, or
  the `command` field of its record in `reports/runs/runs.jsonl`. Every pilot uses
  `OMP_NUM_THREADS=1 uv run python scripts/eNN_*.py ...` at the listed commit, with `uv.lock` frozen.
- **Re-running a 64² sweep.** Use the `command`, `code_rev` and `snapshot` fields of its record in `runs.jsonl`
  (`sbatch ... scripts/hpc/run.sbatch python scripts/eNN_*.py --sweep --out $PRAVAH_HPC_ROOT/results/...`). Row CSVs and
  fields stay under `$PRAVAH_HPC_ROOT/results/`; only summaries are in Git. The gate audits re-executed sampled
  scenarios and matched the committed rows to ≤ 4e-16 (gate-12-17 table "Reproduced numbers").

## Confirmatory (one test)

1. **[E07-1]** In the 64² sweep, in the 8 cells run (veering or mixed training wind × K ∈ {2, 4} × k ∈ {8, 16}, 10 seeds
   per cell, white noise, 24 h-window MAP), the D-optimal network reduced held-out 1 h forecast RMSE against the mean of
   10 random networks by a median 42–84 % per cell.
   - The 95 % percentile CI was above 10 % in 8/8 cells, the distribution-free CI in 7/8, and the mean bootstrap in 8/8.
   - K = 8, k = 32 and seeds 10–19 were not run [E07-2].
   - Required sentence: *a locations-only control did better still, so under this estimator the gain over random is
     attributable to knowing source locations, not to D-optimisation.*
   - Check: `tables/e07-sweep-confirmatory.csv`; job 204894, code dd2c51c.

## Exploratory, 64²

2. **[E07-3, E07-8]** Under a 24 h-window MAP, near-source placement beats D-optimal: 115/120 pairs, median 0.41,
   20/20 layouts. The ordering reverses when the estimator uses all past data: dopt is better in 103/120 pairs, median
   −0.25 [−0.38, −0.18], 19/20 layouts. dopt beats random under both estimators (117/120, 120/120).
3. **[E07-4]** D-optimal networks lose practical rank in held-out windows whose wind direction (southerly) was absent
   from training. At k = 8 the deficient-window fraction is 0.37 vs 0.01 for near-source, and these windows carry 79 % of
   dopt's pooled squared error. This rests on one unseen direction.
4. **[E07-5]** With sites within 1200 m of any source excluded, dopt's gain over random falls from 0.71 to 0.20
   [0.15, 0.33] and no cell clears the 10 % threshold. Near-source still beats dopt (0.375 [0.31, 0.45]).
5. **[E08-1]** Median k\*(0.2) under the 24 h window, for K = 2 / 4 / 8:
   - location-aware designs 2 / 4 / 8;
   - random 8 / 16 / 16;
   - coverage 8 / 8 / 8.
6. **[E08-2..E08-6]** Preregistered Holm family (windowed estimator): H1 supported (19/0/1), H2 contradicted (0/60),
   H3 supported (60/60), H4 supported (58/60), H5 supported (39/40).
7. **[E08-8, E08-9]** With all past data every strategy reaches τ = 0.2 at k = 2, a floor that depends on constant
   emissions and a perfect model, and H5 reverses (dopt 36/40).
8. **[E11-2, E11-3, E11-4]** Under a 24 h window, the wind-robust forecast design with an out-of-sample prior beats dopt
   at k = 8 by +0.34 [0.15, 0.40] (19/20 clusters, Holm p = 0.002, family of 72). It loses to near-source (−0.22
   [−0.54, −0.08], 2/20, Holm p = 0.018). Under all past data it is worse than dopt (−0.30 [−0.40, −0.12], 2/20).
9. **[E09-2]** On E07's 64² fields, all-past-data estimation cuts 1 h forecast RMSE against the 24 h window by a median
   0.85–0.94, with 95 % coverage 0.95.
10. **[E17-1..E17-4]** Synthetic tracer under box-mean ERA5 winds (not validation; 3 cases × 20 clusters sharing one
    wind history per case):
    - dopt and source_near beat random (20/20 at k = 8; 0.35–0.50 under the 24 h window, 0.53–0.64 under all past data);
    - dopt and source_near tie except a monsoon-case edge for source_near that is not Holm-significant (16/20, p = 0.012);
    - with a wrong model wind the forecast error is a transport floor (true strengths do as badly), at 1.8–3.7× oracle
      persistence.

## Exploratory, 32² pilots (6 independent clusters unless stated; upwind numerics)

11. **[E09-1]** All-past-data estimation cuts 1 h forecast RMSE against the 24 h window by a median 58–74 %. A median
    0.99 of the log gain is memory; the prior and the sequential form add essentially nothing. A field-only filter is
    1.6–13× worse **[E09-3]**.
12. **[E10-2, E10-3, E10-4]** With time-varying sources from the filter's own model, the validation-selected random-walk
    filter cuts 1 h RMSE by 0.24–0.55 against all past data at k = 8. It is 12–15 % worse at 6 h under a diurnal cycle.
    Its calibration was measured only with an oracle-matched q.
13. **[E12-1, E12-2]** Direction errors ≥ 30°, lags ≥ 2 h and AR 30° errors put forecasts below oracle persistence in
    0/6 clusters. Speed, +10° and AR 15° errors hide inside a near-perfect fit (ratio 1.01–1.33) while source RMSE rises
    up to 16×.
14. **[E13-2]** A common-mode sensor bias and a background offset give identical readings (to 3.6e-15). With a
    background term, the bias corrupts the forecast field while every truth-free check is nominal (skill −3.9).
15. **[E14-1, E14-2]** The best truth-free early warning has within-configuration AUC 0.71 [0.64, 0.77] (0.74 pooled).
    A × 0.8 speed error is undetectable (AUC 0.44–0.57).
16. **[E15-1]** Robust (CVaR / worst-case) placement gives no tail gain beyond multiplicity: 9 of 648 contrasts at 6/0,
    about 10 expected by chance.
17. **[E16-1, E16-2]** No one-at-a-time ablation reversed any of five conclusions (8 cells, n = 10, 35 uncorrected
    verdicts). The memory flip appears in every cell.

## Negative, withdrawn and not conducted (must be reported)

- **[E03-1]** Fixed, non-optimised networks with a 24 h window were rank deficient in all 648 fits. The identifiable
  half of the hypothesis was never tested.
- **[E05-W1, E05-W2]** The E05 v1 window and sensor-substitution claims and its "contradicted" wind-diversity labels
  were withdrawn after gate-05.
- **[E15-1, E15-4]** No robust-placement tail gain. Criteria and ensembles were indistinguishable.
- **[E10-3]** The drifting-source filter is worse at 6 h under a diurnal cycle.
- **[E07-2, E15-5, E16-5]** Not run: the E07 protocol cells K = 8 and k = 32, and the E12–E16 64² runs.
- **[E18-1]** Observational validation was not conducted: no usable monitoring data exist.

## Not warranted (do not claim)

- "Optimised placement beats random" without the source-location caveat and the 24 h-window qualifier.
- "Knowing source locations matters more than D-optimisation" without "under a 24 h estimation window".
- "Filtering beats batch inversion", or "the moment-matched prior helps".
- "Worse than persistence" without saying that persistence is an oracle full-field bound.
- E17's "15–27× degradation" (a ratio to a near-zero denominator), or E17 as validation.
- Endpoint agreement as a general finding.
- Anything about real sources, PM2.5, exposure, deployment or point sources.

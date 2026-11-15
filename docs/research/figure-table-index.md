# Figure and table index (E19)

All paths are relative to `reports/quarto/`, and all exist (checked by `scripts/validate_report.py`). The shortlist is
proposed for a main text; everything else goes to the appendix or supplement. The ids in the last column of the
shortlist are claim ids in `reports/evidence-matrix.csv`.

## Main-text shortlist

| # | Item | Path | Shows | Claim ids |
|---|---|---|---|---|
| F1 | Benchmark truth panels | `figures/e02-truth-panels.png` | domain, sources, plumes, held-out winds | E02-1 |
| F2 | Confirmatory result | `figures/report-e07-confirmatory.png` | dopt vs random per cell with CIs and the 10 % line | E07-1 |
| T1 | Confirmatory table | `tables/report-e07-confirmatory.md` (data `tables/e07-sweep-confirmatory.csv`) | percentile, distribution-free and mean variants | E07-1 |
| F3 | Cross-objective sweep | `figures/e07-sweep-crossobjective.png` | strategy curves; source_near on top under map24 | E07-3, E07-6 |
| F4 | Mechanism and exclusion control | `figures/e07-followup.png` | rank-deficient windows by regime; 1200 m exclusion | E07-4, E07-5 |
| T2 | Memory reversal | table in ch. 07 "ordering depends on estimator memory" (from `docs/research/gates/gate-09-filter.md`) | map24 vs all-past-data ordering | E07-8 |
| F5 | Sufficiency | `figures/e08-sweep-sufficiency.png` | nRMSE vs k by strategy, noise and schedule | E08-1..E08-7 |
| T3 | k\* and variants | `tables/e08-sweep-breakpoints.csv`, `-excl1200-`, `-mapfull-` | k\*(0.2) table (ch. 08) | E08-1, E08-8, E08-10 |
| F6 | Forecast-optimised placement | `figures/e11-sweep-forecast-design.png` | robust design vs dopt and source_near under both memories | E11-1..E11-4 |
| F7 | Wind mismatch | `figures/e12-local-bias.png` | compensation and loss of skill (pilot) | E12-1, E12-2 |
| F8 | Diagnostics ROC | `figures/e14-local-roc.png` | truth-free warning AUCs (pilot) | E14-1..E14-3 |
| F9 | Regional case | `figures/e17-sweep-results.png` | placement vs random; wrong-wind floor | E17-1..E17-4 |

## Appendix / supplement

- **Numerics.** `figures/e01-convergence.png`; `tables/e01-{grid,order,dt,cfl,mass}.md`; B3 study
  (`docs/research/gates/b3-advection-scheme.md`).
- **Benchmark.** `tables/e02-{splits,gates,info}.md`.
- **Fixed-network inverse (negative).** `figures/e03-source-estimates.png`; `tables/e03-{validity,main,findings,compare,failures,nonid}.md`.
- **Footprints.** `figures/e04-footprints.png`; `tables/e04-{results,identifiability}.md`.
- **Temporal observability.** `figures/e05-rank-vs-window.png`, `figures/e05-source-rmse.png`; `tables/e05-duration-*.csv`.
- **Placement methods.** `figures/e06-designs.png`; `tables/e06-{results,gap,best,undefined,modes,cross,prank,runtime}.md`.
- **E07 pilot.** `figures/e07-local-crossobjective.png`; `tables/e07-local-*`; `tables/report-e07-sourcenear.md`.
- **E08 pilot and variants.** `figures/e08-local-sufficiency.png`; `figures/e08-local-{excl1200,mapfull}-sufficiency.png`;
  `figures/e08-sweep-{excl1200,mapfull}-sufficiency.png`.
- **Filters.** `figures/e09-local-traces.png`, `figures/e10-local-traces.png`; `tables/e09-local-*`, `tables/e10-local-*`.
- **E11 pilot.** `figures/e11-local-forecast-design.png`, `figures/e11-local-v2-forecast-design.png`.
- **Model mismatch.** `figures/e13-local-uncertainty.png`; `tables/e13-local-*`.
- **Calibration.** `figures/e14-local-calibration.png`.
- **Robust placement (negative).** `figures/e15-pareto-local.png`, `figures/e15-sites.png`; `tables/e15-local-*`.
- **Generalization.** `figures/e16-local-regime-ranks.png`; `tables/e16-local-*`.
- **Regional case.** `figures/e17-local-windrose.png`, `figures/e17-local-results.png`; `tables/e17-*`.
- **Observational gap.** `tables/e18-status.csv`; `docs/research/observational-validation-audit.md`.

## Not referenced by any chapter

These exist on disk but are not shown in the book:
- `figures/e08-local-{excl1200,mapfull}-sufficiency.png`;
- `figures/e08-sweep-{excl1200,mapfull}-sufficiency.png`;
- `figures/e11-local-forecast-design.png` (v1, superseded by v2);
- `figures/e15-sites.png`;
- `figures/e17-local-results.png`.

Keep them as supplement or drop them; none carries a claim that is not in a chapter table.

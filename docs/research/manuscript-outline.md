# Manuscript outline (journal-agnostic; E19)

This is an outline, not a manuscript; the human author writes the paper. No target journal has been checked for scope.
Claims and their evidence are in `manuscript-claims.md` and `reports/evidence-matrix.csv`. The figure and table
shortlist is in `figure-table-index.md`.

**Working title:** Where to put a few noisy sensors when the wind is uncertain: placement, memory and model error in a
synthetic advection–diffusion benchmark.

## Abstract (structure only)

1. **Problem.** Sparse low-cost networks must reconstruct and forecast a tracer field from uncertain winds and emissions.
2. **Approach.** A synthetic 2-D benchmark with held-out wind regimes, a preregistered protocol, independent audits, and
   one confirmatory test.
3. **Findings.** Three, as follows.
   - Location-aware placement beats random. This is confirmatory in the cells run, and the gain comes from knowing
     source locations.
   - Which location-aware design wins depends on estimator memory.
   - Wind and model errors dominate placement differences, and truth-free diagnostics catch only some of them.
4. **Limits.** Synthetic, perfect-model world; most plan-3 results are 32² pilots.

## 1. Introduction

- Motivation: low-cost sensor networks; identifiability vs coverage; the Fisher/D-optimal tradition [@green2006],
  data-driven placement [@manohar2018; @desilva2021], and value-of-information placement for PM2.5 [@choi2025].
- Gap: placement is rarely judged on held-out winds, under different estimator memories, or under model mismatch.
- Contributions:
  1. a confirmatory held-out test with a locations-only control;
  2. the memory-dependent reversal of the placement ordering;
  3. a mismatch and diagnostics stress suite, including a structural silent failure;
  4. negative results (robust placement, fixed-network identifiability).

## 2. Related work

Use only verified entries from `reports/quarto/references.bib` (see `reference-audit.md`). Themes:
- inverse source estimation and data assimilation [@mons2017; @zheng2009; @sharma2019; @webster2022];
- identifiability-aware apportionment [@bhardwaj2026] (preprint);
- information-theoretic and GP placement [@krause2008; @mano2022; @zhou2022];
- observability-based placement [@bartos2021];
- Kalman-based field estimation [@roy2018];
- mobile sensing [@cheng2022];
- regional modelling context [@pennington2024; @mues2018; @schlutow2026];
- tools [@raanes2024; @karnik2026].

## 3. Methods

- **PDE and numerics.** Upwind verification; amendment A2 (split VanLeer, dt 30 s) at 64²; the superposition error and
  its measured impact (E01, B3, gate-07 issue 7).
- **Benchmark.** Calendar split, held-out test wind regimes, noise model, amendment A1 source scale and its
  exploratory consequence (E02).
- **Designs and what each knows** (E06/E07 access table). Estimators: 24 h-window MAP, growing-window MAP (≡ augmented
  KF), random-walk filter.
- **Protocol.** Preregistration, the single confirmatory rule, sample units (seed, cluster), multiplicity handling, and
  the gate-audit process.

## 4. Numerical verification

E01 orders and mass conservation, and the B3 effective diffusivity. Appendix material.

## 5. Results, by research question

The structure follows chapter 19, with one subsection per question.
1. **Sensor count and history** (E03 negative, E05 v2, E08).
2. **Placement for which task** (E07 confirmatory plus controls, gate-09 reversal, E08 H5, E11).
3. **Memory and sequential updating** (E09, E10).
4. **Reliability under mismatch** (E12–E14; pilot-level; oracle persistence stated).
5. **Robust placement and transfer** (E15 negative, E16 pilot, E17 regional, not validation).

Each subsection reports the scale, n and unit, the interval, and the audit verdict.

## 6. Discussion and limitations

- Known source locations; uniform wind; constant or filter-generated emissions; perfect transport model.
- One unseen wind direction drives the E07 mechanism; the 1200 m exclusion shrinks dopt's gain to 0.20.
- Pilot-only experiments: E09, E10, E12–E16.
- **Possible negative outcomes the paper must state:**
  - the confirmatory gain is a location-knowledge effect;
  - robust placement did not help;
  - wrong winds erase both placement and memory advantages;
  - E03's fixed-network identifiability is negative.
- The memory reversal means "optimal placement" is conditional on assumed source persistence.

## 7. Reproducibility and availability

- Git commits, `reports/runs/runs.jsonl` (scheduler job ids), `uv.lock`, the evidence matrix and
  `scripts/validate_report.py`.
- 64² row data stay under `$PRAVAH_HPC_ROOT/results/` and are summarised in Git.
- ERA5 (CC-BY 4.0, Copernicus notice) and WorldPop (CC BY 4.0) attribution.

## Mock peer-review checklist

- [ ] Is every "dopt beats random" statement paired with the source_near and 24 h-window qualifiers?
- [ ] Are sample units (seeds, clusters) and dependence stated, and is multiplicity handled or flagged?
- [ ] Is persistence labelled as an oracle bound everywhere?
- [ ] Are pilot (32²) and 64² results visibly separated?
- [ ] Are all negative, withdrawn and not-conducted results present (E03, E05 v1, E15, E18, unrun cells)?
- [ ] Do all post hoc analyses (exclusion control, gate-09 re-scoring, within-config AUC, persistence) say so?
- [ ] Is the A1 source scale disclosed as chosen after seeing data?
- [ ] Are the open bugs from gate-12-17 fixed, or their effect bounded?
- [ ] Is there no implication of real-world exposure or deployment benefit?

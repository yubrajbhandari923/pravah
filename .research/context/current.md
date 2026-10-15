# Research context — pravah

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~2815 tokens.
> Generated 2026-10-15T12:14:11-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Simulate the air pollution field evolution

Git: `master` @ `db3b593298` (uncommitted changes)

## Latest checkpoint

**CP-014** PLAN-007 report written; DYN-00..07 + EXP-027 done, DYN-08 not run (2026-10-14)
- **Understanding:** All registered experiments except DYN-08 have results. Current findings (preliminary): F-044 DYN-06, F-046 DYN-05, F-048 EXP-027 VOID, F-049 DYN-07, F-050 DYN-04, F-051 DYN-00 25/25, F-053 DYN-03, F-054 DYN-02 v3; F-036..F-047 earlier versions superseded after independent review (needs_work) or my own diagnostics. Gate 21/0 on arm64 (final tree) and x86_64 (RUN-0066); 637 default tests pass; pyright 0 errors. HPC: no jobs outstanding; reserved <=~330 CPU-h worst case. Report docs/DYNAMIC_POPULATION_SENSOR_RESEARCH_REPORT.md is PROVISIONAL.
- **Next step:** Ask user about EXP-027b (>=48-96 cases per draw, ~90-190 core-h) and DYN-08; independent review of F-044/F-046/F-048/F-049/F-050/F-053/F-054; constrained-estimator-aware design objective.

## Current baseline

EXP-002 — EXP inverse recovery (frozen reference)

## Active & open questions

- Q-019 [investigating] RQ-D4: Can changing local emissions be distinguished from short-lived transboundary episodes?
- Q-022 [investigating] RQ-D7: How do wind uncertainty and missing sources affect inference and deployment?
- Q-024 [investigating] RQ-D9: Do conclusions transfer from toy geometry to Belauri-like population geography?
- … 14 more (`research question list`)

## Established findings (what we believe)

- F-051 [supported, high] DYN-00 (RUN-0091, 25/25 pre-set checks pass; supersedes F-038 after its independent review F-038/R1): includes the replacement for the vacuous west_no_import check: a constant west inflow basis through a 3600 s wind reversal has cell-summed supply 3.36e-2 (>0) before and exactly 0.0 after, an east basis exactly 0.0 before and 3.36e-2 after, mass closure 2.1e-15 with both on; a non-trivial piecewise history [[0,3,1,2],[1,0,2,4],[0.5,1.5,0,3]] matches direct simulation to 1.2e-15; all other values as in F-038 (mass closure <=4.4e-15 in five dynamic cases, superposition 1.6e-15, closed-box analytic order 0.997, footprints integrate to 1). Verified numerical behaviour of the implicit-Euler/exponential scheme on a 16x12 toy box; not verified: Van Leer scheme, Belauri-scale runs, spatial order beyond monotone convergence. — evidence: EXP-028, RUN-0091
- F-009 [supported, high] Solver routing refactor preserves every deterministic FiPy output bit-for-bit — evidence: EXP-008, RUN-0012, RUN-0013
- F-007 [supported, medium] FINAL: hidden wind error and unmodelled source dominate discretisation mismatch; poor layout unidentifiable — evidence: EXP-007, RUN-0011
- … 24 more (`research finding list`)

## Failed directions (do not repeat without a new reason)

- F-005 [preliminary] Unregularised bounded least squares is unstable for layouts blind to the plumes

## Current experiments

- EXP-031 [needs_review] DYN-03 Value of measurement history — runs 4, ⚠ 4 unsynthesized
- EXP-034 [needs_review] DYN-06 Local activity vs transboundary influx identifiability — runs 4, ⚠ 4 unsynthesized
- EXP-032 [needs_review] DYN-04 Dynamic population-weighted sensor design — runs 3, ⚠ 3 unsynthesized

## Active plans

- PLAN-007 Dynamic Population-Weighted Sensor Deployment and Source-History Inference — 20/30 tasks done · objective: Develop and evaluate a physics-informed dynamical inverse framework for optimal fixed air-quality sensor placement, using time-series observations to infer changing local and transboundary emissions while minimizing population-weighted concentration reconstruction error under uncertain transport.
  - next ready: T-069 Final report, figures, checkpoint (`research task context T-069`)
  - BLOCKED: T-067 Belauri synthetic deployment DYN-08; reconcile EXP-026 — Not run: DYN-08 Belauri synthetic dynamic deployment (and EXP-026 reconciliation) needs a Belauri-scale dynamic response build (~hundreds of core-hours) and the F-046/F-048 conclusion that design ensembles must be much larger first; needs user direction.
- PLAN-004 PRAVAH phase 3 HPC execution, interpretation and synthesis — 1/6 tasks done · objective: Run the deferred heavy Phase 3 experiments as one Slurm job, import results, interpret with independent review, finish synthesis
  - next ready: T-033 H1 PI: set up repo on cluster (slurm/setup_env.sh), copy data/belauri/raw, submit slurm/phase3.sbatch, download results bundle (`research task context T-033`)
- PLAN-003 PRAVAH phase 3: portability, population-aware design, numerical convergence of decisions, wind uncertainty — 6/10 tasks done · objective: Transferable framework whose sensor decisions are robust to discretisation and wind uncertainty, with population-aware design; testbed only, no empirical validation
  - next ready: T-028 S3 EXP population-aware design: area vs pop vs mixed beta vs capped on 5 fixtures; Pareto; held-out (`research task context T-028`)
  - BLOCKED: T-027 S2b EXP convergence of decisions: space, time, joint, scheme; physical-coordinate candidates; displacement, regret, ties vs resampling baseline — Runs as PLAN-004 H1-H3 (EXP-015 stage 2) on HPC
  - BLOCKED: T-031 S5a Conditional: wind-robust and population-robust placement — Removed from scope for now: conditional on a Belauri-scale wind effect; revisit after PLAN-004 H3/H4 (toy shows wind error dominates sensor count for >=10 deg, F-029)

## Needs attention (verification)

- [verification failed] T-070 EXP-027 on a common padded domain: 250/500/2000 m meshes (prepare, config, smoke, dry-run only): no deterministic verifier ran (no task checks and no approved post_implementation hooks): this is not a pass
- [reviewer disagrees] F-035 EXP-015 stage 2: H1 untested (baseline not commensurable); 500 m provisionally admissible on narrow grounds: scientific-reviewer: needs work — H1 is UNTESTED, not supported. The mesh comparison and the resampling baseline are not commensurable: convergence_stage2.py:24
- [awaiting review] F-065 DYN-03 v2 (RUN-0094): causal filter vs causal static fit; interpretation corrected after review F-064/R1: Agent finding awaiting an independent review
- [awaiting review] F-063 DYN-06 v4 identifiability (RUN-0093), predictions restated after review F-060/R1: Agent finding awaiting an independent review
- [awaiting review] F-062 DYN-05 sufficiency (RUN-0078/0080), conclusions corrected after review F-058/R1: Agent finding awaiting an independent review

## ⚠ Needs synthesis before new experiments

- EXP-025 INFRA cluster environment build and production preflight (not a scientific experiment) (4 unsynthesized runs)
- EXP-027 EXP-015v2 decision stability with a COMMENSURABLE paired baseline (pre-registered fix) (4 unsynthesized runs)
- EXP-028 DYN-00 Dynamical transport verification (3 unsynthesized runs)
- EXP-030 DYN-02 Historical source inversion (M0/M1/M2) (4 unsynthesized runs)
- EXP-031 DYN-03 Value of measurement history (4 unsynthesized runs)
- EXP-032 DYN-04 Dynamic population-weighted sensor design (3 unsynthesized runs)
- EXP-033 DYN-05 P0 design-scenario statistical sufficiency (4 unsynthesized runs)
- EXP-034 DYN-06 Local activity vs transboundary influx identifiability (4 unsynthesized runs)
- EXP-035 DYN-07 Robustness and model mismatch (2 unsynthesized runs)

## Active decisions

- D-025 (2026-10-14) DYN-06 v4 (supersedes D-022's scoring), registered before the run, after review F-044/R1. Changes: (1) misattribution is scored only in a (wind, cell) where detection is clearly above the false-alarm level (one-sided binomial test of the detected count against 0.10, p < 0.01); other cells are 'undetected' and reported as such; (2) the summary statistic is POOLED misattribution (sum of wrong labels / sum of detected replicates over the scored winds) alongside the number of winds scored and the mean detection, never an unweighted mean of per-wind rates; (3) cells whose look-alike has no observable change signature (|A cb|/sigma < 1) or an unphysical rescale (c* > 1e3) are flagged detectability_only and reported separately; scenarios whose amplitude matching failed (match_ok False) are invalid, not scored; (4) P4 is evaluated ONLY for S7 (the S1 pair, L4_ref vs L4_interior) as registered in D-019; P1-P3 are adjudicated on the gated pooled metric and on detection (P3 at d=2 is expected to be uninformative because detection is near the false-alarm level there; it is reported as 'not evaluable' where detection is not above null); (5) new noise seeds: null 9500-9599, signal 9600-9699; all other design (d targets, 6 winds, layouts L4/L9/L4_ref, noise 0.02/0.10, frozen ridge alpha, controls) as in D-022. The earlier headline numbers (F-044) are superseded by the v4 finding.
- D-024 (2026-10-14) DYN-03 v2 protocol (supersedes D-014 for EXP-031), registered before the tune or test is run, after review F-053/R1. Changes: (1) CT and CAUSAL are now truly causal: their static prior for bin k is the M0 ridge fit (frozen alpha) on data up to the end of bin k only (previously the prior used the whole window); (2) detection requires the estimate to be on the PRE side at the bin before the change (a constant estimator that is already past the midpoint no longer 'detects'); changes at bin 0 are not scored; (3) test seeds 8600-8695 (7600-7695 consumed by RUN-0092); validation seeds 6000-6047; (4) CAUSAL/RETRO M2 grid reduced to alpha in {0.01,0.1,1,10,100,1e3,1e4} x beta in {0,10,1e2,1e3,1e4,1e5} (the reviewer showed alpha <= 1 is a flat plateau; beta=0 nests M1), CT/M1 alpha grid unchanged; (5) the steady-wind 'negative control' of D-014 is RETIRED as ill-posed (history helps in steady winds through plume carry-over, so the registered stop rule 'steady gain => invalid' fired for a reason unrelated to the analysis); the replacement registered prediction is in RELATIVE terms: the relative gain of CAUSAL over CT and of RETRO over CAUSAL is larger on the four variable winds than on the two steady winds, reported in both absolute and relative terms; (6) primary contrasts, Bonferroni, effect criterion and leave-one-wind-out as in D-014, plus a t-interval over the six wind means as a reported (not decisive) generalisation check. The earlier result remains on record as F-053 (superseded by the v2 finding when written) and its stop-rule firing is acknowledged.

## Blockers

- [run] RUN-0075: RUN-0075 (EXP-027) failed — not yet reviewed
- [run] RUN-0070: RUN-0070 (EXP-027) failed — not yet reviewed
- [run] RUN-0067: RUN-0067 (EXP-025) failed — not yet reviewed

## Project skills

- 9 more skills: `research skills list`, load one with `research skill show NAME`

## Agent policy

- max runs without synthesis: 5
- max failed runs without review: 3
- synthesis required before new experiment: True
- findings: agents create them preliminary; an independent review (`research finding review`) makes them supported
- tasks: `research task done` runs the task's checks; failures block completion
- Protocol: see AGENTS.md

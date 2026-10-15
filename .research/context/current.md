# Research context — pravah

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~2802 tokens.
> Generated 2026-10-15T18:23:14-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Simulate the air pollution field evolution

Git: `master` @ `14d7248a67` (uncommitted changes)

## Latest checkpoint

**CP-014** PLAN-007 report written; DYN-00..07 + EXP-027 done, DYN-08 not run (2026-10-14)
- **Understanding:** All registered experiments except DYN-08 have results. Current findings (preliminary): F-044 DYN-06, F-046 DYN-05, F-048 EXP-027 VOID, F-049 DYN-07, F-050 DYN-04, F-051 DYN-00 25/25, F-053 DYN-03, F-054 DYN-02 v3; F-036..F-047 earlier versions superseded after independent review (needs_work) or my own diagnostics. Gate 21/0 on arm64 (final tree) and x86_64 (RUN-0066); 637 default tests pass; pyright 0 errors. HPC: no jobs outstanding; reserved <=~330 CPU-h worst case. Report docs/DYNAMIC_POPULATION_SENSOR_RESEARCH_REPORT.md is PROVISIONAL.
- **Next step:** Ask user about EXP-027b (>=48-96 cases per draw, ~90-190 core-h) and DYN-08; independent review of F-044/F-046/F-048/F-049/F-050/F-053/F-054; constrained-estimator-aware design objective.

## Current baseline

EXP-002 — EXP inverse recovery (frozen reference)

## Active & open questions

- Q-021 [investigating] RQ-D6: Are optimal sensor decisions stable under changes in discretization?
- Q-019 [investigating] RQ-D4: Can changing local emissions be distinguished from short-lived transboundary episodes?
- Q-022 [investigating] RQ-D7: How do wind uncertainty and missing sources affect inference and deployment?
- … 14 more (`research question list`)

## Established findings (what we believe)

- F-051 [supported, high] DYN-00 (RUN-0091, 25/25 pre-set checks pass; supersedes F-038 after its independent review F-038/R1): includes the replacement for the vacuous west_no_import check: a constant west inflow basis through a 3600 s wind reversal has cell-summed supply 3.36e-2 (>0) before and exactly 0.0 after, an east basis exactly 0.0 before and 3.36e-2 after, mass closure 2.1e-15 with both on; a non-trivial piecewise history [[0,3,1,2],[1,0,2,4],[0.5,1.5,0,3]] matches direct simulation to 1.2e-15; all other values as in F-038 (mass closure <=4.4e-15 in five dynamic cases, superposition 1.6e-15, closed-box analytic order 0.997, footprints integrate to 1). Verified numerical behaviour of the implicit-Euler/exponential scheme on a 16x12 toy box; not verified: Van Leer scheme, Belauri-scale runs, spatial order beyond monotone convergence. — evidence: EXP-028, RUN-0091
- F-009 [supported, high] Solver routing refactor preserves every deterministic FiPy output bit-for-bit — evidence: EXP-008, RUN-0012, RUN-0013
- F-007 [supported, medium] FINAL: hidden wind error and unmodelled source dominate discretisation mismatch; poor layout unidentifiable — evidence: EXP-007, RUN-0011
- … 28 more (`research finding list`)

## Failed directions (do not repeat without a new reason)

- F-005 [preliminary] Unregularised bounded least squares is unstable for layouts blind to the plumes

## Current experiments

- EXP-037 [running] EXP-027b mesh adequacy on common cases (same-ensemble shape error, 2 km falsification gate) — runs 2, ⚠ 1 unsynthesized
- EXP-036 [needs_review] DYN-08 Belauri synthetic population-weighted deployment — runs 2
- EXP-029 [proposed] DYN-01 Temporal event-generation study — runs 0

## Active plans

- PLAN-007 Dynamic Population-Weighted Sensor Deployment and Source-History Inference — 27/30 tasks done · objective: Develop and evaluate a physics-informed dynamical inverse framework for optimal fixed air-quality sensor placement, using time-series observations to infer changing local and transboundary emissions while minimizing population-weighted concentration reconstruction error under uncertain transport.
  - next ready: T-069 Final report, figures, checkpoint (`research task context T-069`)
  - BLOCKED: T-067 Belauri synthetic deployment DYN-08; reconcile EXP-026 — Not run: DYN-08 Belauri synthetic dynamic deployment (and EXP-026 reconciliation) needs a Belauri-scale dynamic response build (~hundreds of core-hours) and the F-046/F-048 conclusion that design ensembles must be much larger first; needs user direction.
- PLAN-004 PRAVAH phase 3 HPC execution, interpretation and synthesis — 1/6 tasks done · objective: Run the deferred heavy Phase 3 experiments as one Slurm job, import results, interpret with independent review, finish synthesis
  - next ready: T-036 H4 Interpret Belauri-scale EXP-016 (exact vs greedy on test, population/mixed/capped, cluster bootstrap, design-split optimism) (`research task context T-036`)
  - BLOCKED: T-033 H1 PI: set up repo on cluster (slurm/setup_env.sh), copy data/belauri/raw, submit slurm/phase3.sbatch, download results bundle — Superseded: per-experiment HPC workflow (RUN-0081, RUN-0098, RUN-0099) replaced phase3.sbatch bundle.
  - BLOCKED: T-034 H2 Register the cluster runs: research run attach per phase from hpc_manifest.json (command, times, exit code, git commit, hashes), artifacts, no rewriting of history — Superseded: cluster runs registered per experiment; no hpc_manifest.json exists.
  - BLOCKED: T-035 H3 Interpret EXP-015 stage 2 (displacement/regret vs replicate baseline, monotone in refinement, tolerance fractions) and decide the design mesh — Superseded by EXP-027 (F-057) and EXP-037; mesh stays provisional per D-006.
- PLAN-003 PRAVAH phase 3: portability, population-aware design, numerical convergence of decisions, wind uncertainty — 6/10 tasks done · objective: Transferable framework whose sensor decisions are robust to discretisation and wind uncertainty, with population-aware design; testbed only, no empirical validation
  - next ready: T-028 S3 EXP population-aware design: area vs pop vs mixed beta vs capped on 5 fixtures; Pareto; held-out (`research task context T-028`)
  - BLOCKED: T-027 S2b EXP convergence of decisions: space, time, joint, scheme; physical-coordinate candidates; displacement, regret, ties vs resampling baseline — Superseded by EXP-027 (F-057, void) and EXP-037 (EXP-027b); no further PLAN-003 work.
  - BLOCKED: T-031 S5a Conditional: wind-robust and population-robust placement — Out of scope: conditional on a Belauri-scale wind effect not shown (F-029, F-034); not pursued.

## Needs attention (verification)

- [reviewer disagrees] F-035 EXP-015 stage 2: H1 untested (baseline not commensurable); 500 m provisionally admissible on narrow grounds: scientific-reviewer: needs work — H1 is UNTESTED, not supported. The mesh comparison and the resampling baseline are not commensurable: convergence_stage2.py:24
- [reviewer disagrees] F-069 EXP-036 / DYN-08 (RUN-0099): STOPPED at registered stop rules 4.5(a) and (b); no H8 verdict; descriptive results only: scientific-reviewer: needs work — Review of RUN-0099 as a confirmatory test of H8: all numbers reproduce (max abs diff 2.8e-17); no pipeline bug behind the cont
- [awaiting review] F-072 EXP-026 (RUN-0098): greedy gaps and design-sample optimism are large; objectives differ rather than tie (replaces F-068): Agent finding awaiting an independent review
- [awaiting review] F-071 EXP-026 (RUN-0098): WorldPop fixture is a joint null with the synthetic fixtures (replaces F-067): Agent finding awaiting an independent review
- [awaiting review] F-070 EXP-026 (RUN-0098): no robust reduction of population-weighted RMSE from population-weighted layouts; result depends on optimiser and is low-powered (replaces F-066): Agent finding awaiting an independent review

## Active decisions

- D-029 (2026-10-15) DYN-08 v2 (EXP-036, corrected rerun) registration D-029, filed BEFORE any v2 run and with no v2 test result seen. v1 (RUN-0099, F-069) stopped at stop rules 4.5(a),(b); the review (F-069/R1) found the controls badly constructed, not a pipeline bug. v1 stays reported; v2 is a labelled POST-HOC-CORRECTED confirmatory rerun, NOT a rescue of v1. Changes: (1) control (a'): replace lowpop by a layout that is bad BY CONSTRUCTION: the worst-surrogate-risk subset among a fixed pool (the 16 existing layouts + 500 random subsets, pool seed 41300), required to have surrogate risk >= 1.5x the median of the pool and checked at the design stage before tuning; it must lose to spread by the registered effect criterion in >=3 of 4 cells, else stop. (2) control (b'): 10 vs 10 random layouts (fresh seeds 41200+) assessed with a TWO-LEVEL (layout x episode) bootstrap; the effect criterion must NOT be met in any cell, else stop. (3) primary claim dyn_opt vs random uses the two-level bootstrap at level 0.0025; contrasts against single baseline layouts (spread, highpop, static_opt, src_opt) are worded as statements about that layout. (4) CT/CAUSAL bin 0 (no observations precede t=7200 s) is filled with the bin-1 estimate and flagged acausal for BOTH arms; H8c stays exploratory. (5) model error: save per-episode rows and the perturbed-truth projection floors, report ratios with episode-bootstrap CIs. (6) Fresh seeds: test 44000-44191, validation 45000-45047; wind windows and design winds unchanged (selection seed 41100); disclosure: the v1 test episodes were viewed, the windows and design layouts are therefore not fresh, only the episodes/noise are. (7) Unchanged: 500 m/dt100 provisional, 54 sites, k in {3,5}, noise 0.02/0.10, tuning on validation with spread only, effect criterion, H8 verdict rule of D-026, 25% edge gate, all other stop rules. Disclosure: M2 tuning grid is larger than M1's (favours M2 slightly).
- D-028 (2026-10-15) EXP-026 (RUN-0097 failed, RUN-0098 rerun): the registered Phase 2 regression check (area greedy layouts == results/geodesign/layouts.csv) is skipped for EXP-026 via configs/population_belauri_hpc.json (frozen.regression_layouts_file=null). Diagnosis (before any test case existed): the check compares layouts selected under different noise scales: EXP-026 uses the validation-ensemble scale (noise_levels.csv validation row, sigma 0.012548 at fraction 0.05; the G2 fix of commit 0370f3e) while results/geodesign/layouts.csv used the design scale (0.015370), so area-greedy layouts legitimately differ at budget 2 (int_c007014+int_c013008 vs int_c013008+int_c009014). It is not an architecture effect (D-005). The check therefore tests a different protocol than the one registered; it is not loosened, it is not applicable. Retained correctness guard: evaluators verified against design.score_layout / greedy_trace (printed in the run log). Disclosure: this is a deviation from the inherited EXP-016 sanity stop, decided after seeing the failure but before any test data; EXP-026 findings must state it.

## Blockers

_none_

## Project skills

- 9 more skills: `research skills list`, load one with `research skill show NAME`

## Agent policy

- max runs without synthesis: 5
- max failed runs without review: 3
- synthesis required before new experiment: True
- findings: agents create them preliminary; an independent review (`research finding review`) makes them supported
- tasks: `research task done` runs the task's checks; failures block completion
- Protocol: see AGENTS.md

# Research context — pravah

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~3029 tokens.
> Generated 2026-10-15T21:18:06-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Simulate the air pollution field evolution

Git: `master` @ `17c0743971` (uncommitted changes)

## Latest checkpoint

**CP-016** CP-016 final scientific review, independent audit, and research closeout (2026-10-15)
- **Understanding:** All planned experiments completed. Syntheses recorded for EXP-036 (DYN-08) and EXP-037 (EXP-027b). Exhaustive skeptical auditor review completed across all 14 corrected dynamic findings (F-051 supported; F-055..F-077 preliminary) and earlier Phase 3 findings (F-029..F-035). Figure index reconciled to include all 20 publication figures and obsolete placeholders removed. Tasks T-032 and T-038 held open with explicit documented blockers regarding non-author review debt and superseded workflows. Q-013 documented as open and partly addressed. Final report and presentation caveats updated. The codebase represents a verified mathematical and computational methods testbed with synthetic inputs; no empirical validation exists.
- **Current problem:** No empirical air quality data; 500 m / dt 100 mesh remains provisional; Q-013 unmeasured replicate spread at Belauri scale; terrain-coupled flow absent; second-round non-author peer reviews remain external debt before manuscript submission.
- **Next step:** Collect empirical station/low-cost sensor data in Belauri; evaluate 250 m mesh inverse with 1 km control; measure replicate spread of Belauri design objective with stratified ensembles; test terrain-coupled flow.

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
- … 31 more (`research finding list`)

## Failed directions (do not repeat without a new reason)

- F-005 [preliminary] Unregularised bounded least squares is unstable for layouts blind to the plumes

## Current experiments

- EXP-029 [proposed] DYN-01 Temporal event-generation study — runs 0
- EXP-016 [proposed] EXP population-aware sensor placement: area vs population vs mixed vs capped objectives (pre-registered) — runs 0

## Active plans

- PLAN-007 Dynamic Population-Weighted Sensor Deployment and Source-History Inference — 30/30 tasks done · objective: Develop and evaluate a physics-informed dynamical inverse framework for optimal fixed air-quality sensor placement, using time-series observations to infer changing local and transboundary emissions while minimizing population-weighted concentration reconstruction error under uncertain transport.
- PLAN-004 PRAVAH phase 3 HPC execution, interpretation and synthesis — 2/6 tasks done · objective: Run the deferred heavy Phase 3 experiments as one Slurm job, import results, interpret with independent review, finish synthesis
  - BLOCKED: T-033 H1 PI: set up repo on cluster (slurm/setup_env.sh), copy data/belauri/raw, submit slurm/phase3.sbatch, download results bundle — Superseded: per-experiment HPC workflow (RUN-0081, RUN-0098, RUN-0099) replaced phase3.sbatch bundle.
  - BLOCKED: T-034 H2 Register the cluster runs: research run attach per phase from hpc_manifest.json (command, times, exit code, git commit, hashes), artifacts, no rewriting of history — Superseded: cluster runs registered per experiment; no hpc_manifest.json exists.
  - BLOCKED: T-035 H3 Interpret EXP-015 stage 2 (displacement/regret vs replicate baseline, monotone in refinement, tolerance fractions) and decide the design mesh — Superseded by EXP-027 (F-057) and EXP-037; mesh stays provisional per D-006.
  - BLOCKED: T-038 H6 Phase 3 synthesis update, revised Phase 4 list, final checkpoint — Phase 3 bundle workflow (phase3.sbatch) was superseded by per-experiment HPC workflows (EXP-015/RUN-0058, EXP-027/RUN-0081, EXP-026/RUN-0098, EXP-036/RUN-0099). Non-author review of EXP-015 stage 2 yielded H1 untested. EXP-016 was superseded by EXP-026 and DYN-08. Final closeout synthesis is recorded in docs/FINAL_RESEARCH_REPORT.md and checkpoint CP-016.
- PLAN-003 PRAVAH phase 3: portability, population-aware design, numerical convergence of decisions, wind uncertainty — 7/10 tasks done · objective: Transferable framework whose sensor decisions are robust to discretisation and wind uncertainty, with population-aware design; testbed only, no empirical validation
  - BLOCKED: T-027 S2b EXP convergence of decisions: space, time, joint, scheme; physical-coordinate candidates; displacement, regret, ties vs resampling baseline — Superseded by EXP-027 (F-057, void) and EXP-037 (EXP-027b); no further PLAN-003 work.
  - BLOCKED: T-031 S5a Conditional: wind-robust and population-robust placement — Out of scope: conditional on a Belauri-scale wind effect not shown (F-029, F-034); not pursued.
  - BLOCKED: T-032 S5b Independent scientific review of all findings, Phase 3 synthesis, docs, notebooks, checkpoint — Phase 3 findings F-029..F-034 never received external non-author review, and F-035 review found H1 untested due to incommensurable baselines (superseded by EXP-027 and EXP-027b; 500 m mesh remains provisional per D-006). Acceptance criterion (reviews recorded by non-author) cannot be legitimately satisfied without external review.

## Needs attention (verification)

- [reviewer disagrees] F-035 EXP-015 stage 2: H1 untested (baseline not commensurable); 500 m provisionally admissible on narrow grounds: scientific-reviewer: needs work — H1 is UNTESTED, not supported. The mesh comparison and the resampling baseline are not commensurable: convergence_stage2.py:24
- [reviewer disagrees] F-069 EXP-036 / DYN-08 (RUN-0099): STOPPED at registered stop rules 4.5(a) and (b); no H8 verdict; descriptive results only: scientific-reviewer: needs work — Review of RUN-0099 as a confirmatory test of H8: all numbers reproduce (max abs diff 2.8e-17); no pipeline bug behind the cont
- [awaiting review] F-077 EXP-027b post-hoc sensitivity (RUN-0102): the INADEQUATE call is method-dependent; 500 m shape error is about 4-12% at noise 0.05 (exploratory): Agent finding awaiting an independent review
- [awaiting review] F-076 EXP-036 / DYN-08 v2 (RUN-0101): surrogate-designed layouts beat random layouts at Belauri scale; static design indistinguishable; H8 supported under the D-026 counting rule, with post-hoc amendments (replaces F-074): Agent finding awaiting an independent review
- [awaiting review] F-075 EXP-027b (EXP-037; RUN-0096, RUN-0100): registered outcome INADEQUATE (rule-based, borderline) for the 500 m / dt 100 mesh; gate passed; replaces F-073: Agent finding awaiting an independent review

## Active decisions

- D-029 (2026-10-15) DYN-08 v2 (EXP-036, corrected rerun) registration D-029, filed BEFORE any v2 run and with no v2 test result seen. v1 (RUN-0099, F-069) stopped at stop rules 4.5(a),(b); the review (F-069/R1) found the controls badly constructed, not a pipeline bug. v1 stays reported; v2 is a labelled POST-HOC-CORRECTED confirmatory rerun, NOT a rescue of v1. Changes: (1) control (a'): replace lowpop by a layout that is bad BY CONSTRUCTION: the worst-surrogate-risk subset among a fixed pool (the 16 existing layouts + 500 random subsets, pool seed 41300), required to have surrogate risk >= 1.5x the median of the pool and checked at the design stage before tuning; it must lose to spread by the registered effect criterion in >=3 of 4 cells, else stop. (2) control (b'): 10 vs 10 random layouts (fresh seeds 41200+) assessed with a TWO-LEVEL (layout x episode) bootstrap; the effect criterion must NOT be met in any cell, else stop. (3) primary claim dyn_opt vs random uses the two-level bootstrap at level 0.0025; contrasts against single baseline layouts (spread, highpop, static_opt, src_opt) are worded as statements about that layout. (4) CT/CAUSAL bin 0 (no observations precede t=7200 s) is filled with the bin-1 estimate and flagged acausal for BOTH arms; H8c stays exploratory. (5) model error: save per-episode rows and the perturbed-truth projection floors, report ratios with episode-bootstrap CIs. (6) Fresh seeds: test 44000-44191, validation 45000-45047; wind windows and design winds unchanged (selection seed 41100); disclosure: the v1 test episodes were viewed, the windows and design layouts are therefore not fresh, only the episodes/noise are. (7) Unchanged: 500 m/dt100 provisional, 54 sites, k in {3,5}, noise 0.02/0.10, tuning on validation with spread only, effect criterion, H8 verdict rule of D-026, 25% edge gate, all other stop rules. Disclosure: M2 tuning grid is larger than M1's (favours M2 slightly).
- D-028 (2026-10-15) EXP-026 (RUN-0097 failed, RUN-0098 rerun): the registered Phase 2 regression check (area greedy layouts == results/geodesign/layouts.csv) is skipped for EXP-026 via configs/population_belauri_hpc.json (frozen.regression_layouts_file=null). Diagnosis (before any test case existed): the check compares layouts selected under different noise scales: EXP-026 uses the validation-ensemble scale (noise_levels.csv validation row, sigma 0.012548 at fraction 0.05; the G2 fix of commit 0370f3e) while results/geodesign/layouts.csv used the design scale (0.015370), so area-greedy layouts legitimately differ at budget 2 (int_c007014+int_c013008 vs int_c013008+int_c009014). It is not an architecture effect (D-005). The check therefore tests a different protocol than the one registered; it is not loosened, it is not applicable. Retained correctness guard: evaluators verified against design.score_layout / greedy_trace (printed in the run log). Disclosure: this is a deviation from the inherited EXP-016 sanity stop, decided after seeing the failure but before any test data; EXP-026 findings must state it.

## Blockers

- [checkpoint] CP-016: No empirical air quality data; 500 m / dt 100 mesh remains provisional; Q-013 unmeasured replicate spread at Belauri scale; terrain-coupled flow absent; second-round non-author peer reviews remain external debt before manuscript submission.

## Project skills

- 9 more skills: `research skills list`, load one with `research skill show NAME`

## Agent policy

- max runs without synthesis: 5
- max failed runs without review: 3
- synthesis required before new experiment: True
- findings: agents create them preliminary; an independent review (`research finding review`) makes them supported
- tasks: `research task done` runs the task's checks; failures block completion
- Protocol: see AGENTS.md

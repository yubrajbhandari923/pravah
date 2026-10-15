# Research context — pravah

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~2972 tokens.
> Generated 2026-10-15T14:09:58-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Simulate the air pollution field evolution

Git: `master` @ `43720e68f5` (uncommitted changes)

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
- Q-024 [investigating] RQ-D9: Do conclusions transfer from toy geometry to Belauri-like population geography?
- Q-017 [investigating] RQ-D2: When does historical information improve source identifiability relative to independent-time or static inference?
- … 12 more (`research question list`)

## Established findings (what we believe)

- F-051 [supported, high] DYN-00 (RUN-0091, 25/25 pre-set checks pass; supersedes F-038 after its independent review F-038/R1): includes the replacement for the vacuous west_no_import check: a constant west inflow basis through a 3600 s wind reversal has cell-summed supply 3.36e-2 (>0) before and exactly 0.0 after, an east basis exactly 0.0 before and 3.36e-2 after, mass closure 2.1e-15 with both on; a non-trivial piecewise history [[0,3,1,2],[1,0,2,4],[0.5,1.5,0,3]] matches direct simulation to 1.2e-15; all other values as in F-038 (mass closure <=4.4e-15 in five dynamic cases, superposition 1.6e-15, closed-box analytic order 0.997, footprints integrate to 1). Verified numerical behaviour of the implicit-Euler/exponential scheme on a 16x12 toy box; not verified: Van Leer scheme, Belauri-scale runs, spatial order beyond monotone convergence. — evidence: EXP-028, RUN-0091
- F-009 [supported, high] Solver routing refactor preserves every deterministic FiPy output bit-for-bit — evidence: EXP-008, RUN-0012, RUN-0013
- F-007 [supported, medium] FINAL: hidden wind error and unmodelled source dominate discretisation mismatch; poor layout unidentifiable — evidence: EXP-007, RUN-0011
- F-006 [supported, medium] FINAL: greedy layouts beat spread/random on fresh test scenarios with alpha frozen on validation (matched, synthetic) — evidence: EXP-005, EXP-006, RUN-0008, RUN-0009
- F-002 [supported, high] Matched synthetic recovery of three amplitudes works; cached and direct routes agree — evidence: EXP-002, RUN-0003
- … 22 more (`research finding list`)

## Failed directions (do not repeat without a new reason)

- F-005 [preliminary] Unregularised bounded least squares is unstable for layouts blind to the plumes

## Current experiments

- EXP-037 [running] EXP-027b mesh adequacy on common cases (same-ensemble shape error, 2 km falsification gate) — runs 1
- EXP-026 [needs_review] EXP-016v population-aware sensor placement at Belauri scale, with the WorldPop fixture and replicated fixture draws (pre-registered variant) — runs 1, ⚠ 1 unsynthesized
- EXP-036 [needs_review] DYN-08 Belauri synthetic population-weighted deployment — runs 1, ⚠ 1 unsynthesized
- EXP-015 [needs_review] EXP convergence of sensor-design decisions with mesh, time step and scheme (pre-registered) — runs 2
- EXP-023 [needs_review] EXP toy wind-error controls from F-029 review (alpha, offset prior grid, model error, base wind) (pre-registered) — runs 12

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

- EXP-026 EXP-016v population-aware sensor placement at Belauri scale, with the WorldPop fixture and replicated fixture draws (pre-registered variant) (1 unsynthesized runs)
- EXP-036 DYN-08 Belauri synthetic population-weighted deployment (1 unsynthesized runs)

## Active decisions

- D-028 (2026-10-15) EXP-026 (RUN-0097 failed, RUN-0098 rerun): the registered Phase 2 regression check (area greedy layouts == results/geodesign/layouts.csv) is skipped for EXP-026 via configs/population_belauri_hpc.json (frozen.regression_layouts_file=null). Diagnosis (before any test case existed): the check compares layouts selected under different noise scales: EXP-026 uses the validation-ensemble scale (noise_levels.csv validation row, sigma 0.012548 at fraction 0.05; the G2 fix of commit 0370f3e) while results/geodesign/layouts.csv used the design scale (0.015370), so area-greedy layouts legitimately differ at budget 2 (int_c007014+int_c013008 vs int_c013008+int_c009014). It is not an architecture effect (D-005). The check therefore tests a different protocol than the one registered; it is not loosened, it is not applicable. Retained correctness guard: evaluators verified against design.score_layout / greedy_trace (printed in the run log). Disclosure: this is a deviation from the inherited EXP-016 sanity stop, decided after seeing the failure but before any test data; EXP-026 findings must state it.
- D-027 (2026-10-15) EXP-026 registration amendment A-1 (filed before any EXP-026 run): (1) worldpop_study_total 184030 is the whole 8 km-buffered grid total; registered quantities are worldpop_grid_total=184029.57 and worldpop_study_total=54370.05 (500 m; 54562.52 at 250 m); (2) stop condition becomes grid totals agree to 1e-6 relative, study totals expected 0.35% apart; (3) mesh gate becomes '500 m dt 100 PROVISIONAL only (D-006; F-057)'; budgets [2,3,4,5,8] as in parameters; (4) seeds unchanged (101, 202, 305, noise 3500, fixture 6501-6506); (5) EXP-026 uses the standard 500 m grid, DYN-08 the padded one; layouts exchanged via UTM site coordinates; only worldpop_real layouts are imported into DYN-08 as a labelled static baseline. Full text: docs/dyn08_plan.md section 5.
- D-026 (2026-10-15) DYN-08 (EXP-036) registration amendment A-1, filed BEFORE any DYN-08 layout is designed. Replaces the registered seeds (5000-5999/6000-6999/7000-7999/8000-8999, which overlap spent seeds) and the stop 'gated on EXP-027' (EXP-027 is void). Full text: docs/dyn08_plan.md section 4 (commit 401f02e), with these choices: reduced design 500 m padded grid dt 100, 12/6/12 ERA5 winds, 192 test episodes, k in {3,5}, noise 0.02/0.10, 54 interior sites; seeds wind selection 41100, random layouts 41200+, validation 46000-46047, test 43000-43191, 44000-45999 reserved, dev >=49000; ERA5 speed clip 2.5 m/s; noise scale = mean at the 54 candidate sites; mesh gate relaxed to '500 m/dt100 PROVISIONAL, mesh250 model-error proxy only'; H8 verdict rule: arms H8a, H8b, H8d count (H8c history arm EXPLORATORY because hyperparameters are borrowed): supported if >=2 of 3 transfer and none reverses with the criterion met, not supported if >=2 fail or a key arm reverses, else unresolved; H8c prediction text 'gain expected from plume carry-over' is WITHDRAWN (F-065: carry-over untested, CAUSAL-vs-CT compares time-varying vs causal static fit). Effect criterion, 20 primary contrasts at 0.0025, controls (lowpop, null replicates), stop rules and 'what this does not say' as in section 4. Stop rule: if the lowpop control fails or >25% of tuned hyperparameters sit on a grid edge, stop and report.

## Blockers

- [run] RUN-0097: RUN-0097 (EXP-026) failed — not yet reviewed

## Project skills

- 9 more skills: `research skills list`, load one with `research skill show NAME`

## Agent policy

- max runs without synthesis: 5
- max failed runs without review: 3
- synthesis required before new experiment: True
- findings: agents create them preliminary; an independent review (`research finding review`) makes them supported
- tasks: `research task done` runs the task's checks; failures block completion
- Protocol: see AGENTS.md

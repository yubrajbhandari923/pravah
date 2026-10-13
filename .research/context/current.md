# Research context — pravah

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~2599 tokens.
> Generated 2026-10-12T23:37:40-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Simulate the air pollution field evolution

Git: `master` @ `d395b78e04`

## Latest checkpoint

**CP-007** Checkpoint — 2026-10-12 (2026-10-12)
- **Understanding:** - Solver routing refactor preserves every deterministic FiPy output bit-for-bit (F-009, supported)
  - FINAL: hidden wind error and unmodelled source dominate discretisation mismatch; poor layout unidentifiable (F-007, supported)
  - FINAL: greedy layouts beat spread/random on fresh test scenarios with alpha frozen on validation (matched, synthetic) (F-006, supported)
  - Matched synthetic recovery of three amplitudes works; cached and direct routes agree (F-002, supported)
  - EXP-015 stage 2: H1 untested (baseline not commensurable); 500 m provisionally admissible on narrow grounds (F-035, preliminary)
  - Toy wind-error controls (reviewed): wind-error penalty and C's gain survive alpha and off-member priors, but C's gain shrinks with distance to the nearest prior member; chi2 detection power is NOT an inverse-crime artefact (specificity and coverage are); base-wind rotation controls are confounded (F-034, preliminary)
  - Toy: greedy layouts cost test error at 2-4 sensors (area RMSE 3.3x / 6.8x / 2.5x the best-found layout at 2 / 3 / 4 sensors, ~1.1x at 5-8); a large design-objective gap carries over to test; uniform stop check passes (F-032, preliminary)
  - Toy prescribed winds, converged domain: for T = 12 h an outer half-width of ~UT/2 (40 km here) suffices (<1% exposure); smaller domains cut off the return of the plume; channel and vortex effects depend on placement; returned mass is good to ~10% (F-031, preliminary)
- **Next step:** EXP-027 — EXP-015v2 decision stability with a COMMENSURABLE paired baseline (pre-registered fix)

## Current baseline

EXP-002 — EXP inverse recovery (frozen reference)

## Active & open questions

- Q-010 [investigating] Q-C: How do prescribed spatiotemporal winds (idealised ridge/reversal) change plume arrival, boundary flux and local/external attribution?
- Q-012 [investigating] Q-E: How does wind error limit source inference (known / wrong-fixed / uncertain-wind estimators)?
- Q-009 [investigating] Q-B: When does population-weighted sensor placement help, and what does it cost region-wide and in attribution?
- Q-011 [investigating] Q-D: Do inverse conclusions and sensor selections converge with mesh, dt and scheme?
- Q-013 [open] Q-F: Is the sensor-design objective identifiable at all with 6 design cases? EXP-015 stage 2 measured eps_sd/J_ref = 1.01-2.52 at noise 0.05, i.e. the replicate-to-replicate spread of the design objective is comparable to or larger than the objective itself. If the objective cannot be estimated from 6 cases, then which layout greedy selects at small budgets is arbitrary, and every small-budget conclusion in EXP-013, EXP-014, EXP-020, EXP-026 and EXP-015 is affected. How many design cases are needed, and does the answer change the published small-budget rankings?
- … 1 more (`research question list`)

## Established findings (what we believe)

- F-009 [supported, high] Solver routing refactor preserves every deterministic FiPy output bit-for-bit — evidence: EXP-008, RUN-0012, RUN-0013
- F-007 [supported, medium] FINAL: hidden wind error and unmodelled source dominate discretisation mismatch; poor layout unidentifiable — evidence: EXP-007, RUN-0011
- F-006 [supported, medium] FINAL: greedy layouts beat spread/random on fresh test scenarios with alpha frozen on validation (matched, synthetic) — evidence: EXP-005, EXP-006, RUN-0008, RUN-0009
- F-002 [supported, high] Matched synthetic recovery of three amplitudes works; cached and direct routes agree — evidence: EXP-002, RUN-0003
- F-035 [preliminary, high] EXP-015 stage 2: H1 untested (baseline not commensurable); 500 m provisionally admissible on narrow grounds — evidence: EXP-015, RUN-0058
- … 13 more (`research finding list`)

## Failed directions (do not repeat without a new reason)

- F-005 [preliminary] Unregularised bounded least squares is unstable for layouts blind to the plumes

## Current experiments

- EXP-015 [needs_review] EXP convergence of sensor-design decisions with mesh, time step and scheme (pre-registered) — runs 2
- EXP-023 [needs_review] EXP toy wind-error controls from F-029 review (alpha, offset prior grid, model error, base wind) (pre-registered) — runs 12
- EXP-021 [needs_review] EXP toy prescribed winds, converged reference domain and deflecting channel (pre-registered) — runs 1
- EXP-020 [needs_review] EXP toy population design with exact selection and replicated settings (pre-registered) — runs 1
- EXP-027 [proposed] EXP-015v2 decision stability with a COMMENSURABLE paired baseline (pre-registered fix) — runs 0

## Active plans

- PLAN-004 PRAVAH phase 3 HPC execution, interpretation and synthesis — 1/6 tasks done · objective: Run the deferred heavy Phase 3 experiments as one Slurm job, import results, interpret with independent review, finish synthesis
  - next ready: T-033 H1 PI: set up repo on cluster (slurm/setup_env.sh), copy data/belauri/raw, submit slurm/phase3.sbatch, download results bundle (`research task context T-033`)
- PLAN-003 PRAVAH phase 3: portability, population-aware design, numerical convergence of decisions, wind uncertainty — 6/10 tasks done · objective: Transferable framework whose sensor decisions are robust to discretisation and wind uncertainty, with population-aware design; testbed only, no empirical validation
  - next ready: T-028 S3 EXP population-aware design: area vs pop vs mixed beta vs capped on 5 fixtures; Pareto; held-out (`research task context T-028`)
  - BLOCKED: T-027 S2b EXP convergence of decisions: space, time, joint, scheme; physical-coordinate candidates; displacement, regret, ties vs resampling baseline — Runs as PLAN-004 H1-H3 (EXP-015 stage 2) on HPC
  - BLOCKED: T-031 S5a Conditional: wind-robust and population-robust placement — Removed from scope for now: conditional on a Belauri-scale wind effect; revisit after PLAN-004 H3/H4 (toy shows wind error dominates sensor count for >=10 deg, F-029)
- PLAN-002 PRAVAH phase 2: solver interchange, external/local, Belauri geography, geographic design — 11/11 tasks done · objective: Solver interchangeability; synthetic external vs local identifiability + indoor box; real Belauri geography and forward sims; geographic sensor design

## Needs attention (verification)

- [reviewer disagrees] F-035 EXP-015 stage 2: H1 untested (baseline not commensurable); 500 m provisionally admissible on narrow grounds: scientific-reviewer: needs work — H1 is UNTESTED, not supported. The mesh comparison and the resampling baseline are not commensurable: convergence_stage2.py:24
- [awaiting review] F-034 Toy wind-error controls (reviewed): wind-error penalty and C's gain survive alpha and off-member priors, but C's gain shrinks with distance to the nearest prior member; chi2 detection power is NOT an inverse-crime artefact (specificity and coverage are); base-wind rotation controls are confounded: Agent finding awaiting an independent review
- [awaiting review] F-032 Toy: greedy layouts cost test error at 2-4 sensors (area RMSE 3.3x / 6.8x / 2.5x the best-found layout at 2 / 3 / 4 sensors, ~1.1x at 5-8); a large design-objective gap carries over to test; uniform stop check passes: Agent finding awaiting an independent review
- [awaiting review] F-031 Toy prescribed winds, converged domain: for T = 12 h an outer half-width of ~UT/2 (40 km here) suffices (<1% exposure); smaller domains cut off the return of the plume; channel and vortex effects depend on placement; returned mass is good to ~10%: Agent finding awaiting an independent review

## ⚠ Needs synthesis before new experiments

- EXP-015 EXP convergence of sensor-design decisions with mesh, time step and scheme (pre-registered) (0 unsynthesized runs)
- EXP-020 EXP toy population design with exact selection and replicated settings (pre-registered) (0 unsynthesized runs)
- EXP-021 EXP toy prescribed winds, converged reference domain and deflecting channel (pre-registered) (0 unsynthesized runs)
- EXP-023 EXP toy wind-error controls from F-029 review (alpha, offset prior grid, model error, base wind) (pre-registered) (0 unsynthesized runs)

## Active decisions

- D-007 (2026-10-12) PRAVAH pre-migration, pre-refactor baseline is frozen at git d395b78 (tag baseline/pre-refactor-2026-10). This is a WORKING ENGINEERING BASELINE, not a scientifically validated model: the solver, inverse, design and HPC pipelines run reproducibly and are regression-locked, but every quantitative atmospheric result remains synthetic and only 4 of 35 findings (F-002, F-006, F-007, F-009) have passed independent review. No empirical validation exists. Future refactors must preserve the invariants listed in this decision.
- D-006 (2026-10-12) EXP-015's acceptance criterion is amended POST HOC (dated 2026-10-12, after results were seen) from the two-sided 'within the replicate-pair IQR' to the one-sided 'no larger than the replicate-pair distribution' that its own hypothesis states; and EXP-015 stage 2 is recorded as NOT having tested H1, because its baseline is not commensurable with its mesh comparison. 500 m / dt 100 is provisionally admissible for EXP-026 on the narrow grounds in F-035 - no evidence against, not shown adequate.
- D-005 (2026-10-12) FiPy regression gate is architecture-scoped: the 21 files / 0 differ requirement is binding WITHIN an architecture; the x86_64 cluster gets its own committed baseline and the arm64 baseline stays authoritative for laptop work. Cross-architecture bit-identity is not achievable and is not required. No tolerance was loosened.

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

# Research context — pravah

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~2908 tokens.
> Generated 2026-10-12T18:02:39-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Simulate the air pollution field evolution

Git: `master` @ `ebc0e1963c` (uncommitted changes)

## Latest checkpoint

**CP-006** Laptop side closed; waiting on HPC job (2026-10-11)
- **Understanding:** Unchanged from CP-005 (toy-scale Phase 3, all synthetic). Since then: EXP-022 (F-032) and EXP-023 (F-034) synthesized; real inputs recorded (WorldPop CC BY with sha256; ERA5 2023 10 m wind with DOI, sha256, licence). Gate: 581 tests pass, pyright 0 errors, FiPy regression 21 files / 0 differ.
- **Current problem:** EXP-015 stage 2 and Belauri-scale EXP-016 not run (need the PI's Slurm job); F-031/F-032/F-034 await non-author review
- **Next step:** PI runs slurm/phase3.sbatch per docs/hpc-runbook.md; then T-034..T-036, T-038

## Current baseline

EXP-002 — EXP inverse recovery (frozen reference)

## Active & open questions

- Q-010 [investigating] Q-C: How do prescribed spatiotemporal winds (idealised ridge/reversal) change plume arrival, boundary flux and local/external attribution?
- Q-012 [investigating] Q-E: How does wind error limit source inference (known / wrong-fixed / uncertain-wind estimators)?
- Q-009 [investigating] Q-B: When does population-weighted sensor placement help, and what does it cost region-wide and in attribution?
- Q-011 [investigating] Q-D: Do inverse conclusions and sensor selections converge with mesh, dt and scheme?
- Q-008 [open] Q-A: Can a new region run the same pipeline by changing only config + assets?

## Established findings (what we believe)

- F-009 [supported, high] Solver routing refactor preserves every deterministic FiPy output bit-for-bit — evidence: EXP-008, RUN-0012, RUN-0013
- F-007 [supported, medium] FINAL: hidden wind error and unmodelled source dominate discretisation mismatch; poor layout unidentifiable — evidence: EXP-007, RUN-0011
- F-006 [supported, medium] FINAL: greedy layouts beat spread/random on fresh test scenarios with alpha frozen on validation (matched, synthetic) — evidence: EXP-005, EXP-006, RUN-0008, RUN-0009
- F-002 [supported, high] Matched synthetic recovery of three amplitudes works; cached and direct routes agree — evidence: EXP-002, RUN-0003
- F-034 [preliminary, medium] Toy wind-error controls (reviewed): wind-error penalty and C's gain survive alpha and off-member priors, but C's gain shrinks with distance to the nearest prior member; chi2 detection power is NOT an inverse-crime artefact (specificity and coverage are); base-wind rotation controls are confounded — evidence: EXP-023
- F-032 [preliminary, medium] Toy: greedy layouts cost test error at 2-4 sensors (area RMSE 3.3x / 6.8x / 2.5x the best-found layout at 2 / 3 / 4 sensors, ~1.1x at 5-8); a large design-objective gap carries over to test; uniform stop check passes — evidence: EXP-022, RUN-0035
- F-031 [preliminary, medium] Toy prescribed winds, converged domain: for T = 12 h an outer half-width of ~UT/2 (40 km here) suffices (<1% exposure); smaller domains cut off the return of the plume; channel and vortex effects depend on placement; returned mass is good to ~10% — evidence: EXP-021, RUN-0034
- F-030 [preliminary, medium] Toy population design with exact selection: greedy is far from optimal at 2-4 sensors; population weighting helps test population RMSE only at 2 sensors on concentrated fixtures and then costs 6-35x area RMSE; at 8 sensors it overfits the design cases — evidence: EXP-020, RUN-0033
- F-029 [preliminary, medium] Toy wind-error response (corrected after review): direction error raises nominal-wind error ~3-4x at 30 deg and exceeds the sensor-count effect for >=10 deg; chi2/dof detects it; ensemble helps only when the truth is near a prior member — evidence: EXP-019, RUN-0031, RUN-0032
- F-025 [preliminary, medium] Belauri field-level convergence (corrected after review): 500 m exponential order ~1.0-1.2, true errors larger than the stated lower bounds, K_eff/K 6.3/3.5/2.3 — evidence: EXP-015, RUN-0028
- F-026 [preliminary, low] Toy population-aware design (corrected after review): no objective effect established; greedy path-dependence and design-sample noise are the same size as the objective differences at budgets 2-3 — evidence: EXP-017, RUN-0030
- F-021 [preliminary, medium] Geographic design robustness: an unknown wind error dominates the sensor-count effect; the identifiability flags cannot show it (saturated by construction); ranking changes are not resolved against noise (synthetic) — evidence: EXP-014, RUN-0027
- F-020 [preliminary, medium] Geographic sensor design on Belauri: small-budget rankings depend on the noise level and on the design noise used for selection; reference-site use depends on it too (synthetic, matched) — evidence: EXP-013, RUN-0026
- F-016 [preliminary, high] Forward scheme: mass-consistent, first order in time, observed spatial order ~1.4 in the periodic test (~2.2-2.7 in the steady strip), large numerical diffusion — evidence: EXP-001, RUN-0001
- F-017 [preliminary, medium] Matched-design layouts are not robust at 2-3 sensors under model mismatch, but per-case counts are mixed (v0 mismatch, synthetic) — evidence: EXP-007, RUN-0011
- … 2 more (`research finding list`)

## Failed directions (do not repeat without a new reason)

- F-005 [preliminary] Unregularised bounded least squares is unstable for layouts blind to the plumes

## Current experiments

- EXP-025 [needs_review] INFRA cluster environment build and production preflight (not a scientific experiment) — runs 3, ⚠ 3 unsynthesized
- EXP-023 [needs_review] EXP toy wind-error controls from F-029 review (alpha, offset prior grid, model error, base wind) (pre-registered) — runs 12
- EXP-021 [needs_review] EXP toy prescribed winds, converged reference domain and deflecting channel (pre-registered) — runs 1
- EXP-020 [needs_review] EXP toy population design with exact selection and replicated settings (pre-registered) — runs 1
- EXP-015 [needs_review] EXP convergence of sensor-design decisions with mesh, time step and scheme (pre-registered) — runs 1
- EXP-016 [proposed] EXP population-aware sensor placement: area vs population vs mixed vs capped objectives (pre-registered) — runs 0

## Active plans

- PLAN-004 PRAVAH phase 3 HPC execution, interpretation and synthesis — 1/6 tasks done · objective: Run the deferred heavy Phase 3 experiments as one Slurm job, import results, interpret with independent review, finish synthesis
  - next ready: T-033 H1 PI: set up repo on cluster (slurm/setup_env.sh), copy data/belauri/raw, submit slurm/phase3.sbatch, download results bundle (`research task context T-033`)
- PLAN-003 PRAVAH phase 3: portability, population-aware design, numerical convergence of decisions, wind uncertainty — 6/10 tasks done · objective: Transferable framework whose sensor decisions are robust to discretisation and wind uncertainty, with population-aware design; testbed only, no empirical validation
  - next ready: T-028 S3 EXP population-aware design: area vs pop vs mixed beta vs capped on 5 fixtures; Pareto; held-out (`research task context T-028`)
  - BLOCKED: T-027 S2b EXP convergence of decisions: space, time, joint, scheme; physical-coordinate candidates; displacement, regret, ties vs resampling baseline — Runs as PLAN-004 H1-H3 (EXP-015 stage 2) on HPC
  - BLOCKED: T-031 S5a Conditional: wind-robust and population-robust placement — Removed from scope for now: conditional on a Belauri-scale wind effect; revisit after PLAN-004 H3/H4 (toy shows wind error dominates sensor count for >=10 deg, F-029)
- PLAN-002 PRAVAH phase 2: solver interchange, external/local, Belauri geography, geographic design — 11/11 tasks done · objective: Solver interchangeability; synthetic external vs local identifiability + indoor box; real Belauri geography and forward sims; geographic sensor design

## Needs attention (verification)

- [awaiting review] F-034 Toy wind-error controls (reviewed): wind-error penalty and C's gain survive alpha and off-member priors, but C's gain shrinks with distance to the nearest prior member; chi2 detection power is NOT an inverse-crime artefact (specificity and coverage are); base-wind rotation controls are confounded: Agent finding awaiting an independent review
- [awaiting review] F-032 Toy: greedy layouts cost test error at 2-4 sensors (area RMSE 3.3x / 6.8x / 2.5x the best-found layout at 2 / 3 / 4 sensors, ~1.1x at 5-8); a large design-objective gap carries over to test; uniform stop check passes: Agent finding awaiting an independent review
- [awaiting review] F-031 Toy prescribed winds, converged domain: for T = 12 h an outer half-width of ~UT/2 (40 km here) suffices (<1% exposure); smaller domains cut off the return of the plume; channel and vortex effects depend on placement; returned mass is good to ~10%: Agent finding awaiting an independent review

## ⚠ Needs synthesis before new experiments

- EXP-015 EXP convergence of sensor-design decisions with mesh, time step and scheme (pre-registered) (0 unsynthesized runs)
- EXP-020 EXP toy population design with exact selection and replicated settings (pre-registered) (0 unsynthesized runs)
- EXP-021 EXP toy prescribed winds, converged reference domain and deflecting channel (pre-registered) (0 unsynthesized runs)
- EXP-023 EXP toy wind-error controls from F-029 review (alpha, offset prior grid, model error, base wind) (pre-registered) (0 unsynthesized runs)
- EXP-025 INFRA cluster environment build and production preflight (not a scientific experiment) (3 unsynthesized runs)

## Active decisions

- D-005 (2026-10-12) FiPy regression gate is architecture-scoped: the 21 files / 0 differ requirement is binding WITHIN an architecture; the x86_64 cluster gets its own committed baseline and the arm64 baseline stays authoritative for laptop work. Cross-architecture bit-identity is not achievable and is not required. No tolerance was loosened.
- D-004 (2026-10-11) Phase 3 baseline and deferrals
- D-003 (2026-10-10) Phase 2 baseline: solver interface (results/solver-regression), external/local study (results/transboundary-rerun), verified Belauri geography (data/belauri/manifest.json, results/geography), Belauri forward (results/belauri-forward), geographic design and robustness (results/geodesign, results/geodesign-robust) — all synthetic emissions/meteorology; v0 baseline D-002 stands
- D-002 (2026-10-10) Baseline v0: results/reference-v0 + results/inverse-v0 (configs/reference.json); comparison evidence is results/alpha-selection, results/benchmark-final, results/mismatch-final (alpha=0.1 frozen on validation, fresh test seeds 33/7000). results/benchmark, benchmark-unregularized, mismatch are exploratory and superseded.

## Blockers

- [checkpoint] CP-006: EXP-015 stage 2 and Belauri-scale EXP-016 not run (need the PI's Slurm job); F-031/F-032/F-034 await non-author review
- [run] RUN-0053: RUN-0053 (EXP-025) failed — not yet reviewed
- [run] RUN-0052: RUN-0052 (EXP-025) failed — not yet reviewed

## Project skills

- 9 more skills: `research skills list`, load one with `research skill show NAME`

## Agent policy

- max runs without synthesis: 5
- max failed runs without review: 3
- synthesis required before new experiment: True
- findings: agents create them preliminary; an independent review (`research finding review`) makes them supported
- tasks: `research task done` runs the task's checks; failures block completion
- Protocol: see AGENTS.md

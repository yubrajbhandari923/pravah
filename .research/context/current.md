# Research context — pravah

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~2643 tokens.
> Generated 2026-10-11T10:29:13-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Simulate the air pollution field evolution

Git: `master` @ `13bbc49053`

## Latest checkpoint

**CP-004** Phase 2 wrap-up (reviewed) (2026-10-10)
- **Understanding:** - Solver routing refactor preserves every deterministic FiPy output bit-for-bit (F-009, supported)
  - FINAL: hidden wind error and unmodelled source dominate discretisation mismatch; poor layout unidentifiable (F-007, supported)
  - FINAL: greedy layouts beat spread/random on fresh test scenarios with alpha frozen on validation (matched, synthetic) (F-006, supported)
  - Matched synthetic recovery of three amplitudes works; cached and direct routes agree (F-002, supported)
  - Geographic design robustness: an unknown wind error dominates the sensor-count effect; the identifiability flags cannot show it (saturated by construction); ranking changes are not resolved against noise (synthetic) (F-021, preliminary)
  - Geographic sensor design on Belauri: small-budget rankings depend on the noise level and on the design noise used for selection; reference-site use depends on it too (synthetic, matched) (F-020, preliminary)
  - Forward scheme: mass-consistent, first order in time, observed spatial order ~1.4 in the periodic test (~2.2-2.7 in the steady strip), large numerical diffusion (F-016, preliminary)
  - Matched-design layouts are not robust at 2-3 sensors under model mismatch, but per-case counts are mixed (v0 mismatch, synthetic) (F-017, preliminary)
- **Next step:** Optional, only if the sensor-design conclusions matter: add a chi-square-per-dof model check to geodesign_robust and rerun geodesign + geodesign_robust with the test noise level fixed from the validation scale (about 1 h compute). Otherwise: decide whether to commit or clear the executed notebook outputs in the working tree (user-owned); apply or drop git stash G2-unshipped.

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
- F-029 [preliminary, medium] Toy wind-error response (corrected after review): direction error raises nominal-wind error ~3-4x at 30 deg and exceeds the sensor-count effect for >=10 deg; chi2/dof detects it; ensemble helps only when the truth is near a prior member — evidence: EXP-019, RUN-0031, RUN-0032
- F-025 [preliminary, medium] Belauri field-level convergence (corrected after review): 500 m exponential order ~1.0-1.2, true errors larger than the stated lower bounds, K_eff/K 6.3/3.5/2.3 — evidence: EXP-015, RUN-0028
- F-026 [preliminary, low] Toy population-aware design (corrected after review): no objective effect established; greedy path-dependence and design-sample noise are the same size as the objective differences at budgets 2-3 — evidence: EXP-017, RUN-0030
- F-027 [preliminary, low] Toy prescribed winds (corrected after review): outer-domain truncation not separated from boundary artefacts; vortex-type psi 'ridge' has no net deflection by construction; eddy mostly raises local exposure — evidence: EXP-018, RUN-0029
- F-021 [preliminary, medium] Geographic design robustness: an unknown wind error dominates the sensor-count effect; the identifiability flags cannot show it (saturated by construction); ranking changes are not resolved against noise (synthetic) — evidence: EXP-014, RUN-0027
- F-020 [preliminary, medium] Geographic sensor design on Belauri: small-budget rankings depend on the noise level and on the design noise used for selection; reference-site use depends on it too (synthetic, matched) — evidence: EXP-013, RUN-0026
- F-016 [preliminary, high] Forward scheme: mass-consistent, first order in time, observed spatial order ~1.4 in the periodic test (~2.2-2.7 in the steady strip), large numerical diffusion — evidence: EXP-001, RUN-0001
- F-017 [preliminary, medium] Matched-design layouts are not robust at 2-3 sensors under model mismatch, but per-case counts are mixed (v0 mismatch, synthetic) — evidence: EXP-007, RUN-0011
- F-018 [preliminary, high] Belauri forward: the external patch reaches the municipality whenever the wind carries it from the WSW side (initially or throughout), not under steady north/east winds; numerical diffusion dominates the nominal K (real geography, synthetic emissions/meteorology) — evidence: EXP-012, RUN-0025
- F-019 [preliminary, medium] Boundary-condition input (whole-side bases) removes part of the error from an omitted localised external plume (synthetic; corrected values) — evidence: EXP-011, RUN-0024

## Failed directions (do not repeat without a new reason)

- F-005 [preliminary] Unregularised bounded least squares is unstable for layouts blind to the plumes

## Current experiments

- EXP-021 [needs_review] EXP toy prescribed winds, converged reference domain and deflecting channel (pre-registered) — runs 1
- EXP-020 [needs_review] EXP toy population design with exact selection and replicated settings (pre-registered) — runs 1
- EXP-015 [needs_review] EXP convergence of sensor-design decisions with mesh, time step and scheme (pre-registered) — runs 1
- EXP-016 [proposed] EXP population-aware sensor placement: area vs population vs mixed vs capped objectives (pre-registered) — runs 0

## Active plans

- PLAN-003 PRAVAH phase 3: portability, population-aware design, numerical convergence of decisions, wind uncertainty — 1/10 tasks done · objective: Transferable framework whose sensor decisions are robust to discretisation and wind uncertainty, with population-aware design; testbed only, no empirical validation
  - next ready: T-025 S1c Experiment contracts: ensembles/seeds/freeze rules, cheap preflight, cache-key audit, benchmark of one Belauri solve at 500/250/125 m (`research task context T-025`)
  - running: T-024 S1b Population layer: grid-aligned loader, conservative regrid, 5 synthetic fixtures, area/population metrics, source registry (claimed by claude-code) — STALE, idle 10.6h
  - running: T-026 S2a Less-diffusive convection scheme (flux-limited) with verification: MMS/periodic, conservation, positivity, order (claimed by claude-code) — STALE, idle 10.2h
- PLAN-002 PRAVAH phase 2: solver interchange, external/local, Belauri geography, geographic design — 11/11 tasks done · objective: Solver interchangeability; synthetic external vs local identifiability + indoor box; real Belauri geography and forward sims; geographic sensor design
- PLAN-001 PRAVAH v0 baseline — 11/11 tasks done · objective: Runnable v0: forward model, inverse, sensor design, notebooks, CLI

## Needs attention (verification)

- [awaiting review] F-029 Toy wind-error response (corrected after review): direction error raises nominal-wind error ~3-4x at 30 deg and exceeds the sensor-count effect for >=10 deg; chi2/dof detects it; ensemble helps only when the truth is near a prior member: Agent finding awaiting an independent review
- [awaiting review] F-027 Toy prescribed winds (corrected after review): outer-domain truncation not separated from boundary artefacts; vortex-type psi 'ridge' has no net deflection by construction; eddy mostly raises local exposure: Agent finding awaiting an independent review
- [awaiting review] F-026 Toy population-aware design (corrected after review): no objective effect established; greedy path-dependence and design-sample noise are the same size as the objective differences at budgets 2-3: Agent finding awaiting an independent review

## ⚠ Needs synthesis before new experiments

- EXP-015 EXP convergence of sensor-design decisions with mesh, time step and scheme (pre-registered) (0 unsynthesized runs)
- EXP-020 EXP toy population design with exact selection and replicated settings (pre-registered) (0 unsynthesized runs)
- EXP-021 EXP toy prescribed winds, converged reference domain and deflecting channel (pre-registered) (0 unsynthesized runs)

## Active decisions

- D-003 (2026-10-10) Phase 2 baseline: solver interface (results/solver-regression), external/local study (results/transboundary-rerun), verified Belauri geography (data/belauri/manifest.json, results/geography), Belauri forward (results/belauri-forward), geographic design and robustness (results/geodesign, results/geodesign-robust) — all synthetic emissions/meteorology; v0 baseline D-002 stands
- D-002 (2026-10-10) Baseline v0: results/reference-v0 + results/inverse-v0 (configs/reference.json); comparison evidence is results/alpha-selection, results/benchmark-final, results/mismatch-final (alpha=0.1 frozen on validation, fresh test seeds 33/7000). results/benchmark, benchmark-unregularized, mismatch are exploratory and superseded.

## Blockers

_none_

## Project skills

- 4 more skills: `research skills list`, load one with `research skill show NAME`

## Agent policy

- max runs without synthesis: 5
- max failed runs without review: 3
- synthesis required before new experiment: True
- findings: agents create them preliminary; an independent review (`research finding review`) makes them supported
- tasks: `research task done` runs the task's checks; failures block completion
- Protocol: see AGENTS.md

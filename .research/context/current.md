# Research context — pravah

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~1465 tokens.
> Generated 2026-10-10T13:09:38-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Simulate the air pollution field evolution

Git: `master` @ `e7c6bf1746` (uncommitted changes)

## Latest checkpoint

**CP-002** Checkpoint — 2026-10-10 (2026-10-10)
- **Understanding:** - Greedy layouts optimised for the matched model are not robust to model error at small budgets (F-008, preliminary)
  - FINAL: greedy layouts beat spread/random on fresh test scenarios with alpha frozen on validation (matched, synthetic) (F-006, preliminary)
  - FINAL: hidden wind error and unmodelled source dominate discretisation mismatch; poor layout unidentifiable (F-007, preliminary)
  - Forward scheme is mass-consistent and converges but carries large numerical diffusion (F-001, preliminary)
  - Matched synthetic recovery of three amplitudes works; cached and direct routes agree (F-002, preliminary)
- **Next step:** Update findings, decision, docs; checkpoint

## Current baseline

EXP-002 — EXP inverse recovery (frozen reference)

## Active & open questions

- Q-005 [investigating] Can total synthetic concentration observations separate local from external contributions, when is the decomposition unidentifiable, and does an upwind/background site help?
- Q-004 [investigating] Is the Belauri Municipality boundary, DEM and feature set acquired from public sources verifiable, and what does it show about the surrounding terrain and national border?
- Q-003 [investigating] Does routing every workflow through one ForwardSolver interface leave all FiPy results bit-identical, and do injected solvers reach every call?
- Q-002 [investigating] Can sparse noisy sensors reconstruct the regional field, and do random, spread and greedy layouts differ on independent cases?
- Q-001 [investigating] Does the FiPy forward model reproduce analytical, conservation and refinement behaviour?

## Established findings (what we believe)

- F-009 [preliminary, high] Solver routing refactor preserves every deterministic FiPy output bit-for-bit — evidence: EXP-008, RUN-0012, RUN-0013
- F-008 [preliminary, medium] Greedy layouts optimised for the matched model are not robust to model error at small budgets — evidence: EXP-007, RUN-0011
- F-006 [preliminary, medium] FINAL: greedy layouts beat spread/random on fresh test scenarios with alpha frozen on validation (matched, synthetic) — evidence: EXP-005, EXP-006, RUN-0008, RUN-0009
- F-007 [preliminary, medium] FINAL: hidden wind error and unmodelled source dominate discretisation mismatch; poor layout unidentifiable — evidence: EXP-007, RUN-0011
- F-001 [preliminary, high] Forward scheme is mass-consistent and converges but carries large numerical diffusion — evidence: EXP-001, RUN-0001
- F-002 [preliminary, medium] Matched synthetic recovery of three amplitudes works; cached and direct routes agree — evidence: EXP-002, RUN-0003

## Failed directions (do not repeat without a new reason)

- F-005 [preliminary] Unregularised bounded least squares is unstable for layouts blind to the plumes

## Current experiments

- EXP-009 [needs_review] EXP geography acquisition and verification (Belauri) — runs 6, ⚠ 2 unsynthesized
- EXP-010 [needs_review] EXP external vs local identifiability (synthetic, domains A/B) — runs 1, ⚠ 1 unsynthesized

## Active plans

- PLAN-002 PRAVAH phase 2: solver interchange, external/local, Belauri geography, geographic design — 5/11 tasks done · objective: Solver interchangeability; synthetic external vs local identifiability + indoor box; real Belauri geography and forward sims; geographic sensor design
  - next ready: T-017 B2: generalised inverse, attribution, indoor box, notebook 04, transboundary study (`research task context T-017`)
  - running: T-018 C2: acquire + verify Belauri geography; geometry checks; docs/geography.md (claimed by claude-code)
  - running: T-020 E: Belauri forward scenarios, domain/mesh sensitivity, notebook 05 (claimed by claude-code)
- PLAN-001 PRAVAH v0 baseline — 11/11 tasks done · objective: Runnable v0: forward model, inverse, sensor design, notebooks, CLI

## Needs attention (verification)

- [awaiting review] F-009 Solver routing refactor preserves every deterministic FiPy output bit-for-bit: Agent finding awaiting an independent review
- [awaiting review] F-008 Greedy layouts optimised for the matched model are not robust to model error at small budgets: Agent finding awaiting an independent review
- [awaiting review] F-007 FINAL: hidden wind error and unmodelled source dominate discretisation mismatch; poor layout unidentifiable: Agent finding awaiting an independent review

## ⚠ Needs synthesis before new experiments

- EXP-009 EXP geography acquisition and verification (Belauri) (2 unsynthesized runs)
- EXP-010 EXP external vs local identifiability (synthetic, domains A/B) (1 unsynthesized runs)

## Active decisions

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

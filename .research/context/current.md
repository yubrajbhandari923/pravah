# Research context — pravah

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~1052 tokens.
> Generated 2026-10-10T10:39:33-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Simulate the air pollution field evolution

Git: `master` @ `dcd75e10df` (uncommitted changes)

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

- Q-002 [investigating] Can sparse noisy sensors reconstruct the regional field, and do random, spread and greedy layouts differ on independent cases?
- Q-001 [investigating] Does the FiPy forward model reproduce analytical, conservation and refinement behaviour?

## Established findings (what we believe)

- F-008 [preliminary, medium] Greedy layouts optimised for the matched model are not robust to model error at small budgets — evidence: EXP-007, RUN-0011
- F-006 [preliminary, medium] FINAL: greedy layouts beat spread/random on fresh test scenarios with alpha frozen on validation (matched, synthetic) — evidence: EXP-005, EXP-006, RUN-0008, RUN-0009
- F-007 [preliminary, medium] FINAL: hidden wind error and unmodelled source dominate discretisation mismatch; poor layout unidentifiable — evidence: EXP-007, RUN-0011
- F-001 [preliminary, high] Forward scheme is mass-consistent and converges but carries large numerical diffusion — evidence: EXP-001, RUN-0001
- F-002 [preliminary, medium] Matched synthetic recovery of three amplitudes works; cached and direct routes agree — evidence: EXP-002, RUN-0003

## Failed directions (do not repeat without a new reason)

- F-005 [preliminary] Unregularised bounded least squares is unstable for layouts blind to the plumes

## Current experiments

_none_

## Active plans

- PLAN-001 PRAVAH v0 baseline — 10/11 tasks done · objective: Runnable v0: forward model, inverse, sensor design, notebooks, CLI
  - running: T-008 G: experiments, mismatch, deployment notebook (claimed by claude-code) — STALE, idle 7.9h

## Needs attention (verification)

- [awaiting review] F-008 Greedy layouts optimised for the matched model are not robust to model error at small budgets: Agent finding awaiting an independent review
- [awaiting review] F-007 FINAL: hidden wind error and unmodelled source dominate discretisation mismatch; poor layout unidentifiable: Agent finding awaiting an independent review
- [awaiting review] F-006 FINAL: greedy layouts beat spread/random on fresh test scenarios with alpha frozen on validation (matched, synthetic): Agent finding awaiting an independent review

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

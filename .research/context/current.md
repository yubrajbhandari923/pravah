# Research context — pravah

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~951 tokens.
> Generated 2026-10-10T09:54:05-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Simulate the air pollution field evolution

Git: `master` @ `f564e25331` (uncommitted changes)

## Latest checkpoint

**CP-001** Checkpoint — 2026-10-10 (2026-10-10)
- **Understanding:** - Hidden wind error and an unmodelled source dominate over discretisation mismatch; more sensors do not fix them (F-004, preliminary)
  - Forward scheme is mass-consistent and converges but carries large numerical diffusion (F-001, preliminary)
  - Matched synthetic recovery of three amplitudes works; cached and direct routes agree (F-002, preliminary)
  - Greedy layouts selected on design scenarios beat spread and random on independent test scenarios on average (matched, synthetic) (F-003, preliminary)
- **Next step:** Final scientific review and freeze

## Current baseline

EXP-002 — EXP inverse recovery (frozen reference)

## Active & open questions

- Q-002 [investigating] Can sparse noisy sensors reconstruct the regional field, and do random, spread and greedy layouts differ on independent cases?
- Q-001 [investigating] Does the FiPy forward model reproduce analytical, conservation and refinement behaviour?

## Established findings (what we believe)

- F-004 [preliminary, medium] Hidden wind error and an unmodelled source dominate over discretisation mismatch; more sensors do not fix them — evidence: EXP-004, RUN-0007
- F-001 [preliminary, high] Forward scheme is mass-consistent and converges but carries large numerical diffusion — evidence: EXP-001, RUN-0001
- F-002 [preliminary, medium] Matched synthetic recovery of three amplitudes works; cached and direct routes agree — evidence: EXP-002, RUN-0003
- F-003 [preliminary, medium] Greedy layouts selected on design scenarios beat spread and random on independent test scenarios on average (matched, synthetic) — evidence: EXP-003, RUN-0005, RUN-0006

## Failed directions (do not repeat without a new reason)

- F-005 [preliminary] Unregularised bounded least squares is unstable for layouts blind to the plumes

## Current experiments

_none_

## Active plans

- PLAN-001 PRAVAH v0 baseline — 9/10 tasks done · objective: Runnable v0: forward model, inverse, sensor design, notebooks, CLI
  - running: T-008 G: experiments, mismatch, deployment notebook (claimed by claude-code) — STALE, idle 7.2h

## Needs attention (verification)

- [awaiting review] F-004 Hidden wind error and an unmodelled source dominate over discretisation mismatch; more sensors do not fix them: Agent finding awaiting an independent review
- [awaiting review] F-003 Greedy layouts selected on design scenarios beat spread and random on independent test scenarios on average (matched, synthetic): Agent finding awaiting an independent review
- [awaiting review] F-002 Matched synthetic recovery of three amplitudes works; cached and direct routes agree: Agent finding awaiting an independent review

## Active decisions

- D-001 (2026-10-10) Freeze results/reference-v0 and results/inverse-v0 with configs/reference.json as the PRAVAH v0 baseline; use results/benchmark and results/mismatch as the reference comparisons

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

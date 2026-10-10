# Research context — pravah

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~1676 tokens.
> Generated 2026-10-10T19:53:40-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Simulate the air pollution field evolution

Git: `master` @ `d7f3d19b14` (uncommitted changes)

## Latest checkpoint

**CP-003** Checkpoint — 2026-10-10 (2026-10-10)
- **Understanding:** - Solver routing refactor preserves every deterministic FiPy output bit-for-bit (F-009, supported)
  - FINAL: hidden wind error and unmodelled source dominate discretisation mismatch; poor layout unidentifiable (F-007, supported)
  - FINAL: greedy layouts beat spread/random on fresh test scenarios with alpha frozen on validation (matched, synthetic) (F-006, supported)
  - Matched synthetic recovery of three amplitudes works; cached and direct routes agree (F-002, supported)
  - Geographic design robustness: an unknown wind error dominates the sensor-count effect; the identifiability flags cannot show it (saturated by construction); ranking changes are not resolved against noise (synthetic) (F-021, preliminary)
  - Geographic sensor design on Belauri: small-budget rankings depend on the noise level and on the design noise used for selection; reference-site use depends on it too (synthetic, matched) (F-020, preliminary)
  - Forward scheme: mass-consistent, first order in time, observed spatial order ~1.4 in the periodic test (~2.2-2.7 in the steady strip), large numerical diffusion (F-016, preliminary)
  - Matched-design layouts are not robust at 2-3 sensors under model mismatch, but per-case counts are mixed (v0 mismatch, synthetic) (F-017, preliminary)
- **Next step:** Final scientific review of the geographic results

## Current baseline

EXP-002 — EXP inverse recovery (frozen reference)

## Active & open questions

_none_

## Established findings (what we believe)

- F-009 [supported, high] Solver routing refactor preserves every deterministic FiPy output bit-for-bit — evidence: EXP-008, RUN-0012, RUN-0013
- F-007 [supported, medium] FINAL: hidden wind error and unmodelled source dominate discretisation mismatch; poor layout unidentifiable — evidence: EXP-007, RUN-0011
- F-006 [supported, medium] FINAL: greedy layouts beat spread/random on fresh test scenarios with alpha frozen on validation (matched, synthetic) — evidence: EXP-005, EXP-006, RUN-0008, RUN-0009
- F-002 [supported, high] Matched synthetic recovery of three amplitudes works; cached and direct routes agree — evidence: EXP-002, RUN-0003
- F-021 [preliminary, medium] Geographic design robustness: an unknown wind error dominates the sensor-count effect; the identifiability flags cannot show it (saturated by construction); ranking changes are not resolved against noise (synthetic) — evidence: EXP-014, RUN-0027
- F-020 [preliminary, medium] Geographic sensor design on Belauri: small-budget rankings depend on the noise level and on the design noise used for selection; reference-site use depends on it too (synthetic, matched) — evidence: EXP-013, RUN-0026
- F-016 [preliminary, high] Forward scheme: mass-consistent, first order in time, observed spatial order ~1.4 in the periodic test (~2.2-2.7 in the steady strip), large numerical diffusion — evidence: EXP-001, RUN-0001
- F-017 [preliminary, medium] Matched-design layouts are not robust at 2-3 sensors under model mismatch, but per-case counts are mixed (v0 mismatch, synthetic) — evidence: EXP-007, RUN-0011
- F-018 [preliminary, high] Belauri forward: the external patch reaches the municipality whenever the wind carries it from the WSW side (initially or throughout), not under steady north/east winds; numerical diffusion dominates the nominal K (real geography, synthetic emissions/meteorology) — evidence: EXP-012, RUN-0025
- F-019 [preliminary, medium] Boundary-condition input (whole-side bases) removes part of the error from an omitted localised external plume (synthetic; corrected values) — evidence: EXP-011, RUN-0024

## Failed directions (do not repeat without a new reason)

- F-005 [preliminary] Unregularised bounded least squares is unstable for layouts blind to the plumes

## Current experiments

_none_

## Active plans

- PLAN-002 PRAVAH phase 2: solver interchange, external/local, Belauri geography, geographic design — 11/11 tasks done · objective: Solver interchangeability; synthetic external vs local identifiability + indoor box; real Belauri geography and forward sims; geographic sensor design
- PLAN-001 PRAVAH v0 baseline — 11/11 tasks done · objective: Runnable v0: forward model, inverse, sensor design, notebooks, CLI

## Needs attention (verification)

- [awaiting review] F-021 Geographic design robustness: an unknown wind error dominates the sensor-count effect; the identifiability flags cannot show it (saturated by construction); ranking changes are not resolved against noise (synthetic): Agent finding awaiting an independent review
- [awaiting review] F-020 Geographic sensor design on Belauri: small-budget rankings depend on the noise level and on the design noise used for selection; reference-site use depends on it too (synthetic, matched): Agent finding awaiting an independent review
- [awaiting review] F-019 Boundary-condition input (whole-side bases) removes part of the error from an omitted localised external plume (synthetic; corrected values): Agent finding awaiting an independent review

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

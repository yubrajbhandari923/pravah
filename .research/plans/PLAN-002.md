---
id: PLAN-002
title: 'PRAVAH phase 2: solver interchange, external/local, Belauri geography, geographic design'
status: active
question: null
skills: []
author_type: agent
author_name: claude-code
author_model: null
created_at: '2026-10-10T10:46:40-04:00'
updated_at: '2026-10-10T10:46:40-04:00'
completed_at: null
---

# PLAN-002 · PRAVAH phase 2: solver interchange, external/local, Belauri geography, geographic design

## Objective

Solver interchangeability; synthetic external vs local identifiability + indoor box; real Belauri geography and forward sims; geographic sensor design

## Success Criteria

docs/phase2-plan.md requirements met with executed evidence; geography blockers named

## Context

<!-- research:generated — everything below is regenerated; edits here are ignored -->

## Tasks (11/11 done)

- ✓ **T-012** A1: capture FiPy regression baseline (before edits) — done
- ✓ **T-013** A2: route every workflow through ForwardSolver; routing tests; compare_runs; docs — done (deps: T-012)
- ✓ **T-014** Contracts: config/source extensions (groups, raster, study, bases, param_table) — done (deps: T-013)
- ✓ **T-015** B1: boundary bases, stream-function wind, tag physics in forward — done (deps: T-014)
- ✓ **T-016** C1: geo fetch/prepare/regionio + GIS fixtures — done (deps: T-014)
- ✓ **T-017** B2: generalised inverse, attribution, indoor box, notebook 04, transboundary study — done (deps: T-015)
- ✓ **T-018** C2: acquire + verify Belauri geography; geometry checks; docs/geography.md — done (deps: T-016)
- ✓ **T-019** D: Opus review of external/local modelling and inference; fixes — done (deps: T-017)
- ✓ **T-020** E: Belauri forward scenarios, domain/mesh sensitivity, notebook 05 — done (deps: T-017, T-018)
- ✓ **T-021** F: geographic candidates, objectives, designs, notebook 06 — done (deps: T-020, T-019)
- ✓ **T-022** G: Opus review of geographic deployment; integration, docs, reviews, synthesis, checkpoint — done (deps: T-021)

_Author: agent (claude-code)_

---
id: PLAN-003
title: 'PRAVAH phase 3: portability, population-aware design, numerical convergence of decisions, wind
  uncertainty'
status: active
question: Q-008
skills: []
author_type: agent
author_name: claude-code
author_model: null
created_at: '2026-10-10T23:46:48-04:00'
updated_at: '2026-10-10T23:46:48-04:00'
completed_at: null
---

# PLAN-003 · PRAVAH phase 3: portability, population-aware design, numerical convergence of decisions, wind uncertainty

## Objective

Transferable framework whose sensor decisions are robust to discretisation and wind uncertainty, with population-aware design; testbed only, no empirical validation

## Success Criteria

Q-A..Q-E each answered or documented as blocked, findings independently reviewed, frozen baselines unchanged (regression gate 0 differ)

## Context

Directive in session 2026-10-10; opens from CP-004; reuses solver interface (F-009) and Belauri geography; stash G2-unshipped holds noise-sigma/chi2 fixes

<!-- research:generated — everything below is regenerated; edits here are ignored -->

## Tasks (6/10 done)

- ✓ **T-023** S1a Transfer audit + second synthetic region fixture and transfer test — done
- ✓ **T-024** S1b Population layer: grid-aligned loader, conservative regrid, 5 synthetic fixtures, area/population metrics, source registry — done
- ✓ **T-025** S1c Experiment contracts: ensembles/seeds/freeze rules, cheap preflight, cache-key audit, benchmark of one Belauri solve at 500/250/125 m — done
- ✓ **T-026** S2a Less-diffusive convection scheme (flux-limited) with verification: MMS/periodic, conservation, positivity, order — done
- ⛔ **T-027** S2b EXP convergence of decisions: space, time, joint, scheme; physical-coordinate candidates; displacement, regret, ties vs resampling baseline — blocked
- ○ **T-028** S3 EXP population-aware design: area vs pop vs mixed beta vs capped on 5 fixtures; Pareto; held-out — todo
- ✓ **T-029** S4a Wind interface audit; time/space-varying and raster wind input; idealised ridge-reversal toy; arrival/flux diagnostics; ERA5 feasibility note — done
- ✓ **T-030** S4b EXP wind-error response: direction/speed grid; strategies A known, B fixed-wrong, C ensemble-marginalised; chi2/dof; coverage — done
- ⛔ **T-031** S5a Conditional: wind-robust and population-robust placement — blocked
- ○ **T-032** S5b Independent scientific review of all findings, Phase 3 synthesis, docs, notebooks, checkpoint — todo

⚠ 2 task(s) blocked

_Author: agent (claude-code)_

---
id: PLAN-006
title: 'PRAVAH phase 3: incremental scientific-code refactor'
status: completed
question: null
skills: []
author_type: agent
author_name: claude-code
author_model: null
created_at: '2026-10-13T00:29:31-04:00'
updated_at: '2026-10-13T01:46:41-04:00'
completed_at: '2026-10-13T01:46:41-04:00'
---

# PLAN-006 · PRAVAH phase 3: incremental scientific-code refactor

## Objective

Implement docs/ARCHITECTURE_WORKFLOW_AUDIT.md section 8/9 without changing scientific behaviour; verification record docs/SCIENTIFIC_REFACTOR_VERIFICATION.md

## Success Criteria

Equivalence evidence per batch; pyright 0; full suite run once; regression gate 21/0; no historical record or HPC change

## Context

<!-- research:generated — everything below is regenerated; edits here are ignored -->

## Tasks (9/9 done)

- ✓ **T-044** B0 Baseline: capture reference outputs and characterization tests — done
- ✓ **T-045** BA Shared utilities (_say, _save_fig, sha256, atomic write) — done (deps: T-044)
- ✓ **T-046** BC Shared scientific protocol (seeds, disjoint check, freeze hash) — done (deps: T-044)
- ✓ **T-047** BB Separate study drivers into pravah.studies with shims — done (deps: T-045, T-046)
- ✓ **T-048** BD Thin script entry points (population_*, prescribed_winds*) — done (deps: T-046)
- ✓ **T-049** BE data_provenance field and config boundaries — done (deps: T-045)
- ✓ **T-050** BF Legacy cleanup (verified-dead only) — done (deps: T-045)
- ✓ **T-051** BG Docs update — done (deps: T-047)
- ✓ **T-052** BV Final verification and record — done (deps: T-048, T-049, T-050, T-051)

_Author: agent (claude-code)_

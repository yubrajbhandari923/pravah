---
id: PLAN-004
title: PRAVAH phase 3 HPC execution, interpretation and synthesis
status: active
question: Q-011
skills: []
author_type: agent
author_name: claude-code
author_model: null
created_at: '2026-10-11T11:54:31-04:00'
updated_at: '2026-10-11T11:54:31-04:00'
completed_at: null
---

# PLAN-004 · PRAVAH phase 3 HPC execution, interpretation and synthesis

## Objective

Run the deferred heavy Phase 3 experiments as one Slurm job, import results, interpret with independent review, finish synthesis

## Success Criteria

EXP-015 stage 2 and Belauri-scale EXP-016 run on the cluster, registered with research run attach, interpreted, independently reviewed; Phase 3 synthesis updated; Phase 4 list revised

## Context

No Research OS or agent on the cluster. PI copies repo, runs ONE sbatch job (slurm/phase3.sbatch -> scripts/hpc_run.py), downloads results/phase3-hpc-bundle.tar.gz and results/hpc_manifest.json; laptop agent registers runs with 'research run attach'. Read docs/hpc-runbook.md, docs/phase3-findings.md, CP-005, D-004 first. Removed from scope (deleted): 125 m design stage (Courant/dt untested, ~10 bases/h), S5a wind-robust placement (conditional on a Belauri-scale wind effect), vanleer as a design arm (nonlinear).

<!-- research:generated — everything below is regenerated; edits here are ignored -->

## Tasks (0/0 done)

_(no tasks yet)_

_Author: agent (claude-code)_

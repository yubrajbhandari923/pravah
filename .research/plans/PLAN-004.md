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

## Tasks (1/6 done)

- ○ **T-033** H1 PI: set up repo on cluster (slurm/setup_env.sh), copy data/belauri/raw, submit slurm/phase3.sbatch, download results bundle — todo
- ○ **T-034** H2 Register the cluster runs: research run attach per phase from hpc_manifest.json (command, times, exit code, git commit, hashes), artifacts, no rewriting of history — todo
- ○ **T-035** H3 Interpret EXP-015 stage 2 (displacement/regret vs replicate baseline, monotone in refinement, tolerance fractions) and decide the design mesh — todo
- ○ **T-036** H4 Interpret Belauri-scale EXP-016 (exact vs greedy on test, population/mixed/capped, cluster bootstrap, design-split optimism) — todo
- ✓ **T-037** H5 Real inputs: WorldPop Belauri extract in data/population (done on laptop) into an EXP-016 variant; ERA5 extract (PI fetches with scripts/fetch_era5.py) through winds.segments_from_series — done
- ○ **T-038** H6 Phase 3 synthesis update, revised Phase 4 list, final checkpoint — todo

_Author: agent (claude-code)_

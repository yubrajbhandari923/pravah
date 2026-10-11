---
title: 'Phase 3 handoff: HPC run, import, interpret (read this first)'
pinned: true
links: [PLAN-004, EXP-015, EXP-016]
created: '2026-10-11T11:54:51-04:00'
---

# Phase 3 handoff: HPC run, import, interpret (read this first)

State: toy-scale Phase 3 done and independently reviewed (CP-005, D-004, docs/phase3-findings.md). Heavy work is deferred to ONE Slurm job (slurm/phase3.sbatch -> scripts/hpc_run.py): EXP-015 stage 2 (decision stability vs mesh/dt with an 8-replicate baseline, parallel via --workers) and Belauri-scale EXP-016 (scripts/population_belauri.py). No Research OS runs on the cluster: the PI downloads results/phase3-hpc-bundle.tar.gz and results/hpc_manifest.json; THEN register with 'research run attach EXP-015 ...' per phase (command, times, exit code, git commit from the manifest), register artifacts, and only then interpret. Rules: findings stay preliminary until a NON-author reviews; recompute numbers from raw CSVs; every first draft in Phase 3 needed corrections (see docs/phase3-findings.md 'What reviews changed'); report cluster-bootstrap/placement-level uncertainty honestly; do not choose objectives, alpha, beta or mesh after seeing test results; Belauri site conclusions are conditional on EXP-015 stage 2. Removed from scope: 125 m design stage, S5a wind-robust placement (T-031 blocked), vanleer as design arm. Done on the laptop in this round: EXP-022 greedy-vs-exact on test, EXP-023 wind-error controls (check their status with research show), WorldPop extract (data/population), ERA5 fetch script (scripts/fetch_era5.py, PI runs it).

Update 2026-10-11 (CP-006): laptop side closed. EXP-022/023 synthesized, T-037 done (ERA5 licence added to data/era5 json), T-027 blocked (runs as PLAN-004 H1-H3). 581 tests, pyright 0, regression 0 differ. Waiting on the PI's Slurm job (T-033).

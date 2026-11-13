# Run manifests

Store compact JSON records for each actual Slurm job. Record exact source
snapshot, dataset manifest, config/seed, environment/lockfile, job ID,
output paths, scheduler outcome, and scientific interpretation separately.
Do not store raw volumes or large checkpoints in this directory.

`legacy-runs.jsonl` indexes the 24 local runs made under ResearchOS (command, commit, code snapshot ref under refs/legacy/). Their metrics and logs are in state/history/researchos-ledger.tar.gz.

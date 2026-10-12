#!/bin/bash
# RUN-0057 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_link_datasets.sh python scripts/convergence_stage2.py --config configs/convergence_stage2.json --out results/phase3-convergence --estimate

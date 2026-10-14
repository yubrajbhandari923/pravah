#!/bin/bash
# RUN-0076 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_link_datasets.sh python scripts/dyn_bench.py --workers 1 2 4 8 16 --episodes 32

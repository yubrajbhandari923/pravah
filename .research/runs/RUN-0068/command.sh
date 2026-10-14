#!/bin/bash
# RUN-0068 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_dyn_tune.sh dyn03 16

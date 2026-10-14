#!/bin/bash
# RUN-0066 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_link_datasets.sh bash scripts/hpc_regression_selfcheck.sh

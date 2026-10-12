#!/bin/bash
# RUN-0053 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_regression_selfcheck.sh

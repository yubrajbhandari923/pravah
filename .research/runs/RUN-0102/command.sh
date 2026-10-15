#!/bin/bash
# RUN-0102 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_exp027b_sens.sh 16

#!/bin/bash
# RUN-0096 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_exp027b.sh 16 convergence_paired_padded_exp027b_s1.json

#!/bin/bash
# RUN-0075 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_exp027.sh 32

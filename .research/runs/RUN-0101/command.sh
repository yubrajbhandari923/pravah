#!/bin/bash
# RUN-0101 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_dyn08.sh all 16 configs/dyn08_v2.json

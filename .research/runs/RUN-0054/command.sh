#!/bin/bash
# RUN-0054 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_preflight_prod.sh

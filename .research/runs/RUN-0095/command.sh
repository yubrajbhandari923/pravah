#!/bin/bash
# RUN-0095 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_dyn08.sh time 1

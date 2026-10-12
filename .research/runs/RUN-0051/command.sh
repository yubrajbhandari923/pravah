#!/bin/bash
# RUN-0051 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_preflight_env.sh

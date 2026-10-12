#!/bin/bash
# RUN-0050 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
python3 scripts/hpc_smoke.py --config configs/hpc_smoke.json

#!/bin/bash
# RUN-0027 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah benchmark --config configs/geodesign_robust.json --out results/geodesign-robust

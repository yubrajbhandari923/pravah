#!/bin/bash
# RUN-0024 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah benchmark --config configs/transboundary.json --out results/transboundary-rerun

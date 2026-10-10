#!/bin/bash
# RUN-0004 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah benchmark --config configs/smoke.json --out results/smoke

#!/bin/bash
# RUN-0008 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah benchmark --config configs/alpha_selection.json --out results/alpha-selection

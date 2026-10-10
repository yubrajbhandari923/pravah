#!/bin/bash
# RUN-0012 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python scripts/capture_regression.py --out results/solver-regression/before

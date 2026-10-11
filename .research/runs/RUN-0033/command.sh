#!/bin/bash
# RUN-0033 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python scripts/population_design_exact.py --profile production

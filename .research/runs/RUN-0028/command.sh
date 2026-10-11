#!/bin/bash
# RUN-0028 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python scripts/convergence_stage1.py --profile production

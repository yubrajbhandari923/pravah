#!/bin/bash
# RUN-0034 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python scripts/prescribed_winds_converged.py --profile production

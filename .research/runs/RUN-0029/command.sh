#!/bin/bash
# RUN-0029 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python scripts/prescribed_winds.py --config configs/prescribed_winds.json --profile production

#!/bin/bash
# RUN-0021 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python scripts/geo_relief_profile.py --out results/geography

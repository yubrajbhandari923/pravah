#!/bin/bash
# RUN-0023 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python scripts/geo_report.py --manifest data/belauri/manifest.json --out results/geography

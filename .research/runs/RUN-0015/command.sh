#!/bin/bash
# RUN-0015 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah geo fetch --config configs/belauri_geo.json

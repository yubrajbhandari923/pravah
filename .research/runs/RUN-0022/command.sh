#!/bin/bash
# RUN-0022 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah geo prepare --config configs/belauri_geo.json

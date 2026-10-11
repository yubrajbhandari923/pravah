#!/bin/bash
# RUN-0047 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python scripts/wind_error.py report --config configs/wind_error_ctrl_base_wind_90.json

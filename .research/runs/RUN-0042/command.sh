#!/bin/bash
# RUN-0042 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python scripts/wind_error.py report --config configs/wind_error_ctrl_model_error_baseline.json

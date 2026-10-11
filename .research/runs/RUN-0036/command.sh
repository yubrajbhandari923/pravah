#!/bin/bash
# RUN-0036 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python scripts/wind_error.py run --config configs/wind_error_ctrl_alpha_sensitivity.json

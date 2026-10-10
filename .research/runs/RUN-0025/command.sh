#!/bin/bash
# RUN-0025 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah benchmark --config configs/belauri_forward.json --out results/belauri-forward

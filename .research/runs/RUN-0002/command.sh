#!/bin/bash
# RUN-0002 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah simulate --config configs/reference.json --out results/reference-v0

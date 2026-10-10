#!/bin/bash
# RUN-0003 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah inverse --config configs/reference.json --direct --out results/inverse-v0

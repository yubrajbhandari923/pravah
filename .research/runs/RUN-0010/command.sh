#!/bin/bash
# RUN-0010 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah benchmark --config configs/benchmark_unregularized.json --out results/benchmark-unregularized-final

#!/bin/bash
# RUN-0091 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python scripts/dyn00_verify.py

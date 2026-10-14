#!/bin/bash
# RUN-0079 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah.studies.toy.dynamic_sufficiency --stage test --design results/dyn05/design.json --workers 6

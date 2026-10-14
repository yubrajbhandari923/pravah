#!/bin/bash
# RUN-0092 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah.studies.toy.dynamic_history --stage test --freeze results/dyn03/frozen_hparams.json --workers 7

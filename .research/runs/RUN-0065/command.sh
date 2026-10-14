#!/bin/bash
# RUN-0065 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah.studies.toy.dynamic_robustness --freeze results/dyn02v2/frozen_hparams.json --workers 6

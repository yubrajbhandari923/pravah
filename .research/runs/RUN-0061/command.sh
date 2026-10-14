#!/bin/bash
# RUN-0061 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah.studies.toy.dynamic_study --stage test --freeze results/dyn02/frozen_hparams.json

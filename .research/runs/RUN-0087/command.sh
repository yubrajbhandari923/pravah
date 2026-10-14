#!/bin/bash
# RUN-0087 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah.studies.toy.dynamic_study --stage test --freeze results/dyn02v3/frozen_hparams.json --workers 7

#!/bin/bash
# RUN-0090 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah.studies.toy.dynamic_design_study --stage test --layouts results/dyn04/layouts.json --freeze results/dyn04/frozen_hparams.json --workers 6

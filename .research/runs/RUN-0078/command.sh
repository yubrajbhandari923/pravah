#!/bin/bash
# RUN-0078 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python -m pravah.studies.toy.dynamic_sufficiency --stage design --workers 4

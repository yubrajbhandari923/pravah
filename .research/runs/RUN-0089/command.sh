#!/bin/bash
# RUN-0089 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_cat.sh RUN-0081 paired paired_verdict.csv paired_summary.json paired.csv

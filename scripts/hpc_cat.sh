#!/usr/bin/env bash
# Print small result files of an earlier cluster run into this job's log (Research OS collects logs, not run-dir files).
#   bash scripts/hpc_cat.sh RUN-0081 paired paired_verdict.csv paired_summary.json paired.csv
set -euo pipefail
RUN="$1"; SUB="$2"; shift 2
D="$HOME/research-os/projects/pravah/runs/$RUN/$SUB"
for f in "$@"; do echo "=====FILE $f"; cat "$D/$f"; echo "=====END $f"; done

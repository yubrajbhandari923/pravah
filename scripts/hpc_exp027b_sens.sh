#!/usr/bin/env bash
# EXP-027b POST-HOC sensitivity analysis on the cluster (reads RUN-0100's paired_work read-only; writes only under $RESEARCH_RUN_DIR).
# Usage: hpc_exp027b_sens.sh [WORKERS] [CONFIG_NAME] [SOURCE_RUN]     (defaults: 16, convergence_paired_padded_exp027b.json, RUN-0100)
# All results are printed to the job log (Research OS collects logs, not run-dir files); the CSV/JSON are also kept in the run dir.
# Submit (the human/lead decides; ~minutes, one node):
#   .research/bin/research hpc submit EXP-037 --dry-run -- bash scripts/hpc_exp027b_sens.sh 16
set -euo pipefail
WORKERS="${1:-16}"
CONFIG="${2:-convergence_paired_padded_exp027b.json}"
SRC_RUN="${3:-RUN-0100}"
WORK="$HOME/research-os/projects/pravah/runs/$SRC_RUN/paired_work"
test -d "$WORK/cases" || { echo "missing $WORK/cases" >&2; exit 2; }
bash scripts/hpc_link_datasets.sh python scripts/exp027b_sensitivity.py --config "configs/$CONFIG" \
  --work "$WORK" --out "$RESEARCH_RUN_DIR/exp027b_sens" --workers "$WORKERS"
echo "=====FILE sens_rows.csv"; cat "$RESEARCH_RUN_DIR/exp027b_sens/sens_rows.csv"; echo "=====END sens_rows.csv"
echo "=====FILE sens_summary.json"; cat "$RESEARCH_RUN_DIR/exp027b_sens/sens_summary.json"; echo "=====END sens_summary.json"

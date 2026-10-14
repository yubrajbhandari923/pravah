#!/usr/bin/env bash
# Validation-only tuning stages on the cluster (outputs under $RESEARCH_RUN_DIR; the frozen file is copied back by Research OS).
#   bash scripts/hpc_dyn_tune.sh dyn03|dyn04 WORKERS
# The confirmatory test stages are NOT run here: tune output is inspected (edge selections, non-convergence) first.
set -euo pipefail
WHICH="$1"; W="$2"
case "$WHICH" in
  dyn03) CMD=(python -m pravah.studies.toy.dynamic_history --stage tune --workers "$W" --out "$RESEARCH_RUN_DIR") ;;
  dyn04) CMD=(python -m pravah.studies.toy.dynamic_design_study --stage tune --workers "$W" --layouts results/dyn04/layouts.json \
              --out "$RESEARCH_RUN_DIR") ;;
  *) echo "unknown $WHICH" >&2; exit 2 ;;
esac
bash scripts/hpc_link_datasets.sh "${CMD[@]}"

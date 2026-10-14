#!/usr/bin/env bash
# EXP-027 on the cluster: link staged datasets, then the paired mesh-vs-resampling test (outputs under $RESEARCH_RUN_DIR).
# Usage: hpc_exp027.sh [WORKERS] [CONFIG_NAME]   (default: 8, convergence_paired_padded.json)
set -euo pipefail
WORKERS="${1:-8}"
CONFIG="${2:-convergence_paired_padded.json}"
bash scripts/hpc_link_datasets.sh python scripts/convergence_stage2.py --config "configs/$CONFIG" \
  --paired --workers "$WORKERS" --out "$RESEARCH_RUN_DIR/paired" --work "$RESEARCH_RUN_DIR/paired_work"

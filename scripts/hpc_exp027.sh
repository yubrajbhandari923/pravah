#!/usr/bin/env bash
# EXP-027 on the cluster: link staged datasets, then the paired mesh-vs-resampling test (outputs under $RESEARCH_RUN_DIR).
set -euo pipefail
WORKERS="${1:-8}"
bash scripts/hpc_link_datasets.sh python scripts/convergence_stage2.py --config configs/convergence_paired.json \
  --paired --workers "$WORKERS" --out "$RESEARCH_RUN_DIR/paired" --work "$RESEARCH_RUN_DIR/paired_work"

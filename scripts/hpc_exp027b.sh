#!/usr/bin/env bash
# EXP-027b on the cluster: link staged datasets, run the paired case/layout build, then the registered same-ensemble evaluation.
# Usage: hpc_exp027b.sh [WORKERS] [CONFIG_NAME] [SEED_CASES_DIR]
#   stage 1: hpc_exp027b.sh 32 convergence_paired_padded_exp027b_s1.json
#   stage 2: hpc_exp027b.sh 32 convergence_paired_padded_exp027b.json PATH/TO/stage1/paired_work/cases   (copies stage-1 cases: none is re-simulated)
# Outputs under $RESEARCH_RUN_DIR. Reservation needs the user's explicit approval (docs/hpc-runbook.md); this script submits nothing itself.
set -euo pipefail
WORKERS="${1:-32}"
CONFIG="${2:-convergence_paired_padded_exp027b_s1.json}"
SEED="${3:-}"
if [ -n "$SEED" ]; then mkdir -p "$RESEARCH_RUN_DIR/paired_work/cases" && cp -n "$SEED"/*.pkl "$RESEARCH_RUN_DIR/paired_work/cases/"; fi
bash scripts/hpc_link_datasets.sh python scripts/convergence_stage2.py --config "configs/$CONFIG" \
  --paired --workers "$WORKERS" --out "$RESEARCH_RUN_DIR/paired" --work "$RESEARCH_RUN_DIR/paired_work"
bash scripts/hpc_link_datasets.sh python scripts/exp027b.py evaluate --config "configs/$CONFIG" \
  --work "$RESEARCH_RUN_DIR/paired_work" --out "$RESEARCH_RUN_DIR/exp027b"

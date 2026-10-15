#!/usr/bin/env bash
# EXP-026 (population-aware design at Belauri scale, WorldPop fixture) on the cluster; amendment D-027 filed before this run.
# Usage: hpc_exp026.sh   (run under `research hpc submit EXP-026 --cpus 8 ...`; artifacts under $RESEARCH_RUN_DIR)
set -euo pipefail
SHARED="${RESEARCH_SHARED_DATA:-/scratch/railabs/yb107/research-os/shared-data}"
IN="$SHARED/belauri-dyn08-inputs/files"
mkdir -p data
ln -sfn "$IN/population" data/population
exec bash scripts/hpc_link_datasets.sh python scripts/population_belauri.py --profile production \
  --config configs/population_belauri_hpc.json --out "$RESEARCH_RUN_DIR/phase3-population-belauri" --work "$RESEARCH_RUN_DIR/work"

#!/usr/bin/env bash
# DYN-08 (reduced) on the cluster.  NOT TO BE RUN WITHOUT the user's reservation approval (AGENTS.md; `research hpc submit EXP-036 --dry-run` first)
# and NOT before the inputs below are staged (a cluster write that needs the user's approval: docs/dyn08_plan.md section 7).
# Usage: hpc_dyn08.sh STAGE [WORKERS] [CONFIG]      STAGE: time | all | design | tune | test | modelerr     (default WORKERS 16, CONFIG configs/dyn08.json)
#   'all' = design -> tune -> gate (stop if > 25 % of selected hyperparameters sit on a grid edge) -> test -> modelerr in ONE job, freezing in-run
#   (layouts and hyperparameter hashes are printed before the test stage opens); separate stages need --layouts/--freeze files from an earlier run.
# Extra staged dataset (read-only link): $RESEARCH_SHARED_DATA/belauri-dyn08-inputs/files/{population,era5}/  (WorldPop padded 500 m extract + ERA5 10 m wind).
set -euo pipefail
STAGE="${1:?stage}"; WORKERS="${2:-16}"; CONFIG="${3:-configs/dyn08.json}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
SHARED="${RESEARCH_SHARED_DATA:-/scratch/railabs/yb107/research-os/shared-data}"
IN="$SHARED/belauri-dyn08-inputs/files"
[ -d "$IN/population" ] && [ -d "$IN/era5" ] || { echo "inputs not staged at $IN (docs/dyn08_plan.md section 7)" >&2; exit 1; }
mkdir -p "$SRC/data"
ln -sfn "$IN/population" "$SRC/data/population"
ln -sfn "$IN/era5" "$SRC/data/era5"
export PRAVAH_RESPONSE_CACHE="${RESEARCH_RUN_DIR:?run under research hpc}/resp_cache"   # artifacts only under $RESEARCH_RUN_DIR
exec bash "$SRC/scripts/hpc_link_datasets.sh" python -m pravah.studies.belauri.dynamic_belauri --stage "$STAGE" --config "$CONFIG" \
  --out "$RESEARCH_RUN_DIR/dyn08_$STAGE" --workers "$WORKERS"

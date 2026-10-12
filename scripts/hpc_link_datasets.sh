#!/bin/bash
# Link cluster-staged datasets into the paths PRAVAH's configs expect, then exec the real command.
#
#     bash scripts/hpc_link_datasets.sh <command...>
#
# PRAVAH configs name data/belauri/manifest.json etc. relative to the repo root, but data/ is
# deliberately excluded from the bundle (datasets belong in the cluster's shared data directory, not
# in a code snapshot). This symlinks the staged dataset into place inside the run's own source tree.
# Read-only use: the link points at shared data that this script never writes to.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
SHARED="${RESEARCH_SHARED_DATA:-/scratch/railabs/yb107/research-os/shared-data}"
GEO="$SHARED/belauri-geography/files"

if [ -d "$GEO" ]; then
  mkdir -p "$SRC/data"
  # -n: never descend into an existing link; -f: replace a stale one from an earlier attempt.
  ln -sfn "$GEO" "$SRC/data/belauri"
  echo "linked data/belauri -> $GEO"
  [ -f "$SRC/data/belauri/manifest.json" ] || { echo "linked dataset has no manifest.json" >&2; exit 1; }
else
  echo "dataset belauri-geography not found at $GEO (stage it: research hpc datasets fetch)" >&2
  exit 1
fi

exec "$@"

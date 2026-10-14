#!/bin/bash
# Link cluster-staged datasets into the paths PRAVAH's configs expect, then exec the real command.
#
#     bash scripts/hpc_link_datasets.sh <command...>
#
# PRAVAH configs name data/belauri/manifest.json etc. relative to the repo root, but data/ is
# deliberately excluded from the bundle (datasets belong in the cluster's shared data directory, not
# in a code snapshot). This symlinks the staged dataset into place inside the run's own source tree.
# Read-only use: the link points at shared data that this script never writes to.
#
# It also puts the persistent cluster venv first on PATH, so a command written as `python ...` gets the
# venv interpreter rather than the system python3 (which has none of the dependencies).
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
SHARED="${RESEARCH_SHARED_DATA:-/scratch/railabs/yb107/research-os/shared-data}"
GEO="$SHARED/belauri-geography/files"
VENV="${PRAVAH_ENV_ROOT:-$HOME/research-os/projects/pravah/env}/venv"

if [ -x "$VENV/bin/python" ]; then
  export PATH="$VENV/bin:$PATH"
  export VIRTUAL_ENV="$VENV"
  echo "using venv $VENV ($("$VENV/bin/python" -V 2>&1))"
else
  echo "no venv at $VENV — run scripts/hpc_preflight_env.sh first" >&2
  exit 1
fi

# The persistent venv holds a NON-editable copy of pravah from an earlier commit; the code being recorded is the bundle's src/.
# Without this the run silently imports the stale copy (RUN-0063 failed with "cannot import name 'dynamic'").
export PYTHONPATH="$SRC/src${PYTHONPATH:+:$PYTHONPATH}"
echo "pravah resolved from: $("$VENV/bin/python" -c 'import pravah;print(pravah.__file__)')"

# One process per independent unit; BLAS must not also thread or the pool oversubscribes the node.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

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

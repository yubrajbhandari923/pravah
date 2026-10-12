#!/bin/bash
# Build (or reuse) PRAVAH's persistent cluster venv and report what it contains.
#
# Run from a bundle's source/ tree via `research hpc submit`. The venv deliberately lives OUTSIDE the
# per-run bundle (each submit creates a fresh source/ tree, so a venv inside one would be thrown away)
# at $PRAVAH_ENV_ROOT, default <project>/env. Idempotent: an existing venv is reused, never rebuilt.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
ENV_ROOT="${PRAVAH_ENV_ROOT:-$HOME/research-os/projects/pravah/env}"
PY="$ENV_ROOT/venv/bin/python"

echo "host $(hostname) job ${SLURM_JOB_ID:-none} cpus ${SLURM_CPUS_PER_TASK:-?} src $SRC"
mkdir -p "$ENV_ROOT"

if [ -x "$PY" ] && "$PY" -c 'import fipy' 2>/dev/null; then
  echo "reusing existing venv at $ENV_ROOT/venv"
else
  echo "building venv at $ENV_ROOT/venv"
  command -v uv >/dev/null || { echo "uv not on PATH ($HOME/.local/bin expected)" >&2; exit 1; }
  [ -d "$ENV_ROOT/venv" ] || uv venv --python 3.12 "$ENV_ROOT/venv"
  # Not editable: the bundle source/ tree is immutable and is replaced on every run.
  uv pip install --python "$PY" "$SRC[dev,geo]" -c "$SRC/requirements-lock-geo.txt"
fi

echo "=== versions ==="
"$PY" - <<'PYEOF'
import importlib, sys
print("python", sys.version.split()[0])
for m in ("numpy", "scipy", "fipy", "pravah", "rasterio", "pyproj", "shapely"):
    try:
        print(m, getattr(importlib.import_module(m), "__version__", "(no __version__)"))
    except Exception as e:
        print(m, "IMPORT FAILED:", type(e).__name__, e)
PYEOF

echo "=== solver smoke (FiPy must actually solve) ==="
cd "$SRC" && "$PY" -c "
from pravah.forward import ForwardSolver
print('ForwardSolver import ok', ForwardSolver)
"
echo PREFLIGHT_ENV_OK

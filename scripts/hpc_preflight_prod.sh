#!/bin/bash
# Production preflight on the cluster: FiPy regression gate + EXP-015 stage-2 runtime estimate.
#
# Run from a bundle source/ tree via `research hpc submit`, using the persistent venv built by
# scripts/hpc_preflight_env.sh. Reports; decides nothing. Writes into $RESEARCH_RUN_DIR so
# `research hpc collect` brings the evidence home (small text/JSON only).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
ENV_ROOT="${PRAVAH_ENV_ROOT:-$HOME/research-os/projects/pravah/env}"
PY="$ENV_ROOT/venv/bin/python"
OUT="${RESEARCH_RUN_DIR:-$SRC/results/hpc-preflight}/preflight"
mkdir -p "$OUT"
cd "$SRC"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "host $(hostname) job ${SLURM_JOB_ID:-none} cpus ${SLURM_CPUS_PER_TASK:-?}"
[ -x "$PY" ] || { echo "no venv at $PY — run scripts/hpc_preflight_env.sh first" >&2; exit 1; }
rc=0

echo
echo "=== 1. environment identity ==="
"$PY" -c "
import sys, numpy, scipy, fipy, json
print(json.dumps({'python': sys.version.split()[0], 'numpy': numpy.__version__,
                  'scipy': scipy.__version__, 'fipy': fipy.__version__}))" | tee "$OUT/versions.json"

echo
echo "=== 2. FiPy regression gate (must print: 21 files compared, 0 differ) ==="
# D-005: the gate is architecture-scoped. On x86_64 compare against the cluster baseline; the arm64
# baseline in results/solver-regression/before is for laptop work and differs here by reassociation only.
BASE=results/solver-regression-x86_64/before
[ "$(uname -m)" = x86_64 ] && [ -d "$BASE" ] || BASE=results/solver-regression/before
echo "baseline: $BASE ($(uname -m))"
if [ -d "$BASE" ]; then
  "$PY" scripts/capture_regression.py --out "$OUT/regression-cluster" > "$OUT/regression-capture.log" 2>&1 \
    && "$PY" scripts/compare_runs.py "$BASE" "$OUT/regression-cluster" \
         2>&1 | tee "$OUT/regression-compare.log" \
    || { echo "regression gate FAILED (see logs)"; rc=1; }
  grep -q "21 files compared, 0 differ" "$OUT/regression-compare.log" 2>/dev/null \
    && echo "REGRESSION_GATE_OK" || { echo "REGRESSION_GATE_DIFFERS — do NOT loosen tolerances; investigate"; rc=1; }
else
  echo "baseline $BASE missing from the bundle" >&2; rc=1
fi

echo
echo "=== 3. EXP-015 stage-2 runtime estimate (the script's own preflight mode) ==="
"$PY" scripts/convergence_stage2.py --config configs/convergence_stage2.json \
      --out "$OUT/stage2-estimate" --estimate 2>&1 | tee "$OUT/stage2-estimate.log" || rc=1

echo
echo "=== 4. parallel units available (ceiling on useful workers) ==="
"$PY" - <<'PYEOF' | tee "$OUT/parallel_units.json"
import json
c = json.load(open("configs/convergence_stage2.json"))["stage2"]
meshes, reps = len(c["meshes"]), c["replicates"]["n"]
print(json.dumps({"mesh_units": meshes, "replicate_units": reps, "total_independent_units": meshes + reps,
                  "note": "process pool maps one worker per unit; workers beyond this sit idle"}, indent=1))
PYEOF

echo
[ $rc = 0 ] && echo PREFLIGHT_PROD_OK || echo "PREFLIGHT_PROD_FAILED rc=$rc"
exit $rc

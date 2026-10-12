#!/bin/bash
# Is the FiPy regression gate deterministic WITHIN this architecture?
#
# The committed baseline in results/solver-regression/before was captured on arm64 (Apple silicon).
# On x86_64 the OpenBLAS DYNAMIC_ARCH kernels reassociate floating-point sums differently, so a
# bit-identical match across architectures is not achievable. What must still hold is that the gate
# is reproducible on THIS machine: capture twice and compare the two captures to each other.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PRAVAH_ENV_ROOT:-$HOME/research-os/projects/pravah/env}/venv/bin/python"
OUT="${RESEARCH_RUN_DIR:-$SRC/results/hpc-selfcheck}/selfcheck"
mkdir -p "$OUT"; cd "$SRC"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "host $(hostname) job ${SLURM_JOB_ID:-none}"
"$PY" -c "import numpy; numpy.show_config()" > "$OUT/numpy_config.txt" 2>&1
grep -m1 "model name" /proc/cpuinfo | tee "$OUT/cpu.txt"

for r in A B; do
  echo "=== capture $r ==="
  "$PY" scripts/capture_regression.py --out "$OUT/cap$r" > "$OUT/cap$r.log" 2>&1 || { echo "capture $r failed"; exit 1; }
done

echo
echo "=== A vs B on the same node (must be: 21 files compared, 0 differ) ==="
"$PY" scripts/compare_runs.py "$OUT/capA" "$OUT/capB" 2>&1 | tee "$OUT/selfcheck.log"
grep -q "21 files compared, 0 differ" "$OUT/selfcheck.log" \
  && { echo SELFCHECK_DETERMINISTIC_OK; } \
  || { echo "SELFCHECK_NONDETERMINISTIC — same node, same code, different output: a real problem"; exit 1; }

echo
echo "=== for the record: this architecture vs the committed arm64 baseline ==="
"$PY" scripts/compare_runs.py results/solver-regression/before "$OUT/capA" 2>&1 | tee "$OUT/vs_baseline.log" | head -3

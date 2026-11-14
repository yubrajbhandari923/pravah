#!/usr/bin/env bash
# Gate-12-17: run the 32^2 re-runs one after another (one process). Arguments: SCRATCH E15_CACHE E16_CACHE
set -u
S=$1; C15=$2; C16=$3
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
PY="uv run --project ${PRAVAH_UV_PROJECT:-../../../..} python"
$PY e08_rerun32.py "$S/e08" > e08_rerun32.out 2>&1
$PY e12_e14_rerun32.py "$S/e12" "$C15/fields" > e12_e14_rerun32.out 2>&1
$PY e13_rerun32.py "$S/e13" "$C15/fields" > e13_rerun32.out 2>&1
$PY e15_e16_check32.py "$S" "$C15" "$C16" > e15_e16_check32.out 2>&1
echo done

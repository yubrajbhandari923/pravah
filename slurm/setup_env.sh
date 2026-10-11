#!/bin/bash
# Idempotent environment set-up (run once on the cluster, from the repo root): venv + locked deps + the 125 m geography level.
# Never overwrites data/belauri/manifest.json (geo prepare gets an explicit --out). No network is needed except for uv/pip packages.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,geo]" -c requirements-lock-geo.txt
G=results/phase3-bench/geo125
if [ ! -f "$G/manifest.json" ]; then
  [ -d data/belauri/raw ] || { echo "data/belauri/raw is missing: copy it from the laptop (docs/hpc-runbook.md 1.3)"; exit 1; }
  mkdir -p "$G"
  .venv/bin/python - <<'PY'
import json
c = json.load(open("configs/belauri_geo.json")); c["grid"]["resolutions_m"] = [500, 250, 125]
json.dump(c, open("results/phase3-bench/geo125/belauri_geo_125.json", "w"), indent=1)
PY
  .venv/bin/python -m pravah geo prepare --config "$G/belauri_geo_125.json" --cache-dir data/belauri/raw --out "$G/processed"
fi
.venv/bin/python scripts/hpc_run.py --preflight --need-125

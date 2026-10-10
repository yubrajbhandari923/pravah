"""Run the default workflows with fixed seeds into one comparison directory.

    python scripts/capture_regression.py --out results/solver-regression/before
    python scripts/capture_regression.py --out results/solver-regression/after [--extra "--solver fipy"]

Runs the CLI exactly as a user would (reference simulate, reference inverse with --direct, smoke benchmark, smoke
alpha-selection, smoke mismatch) and writes a manifest with the commands, the code revision and file checksums.
Compare two captures with scripts/compare_runs.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = [  # (name, cli args)
    ("simulate", ["simulate", "--config", "configs/reference.json"]),
    ("inverse", ["inverse", "--config", "configs/reference.json", "--direct"]),
    ("smoke", ["benchmark", "--config", "configs/smoke.json"]),
    ("smoke-alpha", ["benchmark", "--config", "configs/smoke_alpha_selection.json"]),
    ("smoke-mismatch", ["benchmark", "--config", "configs/smoke_mismatch.json"]),
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="", help="extra CLI flags appended to every command, e.g. '--solver fipy'")
    ap.add_argument("--skip", nargs="*", default=[], help="job names to skip")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True)
    rows = []
    for name, args in JOBS:
        if name in a.skip:
            continue
        run_dir = out / name
        cmd = [sys.executable, "-m", "pravah", *args, "--out", str(run_dir), *shlex.split(a.extra)]
        t0 = time.perf_counter()
        if name == "smoke-mismatch":  # mismatch.layouts_from must point at THIS capture's smoke run
            cfg_path = out / "smoke_mismatch.local.json"
            cfg_path.write_text(json.dumps({"extends": str((ROOT / "configs/smoke_mismatch.json").resolve()),
                                            "mismatch": {"layouts_from": str(out / "smoke")}}, indent=2))
            cmd[cmd.index("configs/smoke_mismatch.json")] = str(cfg_path)
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        rows.append({"job": name, "cmd": cmd, "returncode": r.returncode, "wall_s": time.perf_counter() - t0,
                     "stdout_tail": r.stdout[-400:], "stderr_tail": r.stderr[-400:]})
        print(name, "rc", r.returncode, f"{rows[-1]['wall_s']:.1f}s", flush=True)
        if r.returncode:
            print(r.stderr[-2000:])
            break
    files = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()
             and p.suffix in {".csv", ".json", ".nc", ".npz"}}
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / "manifest.json").write_text(json.dumps({"git_revision": rev, "jobs": rows, "file_sha256": files}, indent=2))
    raise SystemExit(0 if all(r["returncode"] == 0 for r in rows) else 1)


if __name__ == "__main__":
    main()

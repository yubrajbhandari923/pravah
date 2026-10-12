"""HPC infrastructure smoke test: deterministic compute + execution metadata. Standard library only.

    python scripts/hpc_smoke.py --config configs/hpc_smoke.json --out results/hpc-smoke

Computes sum(1..n) and sum of squares, compares them against the closed forms recorded in the config
(so a correct run is verifiable without trusting this script), and writes result.json with where and
how it ran. Exit 0 only when the arithmetic matches; nonzero otherwise, so a broken environment is
reported as a failure rather than a success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hpc_smoke.json")
    ap.add_argument("--out", default=None,
                    help="output dir (default: $RESEARCH_RUN_DIR/results/hpc-smoke inside a job, "
                         "else results/hpc-smoke). Writing under the run dir is what makes "
                         "`research hpc sync` bring the artifact home.")
    ap.add_argument("--fail", action="store_true", help="corrupt the result on purpose (tests failure reporting)")
    a = ap.parse_args()

    cfg_path = Path(a.config)
    cfg = json.loads(cfg_path.read_text())
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    out = Path(a.out) if a.out else Path(run_dir or ".") / "results/hpc-smoke"
    out.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    n = int(cfg["n"])
    total = sum(range(1, n + 1))
    squares = sum(i * i for i in range(1, n + 1))
    if a.fail:
        total += 1
    wall_s = time.monotonic() - t0
    finished = datetime.now(timezone.utc)

    expected = cfg["expected"]
    checks = [
        {"name": "sum", "expected": expected["sum"], "actual": total, "ok": total == expected["sum"]},
        {"name": "sum_squares", "expected": expected["sum_squares"], "actual": squares,
         "ok": squares == expected["sum_squares"]},
    ]
    ok = all(c["ok"] for c in checks)

    result = {
        "schema": 1,
        "ok": ok,
        "checks": checks,
        "input": {"path": str(cfg_path), "sha256": sha256(cfg_path), "n": n},
        "timing": {"started_utc": started.isoformat(timespec="seconds"),
                   "finished_utc": finished.isoformat(timespec="seconds"),
                   "wall_seconds": round(wall_s, 3)},
        "execution": {
            "hostname": socket.gethostname(),
            "cwd": os.getcwd(),
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "slurm": {k: v for k, v in sorted(os.environ.items())
                      if k.startswith("SLURM_") and k in (
                          "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
                          "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_SUBMIT_DIR", "SLURM_JOB_ACCOUNT")},
            "research": {k: os.environ.get(k) for k in ("RESEARCH_RUN_ID", "RESEARCH_RUN_DIR", "RESEARCH_IN_JOB")},
        },
    }
    path = out / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    digest = sha256(path)
    (out / "result.json.sha256").write_text(f"{digest}  result.json\n")
    if run_dir:
        # `research hpc collect` copies the artifacts listed here; nothing else travels home.
        with open(Path(run_dir) / "artifacts.jsonl", "a") as f:
            for name in ("result.json", "result.json.sha256"):
                f.write(json.dumps({"path": str((out / name).resolve()),
                                    "sha256": digest if name == "result.json" else sha256(out / name),
                                    "description": f"HPC smoke test {name}"}) + "\n")

    for c in checks:
        print(f"{'ok  ' if c['ok'] else 'FAIL'} {c['name']}: {c['actual']} (expected {c['expected']})")
    print(f"host {result['execution']['hostname']} job {result['execution']['slurm'].get('SLURM_JOB_ID', 'none')} "
          f"python {result['execution']['python_version']} wall {wall_s:.2f}s")
    print(f"wrote {path}")
    if not ok:
        print("smoke test FAILED: the arithmetic does not match the closed forms", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

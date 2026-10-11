"""Single-job orchestrator for the heavy Phase 3 work (one Slurm job, phases strictly in sequence, parallelism inside Python).

    python scripts/hpc_run.py --config configs/hpc_phase3.json [--dry-run] [--only NAME] [--max-workers N] [--no-bundle]

Config: {"name", "results_root", "phases": [{"name", "command": [argv with {python} {workers} {out}], "requires": [files],
"done_marker": file, "est_minutes"}]}.  {out} = results_root.  A phase is skipped (resume) when its done_marker exists AND
results_root/hpc_manifest.json holds a successful entry with the same phase config hash (phase dict + contents of any *.json
file named in argv).  A phase whose script (a *.py in argv) is missing is SKIPPED with a message, not failed.  The marker is
created by the orchestrator when the command did not create it.  Each phase's stdout/stderr go to results_root/hpc_logs/<name>.log
(and the console).  After every phase one entry is written to hpc_manifest.json (provenance for `research run attach` on a
machine that has Research OS).  At the end results_root/phase3-hpc-bundle.tar.gz (+ .sha256) holds the listed outputs, the manifest and
logs; work/cache dirs and .npz files > 50 MB are left out (their sizes are recorded).
Worker count: $PRAVAH_WORKERS, else $SLURM_CPUS_PER_TASK, else os.cpu_count(), capped by --max-workers.  The code is CPU-only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import resource
import shlex
import socket
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
BIG_NPZ = 50e6
GEO125 = "results/phase3-bench/geo125/manifest.json"
REBUILD_125 = """mkdir -p results/phase3-bench/geo125 && python - <<'EOF'
import json
c = json.load(open("configs/belauri_geo.json")); c["grid"]["resolutions_m"] = [500, 250, 125]
json.dump(c, open("results/phase3-bench/geo125/belauri_geo_125.json", "w"), indent=1)
EOF
python -m pravah geo prepare --config results/phase3-bench/geo125/belauri_geo_125.json --cache-dir data/belauri/raw --out results/phase3-bench/geo125/processed
(or: bash slurm/setup_env.sh; docs/hpc-runbook.md section 1.4. Always pass --out explicitly: the default overwrites data/belauri/manifest.json)"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def n_workers(max_workers: int | None) -> int:
    w = int(os.environ.get("PRAVAH_WORKERS") or os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    return max(1, min(w, max_workers) if max_workers else w)


def expand(cmd: list[str], workers: int, out: str) -> list[str]:
    return [c.format(python=sys.executable, workers=workers, out=out) for c in cmd]


def phase_hash(phase: dict[str, Any], argv: list[str]) -> str:
    h = hashlib.sha256(json.dumps(phase, sort_keys=True).encode())
    for a in argv:
        if a.endswith(".json") and (ROOT / a).is_file():
            h.update(sha256(ROOT / a).encode())
    return h.hexdigest()


def missing_script(argv: list[str]) -> str | None:
    for a in argv:
        if a.endswith(".py"):
            return None if (ROOT / a).is_file() or Path(a).is_file() else a
    return None


def git_info() -> dict[str, Any]:
    def g(*a: str) -> str | None:
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
    head, st, diff = g("rev-parse", "HEAD"), g("status", "--porcelain", "--untracked-files=no"), g("diff", "HEAD")
    return {"commit": head.strip() if head else None, "dirty": bool(st.strip()) if st is not None else None,
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff is not None else None}


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for p in ("numpy", "scipy", "fipy"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def is_cache(p: Path) -> bool:
    return any(part in ("cache", "work") or part.endswith("_work") for part in p.parts)


def scan(root: Path, since: float) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Files under root modified since `since` (cache/work dirs summed instead of listed; huge .npz listed as not bundled)."""
    files, caches = [], {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "hpc_logs" in p.parts or p.name.startswith("phase3-hpc-bundle") or p.name == "hpc_manifest.json":
            continue
        rel = p.relative_to(root)
        if is_cache(rel):
            top = str(next(Path(*rel.parts[: i + 1]) for i, s in enumerate(rel.parts) if s in ("cache", "work") or s.endswith("_work")))
            caches[top] = caches.get(top, 0) + p.stat().st_size
        elif p.stat().st_mtime >= since:
            size = p.stat().st_size
            files.append({"path": str(p.relative_to(ROOT)), "bytes": size, "sha256": sha256(p),
                          "in_bundle": not (p.suffix == ".npz" and size > BIG_NPZ)})
    return files, caches


def run_phase(phase: dict[str, Any], argv: list[str], log: Path, env: dict[str, str]) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as lf:
        p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        return p.wait()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {"schema": 1, "phases": []}


def save_manifest(path: Path, m: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(m, indent=1))
    os.replace(tmp, path)


def make_bundle(root: Path, m: dict[str, Any]) -> tuple[Path, str]:
    b = root / "phase3-hpc-bundle.tar.gz"
    names = {f["path"] for ph in m["phases"] for f in ph.get("outputs", []) if f["in_bundle"]}
    with tarfile.open(b, "w:gz") as t:
        for n in sorted(names):
            if (ROOT / n).is_file():
                t.add(ROOT / n, arcname=n)
        for extra in [root / "hpc_manifest.json", *sorted((root / "hpc_logs").glob("*.log"))]:
            t.add(extra, arcname=str(extra.relative_to(ROOT)))
    h = sha256(b)
    (root / (b.name + ".sha256")).write_text(f"{h}  {b.name}\n")
    return b, h


def preflight(need_125: bool) -> int:
    ok = True
    for mod in ("fipy", "numpy", "shapely", "rasterio", "pyproj", "matplotlib"):
        try:
            __import__(mod)
            print(f"import {mod}: ok")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"import {mod}: FAILED {e!r}")
    try:
        from pravah.belauri import resolve_manifest
        from pravah.geo.prepare import load_region
        mp = resolve_manifest("data/belauri/manifest.json")
        reg = load_region(mp)  # verifies the recorded file hashes
        print(f"{mp}: present, loaded and verified ({len(reg.levels)} levels)" if hasattr(reg, "levels") else f"{mp}: loaded and verified")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"data/belauri/manifest.json: FAILED {e!r}")
    if need_125 and not (ROOT / GEO125).is_file():
        ok = False
        print(f"the 125 m geography level is missing ({GEO125}). Rebuild it exactly:\n{REBUILD_125}")
    print("preflight " + ("OK (CPU-only code: no GPU needed or used)" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/hpc_phase3.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="run only this phase name")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--no-bundle", action="store_true")
    ap.add_argument("--preflight", action="store_true", help="(internal) the preflight checks")
    ap.add_argument("--need-125", action="store_true", help="(internal, with --preflight) require the 125 m level")
    a = ap.parse_args()
    if a.preflight:
        return preflight(a.need_125)
    cfg = json.loads((ROOT / a.config).read_text())
    cfg_sha = sha256(ROOT / a.config)
    out = cfg.get("results_root", "results")
    root = ROOT / out
    workers = n_workers(a.max_workers)
    phases = [p for p in cfg["phases"] if a.only in (None, p["name"])]
    if a.only and not phases:
        raise SystemExit(f"no phase named {a.only}")
    mpath = root / "hpc_manifest.json"
    man = load_manifest(mpath)
    done = {p["phase"]: p for p in man["phases"] if p.get("exit_code") == 0 and not p.get("skipped")}
    print(f"hpc_run: {cfg.get('name')} | workers {workers} | host {socket.gethostname()} | SLURM_JOB_ID {os.environ.get('SLURM_JOB_ID')}"
          f" | est {sum(p.get('est_minutes', 0) for p in phases)} min")
    env = {**os.environ, **{k: "1" for k in THREAD_VARS}, "PRAVAH_WORKERS": str(workers)}
    for ph in phases:
        argv = expand(ph["command"], workers, out)
        h = phase_hash(ph, argv)
        marker = ROOT / ph["done_marker"].format(out=out)
        miss = missing_script(argv)
        state = ("SKIP (script missing: " + miss + ")" if miss else
                 "SKIP (done, config hash matches)" if marker.exists() and done.get(ph["name"], {}).get("phase_config_sha256") == h else "RUN")
        print(f"- {ph['name']:<14} est {ph.get('est_minutes', '?'):>4} min  {state}\n    {shlex.join(argv)}")
        if a.dry_run or state != "RUN":
            continue
        for r in ph.get("requires", []):
            if not (ROOT / r.format(out=out)).exists():
                print(f"phase {ph['name']}: required file {r} missing")
                return 2
        t0, w0 = time.time(), time.perf_counter()
        print(f"=== {ph['name']} start {now()}")
        rc = run_phase(ph, argv, root / "hpc_logs" / f"{ph['name']}.log", env)
        wall = time.perf_counter() - w0
        outs, caches = scan(root, t0 - 1)
        if rc == 0 and not marker.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"phase": ph["name"], "finished": now(), "phase_config_sha256": h}) + "\n")
        entry = {"phase": ph["name"], "argv": argv, "start": datetime.fromtimestamp(t0, timezone.utc).isoformat(timespec="seconds"),
                 "end": now(), "wall_seconds": round(wall, 1), "exit_code": rc, "host": socket.gethostname(),
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cpu_count": os.cpu_count(), "workers": workers,
                 "versions": versions(), "git": git_info(), "hpc_config": a.config, "hpc_config_sha256": cfg_sha,
                 "phase_config_sha256": h, "done_marker": str(marker.relative_to(ROOT)), "outputs": outs,
                 "cache_dirs_bytes": caches, "children_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                                                                        / (1e6 if sys.platform == "darwin" else 1e3), 1)}
        man["phases"] = [p for p in man["phases"] if p["phase"] != ph["name"]] + [entry]
        root.mkdir(parents=True, exist_ok=True)
        save_manifest(mpath, man)
        done = {p["phase"]: p for p in man["phases"] if p.get("exit_code") == 0}
        print(f"=== {ph['name']} exit {rc} in {wall:.0f} s ({len(outs)} output files)")
        if rc != 0:
            print(f"FAILED; fix and resubmit: finished phases are skipped. Log: {root / 'hpc_logs' / (ph['name'] + '.log')}")
            return rc
    if not a.dry_run and not a.no_bundle and man["phases"]:
        b, h = make_bundle(root, man)
        man["bundle"] = {"path": str(b.relative_to(ROOT)), "sha256": h, "bytes": b.stat().st_size, "created": now()}
        save_manifest(mpath, man)
        print(f"bundle {b} sha256 {h}\nscp CLUSTER:{ROOT}/{out}/phase3-hpc-bundle.tar.gz CLUSTER:{ROOT}/{out}/hpc_manifest.json LOCAL_REPO/{out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

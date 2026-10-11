"""Wall time / peak RSS of one production-like Belauri forward solve (configs/geodesign.json, draw 'd0' of the design ensemble)
per (resolution, dt) case, each in a fresh subprocess.  Appends rows to a CSV; cases already in the CSV are skipped (resumable).
A case exceeding --max-min minutes or --max-gb RSS is killed and recorded 'infeasible'.  `--mode basis` times
compute_responses (n_params+1 solves = one source-response basis) instead of one solve."""
from __future__ import annotations

import argparse, csv, json, resource, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

FIELDS = ["mode", "resolution_m", "dt_s", "nx", "ny", "cells", "steps", "n_solves", "solver", "status", "wall_s", "peak_rss_gb", "note"]


def child(a: argparse.Namespace) -> None:
    import numpy as np
    from pravah.config import SolverOptions
    from pravah.forward import FIPY
    from pravah.geo.prepare import load_region
    from pravah.geodesign import Setup, draw_cases
    from pravah.inverse import compute_responses, run_forward
    cfg = json.loads(Path(a.config).read_text())
    cfg["mesh"] = {"resolution_m": a.res, "dt_s": a.dt}
    d = draw_cases(cfg, 1, int(cfg["ensembles"]["design"]["seed"]), "d")[0]
    ph = cfg["physics"]
    me = SimpleNamespace(cfg=cfg, geo=SimpleNamespace(region=load_region(a.manifest)), res=a.res, dt=a.dt,
                         T=float(ph["T"]), out_dt=float(ph["output_interval"]))
    sc = Setup.scenario(me, d)  # type: ignore[arg-type]  # shortcut: unbound call avoids loading candidates/held-out points
    solver = SolverOptions(**cfg.get("solver", {}))
    t0 = time.perf_counter()
    if a.mode == "forward":
        run_forward(FIPY, sc.scenario, sc.theta, solver); n = 1
    else:
        compute_responses(sc.scenario, solver); n = len(sc.theta) + 1
    wall = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    dom = sc.scenario.domain
    print("RESULT " + json.dumps({"nx": dom.nx, "ny": dom.ny, "cells": dom.nx * dom.ny, "steps": round(float(ph["T"]) / a.dt),
          "n_solves": n, "solver": f"fipy LinearLUSolver(tol={solver.tolerance:g}), {solver.convection} convection",
          "wall_s": round(wall, 2), "peak_rss_gb": round(peak, 3)}))


def rss_gb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) / 1e6 if out else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/geodesign.json")
    p.add_argument("--manifest", default="results/phase3-bench/geo125/manifest.json", help="manifest with a 125 m level")
    p.add_argument("--out", default="results/phase3-bench/resolution_cost.csv")
    p.add_argument("--cases", default="500:100,500:50,250:100,125:100", help="res_m:dt_s,...")
    p.add_argument("--mode", choices=["forward", "basis"], default="forward")
    p.add_argument("--max-min", type=float, default=10.0)
    p.add_argument("--max-gb", type=float, default=5.0)
    p.add_argument("--child", action="store_true")
    p.add_argument("--res", type=float); p.add_argument("--dt", type=float)
    a = p.parse_args()
    if a.child:
        return child(a)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["mode"], float(r["resolution_m"]), float(r["dt_s"])) for r in csv.DictReader(out.open())}
    for c in a.cases.split(","):
        res, dt = (float(v) for v in c.split(":"))
        if (a.mode, res, dt) in done:
            continue
        cmd = [sys.executable, __file__, "--child", "--config", a.config, "--manifest", a.manifest, "--mode", a.mode,
               "--res", str(res), "--dt", str(dt)]
        t0, peak, status = time.time(), 0.0, "ok"
        pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while pr.poll() is None:
            peak = max(peak, rss_gb(pr.pid)); time.sleep(0.5)
            if time.time() - t0 > a.max_min * 60: status = f"infeasible: >{a.max_min:g} min"
            elif peak > a.max_gb: status = f"infeasible: RSS>{a.max_gb:g} GB"
            if status != "ok": pr.kill(); break
        so, se = pr.communicate()
        row = {"mode": a.mode, "resolution_m": res, "dt_s": dt, "status": status, "wall_s": round(time.time() - t0, 1) if status != "ok" else "",
               "peak_rss_gb": round(peak, 3) if status != "ok" else ""}
        line = [l for l in so.splitlines() if l.startswith("RESULT ")]
        if line and status == "ok":
            row.update(json.loads(line[0][7:]))
        elif status == "ok":
            row.update(status="error", note=(se.strip().splitlines() or ["no output"])[-1][:200])
        new = not out.exists()
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, FIELDS); new and w.writeheader(); w.writerow(row)
        print(row, flush=True)


if __name__ == "__main__":
    main()

"""EXP-015 stage 1: field-level numerical verification of the production-like Belauri forward case (configs/geodesign.json,
draw 'd0' of the design ensemble, as in scripts/bench_resolution.py).  Only mesh, dt and scheme vary.

    python scripts/convergence_stage1.py --profile smoke|production [--out DIR] [--manifest M] [--summary-only]

Matrix (unique solves; `modes` lists which of A/B/C/F a solve serves):
  A spatial, dt fixed:        res 500/250/125 m x {exponential, vanleer}
  B temporal, res 500 m:      dt 200/100/50/25/12.5 s, exponential
  C joint, constant ratio:    (500,100) (250,50) (125,25), exponential
  F fingerprints (exp only):  each source group alone (road, builtup, external_patch, bg) at every res, dt fixed
Each solve runs in a fresh subprocess (clean peak RSS); finished keys (res, dt, scheme, group) in stage1_runs.csv are skipped.
Per-case fields (block-mean to the common 500 m grid) go to fields/*.npz.  The summary/figures are rebuilt at the end."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COARSE = 500.0
GROUPS = ("road", "builtup", "external_patch", "bg")
MASS_TOL = 1e-3
COST_FACTOR = 3.0
RUN_FIELDS = ["res_m", "dt_s", "scheme", "group", "modes", "nx", "ny", "cells", "steps", "max_substeps", "dt_eff_s", "wall_s",
              "peak_rss_gb", "rel_budget_residual", "min_conc", "n_negative", "wind_speed_ms", "keff_over_k", "keff_over_k_dteff",
              "mass_road_ug", "mass_builtup_ug", "mass_external_patch_ug", "instudy_road", "instudy_builtup",
              "instudy_external_patch", "field_file"]
PROFILES: dict[str, dict[str, Any]] = {
    "production": dict(res=(500.0, 250.0, 125.0), dt0=25.0, b_dts=(200.0, 100.0, 50.0, 25.0, 12.5),
                       c=((500.0, 100.0), (250.0, 50.0), (125.0, 25.0)), overrides={}),
    "smoke": dict(res=(500.0, 250.0), dt0=50.0, b_dts=(100.0, 50.0), c=((500.0, 100.0), (250.0, 50.0)),
                  overrides={"T": 7200.0, "t_exclude": 1800.0, "profile_break": 1800.0}),  # as configs/geodesign_smoke.json
}
Key = tuple[float, float, str, str]  # (res, dt, scheme, group); group '' = all sources


# ------------------------------------------------------------------------------------------------ plan / aggregation
def plan(profile: str) -> dict[Key, set[str]]:
    p = PROFILES[profile]
    out: dict[Key, set[str]] = {}

    def add(res: float, dt: float, scheme: str, mode: str, group: str = "") -> None:
        out.setdefault((res, dt, scheme, group), set()).add(mode)

    for r in p["res"]:
        for s in ("exponential", "vanleer"):
            add(r, p["dt0"], s, "A")
        for g in GROUPS:
            add(r, p["dt0"], "exponential", "F", g)
    for dt in p["b_dts"]:
        add(COARSE, dt, "exponential", "B")
    for r, dt in p["c"]:
        add(r, dt, "exponential", "C")
    return out


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    """(..., ny*k, nx*k) -> (..., ny, nx): exact conservative aggregation of cell averages (equal-area cells)."""
    *lead, ny, nx = a.shape
    assert ny % k == 0 and nx % k == 0, f"{a.shape} not divisible by {k}"
    return a.reshape(*lead, ny // k, k, nx // k, k).mean(axis=(-3, -1))


def assert_nested(centres: dict[float, tuple[np.ndarray, np.ndarray]], tol: float = 1e-6) -> None:
    """centres[res] = (x, y) 1-D cell-centre coordinates (m, same origin).  Every grid must be an integer refinement of the
    coarsest and its block-mean centres must equal the coarse centres (same SW corner, same extent)."""
    res0 = max(centres)
    x0, y0 = centres[res0]
    for r, (x, y) in sorted(centres.items(), reverse=True):
        k = round(res0 / r)
        assert abs(k * r - res0) < 1e-9, f"{r} m is not an integer divisor of {res0} m"
        assert len(x) == k * len(x0) and len(y) == k * len(y0), f"{r} m grid shape {len(x)}x{len(y)} is not {k}x the coarse grid"
        dx, dy = np.abs(x.reshape(-1, k).mean(1) - x0).max(), np.abs(y.reshape(-1, k).mean(1) - y0).max()
        assert max(dx, dy) < tol, f"{r} m grid not aligned with {res0} m grid (max centre offset {max(dx, dy):.3g} m)"
        print(f"nested {r:g} m: SW corner offset {x[0] - r / 2 - (x0[0] - res0 / 2):+.3g} m (x), {y[0] - r / 2 - (y0[0] - res0 / 2):+.3g} m (y); "
              f"cells {len(x)}x{len(y)}; max block-centre offset {max(dx, dy):.2g} m")


# ------------------------------------------------------------------------------------------------ one solve (child)
def case_config(profile: str) -> dict[str, Any]:
    cfg = json.loads((ROOT / "configs/geodesign.json").read_text())
    ov = PROFILES[profile]["overrides"]
    if ov:
        b = ov["profile_break"]
        cfg["physics"].update(T=ov["T"], t_exclude=ov["t_exclude"])
        cfg["model"]["external_profile"] = [[0.0, 0.0], [b, 1.0]]
        cfg["model"]["bg"]["profile"] = [[0.0, 0.0], [b, 1.0]]
        cfg["ensemble"]["switch_step_s"] = b
    return cfg


def child(profile: str, manifest: str, key: Key, out: Path) -> dict[str, Any]:
    from pravah.config import SolverOptions
    from pravah.forward import FIPY, step_times
    from pravah.geo.prepare import load_region
    from pravah.geo.scenarios import numerical_diffusion_estimate
    from pravah.geodesign import Setup, draw_cases
    from pravah.inverse import run_forward
    from pravah.sources import emission_profile, source_fields
    res, dt, scheme, group = key
    cfg = case_config(profile)
    cfg["mesh"] = {"resolution_m": res, "dt_s": dt}
    d = draw_cases(cfg, 1, int(cfg["ensembles"]["design"]["seed"]), "d")[0]
    ph = cfg["physics"]
    me = SimpleNamespace(cfg=cfg, geo=SimpleNamespace(region=load_region(manifest)), res=res, dt=dt, T=float(ph["T"]),
                         out_dt=float(ph["output_interval"]))
    bc = Setup.scenario(me, d)  # type: ignore[arg-type]  # shortcut: unbound call avoids loading candidates/held-out points
    sc, theta = bc.scenario, bc.theta.copy()
    names = [lab["name"] for lab in bc.labels]
    assert tuple(names) == GROUPS, names
    if group:  # fingerprint: only this group, everything else switched off
        keep = names.index(group)
        theta = np.where(np.arange(len(theta)) == keep, theta, 0.0)
    solver = SolverOptions(convection=scheme)
    t0 = time.perf_counter()
    res_f = run_forward(FIPY, sc, theta, solver)
    wall = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)
    dg = res_f.diagnostics
    conc = np.asarray(res_f.concentration.transpose("time", "y", "x").values)
    k = round(COARSE / res)
    fname = f"fields/r{res:g}_dt{dt:g}_{scheme}_{group or 'all'}.npz"
    (out / "fields").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / fname, field=block_mean(conc, k), times=np.asarray(res_f.concentration["time"].values))
    n_sub = int(dg.get("max_substeps", 1))
    u = max(math.hypot(v.u, v.v) for v in sc.transport.wind)
    kk = lambda dtx: 1.0 + numerical_diffusion_estimate(u, res, dtx, sc.transport.K)  # noqa: E731
    f = source_fields(sc)
    dA = sc.domain.dx * sc.domain.dy
    ts = step_times(sc)
    w_study = np.asarray(sc.study.weights, dtype=float)  # type: ignore[union-attr]
    row: dict[str, Any] = dict(res_m=res, dt_s=dt, scheme=scheme, group=group, nx=sc.domain.nx, ny=sc.domain.ny,
                               cells=sc.domain.nx * sc.domain.ny, steps=int(dg["n_steps"]), max_substeps=n_sub, dt_eff_s=dt / n_sub,
                               wall_s=round(wall, 2), peak_rss_gb=round(peak, 3), rel_budget_residual=dg["max_rel_budget_residual"],
                               min_conc=dg["min_concentration"], n_negative=dg["n_negative"], wind_speed_ms=round(u, 4),
                               keff_over_k=kk(dt), keff_over_k_dteff=kk(dt / n_sub), field_file=fname)
    for j, s in enumerate(sc.sources):  # total emitted mass = q_j * sum(f_j) dA * int s_j(t) dt (what the solver injects)
        if group and names[j] != group:
            continue
        integ = sum(float(h) * emission_profile(s, 0.5 * (a + b)) for a, b, h in zip(ts[:-1], ts[1:], np.diff(ts)))
        row[f"mass_{s.name}_ug"] = theta[j] * float(f[j].sum()) * dA * integ
        row[f"instudy_{s.name}"] = float((f[j] * w_study).sum()) * dA
    return row


# ------------------------------------------------------------------------------------------------ parent loop
def bench_cost_per_step(res: float) -> float:
    rows = [r for r in csv.DictReader((ROOT / "results/phase3-bench/resolution_cost.csv").open())
            if r["mode"] == "forward" and float(r["resolution_m"]) == res and r["status"] == "ok"]
    return float(np.mean([float(r["wall_s"]) / float(r["steps"]) for r in rows]))


def key_of(r: dict[str, Any]) -> Key:
    return (float(r["res_m"]), float(r["dt_s"]), str(r["scheme"]), str(r["group"]))


def subprocess_runner(profile: str, manifest: str, out: Path) -> Callable[[Key], dict[str, Any]]:
    def run(key: Key) -> dict[str, Any]:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(key), "--profile", profile, "--manifest", manifest,
               "--out", str(out)]
        pr = subprocess.run(cmd, capture_output=True, text=True)
        line = [ln for ln in pr.stdout.splitlines() if ln.startswith("RESULT ")]
        if pr.returncode or not line:
            raise RuntimeError(f"case {key} failed:\n{pr.stderr[-2000:]}")
        return json.loads(line[0][7:])
    return run


def execute(pl: dict[Key, set[str]], runs_csv: Path, runner: Callable[[Key], dict[str, Any]],
            cost_check: Callable[[dict[str, Any]], None] | None = None) -> int:
    """Run every planned key not yet in runs_csv, appending one row each; returns the number of solves performed."""
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    done = {key_of(r) for r in csv.DictReader(runs_csv.open())} if runs_csv.exists() else set()
    n = 0
    for key in sorted(pl, key=lambda k: (k[0] != 500.0, -k[0], k[3] != "", k)):  # coarse first: cheap failures first
        if key in done:
            continue
        row = runner(key)
        row["modes"] = ";".join(sorted(pl[key]))
        new = not runs_csv.exists()
        with runs_csv.open("a", newline="") as f:
            w = csv.DictWriter(f, RUN_FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in RUN_FIELDS})
        n += 1
        print("done", key, f"wall {row['wall_s']} s, rss {row['peak_rss_gb']} GB", flush=True)
        if cost_check:
            cost_check(row)
    return n


# ------------------------------------------------------------------------------------------------ analysis
def load_runs(out: Path) -> dict[Key, dict[str, Any]]:
    return {key_of(r): r for r in csv.DictReader((out / "stage1_runs.csv").open())}


def rel_l2(a: np.ndarray, r: np.ndarray) -> float:
    return float(np.sqrt(((a - r) ** 2).sum() / (r**2).sum()))


def arrival(t: np.ndarray, m: np.ndarray, thr: float) -> float:
    """First time the series exceeds thr (linear interpolation between stored times); nan if never."""
    for i in np.flatnonzero(m > thr):
        return float(t[0] if i == 0 else t[i - 1] + (thr - m[i - 1]) / (m[i] - m[i - 1]) * (t[i] - t[i - 1]))
    return float("nan")


def order_status(e_coarse: float, e_fine: float) -> tuple[float, str]:
    if not (e_coarse > e_fine > 0):
        return float("nan"), "non-monotone (error does not decrease)"
    return math.log(e_coarse / e_fine) / math.log(2.0), "ok"


def richardson(u0: float, u1: float, u2: float) -> tuple[float, str]:
    d1, d2 = u0 - u1, u1 - u2
    if d1 * d2 <= 0:
        return float("nan"), "non-monotone (differences change sign or vanish)"
    ratio = d1 / d2
    return math.log2(ratio), "ok" if ratio > 1 else "not converging (successive change grows)"


def summarise(out: Path, profile: str, manifest: str, run_id: str) -> None:
    from pravah.geo.prepare import load_region
    p = PROFILES[profile]
    runs = load_runs(out)
    te = case_config(profile)["physics"]["t_exclude"]
    w = np.asarray(load_region(manifest).level(COARSE).arrays["study_fraction"], dtype=float)

    def fld(key: Key) -> tuple[np.ndarray, np.ndarray]:
        z = np.load(out / runs[key]["field_file"])
        return z["times"], z["field"]

    def mean_series(f: np.ndarray) -> np.ndarray:
        return (f * w).sum(axis=(1, 2)) / w.sum()

    res_ref, dt_ref = min(p["res"]), p["dt0"]
    refs = {"R_exp": (res_ref, dt_ref, "exponential", ""), "R_vl": (res_ref, dt_ref, "vanleer", "")}
    rows: list[dict[str, Any]] = []
    series: dict[Key, np.ndarray] = {}
    for key in runs:
        if not key[3]:
            series[key] = mean_series(fld(key)[1])
    for key, r in runs.items():
        if key[3]:
            continue
        t, f = fld(key)
        sel = t >= te
        for rn, rk in refs.items():
            fr = fld(rk)[1]
            m, mr = series[key], series[rk]
            thr = 0.1 * mr.max()
            ta, tar = arrival(t, m, thr), arrival(t, mr, thr)  # whole record: local sources already act before t_exclude
            err = m[sel] - mr[sel]
            rows.append(dict(res_m=key[0], dt_s=key[1], scheme=key[2], modes=r["modes"], ref=rn, rel_L2=rel_l2(f[sel], fr[sel]),
                             mean_err_max=float(np.abs(err).max()), mean_err_rms=float(np.sqrt((err**2).mean())),
                             mean_err_max_rel=float(np.abs(err).max() / mr[sel].max()), arrival_s=ta, ref_arrival_s=tar,
                             arrival_err_s=ta - tar, keff_over_k=r["keff_over_k"], keff_over_k_dteff=r["keff_over_k_dteff"],
                             wall_s=r["wall_s"], peak_rss_gb=r["peak_rss_gb"], rel_budget_residual=r["rel_budget_residual"],
                             min_conc=r["min_conc"], n_negative=r["n_negative"], max_substeps=r["max_substeps"]))
    for key, r in runs.items():  # fingerprints vs the finest-mesh exponential fingerprint of the same group
        if not key[3]:
            continue
        t, f = fld(key)
        sel = t >= te
        fr = fld((res_ref, dt_ref, "exponential", key[3]))[1]
        rows.append(dict(res_m=key[0], dt_s=key[1], scheme=key[2], group=key[3], modes="F", ref="F_ref(finest exp)",
                         rel_L2=rel_l2(f[sel], fr[sel]), wall_s=r["wall_s"], peak_rss_gb=r["peak_rss_gb"],
                         rel_budget_residual=r["rel_budget_residual"], min_conc=r["min_conc"], n_negative=r["n_negative"]))
    cols = ["res_m", "dt_s", "scheme", "group", "modes", "ref", "rel_L2", "mean_err_max", "mean_err_rms", "mean_err_max_rel", "arrival_s",
            "ref_arrival_s", "arrival_err_s", "keff_over_k", "keff_over_k_dteff", "wall_s", "peak_rss_gb", "rel_budget_residual",
            "min_conc", "n_negative", "max_substeps"]
    write_csv(out / "stage1_summary.csv", cols, rows)

    # observed orders
    orders: list[dict[str, Any]] = []

    def get(res: float, dt: float, scheme: str) -> Key | None:
        return (res, dt, scheme, "") if (res, dt, scheme, "") in runs else None

    def sel_mean(key: Key) -> float:
        t = fld(key)[0]
        return float(series[key][t >= te].mean())

    for scheme in ("exponential", "vanleer"):
        ks = [get(r, p["dt0"], scheme) for r in sorted(p["res"], reverse=True)]
        if None in ks:
            continue
        for rn in refs:
            e = {k[0]: next(x["rel_L2"] for x in rows if key_of_row(x) == k and x["ref"] == rn) for k in ks if k}  # type: ignore[union-attr]
            for hi, lo in zip(sorted(e, reverse=True), sorted(e, reverse=True)[1:]):
                if e[lo] > 0:  # the reference level itself has zero error: no order from it
                    pv, st = order_status(e[hi], e[lo])
                    orders.append(dict(study="A_space", scheme=scheme, ref=rn, levels=f"{hi:g}->{lo:g} m", method="L2 error ratio", p=pv, status=st))
        if len(ks) >= 3:
            u = [sel_mean(k) for k in ks[:3] if k]
            pv, st = richardson(*u)
            orders.append(dict(study="A_space", scheme=scheme, ref="none", levels="/".join(f"{k[0]:g}" for k in ks[:3] if k) + " m",  # type: ignore[index]
                               method="Richardson on time-mean study-mean", p=pv, status=st))
    dts = sorted(p["b_dts"], reverse=True)
    kb = [get(COARSE, d, "exponential") for d in dts]
    if None not in kb:
        selfref = fld(kb[-1])[1]  # type: ignore[index]
        t = fld(kb[-1])[0]  # type: ignore[index]
        eb = {k[1]: rel_l2(fld(k)[1][t >= te], selfref[t >= te]) for k in kb[:-1]}  # type: ignore[index]
        for a, b in zip(list(eb), list(eb)[1:]):
            pv, st = order_status(eb[a], eb[b])
            orders.append(dict(study="B_time", scheme="exponential", ref=f"self dt={dts[-1]:g}", levels=f"{a:g}->{b:g} s", method="L2 error ratio", p=pv, status=st))
        for i in range(len(dts) - 2):
            pv, st = richardson(*[sel_mean(k) for k in kb[i:i + 3]])  # type: ignore[arg-type]
            orders.append(dict(study="B_time", scheme="exponential", ref="none", levels="/".join(f"{d:g}" for d in dts[i:i + 3]) + " s",
                               method="Richardson on time-mean study-mean", p=pv, status=st))
    write_csv(out / "stage1_orders.csv", ["study", "scheme", "ref", "levels", "method", "p", "status"], orders)

    # source-mass consistency across meshes
    mrows = []
    for g in ("road", "builtup", "external_patch"):
        # same dt only: the profile breakpoints are on the step grid, so mass must not depend on the mesh or dt at all
        vals = {(k[0], k[1], k[2]): float(r[f"mass_{g}_ug"]) for k, r in runs.items() if r[f"mass_{g}_ug"] != "" and (not k[3] or k[3] == g)}
        ref = vals[(res_ref, dt_ref, "exponential")]
        rel = max(abs(v - ref) / ref for v in vals.values())
        ins = [float(r[f"instudy_{g}"]) for r in runs.values() if r[f"instudy_{g}"] != ""]
        mrows.append(dict(group=g, mass_ref_ug=ref, max_rel_diff_over_all_runs=rel, flag=("MISMATCH" if rel > MASS_TOL else "ok"),
                          instudy_fraction_min=min(ins), instudy_fraction_max=max(ins)))
    write_csv(out / "stage1_mass_check.csv", ["group", "mass_ref_ug", "max_rel_diff_over_all_runs", "flag", "instudy_fraction_min",
                                              "instudy_fraction_max"], mrows)
    figures(out, rows, orders, run_id, p)


def key_of_row(x: dict[str, Any]) -> Key:
    return (float(x["res_m"]), float(x["dt_s"]), x["scheme"], x.get("group", "") or "")


def write_csv(path: Path, cols: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def figures(out: Path, rows: list[dict[str, Any]], orders: list[dict[str, Any]], run_id: str, p: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag = f"EXP-015 stage1, {run_id}"
    fd = out / "figures"
    fd.mkdir(exist_ok=True)

    def line(ax: Any, xs: list[float], ys: list[float], label: str, **kw: Any) -> None:
        if len(xs) < 2:
            return
        sl = np.polyfit(np.log(xs), np.log(ys), 1)[0]
        ax.loglog(xs, ys, marker="o", label=f"{label} (slope {sl:.2f})", **kw)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for scheme, ls in (("exponential", "-"), ("vanleer", "--")):
        for rn in ("R_exp", "R_vl"):
            pts = sorted((r["res_m"], r["rel_L2"]) for r in rows if r["scheme"] == scheme and r["ref"] == rn and r["modes"] != "F"
                         and r["dt_s"] == p["dt0"] and r["rel_L2"] > 0 and "A" in r["modes"])
            line(a1, [x for x, _ in pts], [y for _, y in pts], f"{scheme} vs {rn}", ls=ls)
    a1.set(xlabel="mesh size dx (m), dt fixed", ylabel="relative space-time L2 error", title="A: spatial")
    pts = sorted((r["dt_s"], r["rel_L2"]) for r in rows if r["ref"] == "R_exp" and r["res_m"] == COARSE and r["scheme"] == "exponential" and "B" in r["modes"])
    line(a2, [x for x, _ in pts], [y for _, y in pts], "500 m, vs R_exp (mesh error floor)")
    fdts = sorted(p["b_dts"])
    sf = np.load(out / "fields" / f"r{COARSE:g}_dt{fdts[0]:g}_exponential_all.npz")
    te = case_config("production" if p["dt0"] == 25.0 else "smoke")["physics"]["t_exclude"]
    ys = []
    for d in fdts[1:]:
        z = np.load(out / "fields" / f"r{COARSE:g}_dt{d:g}_exponential_all.npz")
        s = z["times"] >= te
        ys.append(rel_l2(z["field"][s], sf["field"][s]))
    line(a2, fdts[1:], ys, f"500 m, vs own dt={fdts[0]:g} s")
    a2.set(xlabel="time step dt (s), 500 m mesh", ylabel="relative space-time L2 error", title="B: temporal")
    for a in (a1, a2):
        a.legend(fontsize=7)
        a.grid(True, which="both", alpha=0.3)
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(fd / "error_vs_dx_dt.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for scheme, mk, key, lab in (("exponential", "o-", "keff_over_k", "exponential: formula at dt"),
                                 ("vanleer", "s--", "keff_over_k_dteff", "vanleer: exponential-scheme formula at dt_eff\n(not a vanleer measurement)")):
        pts = sorted((r["res_m"], r[key]) for r in rows if r["scheme"] == scheme and r["ref"] == "R_exp" and "A" in r["modes"])
        ax.plot([x for x, _ in pts], [y for _, y in pts], mk, mfc="none" if scheme == "vanleer" else None, label=lab)
    ax.set(xscale="log", xlabel="mesh size dx (m), dt fixed", ylabel="K_eff / K (-)", title=f"{tag}\nK_eff/K, wind speed of case d0")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fd / "keff_vs_mesh.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None, runner: Callable[[Key], dict[str, Any]] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=list(PROFILES), default="smoke")
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default="results/phase3-bench/geo125/manifest.json")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    out = Path(a.out or ("results/phase3-convergence" if a.profile == "production" else "results/phase3-convergence-smoke"))
    if a.child:
        print("RESULT " + json.dumps(child(a.profile, a.manifest, tuple(json.loads(a.child)), out)))  # type: ignore[arg-type]
        return 0
    from pravah.geo.prepare import load_region
    region = load_region(a.manifest)
    cen: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for r in PROFILES[a.profile]["res"]:
        dom = region.domain(r)
        cen[r] = (dom.dx * (np.arange(dom.nx) + 0.5), dom.dy * (np.arange(dom.ny) + 0.5))
    assert_nested(cen)
    print(f"common origin (all levels): x0={region.georef.x0:.3f} y0={region.georef.y0:.3f} (UTM {region.georef.epsg}); domain {[(r, region.domain(r).lx, region.domain(r).ly) for r in cen]}")
    if not a.summary_only:
        def cost(row: dict[str, Any]) -> None:
            limit = COST_FACTOR * bench_cost_per_step(float(row["res_m"])) * int(row["steps"]) * int(row["max_substeps"])
            if float(row["wall_s"]) > limit:
                msg = f"COST GUARD: wall {row['wall_s']} s > {COST_FACTOR:g} x benchmark-scaled {limit / COST_FACTOR:.1f} s for {row['res_m']} m dt {row['dt_s']}"
                if a.profile == "production":
                    print(msg, file=sys.stderr)
                    raise SystemExit(4)
                print("warning (smoke):", msg)
        execute(plan(a.profile), out / "stage1_runs.csv", runner or subprocess_runner(a.profile, a.manifest, out), cost)
    summarise(out, a.profile, a.manifest, os.environ.get("RESEARCH_RUN_ID", "local"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

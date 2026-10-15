"""EXP-027b helpers: cost plan (no data needed), local resampling power analysis, and the registered same-ensemble evaluation.

  python scripts/exp027b.py plan  [--draws K --cases N --contention X --eff E]
  python scripts/exp027b.py power --config CFG --work WORK --ref 500:100 --control 2000:400 [--main 250:50] --sizes 12,24 [--reps B]
  python scripts/exp027b.py evaluate --config CFG --work WORK [--out DIR]       (after scripts/convergence_stage2.py --paired)

Design (docs/exp027b_design.md): ALL the PDE work is the driver's `--paired` run (design cases per mesh, pickled in WORK/cases). Here the cases of
all draws are pooled (P cases per mesh) and, per cell (noise fraction x budget), a fixed MENU of candidate layouts is built from greedy
selections at every mesh (the driver's per-draw layouts, greedy on the pooled cases, and B random sub-ensembles of the pool). Each layout is
scored case by case on every mesh. Because the SAME cases are scored at every mesh, mesh effects carry no design-sampling noise; case-to-case
variation enters only through a bootstrap over cases. Statistics, mesh m vs reference (J_m(L) = mean over the cases of the design objective):
  R_m = J_ref(argmin_L J_m(L)) / min_L J_ref(L) - 1                      decision regret of choosing from the menu on mesh m instead of ref
  E_m = median_L | (J_m(L)/median_L' J_m(L')) / (J_ref(L)/median_L' J_ref(L')) - 1 |       shape error: ranking-relevant J error, scale removed
  rho_m = Spearman(J_m, J_ref) over the menu (reported only).
Scoring costs ~1 ms per layout-case; no cluster needed."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # scripts/ is not a package; convergence_stage2 is imported by path
import convergence_stage2 as C  # noqa: E402

D, G, ROOT = C.D, C.G, C.ROOT
MEASURED_S = {"250": 288.0, "500": 43.0, "2000": 6.0}  # local single-core seconds per design case (T-079, this workstation)
CLUSTER_PER_TRIPLE_CPU_H = 23.5 / 48  # RUN-0081 billed (32 cores x 44 min) per 250 + 500 + 2000 m case triple: idle tail included


def mesh(s: str) -> tuple[float, float]:
    a, b = s.split(":")
    return float(a), float(b)


# ------------------------------------------------------------------------------------------------------ plan (cost model)
def plan(a: argparse.Namespace) -> None:
    k, n = a.draws, a.cases
    if a.config:  # dry run: read the draw structure from the config (no data, no solves)
        r = C.load_config(ROOT / a.config)["stage2"]["replicates"]
        k, n = int(r["n"]), int(r["n_cases"])
        print(f"config {a.config}: {k} draws, seeds {r['seed0']}..{r['seed0'] + k - 1}, {n} cases each; units = {3 * k * n} case builds + {3 * k} layout units + {k} objective units")
    cases = k * n
    sec = {"250": a.sec250, "500": a.sec500, "2000": a.sec2000}
    pure = cases * sum(sec.values()) / 3600
    print(f"EXP-027b plan: {k} draws x {n} cases = {cases} cases per mesh, meshes 250/500/2000 m")
    print(f"  PDE work, serial local seconds/case 250/500/2000 = {sec['250']:g}/{sec['500']:g}/{sec['2000']:g}: {pure:.1f} core-h of compute")
    print(f"  x {a.contention:g} (12-worker memory-bandwidth contention measured locally: 78 s vs 43 s per 500 m case) = {pure * a.contention:.1f} core-h")
    print(f"  billed, optimistic (efficiency {a.eff:g}, case-level packing): {pure * a.contention / a.eff:.1f} core-h")
    print(f"  billed, as observed in RUN-0081 ({CLUSTER_PER_TRIPLE_CPU_H:.3f} core-h per case triple): {cases * CLUSTER_PER_TRIPLE_CPU_H:.1f} core-h")
    print(f"  evaluation (greedy + scoring): ~{cases * 1.0 / 3600:.2f} core-h (about 1 s per case for all cells)")
    print("  wall time on 32 cores: observed-rate cost / 32 =", f"{cases * CLUSTER_PER_TRIPLE_CPU_H / 32 * 60:.0f} min")


# ------------------------------------------------------------------------------------------------------ scoring machinery
DELTA, EPS = 0.05, 0.05  # EPS decides (shape error); DELTA (regret) is descriptive. Registered margins (docs/exp027b_design.md section 3)


class Scorer:
    """Per-case objective of a layout on one mesh's cases (cached by (fraction, layout))."""

    def __init__(self, run: Any, m: tuple[float, float], cases: list[Any]) -> None:
        self.st, self.cases, self.cache = run.setup(m), cases, {}
        self.sg, self.alpha = run.frozen_sigma(), run.alpha

    def j(self, f: str, ids: tuple[str, ...]) -> np.ndarray:
        key = (f, ids)
        if key not in self.cache:
            idx = tuple(sorted(self.st.cands.ids.index(i) for i in ids))
            pc = D.score_layout(idx, self.cases, self.st.opts(self.sg[float(f)], self.alpha), "total")["per_case"]
            self.cache[key] = np.array([pc[c.name] for c in self.cases])
        return self.cache[key]


def stats(ja: np.ndarray, jm: np.ndarray) -> tuple[float, float, float]:
    """ja, jm: layouts x cases (ref, mesh m) -> (R, E, rho) on these cases."""
    a, b = ja.mean(1), jm.mean(1)
    shape = (b / np.median(b)) / (a / np.median(a)) - 1.0
    rho = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    return float(a[b.argmin()] / a.min() - 1.0), float(np.median(np.abs(shape))), rho


def cell_table(menu: dict[tuple[str, str], list[tuple[str, ...]]], scorers: dict[str, Scorer], boot: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (f, b), lays in sorted(menu.items()):
        J = {nm: np.array([sc.j(f, ly) for ly in lays]) for nm, sc in scorers.items()}  # layouts x cases
        p = J["ref"].shape[1]
        gs = [rng.integers(0, p, p) for _ in range(boot)]
        for nm in (n for n in J if n != "ref"):
            r, e, rho = stats(J["ref"], J[nm])
            bs = np.array([stats(J["ref"][:, g], J[nm][:, g])[:2] for g in gs])
            rows.append({"noise": f, "budget": b, "mesh": nm, "menu": len(lays), "cases": p, "R": r, "R_lo95": np.percentile(bs[:, 0], 2.5),
                         "R_hi95": np.percentile(bs[:, 0], 97.5), "E": e, "E_lo95": np.percentile(bs[:, 1], 2.5), "E_hi95": np.percentile(bs[:, 1], 97.5),
                         "rho": rho, "J_ratio_median": float(np.median(J[nm].mean(1) / J["ref"].mean(1)))})
    return pd.DataFrame(rows)


def verdict(df: pd.DataFrame, delta: float = DELTA, eps: float = EPS) -> dict[str, Any]:
    """Registered rules (docs/exp027b_design.md section 3)."""
    c = df[df.mesh == "control"]
    fail = c.E_lo95 > eps  # the control fails a cell if its shape error is certainly above the margin (R is descriptive only)
    out: dict[str, Any] = {"delta": delta, "eps": eps, "control_cells_failing": int(fail.sum()), "n_cells": len(c)}
    out["gate_control_can_fail"] = bool(fail.sum() >= 3)
    if (df.mesh == "main").any():
        m = df[df.mesh == "main"]
        ok = m.E_hi95 <= eps
        bad = m.E_lo95 > eps
        out |= {"main_cells_adequate": int(ok.sum()), "main_cells_inadequate": int(bad.sum())}
    out["status"] = ("VOID (control cannot fail)" if not out["gate_control_can_fail"] else "control ok; main not tested" if "main_cells_adequate" not in out else
                     "main ADEQUATE" if out["main_cells_adequate"] == len(m) else "main INADEQUATE" if out["main_cells_inadequate"] > 0 else "main INCONCLUSIVE")
    return out


def load_run(cfg_path: str, work: str, ref: tuple[float, float]) -> Any:
    cfg = C.load_config(ROOT / cfg_path)
    cfg["region"]["manifest"] = str(ROOT / cfg["region"]["manifest"])
    cfg["stage2"]["reference"] = list(ref)
    return C.Run(cfg, Path(work) / "_exp027b_out", Path(work))


def load_cases(work: Path, m: tuple[float, float], nd: int, n: int) -> list[Any]:
    return [pickle.loads((work / "cases" / f"{C.mname(*m)}_{r}_{i}.pkl").read_bytes()) for r in range(1, nd + 1) for i in range(n)]


def build_menu(run: Any, pool: dict[str, list[Any]], meshes: dict[str, tuple[float, float]], extra: list[dict[str, Any]], b_sub: int, sub_size: int,
               seed: int) -> dict[tuple[str, str], list[tuple[str, ...]]]:
    """Per cell: layouts given in `extra` (driver layouts, {fraction: {budget: ids}}), plus greedy on the pool and on b_sub random sub-ensembles, at every mesh."""
    rng = np.random.default_rng(seed)
    sg = run.frozen_sigma()
    menu: dict[tuple[str, str], set[tuple[str, ...]]] = {}

    def add(lay: dict[str, dict[str, list[str]]]) -> None:
        for f, pb in lay.items():
            for b, ids in pb.items():
                menu.setdefault((f, b), set()).add(tuple(sorted(ids)))
    for lay in extra:
        add(lay)
    p = len(pool["ref"])
    subsets = [np.arange(p)] + [rng.permutation(p)[:sub_size] for _ in range(b_sub)]
    for nm, m in meshes.items():
        st = run.setup(m)
        for ss in subsets:
            add(C.greedy_layouts(st, [pool[nm][g] for g in ss], sg, run.alpha, run.budgets))
    return {k: sorted(v) for k, v in menu.items()}


# ------------------------------------------------------------------------------------------------------ power (resampling) and evaluation
def power(a: argparse.Namespace) -> None:
    ref, ctrl = mesh(a.ref), mesh(a.control)
    meshes = {"ref": ref, "control": ctrl} | ({"main": mesh(a.main)} if a.main else {})
    run, work = load_run(a.config, a.work, ref), Path(a.work)
    nd, nc = int(run.s2["replicates"]["n"]), int(run.s2["replicates"]["n_cases"])
    pool = {nm: load_cases(work, m, nd, nc) for nm, m in meshes.items()}
    scorers = {nm: Scorer(run, m, pool[nm]) for nm, m in meshes.items()}
    rng, rows = np.random.default_rng(a.seed), []
    for n in (int(s) for s in a.sizes.split(",")):
        for rep in range(a.reps):
            idx = rng.permutation(len(pool["ref"]))[:n]
            sub = {nm: [pool[nm][g] for g in idx] for nm in pool}
            sc = {nm: Scorer(run, m, sub[nm]) for nm, m in meshes.items()}
            menu = build_menu(run, sub, meshes, [], a.b_sub, n // 2, seed=rep)
            df = cell_table(menu, sc, a.boot, seed=rep)
            df["N"], df["rep"] = n, rep
            rows.append(df)
            print(f"N={n} rep {rep + 1}/{a.reps}: control cells failing {int((df[df.mesh == 'control'].E_lo95 > EPS).sum())}/4", flush=True)
    res = pd.concat(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "power_rows.csv", index=False)
    print(res.groupby(["N", "mesh", "noise", "budget"])[["R", "R_lo95", "E", "E_lo95", "rho"]].median().round(3).to_string())
    g = res[res.mesh == "control"].assign(f=lambda d: d.E_lo95 > EPS).groupby(["N", "rep"]).f.sum()
    print("control cells failing per partition, by N (gate needs >= 3 of 4):\n", g.groupby("N").agg(["mean", "min", lambda s: (s >= 3).mean()]).to_string())


def evaluate(a: argparse.Namespace) -> None:
    cfg = C.load_config(ROOT / a.config)
    pm = cfg["stage2"]["paired"]
    meshes = {"ref": tuple(float(v) for v in pm["ref"]), "main": tuple(float(v) for v in pm["main"]), "control": tuple(float(v) for v in pm["control"])}
    run, work = load_run(a.config, a.work, meshes["ref"]), Path(a.work)
    nd, nc = int(run.s2["replicates"]["n"]), int(run.s2["replicates"]["n_cases"])
    pool = {nm: load_cases(work, m, nd, nc) for nm, m in meshes.items()}
    extra = [json.loads((work / f"pair_{C.mname(*m)}_{r}.json").read_text())["layouts"] for m in meshes.values() for r in range(1, nd + 1)]
    menu = build_menu(run, pool, meshes, extra, a.b_sub, len(pool["ref"]) // nd, seed=27)
    df = cell_table(menu, {nm: Scorer(run, m, pool[nm]) for nm, m in meshes.items()}, a.boot)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "exp027b_cells.csv", index=False)
    v = verdict(df)
    (out / "exp027b_verdict.json").write_text(json.dumps(v, indent=1))
    print(df.round(4).to_string())
    print(json.dumps(v, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--config", default=None, help="dry run: take draws/cases from this config")
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--cases", type=int, default=24)
    p.add_argument("--sec250", type=float, default=MEASURED_S["250"])
    p.add_argument("--sec500", type=float, default=MEASURED_S["500"])
    p.add_argument("--sec2000", type=float, default=MEASURED_S["2000"])
    p.add_argument("--contention", type=float, default=1.8)
    p.add_argument("--eff", type=float, default=0.7)
    for name in ("power", "evaluate"):
        q = sub.add_parser(name)
        q.add_argument("--config", required=True)
        q.add_argument("--work", required=True)
        q.add_argument("--out", default=None)
        q.add_argument("--boot", type=int, default=400)
        q.add_argument("--b-sub", type=int, default=8)
        if name == "power":
            q.add_argument("--ref", required=True)
            q.add_argument("--control", required=True)
            q.add_argument("--main", default=None)
            q.add_argument("--sizes", default="12,24")
            q.add_argument("--reps", type=int, default=4)
            q.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(a)
    else:
        a.out = a.out or str(Path(a.work) / "exp027b_out")
        power(a) if a.cmd == "power" else evaluate(a)


if __name__ == "__main__":
    main()

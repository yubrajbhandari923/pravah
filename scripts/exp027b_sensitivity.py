"""EXP-027b POST-HOC SENSITIVITY ANALYSIS (not part of the registered evaluation; does not change scripts/exp027b.py semantics).

  python scripts/exp027b_sensitivity.py --config configs/convergence_paired_padded_exp027b.json --work WORK --out DIR [--workers N]

Motivation (F-075, review F-073/R1): the percentile interval of the median-absolute shape error E is biased toward INADEQUATE; the 0.05/5 lower bound
(0.0516) is within bootstrap Monte Carlo error; the stage-1 -> stage-2 shift may be driven by few cases; the menu is built and scored on the same cases
and partly from the 2 km control mesh. Reuses the loading / menu / scoring code of scripts/exp027b.py (E, J ratio, verdict thresholds identical).
Everything here is exploratory: a different answer from the registered one does NOT change the registered outcome, it qualifies how much it means.

Analyses, each for the 4 cells and for main (500 m) and control (2 km) against the 250 m reference:
  A1  registered percentile interval re-run with SEEDS different bootstrap seeds at B=400 (range of E_lo95/E_hi95, fraction INADEQUATE) and B=10000
  A2  basic (reflected) and BCa intervals at B=10000
  A3  per draw (E, J ratio), leave-one-draw-out, and drop-top-5-influential-cases (influence = E - E without the case)
  A4  menu restricted to layouts derived from the reference or main mesh (no layout whose only source is the 2 km control)
  A5  cross-fit: menu built (greedy, driver layouts, random sub-ensembles) on the other draws only, scored on the held-out draw (each rotation)
  A6  noise-debiased E: split-half cross-fit of the case set, d_L = sqrt(max(s1_L * s2_L, 0)) from the signed shape errors of two independent halves
Only the PDE-free scoring/greedy of exp027b.py is used (no new solves; a response-cache dir is created fresh under --out).
Outputs: DIR/sens_rows.csv, DIR/sens_summary.json; compact tables are printed (the cluster collects logs only)."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EPS = 0.05  # registered margin (docs/exp027b_design.md section 3); same value as exp027b.EPS
ND = NormalDist()


# ---------------------------------------------------------------------------------------------- pure numpy core (tested without any data)
def shape_err(ja: np.ndarray, jm: np.ndarray) -> np.ndarray:
    """Signed shape error per layout; ja, jm: layouts x cases (same definition as exp027b.stats)."""
    a, b = ja.mean(1), jm.mean(1)
    return (b / np.median(b)) / (a / np.median(a)) - 1.0


def E_of(ja: np.ndarray, jm: np.ndarray) -> float:
    return float(np.median(np.abs(shape_err(ja, jm))))


def jr_of(ja: np.ndarray, jm: np.ndarray) -> float:
    return float(np.median(jm.mean(1) / ja.mean(1)))


def classify(lo: float, hi: float, eps: float = EPS) -> str:
    return "INADEQUATE" if lo > eps else "ADEQUATE" if hi <= eps else "INCONCLUSIVE"


def boot_idx(rng: np.random.Generator, B: int, p: int) -> np.ndarray:
    """B x p case indices; B sequential rng.integers(0, p, p) calls, i.e. the exact stream of exp027b.cell_table."""
    return np.stack([rng.integers(0, p, p) for _ in range(B)])


def boot_stats(ja: np.ndarray, jm: np.ndarray, idx: np.ndarray, chunk: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """(E_b, J-ratio_b) for every bootstrap row of `idx` (B x p), via resample counts (matrix products, no layouts x B x p tensor)."""
    p = ja.shape[1]
    es, js = [], []
    for s in range(0, idx.shape[0], chunk):
        ix = idx[s:s + chunk]
        cnt = np.zeros((ix.shape[0], p))
        np.add.at(cnt, (np.arange(ix.shape[0])[:, None], ix), 1.0)
        a, b = ja @ cnt.T / p, jm @ cnt.T / p  # layouts x B
        sh = (b / np.median(b, 0)) / (a / np.median(a, 0)) - 1.0
        es.append(np.median(np.abs(sh), 0))
        js.append(np.median(b / a, 0))
    return np.concatenate(es), np.concatenate(js)


def intervals(theta: float, boot: np.ndarray, jack: np.ndarray | None, level: float = 0.95) -> dict[str, tuple[float, float]]:
    """percentile, basic (reflected) and BCa intervals of a bootstrap distribution."""
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    pl, ph = np.percentile(boot, [100 * lo_q, 100 * hi_q])
    out = {"pct": (float(pl), float(ph)), "basic": (float(2 * theta - ph), float(2 * theta - pl))}
    if jack is not None and len(jack) > 2:
        frac = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / len(boot)
        z0 = ND.inv_cdf(float(np.clip(frac, 1e-4, 1 - 1e-4)))
        d = jack.mean() - jack
        den = 6.0 * (np.sum(d ** 2) ** 1.5)
        acc = float(np.sum(d ** 3) / den) if den > 0 else 0.0
        qs = []
        for q in (lo_q, hi_q):
            z = z0 + ND.inv_cdf(q)
            qs.append(ND.cdf(z0 + z / (1 - acc * z)))
        out["bca"] = (float(np.percentile(boot, 100 * qs[0])), float(np.percentile(boot, 100 * qs[1])))
    return out


def jackknife_E(ja: np.ndarray, jm: np.ndarray) -> np.ndarray:
    p = ja.shape[1]
    return np.array([E_of(np.delete(ja, i, 1), np.delete(jm, i, 1)) for i in range(p)])


def summarize(ja: np.ndarray, jm: np.ndarray, B: int, seed: int, types: tuple[str, ...] = ("pct",)) -> dict[str, Any]:
    """E, J ratio with percentile (and optionally basic/BCa) intervals for one layouts x cases pair."""
    p = ja.shape[1]
    e, jr = E_of(ja, jm), jr_of(ja, jm)
    be, bj = boot_stats(ja, jm, boot_idx(np.random.default_rng(seed), B, p))
    jack = jackknife_E(ja, jm) if "bca" in types else None
    iv = intervals(e, be, jack)
    r: dict[str, Any] = {"n_layouts": ja.shape[0], "n_cases": p, "E": e, "J_ratio": jr, "JR_lo95": float(np.percentile(bj, 2.5)), "JR_hi95": float(np.percentile(bj, 97.5))}
    for t in types:
        r[f"E_lo95_{t}"], r[f"E_hi95_{t}"] = iv[t]
    r["verdict"] = classify(*iv[types[0]])
    return r


def influence(ja: np.ndarray, jm: np.ndarray) -> np.ndarray:
    """E - E(without case i), per case: the contribution of each case to the point estimate."""
    return E_of(ja, jm) - jackknife_E(ja, jm)


def debiased_E(ja: np.ndarray, jm: np.ndarray, splits: int, rng: np.random.Generator) -> float:
    """Split-half cross-fit E: mean over random splits of median_L sqrt(max(s1_L * s2_L, 0)); s_h = signed shape error from half h only.
    Case noise that adds |noise| to the plain E enters s1 and s2 independently, so E[s1 s2] = s^2 (up to normalisation noise)."""
    p = ja.shape[1]
    vals = []
    for _ in range(splits):
        perm = rng.permutation(p)
        h1, h2 = perm[: p // 2], perm[p // 2:2 * (p // 2)]
        s1, s2 = shape_err(ja[:, h1], jm[:, h1]), shape_err(ja[:, h2], jm[:, h2])
        vals.append(float(np.median(np.sqrt(np.maximum(s1 * s2, 0.0)))))
    return float(np.mean(vals))


def debiased_interval(ja: np.ndarray, jm: np.ndarray, B: int, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile interval of the debiased E: each replicate = fresh random split of the cases, then an independent bootstrap within each half."""
    p = ja.shape[1]
    vals = []
    for _ in range(B):
        perm = rng.permutation(p)
        h1, h2 = perm[: p // 2], perm[p // 2:2 * (p // 2)]
        g1, g2 = rng.choice(h1, len(h1)), rng.choice(h2, len(h2))
        s1, s2 = shape_err(ja[:, g1], jm[:, g1]), shape_err(ja[:, g2], jm[:, g2])
        vals.append(float(np.median(np.sqrt(np.maximum(s1 * s2, 0.0)))))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------------------------- analyses on precomputed J matrices
Cell = tuple[str, str]


def row(analysis: str, variant: str, cell: Cell, mesh: str, **kw: Any) -> dict[str, Any]:
    return {"analysis": analysis, "variant": variant, "noise": cell[0], "budget": cell[1], "mesh": mesh, **kw}


def analyse_cell(cell: Cell, J: dict[str, np.ndarray], draw_of: np.ndarray, names: list[str], keep: np.ndarray, o: argparse.Namespace) -> list[dict[str, Any]]:
    """A1-A4, A6 for one cell. J[mesh]: layouts x cases (full menu); keep: boolean mask of layouts derived from ref/main."""
    rows: list[dict[str, Any]] = []
    ja = J["ref"]
    p = ja.shape[1]
    meshes = [m for m in ("main", "control") if m in J]
    stream = np.random.default_rng  # registered stream: the replicates are drawn once per cell and shared by the meshes
    for m in meshes:
        jm = J[m]
        # A1 seeds at B=400 (seed 0 reproduces the registered run's stream) and B=10000
        lo, hi, inad = [], [], []
        for s in [0] + list(range(1, o.seeds + 1)):
            be, _ = boot_stats(ja, jm, boot_idx(stream(s), o.boot_reg, p))
            l, h = np.percentile(be, [2.5, 97.5])
            lo.append(l), hi.append(h), inad.append(l > EPS)
        lo_a, hi_a, in_a = np.array(lo), np.array(hi), np.array(inad)
        e = E_of(ja, jm)
        rows.append(row("A1", f"pct B={o.boot_reg} seed0 (registered stream)", cell, m, E=e, E_lo95=lo_a[0], E_hi95=hi_a[0], verdict=classify(lo_a[0], hi_a[0]),
                        n_cases=p, n_layouts=ja.shape[0], J_ratio=jr_of(ja, jm)))
        rows.append(row("A1", f"pct B={o.boot_reg} x{o.seeds} seeds", cell, m, E=e, E_lo95=float(lo_a[1:].mean()), E_hi95=float(hi_a[1:].mean()), E_lo95_min=float(lo_a[1:].min()),
                        E_lo95_max=float(lo_a[1:].max()), E_hi95_min=float(hi_a[1:].min()), E_hi95_max=float(hi_a[1:].max()), frac_inadequate=float(in_a[1:].mean()),
                        verdict=f"{int(in_a[1:].sum())}/{o.seeds} INADEQUATE", n_cases=p, n_layouts=ja.shape[0]))
        # A1/A2 at large B: pct, basic, bca
        big = summarize(ja, jm, o.boot_big, seed=777, types=("pct", "basic", "bca"))
        for t in ("pct", "basic", "bca"):
            rows.append(row("A2" if t != "pct" else "A1", f"{t} B={o.boot_big}", cell, m, E=big["E"], E_lo95=big[f"E_lo95_{t}"], E_hi95=big[f"E_hi95_{t}"],
                            verdict=classify(big[f"E_lo95_{t}"], big[f"E_hi95_{t}"]), n_cases=p, n_layouts=ja.shape[0], J_ratio=big["J_ratio"], JR_lo95=big["JR_lo95"], JR_hi95=big["JR_hi95"]))
        # A3 per draw, leave-one-draw-out
        draws = sorted(set(draw_of.tolist()))
        for d in draws:
            for tag, sel in ((f"draw {d} only", draw_of == d), (f"without draw {d}", draw_of != d)):
                r = summarize(ja[:, sel], jm[:, sel], o.boot, seed=100 + d)
                rows.append(row("A3", tag, cell, m, E=r["E"], E_lo95=r["E_lo95_pct"], E_hi95=r["E_hi95_pct"], verdict=r["verdict"], n_cases=r["n_cases"], n_layouts=r["n_layouts"],
                                J_ratio=r["J_ratio"], JR_lo95=r["JR_lo95"], JR_hi95=r["JR_hi95"]))
        inf = influence(ja, jm)
        top = np.argsort(-np.abs(inf))[: o.top]
        sel = np.ones(p, bool)
        sel[top] = False
        r = summarize(ja[:, sel], jm[:, sel], o.boot, seed=200)
        rows.append(row("A3", f"drop top-{o.top} influential cases", cell, m, E=r["E"], E_lo95=r["E_lo95_pct"], E_hi95=r["E_hi95_pct"], verdict=r["verdict"], n_cases=r["n_cases"],
                        n_layouts=r["n_layouts"], J_ratio=r["J_ratio"], dropped=";".join(f"{names[i]}({inf[i]:+.4f})" for i in top), influence_sum_top=float(inf[top].sum()),
                        influence_max=float(np.abs(inf).max()), E_full=e))
        # A4 menu without control-derived layouts
        if keep.sum() >= 3:
            r = summarize(ja[keep], jm[keep], o.boot, seed=300)
            rows.append(row("A4", "menu = layouts from ref/main only", cell, m, E=r["E"], E_lo95=r["E_lo95_pct"], E_hi95=r["E_hi95_pct"], verdict=r["verdict"], n_cases=r["n_cases"],
                            n_layouts=r["n_layouts"], J_ratio=r["J_ratio"], JR_lo95=r["JR_lo95"], JR_hi95=r["JR_hi95"], n_layouts_full=ja.shape[0]))
        # A6 debiased
        rng = np.random.default_rng(400)
        ed = debiased_E(ja, jm, o.splits, rng)
        lo6, hi6 = debiased_interval(ja, jm, o.boot_deb, rng)
        rows.append(row("A6", f"split-half cross-fit ({o.splits} splits)", cell, m, E=ed, E_lo95=lo6, E_hi95=hi6, verdict=classify(lo6, hi6), n_cases=p, n_layouts=ja.shape[0], E_plain=e))
    return rows


# ---------------------------------------------------------------------------------------------- data-dependent part (reuses scripts/exp027b.py)
_CTX: dict[str, Any] = {}


def _greedy_task(key: tuple[str, tuple[int, ...]]) -> tuple[tuple[str, tuple[int, ...]], dict[str, dict[str, list[str]]]]:
    import convergence_stage2 as C
    nm, ss = key
    c = _CTX
    return key, C.greedy_layouts(c["st"][nm], [c["pool"][nm][g] for g in ss], c["sg"], c["run"].alpha, c["run"].budgets)


def make_subsets(rng: np.random.Generator, pool_idx: np.ndarray, b_sub: int, sub_size: int) -> list[np.ndarray]:
    """Same recipe as exp027b.build_menu: the whole pool, then b_sub random sub-ensembles (indices into the pooled case list)."""
    return [pool_idx] + [pool_idx[rng.permutation(len(pool_idx))[:sub_size]] for _ in range(b_sub)]


def assemble(extra: list[tuple[str, dict[str, Any]]], greedy: dict[tuple[str, tuple[int, ...]], Any], subsets: list[np.ndarray],
             mesh_names: list[str]) -> dict[Cell, dict[tuple[str, ...], set[str]]]:
    """Menu with provenance: cell -> layout -> set of source meshes (driver layout of that mesh, or greedy at that mesh)."""
    menu: dict[Cell, dict[tuple[str, ...], set[str]]] = {}

    def add(lay: dict[str, dict[str, list[str]]], src: str) -> None:
        for f, pb in lay.items():
            for b, ids in pb.items():
                menu.setdefault((f, b), {}).setdefault(tuple(sorted(ids)), set()).add(src)
    for src, lay in extra:
        add(lay, src)
    for nm in mesh_names:
        for ss in subsets:
            add(greedy[(nm, tuple(int(i) for i in ss))], nm)
    return menu


def run_all(o: argparse.Namespace) -> None:
    import convergence_stage2 as C
    import exp027b as X
    t0 = time.time()

    def say(msg: str) -> None:
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    cfg = C.load_config(X.ROOT / o.config)
    pm = cfg["stage2"]["paired"]
    meshes = {"ref": tuple(float(v) for v in pm["ref"]), "main": tuple(float(v) for v in pm["main"]), "control": tuple(float(v) for v in pm["control"])}
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(o.work)  # read-only: only cases/ and pair_*.json are read; the response cache is a fresh dir under --out
    cfg["region"]["manifest"] = str(X.ROOT / cfg["region"]["manifest"])
    cfg["stage2"]["reference"] = list(meshes["ref"])
    run = C.Run(cfg, out / "_run", out / "_run_work")
    nd = o.draws or int(run.s2["replicates"]["n"])
    nc = o.cases or int(run.s2["replicates"]["n_cases"])
    pool = {nm: X.load_cases(work, m, nd, nc) for nm, m in meshes.items()}
    P = len(pool["ref"])
    draw_of = np.repeat(np.arange(1, nd + 1), nc)
    names = [c.name for c in pool["ref"]]
    pairs = {(nm, r): json.loads((work / f"pair_{C.mname(*m)}_{r}.json").read_text())["layouts"] for nm, m in meshes.items() for r in range(1, nd + 1)}
    say(f"loaded {nd} draws x {nc} cases x {len(meshes)} meshes; menu seed {o.menu_seed}, b_sub {o.b_sub}")

    # menus: full (registered recipe, seed 27) and one per held-out draw (built on the other draws only)
    plans: dict[str, tuple[list[tuple[str, dict[str, Any]]], list[np.ndarray]]] = {}
    plans["full"] = ([(nm, pairs[(nm, r)]) for nm in meshes for r in range(1, nd + 1)], make_subsets(np.random.default_rng(o.menu_seed), np.arange(P), o.b_sub, P // nd))
    for d in range(1, nd + 1):
        tr = np.where(draw_of != d)[0]
        plans[f"cf{d}"] = ([(nm, pairs[(nm, r)]) for nm in meshes for r in range(1, nd + 1) if r != d], make_subsets(np.random.default_rng(o.menu_seed), tr, o.b_sub, nc))
    keys = sorted({(nm, tuple(int(i) for i in ss)) for _, subs in plans.values() for ss in subs for nm in meshes})
    _CTX.update(run=run, pool=pool, sg=run.frozen_sigma(), st={nm: run.setup(m) for nm, m in meshes.items()})
    say(f"greedy tasks: {len(keys)} on {o.workers} workers")
    if o.workers > 1:
        with mp.get_context("fork").Pool(o.workers) as pl:
            greedy = dict(pl.map(_greedy_task, keys, chunksize=1))
    else:
        greedy = dict(map(_greedy_task, keys))
    say("greedy done")
    menus = {k: assemble(ex, greedy, subs, list(meshes)) for k, (ex, subs) in plans.items()}
    scorers = {nm: X.Scorer(run, m, pool[nm]) for nm, m in meshes.items()}

    def Jmat(cell: Cell, lays: list[tuple[str, ...]]) -> dict[str, np.ndarray]:
        return {nm: np.array([sc.j(cell[0], ly) for ly in lays]) for nm, sc in scorers.items()}  # layouts x ALL pooled cases

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"args": {k: v for k, v in vars(o).items()}, "cells": {}}
    cells = sorted(menus["full"])
    # consistency with the registered menu (same recipe and seed => identical layouts)
    reg = X.build_menu(run, pool, meshes, [lay for (_, lay) in plans["full"][0]], o.b_sub, P // nd, seed=o.menu_seed) if o.check_registered else None
    for cell in cells:
        pm_full = menus["full"][cell]
        lays = sorted(pm_full)
        if reg is not None:
            assert lays == reg[cell], f"menu differs from exp027b.build_menu in cell {cell}"
        J = Jmat(cell, lays)
        keep = np.array([bool(pm_full[ly] & {"ref", "main"}) for ly in lays])
        summary["cells"]["/".join(cell)] = {"menu": len(lays), "menu_ref_main": int(keep.sum()), "sources": {s: int(sum(s in pm_full[ly] for ly in lays)) for s in meshes}}
        rows += analyse_cell(cell, J, draw_of, names, keep, o)
        # A5 cross-fit: menu from the other draws, scored on the held-out draw
        for d in range(1, nd + 1):
            lays_d = sorted(menus[f"cf{d}"][cell])
            Jd = Jmat(cell, lays_d)
            sel = draw_of == d
            for m in ("main", "control"):
                r = summarize(Jd["ref"][:, sel], Jd[m][:, sel], o.boot, seed=500 + d)
                rows.append(row("A5", f"cross-fit: menu from other draws, scored on draw {d}", cell, m, E=r["E"], E_lo95=r["E_lo95_pct"], E_hi95=r["E_hi95_pct"], verdict=r["verdict"],
                                n_cases=r["n_cases"], n_layouts=r["n_layouts"], J_ratio=r["J_ratio"], JR_lo95=r["JR_lo95"], JR_hi95=r["JR_hi95"]))
        a5 = [r for r in rows if r["analysis"] == "A5" and (r["noise"], r["budget"]) == cell]
        for m in ("main", "control"):
            es = [r["E"] for r in a5 if r["mesh"] == m and "mean over" not in r["variant"]]
            rows.append(row("A5", f"cross-fit: mean E over the {nd} held-out draws", cell, m, E=float(np.mean(es)), n_cases=P, n_layouts=int(np.mean([r["n_layouts"] for r in a5 if r["mesh"] == m])),
                            verdict="(point estimate only)"))
        say(f"cell {cell} done ({len(lays)} layouts)")
    df = pd.DataFrame(rows)
    df.to_csv(out / "sens_rows.csv", index=False)
    summary["runtime_s"] = time.time() - t0
    (out / "sens_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    report(df, summary)


def report(df: pd.DataFrame, summary: dict[str, Any]) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("\n===== EXP-027b POST-HOC SENSITIVITY (exploratory; the registered outcome is unchanged by anything below) =====")
    print("menu sizes:", json.dumps(summary["cells"]))
    for a, title in (("A1", "A1/A2 interval type and bootstrap seed"), ("A2", None), ("A3", "A3 per draw / leave-one-draw-out / top-case influence"), ("A4", "A4 menu without 2 km-derived layouts"),
                     ("A5", "A5 cross-fit (out-of-sample menu)"), ("A6", "A6 noise-debiased E (split-half)")):
        if title is None:
            continue
        d = df[df.analysis.isin(["A1", "A2"])] if a == "A1" else df[df.analysis == a]
        cols = [c for c in ("noise", "budget", "mesh", "variant", "n_cases", "n_layouts", "E", "E_lo95", "E_hi95", "E_lo95_min", "E_lo95_max", "E_hi95_min", "E_hi95_max", "frac_inadequate", "J_ratio", "verdict") if c in d]
        print(f"\n--- {title} ---")
        print(d.sort_values(["mesh", "noise", "budget"], kind="stable")[cols].round(4).to_string(index=False))
    t = df[df.variant.str.startswith("drop top")]
    if len(t):
        print("\n--- A3 dropped cases (case(influence = E - E without case)) ---")
        for _, r in t.iterrows():
            print(f"{r.noise}/{r.budget} {r.mesh}: E {r.E_full:.4f} -> {r.E:.4f}; {r.dropped}")


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--work", required=True, help="registered run's paired_work dir (read-only: cases/ and pair_*.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1, help="processes for the greedy menu builds")
    ap.add_argument("--seeds", type=int, default=20, help="A1: number of alternative bootstrap seeds at B=--boot-reg")
    ap.add_argument("--boot-reg", type=int, default=400)
    ap.add_argument("--boot-big", type=int, default=10000)
    ap.add_argument("--boot", type=int, default=2000, help="B for the A3-A5 intervals")
    ap.add_argument("--boot-deb", type=int, default=1000, help="B for the A6 interval")
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--b-sub", type=int, default=8)
    ap.add_argument("--menu-seed", type=int, default=27)
    ap.add_argument("--draws", type=int, default=0, help="override the config's draws (pilot data)")
    ap.add_argument("--cases", type=int, default=0, help="override the config's cases per draw (pilot data)")
    ap.add_argument("--check-registered", action="store_true", help="assert that the full menu equals exp027b.build_menu (serial greedy: slow, for local checks)")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run_all(parse())

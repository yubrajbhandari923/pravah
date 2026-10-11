# pyright: reportGeneralTypeIssues=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportOperatorIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportReturnType=false, reportAssignmentType=false
"""EXP-020: exact vs greedy site selection and replicated population-aware design on the Phase-1 toy domain (synthetic).

    python scripts/population_design_exact.py --profile {smoke,pilot,production} [--config FILE] [--out DIR] [--summary-only]

Follow-up to EXP-017 (scripts/population_design.py, imported, not edited).  Differences:
* Selection.  Each weighted design objective (area, mixed_0.5, population, capped) is minimised over ALL candidate subsets for
  budgets <= search.exact_max_budget; for larger budgets by greedy + best-improvement single-swap local search from the greedy
  layout and n_restarts random layouts, best found reported (NOT provably optimal).  The objective is the one design.greedy_trace
  uses: mean over design cases and noise reps of the weighted MSE of the bounded-LS (alpha-ridge, q >= 0) fit.  The fit is
  re-implemented in closed form (3 parameters; unconstrained solve, else best feasible active set) and cross-checked against
  design.score_layout / design.greedy_trace on every placement.
* Replicates.  n_placements source placements (3 sources drawn by seed on the Domain; new response bases, cached) x
  len(fixture_draw_seeds) population draws per fixture type; design / validation / test cases are drawn per placement with fresh
  disjoint seeds; alpha is chosen once per placement on VALIDATION with the area objective, then everything is frozen (hash).
* Primary unit of analysis = setting (placement x fixture draw): test metrics are averaged over test cases first, paired
  differences against the exact-area layout are bootstrapped over settings.
Order: seeds/out guards -> bases -> alpha per placement (validation only) -> freeze -> per placement: selection, check, test
cases, evaluation -> freeze guard -> tables, figures.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pravah import design as D
from pravah import population as P
from pravah.config import _merge, load_config
from pravah.geo.regionio import GridSpec
from pravah.inverse import SCALE_BY_KIND, ResponseCache
from pravah.io import provenance, write_json

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("population_design", ROOT / "scripts/population_design.py")
PD = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["population_design"] = PD
_spec.loader.exec_module(PD)  # type: ignore[union-attr]

STAMP = "EXP-020"
CONFIGS = {"smoke": "configs/population_toy_exact_smoke.json", "pilot": "configs/population_toy_exact.json",
           "production": "configs/population_toy_exact.json"}
OUTS = {"smoke": "results/phase3-population-toy-exact-smoke", "pilot": "results/phase3-population-toy-exact-pilot",
        "production": "results/phase3-population-toy-exact"}
METRICS = ("pop_rmse", "area_rmse", "lowq_rmse", "worst_cell_rmse")
BOOT_LABEL = ("percentile bootstrap over settings (placement x fixture draw; settings in one placement share cases, so they are "
              "not independent; not a confidence interval for new scenarios)")
T0 = time.perf_counter()
QS = SCALE_BY_KIND["emission"]


def say(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ seeds, placements, guards
def draw_sources(lx: float, ly: float, n: int, seed: int, margin: float, min_sep: float) -> np.ndarray:
    """n source positions uniform in the inner (1 - 2 margin) box, rejection-sampled for pairwise distance >= min_sep."""
    rng = np.random.default_rng(seed)
    lo, hi = np.array([margin * lx, margin * ly]), np.array([(1 - margin) * lx, (1 - margin) * ly])
    for _ in range(100000):
        pts = np.round(rng.uniform(lo, hi, (n, 2)), 1)
        d = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
        if (d[np.triu_indices(n, 1)] >= min_sep).all():
            return pts
    raise RuntimeError("could not place sources with the requested separation")


def placement_cfg(cfg: dict[str, Any], p: int) -> dict[str, Any]:
    """Config of placement p: sources moved, seeds shifted by p * stride (design/validation/test seeds and noise-seed bases)."""
    r, d = cfg["replicates"], cfg["design"]
    out = json.loads(json.dumps(cfg))
    dom = out["scenario"]["domain"]
    pts = draw_sources(float(dom["lx"]), float(dom["ly"]), len(out["scenario"]["sources"]), int(r["placement_seed"]) + p,
                       float(r["margin"]), float(r["min_sep"]))
    for s, (x, y) in zip(out["scenario"]["sources"], pts):
        s["x"], s["y"] = float(x), float(y)
    for role in ("design", "validation", "test"):
        out["design"][f"{role}_seed"] = int(d[f"{role}_seed"]) + p * int(r["seed_stride"])
        out["design"][f"{role}_noise_seed"] = int(d[f"{role}_noise_seed"]) + p * int(r["noise_stride"])
    out["population"]["fixture_seeds"] = {"uniform": 0}  # PD.make_setup needs the uniform fixture; real draws are made here
    return out


def seed_blocks(cfg: dict[str, Any]) -> list[tuple[str, int, int]]:
    """(label, first, n) of every case-seed and noise-seed block of the experiment: all placements x roles."""
    d, r = cfg["design"], cfg["replicates"]
    n = {"design": d["n_design"], "validation": d["n_validation"], "test": d["n_test"]}
    blocks = []
    for p in range(int(r["n_placements"])):
        for role in n:
            blocks.append((f"{role}_seed p{p}", int(d[f"{role}_seed"]) + p * int(r["seed_stride"]), 1))
            blocks.append((f"{role}_noise p{p}", int(d[f"{role}_noise_seed"]) + p * int(r["noise_stride"]), int(n[role])))
    return blocks


def historic_seeds() -> list[tuple[int, int]]:
    """(first, width) of every seed in the other configs (case seeds: width 1; noise seeds: width 1000)."""
    out = []

    def walk(o: Any, key: str = "") -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, key)
        elif isinstance(o, int) and not isinstance(o, bool) and "seed" in key:
            out.append((o, 1000 if "noise" in key else 1))
    for f in sorted((ROOT / "configs").glob("*.json")):
        if not f.name.startswith("population_toy_exact"):
            walk(json.loads(f.read_text()))
    return out


def check_seeds(cfg: dict[str, Any], historic: list[tuple[int, int]] | None = None) -> None:
    """All case/noise seed blocks pairwise disjoint, and disjoint from the seeds of every other config (before any solve)."""
    blocks = seed_blocks(cfg)
    ranges = [(lab, set(range(a, a + n))) for lab, a, n in blocks]
    for i, (la, ra) in enumerate(ranges):
        for lb, rb in ranges[i + 1:]:
            if ra & rb:
                raise ValueError(f"seed blocks overlap: {la} / {lb}")
    for a, w in (historic_seeds() if historic is None else historic):
        for lab, r in ranges:
            if ("noise" in lab) == (w > 1) and r & set(range(a, a + w)):  # noise seeds vs noise seeds, case seeds vs case seeds
                raise ValueError(f"{lab} overlaps seed {a} (+{w}) used by another config")


def frozen_payload(cfg: dict[str, Any], alphas: dict[str, float] | None) -> dict[str, Any]:
    out = {**PD.frozen_payload(cfg, 0.0), "replicates": cfg["replicates"], "search": cfg["search"]}
    out.pop("alpha")
    return {**out, "alphas": alphas}


def frozen_hash(cfg: dict[str, Any], alphas: dict[str, float] | None) -> str:
    return hashlib.sha256(json.dumps(frozen_payload(cfg, alphas), sort_keys=True, default=str).encode()).hexdigest()


def assert_frozen(out: Path, cfg: dict[str, Any], alphas: dict[str, float]) -> str:
    rec = json.loads((out / "provenance.json").read_text())["frozen_config_hash"]
    now = frozen_hash(cfg, alphas)
    if rec != now:
        raise RuntimeError(f"freeze guard: frozen config hash changed ({rec} -> {now})")
    return now


def guard_out(out: Path, cfg: dict[str, Any], selection: dict[str, Any]) -> None:
    """Output dir must be empty (new) or carry run_guard.json with the same config hash and run selection (resume)."""
    gp = out / "run_guard.json"
    rec = {"config_hash_without_alpha": frozen_hash(cfg, None), "selection": selection}
    if out.exists() and any(out.iterdir()):
        if not gp.exists():
            raise RuntimeError(f"{out} is not empty and has no run_guard.json; use a new --out")
        old = json.loads(gp.read_text())
        if old != rec:
            raise RuntimeError(f"{out} was made with a different frozen config or run selection; use a new --out")
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(gp, rec)


# ------------------------------------------------------------------------------------------------ batched bounded fit
def _inv3(H: np.ndarray) -> np.ndarray:
    a, b, c = H[:, 0, 0], H[:, 0, 1], H[:, 0, 2]
    d, e, f = H[:, 1, 1], H[:, 1, 2], H[:, 2, 2]
    c00, c01, c02 = d * f - e * e, c * e - b * f, b * e - c * d
    det = a * c00 + b * c01 + c * c02
    c11, c12, c22 = a * f - c * c, b * c - a * e, a * d - b * b
    return np.stack([np.stack([c00, c01, c02], -1), np.stack([c01, c11, c12], -1), np.stack([c02, c12, c22], -1)], -2) \
        / det[:, None, None]


def _sub_solve(Hp: np.ndarray, rp: np.ndarray, F: tuple[int, ...]) -> np.ndarray:
    if len(F) == 1:
        return (rp[:, F[0]] / Hp[:, F[0], F[0]])[:, None]
    i, j = F
    a, b, d = Hp[:, i, i], Hp[:, i, j], Hp[:, j, j]
    det = a * d - b * b
    return np.stack([(d * rp[:, i] - b * rp[:, j]) / det, (a * rp[:, j] - b * rp[:, i]) / det], 1)


def nnls3(H: np.ndarray, r: np.ndarray) -> np.ndarray:
    """min over theta >= 0 of 0.5 theta'H theta - r.theta for SPD 3x3 H.  H (N,3,3), r (N,R,3) -> (N,R,3).
    Unconstrained solution if non-negative; otherwise the feasible active-set solution (proper subsets) with the lowest value."""
    th = np.einsum("nij,nrj->nri", _inv3(H), r)
    bad = np.argwhere((th < 0).any(-1))
    if len(bad):
        n_i, r_i = bad[:, 0], bad[:, 1]
        Hp, rp = H[n_i], r[n_i, r_i]
        best_f, best = np.zeros(len(bad)), np.zeros((len(bad), 3))  # empty set: theta = 0
        for k in (1, 2):
            for F in itertools.combinations(range(3), k):
                t = _sub_solve(Hp, rp, F)
                f = -0.5 * (rp[:, list(F)] * t).sum(1)
                ok = (t >= 0).all(1) & (f < best_f)
                best_f = np.where(ok, f, best_f)
                full = np.zeros((len(bad), 3))
                full[:, list(F)] = t
                best = np.where(ok[:, None], full, best)
        th[n_i, r_i] = best
    return th


class Evaluator:
    """Design objective of any layout, for K weight columns at once: mean over cases and noise reps of the weighted MSE of the
    bounded fit.  Per case: objective = c0 + 2 h.qbar + <G, mean_r q q'> with the weighted forms of that column."""

    def __init__(self, cases: list[D.DesignCase], forms: list[PD.CellForm], weights: np.ndarray, alpha: float, sigma: float):
        if not (alpha > 0 and sigma > 0):
            raise ValueError("exact enumeration needs alpha > 0 and noise_std > 0")
        if any(k != "emission" for k in cases[0].param_kinds) or cases[0].n_params != 3:
            raise ValueError("Evaluator supports 3 emission parameters")
        nc, ns, R = len(cases), len(cases[0].candidate_ids), cases[0].n_noise
        self.nc, self.ns, self.R, self.alpha, self.K = nc, ns, R, alpha, weights.shape[0]
        self.Hs, self.rs = np.zeros((ns, nc, 3, 3)), np.zeros((ns, nc, R, 3))
        coef = np.zeros((nc, 12, self.K))
        w = weights / weights.sum(axis=(1, 2), keepdims=True)
        for ci, (c, cf) in enumerate(zip(cases, forms)):
            Aw = c.A_site * QS / sigma
            self.Hs[:, ci] = np.einsum("snp,snq->spq", Aw, Aw)
            rhs = (c.truth_readings[None] + sigma * c.noise - c.b_site[None]) / sigma
            self.rs[:, ci] = np.einsum("snp,rsn->srp", Aw, rhs)
            coef[ci, :3] = 2 * np.einsum("kyx,jyx->jk", w, cf.h)
            coef[ci, 3:] = np.einsum("kyx,jlyx->jlk", w, cf.G).reshape(9, self.K)
        self.const = sum(np.einsum("kyx,yx->k", w, cf.c0) for cf in forms) / nc
        self.coef = coef.reshape(nc * 12, self.K) / nc

    def values(self, subsets: np.ndarray, cols: list[int] | None = None, chunk: int = 20000) -> np.ndarray:
        """(L, b) layouts -> (L, K or len(cols)) design objectives."""
        subsets = np.asarray(subsets)
        co = self.coef if cols is None else self.coef[:, cols]
        cs = self.const if cols is None else self.const[cols]
        out = np.empty((len(subsets), co.shape[1]))
        eye = self.alpha * np.eye(3)
        for a in range(0, len(subsets), chunk):
            s = subsets[a:a + chunk]
            L = len(s)
            H, r = self.Hs[s[:, 0]].copy(), self.rs[s[:, 0]].copy()
            for j in range(1, s.shape[1]):
                H += self.Hs[s[:, j]]
                r += self.rs[s[:, j]]
            q = nnls3((H + eye).reshape(L * self.nc, 3, 3), r.reshape(L * self.nc, self.R, 3)) * QS
            qbar = q.mean(1)
            M = np.einsum("nri,nrj->nij", q, q).reshape(-1, 9) / self.R
            out[a:a + chunk] = np.concatenate([qbar, M], 1).reshape(L, self.nc * 12) @ co + cs
        return out


# ------------------------------------------------------------------------------------------------ selection
def unrank(ordinal: int, n: int, b: int) -> tuple[int, ...]:
    """The ordinal-th combination of range(n) in itertools.combinations (lexicographic) order."""
    out, x = [], 0
    for i in range(b):
        while True:
            c = math.comb(n - x - 1, b - i - 1)
            if ordinal < c:
                break
            ordinal -= c
            x += 1
        out.append(x)
        x += 1
    return tuple(out)


def exact_enum(ev: Evaluator, b: int, keep: int, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """All C(n, b) layouts: per column the `keep` smallest (value, lexicographic ordinal), sorted ascending: (keep, K) each."""
    it, start = itertools.combinations(range(ev.ns), b), 0
    vk = ok = None
    while True:
        blk = list(itertools.islice(it, chunk))
        if not blk:
            break
        v = ev.values(np.array(blk, dtype=np.int64))
        o = np.repeat(np.arange(start, start + len(blk))[:, None], ev.K, 1)
        if vk is not None:
            v, o = np.concatenate([vk, v]), np.concatenate([ok, o])
        if len(v) > keep:
            idx = np.argpartition(v, keep - 1, axis=0)[:keep]
            v, o = np.take_along_axis(v, idx, 0), np.take_along_axis(o, idx, 0)
        vk, ok, start = v, o, start + len(blk)
    order = [np.lexsort((ok[:, k], vk[:, k])) for k in range(ev.K)]
    return (np.stack([vk[order[k], k] for k in range(ev.K)], 1), np.stack([ok[order[k], k] for k in range(ev.K)], 1))


def greedy_order(ev: Evaluator, col: int, bmax: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Same rule as design.greedy_trace: add the site with the lowest objective, ties (rel 1e-12) to the lowest index."""
    chosen: list[int] = []
    scores: list[float] = []
    for _ in range(bmax):
        rest = [s for s in range(ev.ns) if s not in chosen]
        sc = ev.values(np.array([sorted(chosen + [s]) for s in rest]), [col])[:, 0]
        best = int(np.flatnonzero(sc <= sc.min() * (1 + 1e-12))[0])
        chosen.append(rest[best])
        scores.append(float(sc[best]))
    return tuple(chosen), tuple(scores)


def local_search(ev: Evaluator, col: int, start: tuple[int, ...]) -> tuple[tuple[int, ...], float]:
    """Best-improvement single-swap descent to a local optimum."""
    cur = np.array(sorted(start))
    val = float(ev.values(cur[None], [col])[0, 0])
    while True:
        inn = np.setdiff1d(np.arange(ev.ns), cur)
        nb = np.repeat(cur[None], len(cur) * len(inn), 0)
        nb[np.arange(len(nb)), np.repeat(np.arange(len(cur)), len(inn))] = np.tile(inn, len(cur))
        nb.sort(1)
        v = ev.values(nb, [col])[:, 0]
        i = int(np.argmin(v))
        if not v[i] < val * (1 - 1e-12):
            return tuple(int(x) for x in cur), val
        cur, val = nb[i], float(v[i])


def search_layout(ev: Evaluator, col: int, greedy_layout: tuple[int, ...], starts: list[tuple[int, ...]]) -> dict[str, Any]:
    """Greedy start + random restarts, each descended to a local optimum; best found (not provably optimal)."""
    res = [local_search(ev, col, s) for s in [greedy_layout] + starts]
    vals = np.array([v for _, v in res])
    i = int(np.argmin(vals))
    return {"layout": res[i][0], "value": float(vals[i]), "n_starts": len(res), "hit_starts": int((vals <= vals[i] * (1 + 1e-9)).sum())}


def random_starts(n: int, b: int, count: int, seed: int) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    return [tuple(sorted(int(x) for x in rng.choice(n, b, replace=False))) for _ in range(count)]


def tie_pairs(rec: pd.DataFrame, frac: float, objs: list[str]) -> pd.DataFrame:
    """Per (type, placement, draw, budget) and objective pair: both cross-excess values and the tie flag (within `frac` of the
    optimum under BOTH objectives).  rec rows carry columns cross_<obj> = value under <obj> of that row's layout."""
    rows = []
    for key, g in rec.groupby(["type", "placement", "draw", "budget"]):
        g = g.set_index("objective")
        for a, b in itertools.combinations(objs, 2):
            ea = g.loc[b, f"cross_{a}"] / g.loc[a, "value"] - 1  # excess under a of the layout optimal for b
            eb = g.loc[a, f"cross_{b}"] / g.loc[b, "value"] - 1
            rows.append({"type": key[0], "placement": key[1], "draw": key[2], "budget": key[3], "a": a, "b": b,
                         "same_layout": g.loc[a, "layout"] == g.loc[b, "layout"], "excess_a_at_layout_b": ea,
                         "excess_b_at_layout_a": eb, "tie": bool(max(ea, eb) <= frac), "method": g.loc[a, "method"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ settings
def moderator(counts: np.ndarray, sources: np.ndarray, lx: float, ly: float) -> dict[str, float]:
    """Distance (km) of the population from the nearest source: population-weighted mean over cells of the distance to the
    nearest source, and the distance of the population centroid to its nearest source."""
    ny, nx = counts.shape
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * lx / nx, (np.arange(ny) + 0.5) * ly / ny)
    dmin = np.min([np.hypot(X - sx, Y - sy) for sx, sy in sources], axis=0) / 1e3
    n = counts / counts.sum()
    cx, cy = float((n * X).sum()), float((n * Y).sum())
    return {"dist_mean_km": float((n * dmin).sum()),
            "dist_centroid_km": float(min(math.hypot(cx - sx, cy - sy) for sx, sy in sources) / 1e3)}


def _warm(args: tuple[Any, Any, Any]) -> None:
    ResponseCache(args[2]).get(args[0], args[1])


def warm_cache(st: Any, workers: int) -> None:
    """Compute the response bases of all cases of a placement in parallel (workers write distinct cache files)."""
    todo = [(s.scenario, st.solver, st.cache.directory) for r in st.specs.values() for s in r]
    if workers <= 1:
        return
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_warm, todo))


class Placement:
    """One source placement: setup (cases, cache), fixtures, columns."""

    def __init__(self, cfg: dict[str, Any], p: int, work: Path, types: list[str], draws: list[int]):
        self.p, self.pc = p, placement_cfg(cfg, p)
        self.st = PD.make_setup(self.pc, work)
        r, pc = cfg["replicates"], self.pc["population"]
        dom = self.pc["scenario"]["domain"]
        self.sources = np.array([[s["x"], s["y"]] for s in self.pc["scenario"]["sources"]])
        grid = GridSpec(0.0, 0.0, dom["nx"], dom["ny"], dom["lx"] / dom["nx"], dom["ly"] / dom["ny"], int(pc["epsg"]))
        self.objs = PD.objective_names(pc["betas"])
        self.units: list[dict[str, Any]] = []  # one per (type, draw): counts, weights, low mask, moderator
        for t in types:
            for k in draws:
                cnt = P.FIXTURES[t](grid, int(pc["total"]), int(r["fixture_draw_seeds"][k]))
                self.units.append({"type": t, "draw": k, "counts": cnt, "low": PD.lowpop_mask(cnt, float(pc["lowpop_quantile"]), float(pc["support_min"])),
                                   "w": PD.objective_weights(cnt, pc["betas"], float(pc["cap_c"])),
                                   **moderator(cnt, self.sources, float(dom["lx"]), float(dom["ly"]))})
        self.cols = [(u["type"], u["draw"], o) for u in self.units for o in self.objs]
        self.W = np.stack([u["w"][o] for u in self.units for o in self.objs])
        for c in range(len(self.cols)):
            if not np.isclose(self.W[c].sum(), 1.0) or (self.W[c] < 0).any():
                raise RuntimeError("objective weights must be non-negative and sum to 1")

    def col(self, t: str, k: int, o: str) -> int:
        return self.cols.index((t, k, o))


def selection_file(work: Path, p: int, b: int) -> Path:
    return work / "select" / f"p{p}_b{b}.csv"


def eval_file(work: Path, p: int, b: int) -> Path:
    return work / "eval" / f"p{p}_b{b}.csv"


def select_stage(cfg: dict[str, Any], pl: Placement, alpha: float, work: Path, timing: dict[str, Any]) -> None:
    sc, d, st = cfg["search"], cfg["design"], pl.st
    budgets = sorted(int(b) for b in d["budgets"])
    if all(selection_file(work, pl.p, b).exists() for b in budgets):
        return
    t0 = time.perf_counter()
    st.alpha = alpha
    dcs = PD.build_cases(st, "design", None)
    forms = [PD.cell_form(st.cache.get(s.scenario, st.solver), PD.truth_field(st, s), float(d["t_exclude"])) for s in st.specs["design"]]
    ev = Evaluator(dcs, forms, pl.W, alpha, st.noise_std)
    # cross-check against design.score_layout / greedy_trace (weights built into the cases)
    c0 = pl.col(pl.units[0]["type"], pl.units[0]["draw"], "mixed_0.5" if "mixed_0.5" in pl.objs else pl.objs[0])
    w0 = pl.W[c0]
    dcs_w = PD.build_cases(st, "design", w0)
    nb = min(3, budgets[0] + 1)
    for lay in random_starts(ev.ns, nb, int(sc["verify_layouts"]), 1):
        ref = D.score_layout(lay, dcs_w, st.opts(w0))["value"]
        mine = float(ev.values(np.array([lay]), [c0])[0, 0])
        if abs(ref - mine) > 1e-8 * abs(ref):
            raise RuntimeError(f"vectorised objective differs from design.score_layout: {mine} vs {ref} for {lay}")
    tr = D.greedy_trace(st.cands, nb, dcs_w, D.Constraints(), st.opts(w0), "total")
    go, gs = greedy_order(ev, c0, nb)
    if go != tr.order and not np.allclose(gs, tr.scores, rtol=1e-8):
        raise RuntimeError("vectorised greedy differs from design.greedy_trace")
    timing[f"p{pl.p}_setup_check_s"] = time.perf_counter() - t0
    bmax = max(budgets)
    greedy = {c: greedy_order(ev, c, bmax) for c in range(ev.K)}
    for b in budgets:
        f = selection_file(work, pl.p, b)
        if f.exists():
            continue
        t1 = time.perf_counter()
        exact = b <= int(sc["exact_max_budget"])
        res: dict[int, dict[str, Any]] = {}
        if exact:
            vals, ords = exact_enum(ev, b, int(sc["keep"]), int(sc["chunk"]))
            for c in range(ev.K):
                v = vals[:, c]
                near = int((v <= v[0] * (1 + float(sc["near_tie_frac"]))).sum())
                res[c] = {"layout": unrank(int(ords[0, c]), ev.ns, b), "value": float(v[0]), "margin_rel": float(v[1] / v[0] - 1),
                          "n_within": near, "n_within_censored": near >= len(v)}
        t_enum = time.perf_counter() - t1
        starts = random_starts(ev.ns, b, int(sc["n_restarts"]), int(sc["restart_seed"]) + 1000 * pl.p + b)
        t2 = time.perf_counter()
        for c in range(ev.K):
            if exact and pl.cols[c][1] >= int(sc["search_check_draws"]):
                continue
            s = search_layout(ev, c, tuple(sorted(greedy[c][0][:b])), starts)
            if exact:
                res[c].update(search_value=s["value"], search_hit_exact=bool(s["value"] <= res[c]["value"] * (1 + 1e-9)))
            else:
                res[c] = {**s, "margin_rel": np.nan, "n_within": -1, "n_within_censored": False}
        t_search = time.perf_counter() - t2
        rows = []
        for c, (t, k, o) in enumerate(pl.cols):
            gl = tuple(sorted(greedy[c][0][:b]))
            gv = float(ev.values(np.array([gl]), [c])[0, 0])
            r = res[c]
            rows.append({"placement": pl.p, "type": t, "draw": k, "objective": o, "budget": b, "method": "exact" if exact else "heuristic",
                         "layout": json.dumps(list(r["layout"])), "value": r["value"], "greedy_layout": json.dumps(list(gl)),
                         "greedy_value": gv, "gap_rel": gv / r["value"] - 1, **{x: r.get(x, np.nan) for x in (
                             "margin_rel", "n_within", "n_within_censored", "n_starts", "hit_starts", "search_value", "search_hit_exact")}})
        df = pd.DataFrame(rows)
        for (t, k), g in df.groupby(["type", "draw"]):  # value of every objective's layout under every objective
            cc = [pl.col(t, k, o) for o in pl.objs]
            for i in g.index:
                lay = np.array([json.loads(g.loc[i, "layout"])])
                for o, v in zip(pl.objs, ev.values(lay, cc)[0]):
                    df.loc[i, f"cross_{o}"] = v
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f.with_suffix(".tmp"), index=False)
        f.with_suffix(".tmp").replace(f)
        timing[f"p{pl.p}_b{b}"] = {"enum_s": t_enum, "search_s": t_search, "n_columns": ev.K, "exact": exact}
        say(f"placement {pl.p} budget {b}: {'exact' if exact else 'heuristic'} selection, {t_enum:.1f} s enumeration, {t_search:.1f} s search")


def eval_stage(cfg: dict[str, Any], pl: Placement, alpha: float, work: Path, timing: dict[str, Any]) -> None:
    d, st = cfg["design"], pl.st
    budgets = sorted(int(b) for b in d["budgets"])
    todo = [b for b in budgets if not eval_file(work, pl.p, b).exists()]
    if not todo:
        return
    t0 = time.perf_counter()
    st.alpha = alpha
    tcs = PD.build_cases(st, "test", None)
    forms = {s.name: PD.cell_form(st.cache.get(s.scenario, st.solver), PD.truth_field(st, s), float(d["t_exclude"])) for s in st.specs["test"]}
    D.assert_disjoint(PD.build_cases(st, "design", None), tcs, PD.build_cases(st, "validation", None))
    topts = st.opts(None)
    timing[f"p{pl.p}_test_cases_s"] = time.perf_counter() - t0
    qcache: dict[tuple[tuple[int, ...], str, int], np.ndarray] = {}
    for b in todo:
        t1 = time.perf_counter()
        sel = pd.read_csv(selection_file(work, pl.p, b))
        rows = []
        for u in pl.units:
            for _, r in sel[(sel["type"] == u["type"]) & (sel["draw"] == u["draw"])].iterrows():
                lay = tuple(json.loads(r["layout"]))
                acc = {c.name: {m: [] for m in METRICS} for c in tcs}
                for c in tcs:
                    for rep in range(c.n_noise):
                        key = (lay, c.name, rep)
                        if key not in qcache:
                            f = D.fit_layout(c, lay, rep, topts)
                            qcache[key] = f["q_hat"], f["mse_w"]
                        q, mse_w = qcache[key]
                        err = np.sqrt(forms[c.name].e2(q))
                        area = P.area_weighted_rmse(err, 1.0)
                        if abs(area ** 2 - mse_w) > 1e-6 * mse_w + 1e-18:
                            raise RuntimeError("per-cell error form disagrees with design.fit_layout mse_w")
                        for m, v in zip(METRICS, (P.pop_weighted_rmse(err, u["counts"]), area,
                                                  P.area_weighted_rmse(err, 1.0, u["low"].astype(float)), float(err.max()))):
                            acc[c.name][m].append(v)
                for name, a in acc.items():
                    rows.append({"placement": pl.p, "type": u["type"], "draw": u["draw"], "objective": r["objective"], "budget": b,
                                 "case": name, "layout": r["layout"], **{m: float(np.mean(v)) for m, v in a.items()}})
        f = eval_file(work, pl.p, b)
        f.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(f.with_suffix(".tmp"), index=False)
        f.with_suffix(".tmp").replace(f)
        timing[f"p{pl.p}_b{b}_eval_s"] = time.perf_counter() - t1
        say(f"placement {pl.p} budget {b}: test evaluation {time.perf_counter() - t1:.1f} s")


# ------------------------------------------------------------------------------------------------ analysis
def setting_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean over test cases first: one row per (type, placement, draw, objective, budget)."""
    return raw.groupby(["type", "placement", "draw", "objective", "budget"], as_index=False)[list(METRICS)].mean()


def paired_tables(sm: pd.DataFrame, ties: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    """Per type, objective != area, budget, metric: setting-wise difference to the exact-area layout, mean, sd and bootstrap
    interval over settings, fraction of settings lower / higher.  Second block: settings whose objective pair is a tie dropped."""
    key = ["type", "placement", "draw", "budget"]
    ref = sm[sm["objective"] == "area"].set_index(key)
    tied = set()
    if len(ties):
        tt = ties[(ties["a"] == "area") & ties["tie"]]
        tied = {(r.type, r.placement, r.draw, r.budget, r.b) for r in tt.itertuples()}
    rows = []
    for (t, o, b), g in sm[sm["objective"] != "area"].groupby(["type", "objective", "budget"]):
        g = g.set_index(key)
        for subset in ("all", "excluding_ties"):
            gg = g if subset == "all" else g[[(k[0], k[1], k[2], k[3], o) not in tied for k in g.index]]
            for m in METRICS:
                diff = (gg[m] - ref.loc[gg.index, m]).to_numpy()
                if not diff.size:
                    continue
                lo, hi = PD.bootstrap_ci(diff, int(stats["n_boot"]), int(stats["boot_seed"]), float(stats["level"]))
                rows.append({"type": t, "objective": o, "budget": b, "metric": m, "subset": subset, "n_settings": diff.size,
                             "mean_diff": diff.mean(), "sd_diff": diff.std(ddof=1) if diff.size > 1 else np.nan, "boot_lo": lo,
                             "boot_hi": hi, "frac_lower": float((diff < 0).mean()), "frac_higher": float((diff > 0).mean()),
                             "mean_area_layout": float(ref.loc[gg.index, m].mean()), "interval": BOOT_LABEL})
    return pd.DataFrame(rows)


def moderator_tables(sm: pd.DataFrame, units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benefit of each objective's layout over the exact-area layout (area-layout pop RMSE minus own; positive = better) per
    setting with the distance moderator, and its Spearman correlation per type / budget / objective (descriptive)."""
    key = ["type", "placement", "draw", "budget"]
    ref = sm[sm["objective"] == "area"].set_index(key)["pop_rmse"]
    b = sm[sm["objective"] != "area"].copy()
    b["benefit"] = ref.loc[pd.MultiIndex.from_frame(b[key])].to_numpy() - b["pop_rmse"].to_numpy()
    b = b.merge(units, on=["type", "placement", "draw"])
    b["benefit_rel"] = b["benefit"] / ref.loc[pd.MultiIndex.from_frame(b[key])].to_numpy()
    cor = [{"type": t, "objective": o, "budget": bb, "n": len(g), "spearman_dist_mean": g["benefit"].corr(g["dist_mean_km"], method="spearman"),
            "spearman_dist_centroid": g["benefit"].corr(g["dist_centroid_km"], method="spearman")}
           for (t, o, bb), g in b.groupby(["type", "objective", "budget"])]
    return b, pd.DataFrame(cor)


def design_gap_table(rec: pd.DataFrame) -> pd.DataFrame:
    return rec.groupby(["type", "objective", "budget", "method"], as_index=False)["gap_rel"].agg(
        mean_gap="mean", median_gap="median", max_gap="max", frac_greedy_suboptimal=lambda x: float((x > 1e-9).mean()), n="size")


# ------------------------------------------------------------------------------------------------ figures
def figures(out: Path, tabs: dict[str, pd.DataFrame], rec: pd.DataFrame, sm: pd.DataFrame, profile: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    tag = f"{STAMP} ({profile}, synthetic fixtures)"
    types = sorted(sm["type"].unique())
    budgets = sorted(sm["budget"].unique())
    colors = dict(zip(types, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]))

    def finish(fig: Any, name: str) -> None:
        fig.text(0.995, 0.005, tag, ha="right", va="bottom", fontsize=7, color="0.4")
        fig.savefig(fd / name, dpi=130)
        plt.close(fig)

    mod = tabs["moderator"]
    fig, ax = plt.subplots(1, len(budgets), figsize=(3.2 * len(budgets), 3.4), squeeze=False, sharey=False)
    for j, b in enumerate(budgets):
        for t in types:
            g = mod[(mod["budget"] == b) & (mod["objective"] == "population") & (mod["type"] == t)]
            ax[0, j].plot(g["dist_mean_km"], g["benefit"], "o", ms=4, color=colors[t], label=t)
        ax[0, j].axhline(0, color="0.5", lw=0.8)
        ax[0, j].set(title=f"budget {b}", xlabel="pop.-weighted distance to nearest source (km)", ylabel="pop RMSE: area layout - population layout" if j == 0 else "")
        ax[0, j].title.set_fontsize(8)
    ax[0, 0].legend(fontsize=5)
    fig.suptitle("benefit of the exact population layout (positive = better) vs distance of the population from the sources", fontsize=8)
    fig.tight_layout()
    finish(fig, "fig1_benefit_vs_distance.png")

    objs = [o for o in ["area", "mixed_0.5", "population", "capped"] if o in set(rec["objective"])]
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.6))
    for i, o in enumerate(objs):
        for j, b in enumerate(budgets):
            g = rec[(rec["objective"] == o) & (rec["budget"] == b)]["gap_rel"].clip(lower=1e-9)
            ax.boxplot(g, positions=[j + (i - 1.5) * 0.18], widths=0.14, showfliers=True, flierprops={"ms": 2},
                       boxprops={"color": f"C{i}"}, medianprops={"color": f"C{i}"}, whiskerprops={"color": f"C{i}"}, capprops={"color": f"C{i}"})
    ax.set(xticks=range(len(budgets)), xticklabels=[f"{b}\n({rec[rec['budget'] == b]['method'].iloc[0]})" for b in budgets], yscale="log",
           xlabel="budget (selection method)", ylabel="(greedy - best) / best (design obj.)", ylim=(1e-4, None))
    ax.legend(handles=[plt.Line2D([], [], color=f"C{i}", label=o) for i, o in enumerate(objs)], fontsize=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    finish(fig, "fig2_greedy_vs_exact_gap.png")

    pr = tabs["paired"]
    pr = pr[(pr["metric"] == "pop_rmse") & (pr["subset"] == "all")]
    fig, ax = plt.subplots(1, len(types), figsize=(3.4 * len(types), 3.5), squeeze=False, sharey=True)
    for i, t in enumerate(types):
        for k, o in enumerate([x for x in objs if x != "area"]):
            g = pr[(pr["type"] == t) & (pr["objective"] == o)].sort_values("budget")
            x = np.arange(len(g)) + (k - 1) * 0.12
            ax[0, i].errorbar(x, g["mean_diff"], yerr=[g["mean_diff"] - g["boot_lo"], g["boot_hi"] - g["mean_diff"]], fmt="o", ms=3, capsize=2, color=f"C{k + 1}", label=o)
            ax[0, i].set_xticks(range(len(g)), g["budget"].tolist())
        ax[0, i].axhline(0, color="0.5", lw=0.8)
        ax[0, i].set(title=t, xlabel="budget")
        ax[0, i].title.set_fontsize(8)
    ax[0, 0].set_ylabel("test pop RMSE minus exact-area layout")
    ax[0, 0].legend(fontsize=6)
    fig.suptitle("paired differences per setting (mean, 95% percentile bootstrap over settings; negative = better than exact area layout)", fontsize=8)
    fig.tight_layout()
    finish(fig, "fig3_paired_differences.png")

    mk = dict(zip(objs, ["o", "s", "^", "*"]))
    fig, ax = plt.subplots(1, len(budgets), figsize=(3.2 * len(budgets), 3.5), squeeze=False)
    for j, b in enumerate(budgets):
        for t in types:
            for o in objs:
                g = sm[(sm["budget"] == b) & (sm["type"] == t) & (sm["objective"] == o)]
                ax[0, j].plot(g["area_rmse"].mean(), g["pop_rmse"].mean(), mk[o], color=colors[t], ms=6)
        ax[0, j].set(title=f"budget {b}", xlabel="test area RMSE", ylabel="test pop RMSE" if j == 0 else "")
        ax[0, j].title.set_fontsize(8)
        ax[0, j].grid(alpha=0.3)
    ax[0, 0].legend(handles=[plt.Line2D([], [], marker=mk[o], ls="", color="k", label=o) for o in objs] +
                    [plt.Line2D([], [], marker="o", ls="", color=colors[t], label=t) for t in types], fontsize=5)
    fig.suptitle("Pareto: mean over settings of test population vs area RMSE, best found layout per objective", fontsize=8)
    fig.tight_layout()
    finish(fig, "fig4_pareto.png")


def finalise(out: Path, cfg: dict[str, Any], budgets: list[int], placements: list[int], profile: str, work: Path, units: pd.DataFrame) -> None:
    raw = pd.concat([pd.read_csv(eval_file(work, p, b)) for p in placements for b in budgets], ignore_index=True)
    rec = pd.concat([pd.read_csv(selection_file(work, p, b)) for p in placements for b in budgets], ignore_index=True)
    objs = [c[6:] for c in rec.columns if c.startswith("cross_")]
    ties = tie_pairs(rec, float(cfg["search"]["near_tie_frac"]), objs)
    sm = setting_metrics(raw)
    mod, cor = moderator_tables(sm, units)
    tabs = {"test_raw": raw, "selection": rec, "near_ties": ties, "setting_metrics": sm, "design_gap": design_gap_table(rec),
            "paired": paired_tables(sm, ties, cfg["stats"]), "moderator": mod, "moderator_corr": cor,
            "pareto": sm.groupby(["type", "objective", "budget"], as_index=False)[list(METRICS)].mean(), "units": units}
    for k, t in tabs.items():
        t.to_csv(out / f"{k}.csv", index=False)
    figures(out, tabs, rec, sm, profile)


# ------------------------------------------------------------------------------------------------ main
def run(profile: str, cfg_path: str | None, out_arg: str | None, summary_only: bool) -> int:
    cfg = load_config(ROOT / (cfg_path or CONFIGS[profile]))
    if profile == "pilot":
        cfg = _merge(cfg, cfg["pilot"])
    r = cfg["replicates"]
    sel = {"types": cfg["run"]["types"] or list(r["types"]), "placements": cfg["run"]["placements"] or list(range(int(r["n_placements"]))),
           "draws": cfg["run"]["draws"] or list(range(len(r["fixture_draw_seeds"])))}
    out = ROOT / (out_arg or OUTS[profile])
    check_seeds(cfg)
    if min(float(a) for a in cfg["design"]["alpha_grid"]) <= 0:
        raise ValueError("alpha_grid must be > 0 (exact enumeration needs a unique bounded fit)")
    guard_out(out, cfg, sel)
    work = out / "work"
    work.mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    budgets = sorted(int(b) for b in cfg["design"]["budgets"])
    timing: dict[str, Any] = {"profile": profile}
    write_json(out / "config.json", cfg)
    pls = {p: Placement(cfg, p, work, sel["types"], sel["draws"]) for p in sel["placements"]}
    units = pd.DataFrame([{"type": u["type"], "placement": p, "draw": u["draw"], "dist_mean_km": u["dist_mean_km"], "dist_centroid_km": u["dist_centroid_km"]}
                          for p, pl in pls.items() for u in pl.units])
    # 1. bases and alpha (validation only), once per placement
    alphas: dict[str, float] = {}
    for p, pl in pls.items():
        ap = out / f"alpha_selected_p{p}.json"
        if not ap.exists():
            t0 = time.perf_counter()
            warm_cache(pl.st, int(cfg["run"]["workers"]))
            timing[f"p{p}_bases_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            say(f"placement {p}: verifying affine truth against FiPy, selecting alpha on validation")
            timing[f"p{p}_truth_check_rel_err"] = PD.verify_truth(pl.st)
            s = PD.select_alpha(pl.st)
            s["config_hash_without_alpha"] = frozen_hash(cfg, None)
            s["sources"] = pl.sources.tolist()
            write_json(ap, s)
            timing[f"p{p}_alpha_s"] = time.perf_counter() - t0
        s = json.loads(ap.read_text())
        if s["config_hash_without_alpha"] != frozen_hash(cfg, None):
            raise RuntimeError("alpha_selected file made with a different frozen config; use a new --out")
        alphas[str(p)] = float(s["alpha"])
    say(f"alphas {alphas}")
    # 2. freeze before any test evaluation
    pp, fh = out / "provenance.json", frozen_hash(cfg, alphas)
    if pp.exists():
        assert_frozen(out, cfg, alphas)
    elif summary_only:
        raise RuntimeError("--summary-only needs a provenance.json from a full run")
    else:
        write_json(pp, {**provenance(cfg, {k: v for k, v in cfg["design"].items() if k.endswith("seed")},
                                     {"experiment": STAMP, "profile": profile, "note": "synthetic idealised experiment"}),
                        "frozen_config_hash": fh, "frozen_payload": frozen_payload(cfg, alphas), "alphas": alphas,
                        "frozen_before_test_utc": pd.Timestamp.now("UTC").isoformat()})
    # 3. selection and evaluation per placement
    if not summary_only:
        for p, pl in pls.items():
            select_stage(cfg, pl, alphas[str(p)], work, timing)
            eval_stage(cfg, pl, alphas[str(p)], work, timing)
    assert_frozen(out, cfg, alphas)
    finalise(out, cfg, budgets, list(pls), profile, work, units)
    timing.update(total_s=time.perf_counter() - T0, freeze_hash_verified=fh, n_columns_per_placement=len(next(iter(pls.values())).cols))
    tp = out / "timing.json"
    write_json(tp, {**(json.loads(tp.read_text()) if tp.exists() else {}), **timing})
    write_json(pp, {**json.loads(pp.read_text()), "freeze_guard_passed_after_test": True})
    say(f"done -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=list(CONFIGS), default="smoke")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)
    return run(a.profile, a.config, a.out, a.summary_only)


if __name__ == "__main__":
    sys.exit(main())

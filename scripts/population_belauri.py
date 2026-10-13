# pyright: reportGeneralTypeIssues=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportOperatorIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportReturnType=false, reportAssignmentType=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportPossiblyUnboundVariable=false, reportPrivateImportUsage=false
"""EXP-016: Belauri-scale population-aware sensor design (real geography; synthetic emissions, meteorology, population fixtures
plus one real WorldPop field).

    python scripts/population_belauri.py --profile {smoke,pilot,production} [--config FILE] [--out DIR] [--work DIR]

Selection: every weighted design objective (area, mixed, population, capped; one column per distinct weight map) is minimised EXACTLY
over all subsets of the INTERIOR candidate table for budgets <= search.exact_max_budget (refused above search.exact_max_combinations),
and by greedy + swap search + random restarts (best found, not provably optimal) above.  Greedy layouts are recorded and ALSO evaluated
on test, so the test-error cost of greedy suboptimality is measured.  The design objective is the one design.greedy_trace uses (mean over
design cases and noise reps of the weighted MSE of the ridge, q >= 0 fit), computed in closed form (4 parameters: 3 emissions + bg inflow)
and cross-checked against design.score_layout / design.greedy_trace.
Order: seed/out guards -> validation + design cases (process pool) -> alpha cross-check (validation only) -> selection ->
sanity stops (uniform fixture layouts == area layouts) -> freeze hash -> TEST cases -> evaluation -> freeze guard -> tables, figures.
Statistics: test metrics are averaged over test cases and noise reps first; a paired difference is a matrix (fixture draws x test
scenarios).  The 6 fixture draws are independent populations but share the SAME 12 test scenarios, so the independent unit for scenario
variability is the 12 scenarios and for fixture variability the 6 draws; we report a two-way cluster bootstrap (primary, resamples both
independently) and both one-way bootstraps.  Mesh/dt are config keys (EXP-015 stage 2 decides whether 500 m / dt 100 is adequate).
Attribution: local/external errors are the quadratic forms of the case truth components, exactly as geodesign.evaluate_test does.
"""
from __future__ import annotations

import os

if __name__ == "__mp_main__":  # spawned pool workers: single-threaded BLAS (set before numpy is imported)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Iterator  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pravah import design as D  # noqa: E402
from pravah import geodesign as G  # noqa: E402
from pravah import population as P  # noqa: E402
from pravah.config import _merge, load_config  # noqa: E402
from pravah.forward import FIPY  # noqa: E402
from pravah.inverse import SCALE_BY_KIND, ResponseCache, run_forward  # noqa: E402
from pravah.io import provenance, write_json  # noqa: E402
from pravah.runner import check_frozen, guard_out as guard_out_rec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


PDE = _load("population_design_exact", "scripts/population_design_exact.py")  # imported read-only
PD = PDE.PD

STAMP = "EXP-016"
CONFIGS = {"smoke": "configs/population_belauri_smoke.json", "pilot": "configs/population_belauri.json",
           "production": "configs/population_belauri.json"}
OUTS = {"smoke": "results/phase3-population-belauri-smoke", "pilot": "results/phase3-population-belauri-pilot",
        "production": "results/phase3-population-belauri"}
METRICS = ("pop_rmse", "area_rmse", "lowq_rmse", "worst_cell_rmse", "heldout_rmse", "heldout_random_rmse", "heldout_block_rmse",
           "rmse_local", "rmse_external", "attr_err")
BOOT_LABEL = ("percentile cluster bootstrap; 'both' resamples the fixture draws and the test scenarios independently (primary); the draws "
              "are independent populations but share the same test scenarios; not a confidence interval for new scenarios or regions")
LABEL = "Real geography; synthetic emissions, meteorology and synthetic population fixtures (plus one real WorldPop field)"
T0 = time.perf_counter()
RSS_LIMIT_GB = 2.0


def rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1e9 if sys.platform == "darwin" else 1e6)


def say(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.0f}s rss {rss_gb():.2f}GB] {msg}", flush=True)
    assert rss_gb() < RSS_LIMIT_GB, f"peak RSS {rss_gb():.2f} GB exceeds {RSS_LIMIT_GB} GB"


# ------------------------------------------------------------------------------------------------ process pool, files
def atomic_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps(obj))
    os.replace(tmp, path)


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def n_workers(cfg: dict[str, Any]) -> int:
    w = cfg["run"]["workers"]
    w = int(w) if w is not None else int(os.environ.get("PRAVAH_WORKERS") or os.cpu_count() or 1)
    return max(1, min(w, int(cfg["run"]["workers_cap"])))


def _pinit(limit: float, init: Callable[..., None] | None, initargs: tuple[Any, ...]) -> None:
    global RSS_LIMIT_GB
    RSS_LIMIT_GB = limit
    if init is not None:
        init(*initargs)


def _run(fn: Callable[[Any], Any], arg: Any) -> Any:
    out = fn(arg)
    assert rss_gb() < RSS_LIMIT_GB, f"worker peak RSS {rss_gb():.2f} GB exceeds {RSS_LIMIT_GB} GB"
    return out


class Pool:
    """workers <= 1 runs in-process (same code path); otherwise a spawn process pool whose initializer sets worker globals."""

    def __init__(self, workers: int, init: Callable[..., None] | None = None, initargs: tuple[Any, ...] = (), rss_limit: float = 2.0):
        global RSS_LIMIT_GB
        RSS_LIMIT_GB = rss_limit
        self.ex = None
        if workers > 1:
            self.ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_pinit, initargs=(rss_limit, init, initargs))
        elif init is not None:
            init(*initargs)

    def imap(self, fn: Callable[[Any], Any], items: list[Any]) -> Iterator[tuple[Any, Any]]:
        if self.ex is None:
            for it in items:
                yield it, _run(fn, it)
            return
        futs = {self.ex.submit(_run, fn, it): it for it in items}
        for f in as_completed(futs):
            yield futs[f], f.result()

    def __enter__(self) -> "Pool":
        return self

    def __exit__(self, *a: Any) -> None:
        if self.ex is not None:
            self.ex.shutdown()


def run_units(pool: Pool, fn: Callable[[Any], Any], units: list[tuple[Path, Any]]) -> dict[Path, Any]:
    """Resumable: a unit whose file exists is loaded, not recomputed; results are written (atomically) as they complete."""
    out: dict[Path, Any] = {}
    todo = {}
    for path, arg in units:
        if path.exists():
            out[path] = pickle.loads(path.read_bytes())
        else:
            todo[id(arg)] = (path, arg)
    args = [a for _, a in todo.values()]
    for arg, res in pool.imap(fn, args):
        path = todo[id(arg)][0]
        atomic_pickle(path, res)
        out[path] = res
    return out


# ------------------------------------------------------------------------------------------------ config, seeds, guards
def load_cfg(profile: str, path: str | None) -> dict[str, Any]:
    cfg = load_config(ROOT / (path or CONFIGS[profile]))
    return _merge(cfg, cfg["pilot"]) if profile == "pilot" else cfg


def cfg_hash(cfg: dict[str, Any]) -> str:
    return G.config_hash(cfg, ("run", "pilot", "notes", "name"))


def guard_out(out: Path, cfg: dict[str, Any], profile: str) -> None:
    """Output dir must be empty (new) or carry run_guard.json with the same config hash and profile (resume)."""
    rec = {"config_hash": cfg_hash(cfg), "profile": profile, "types": cfg["run"]["types"], "draws": cfg["run"]["draws"]}
    guard_out_rec(out, rec)


def historic_seeds(skip: str = "population_belauri") -> tuple[set[int], list[tuple[int, int]]]:
    """(case-like seeds, noise ranges (first, width)) of every OTHER config: exact blocks for 'ensembles' and the stage-2 replicates,
    width 64 for any other *noise* seed key (shortcut: a bound on the toy configs' ensemble sizes)."""
    seeds: set[int] = set()
    noise: list[tuple[int, int]] = []

    def walk(o: Any, key: str = "") -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, key)
        elif isinstance(o, int) and not isinstance(o, bool) and "seed" in key:
            (noise.append((o, 64)) if "noise" in key else seeds.add(o))
    for f in sorted((ROOT / "configs").glob("*.json")):
        if f.name.startswith(skip):
            continue
        c = load_config(f)
        walk(c)
        for e in (c.get("ensembles") or {}).values():
            noise.append((int(e["noise_seed"]), int(e["n_cases"])))
        r = (c.get("stage2") or {}).get("replicates")
        if r:
            for i in range(int(r["n"])):
                seeds.add(int(r["seed0"]) + i)
                noise.append((int(r["noise_seed0"]) + i * int(r["noise_seed_stride"]), int(r["n_cases"])))
    return seeds, noise


def check_seeds(cfg: dict[str, Any], historic: tuple[set[int], list[tuple[int, int]]] | None = None) -> None:
    """Case seeds, fixture draw seeds and noise-seed ranges: pairwise disjoint, and disjoint from every other config's seeds.  Only the
    design/validation case seeds listed in seeds.allowed_reuse may equal an earlier config's (the registration says 101/202); the
    test ensemble and all fixture seeds must be fresh."""
    e, p = cfg["ensembles"], cfg["population"]
    reuse = set(int(x) for x in cfg["seeds"]["allowed_reuse"])
    case = {r: int(e[r]["seed"]) for r in ("design", "validation", "test")}
    fix = [int(s) for s in p["fixture_draw_seeds"]]
    allseeds = list(case.values()) + fix
    if len(set(allseeds)) != len(allseeds):
        raise ValueError("case seeds and fixture draw seeds must be pairwise distinct")
    if case["test"] in reuse:
        raise ValueError("the test seed must be fresh (not in seeds.allowed_reuse)")
    rng = {r: set(range(int(e[r]["noise_seed"]), int(e[r]["noise_seed"]) + int(e[r]["n_cases"]))) for r in case}
    for a, b in itertools.combinations(rng, 2):
        if rng[a] & rng[b]:
            raise ValueError(f"noise seed ranges of {a} and {b} overlap")
    hs, hn = historic if historic is not None else historic_seeds()
    for r, s in case.items():
        if s in hs and s not in reuse:
            raise ValueError(f"{r} seed {s} is used by another config")
        if s not in reuse and any(rng[r] & set(range(a, a + w)) for a, w in hn):
            raise ValueError(f"{r} noise seeds overlap another config's")
    bad = [s for s in fix if s in hs]
    if bad:
        raise ValueError(f"fixture draw seeds {bad} are used by another config")


def historic_ensembles(cfg: dict[str, Any]) -> dict[str, tuple[list[G.CaseDraw], int, int]]:
    """The EXP-015 draws (Phase 2 test 303, held 304, replicates 1101-1108) for geodesign.check_ensembles."""
    out = {"phase2_test": (G.draw_cases(cfg, 6, 303, "x"), 303, 3000), "held_304": (G.draw_cases(cfg, 6, 304, "y"), 304, 3100)}
    f = ROOT / "configs/convergence_stage2.json"
    if f.exists():
        r = (load_config(f).get("stage2") or {}).get("replicates")
        for i in range(int(r["n"]) if r else 0):
            s = int(r["seed0"]) + i
            out[f"rep{i}"] = (G.draw_cases(cfg, int(r["n_cases"]), s, f"r{i}_"), s, int(r["noise_seed0"]) + i * int(r["noise_seed_stride"]))
    return out


def check_ensemble_draws(cfg: dict[str, Any]) -> None:
    e = cfg["ensembles"]
    ens = {r: (G.draw_cases(cfg, int(e[r]["n_cases"]), int(e[r]["seed"]), r[0]), int(e[r]["seed"]), int(e[r]["noise_seed"]))
           for r in ("design", "validation", "test")}
    G.check_ensembles({**ens, **historic_ensembles(cfg)})


# ------------------------------------------------------------------------------------------------ population
def objective_weights(N: np.ndarray, area_w: np.ndarray, sf: np.ndarray, betas: list[float], cap_c: float) -> dict[str, np.ndarray]:
    """name -> (ny, nx) weight summing to 1.  area: A_i*study_i/sum; population N/sum N (N already masked, study fraction is NOT applied
    twice); mixed (1-beta) w_area + beta w_pop; capped min(N, c * mean N over study cells)/sum."""
    wa = area_w / area_w.sum()
    w = {PD.beta_name(b): (1 - b) * wa + b * N / N.sum() for b in betas}
    c = np.minimum(N, cap_c * N[sf > 0].mean())
    w["capped"] = c / c.sum()
    return w


def make_units(cfg: dict[str, Any], grid: Any, sf: np.ndarray, res: float, types: list[str] | None = None,
               draws: list[int] | None = None) -> list[dict[str, Any]]:
    """One unit per (fixture type, draw): people per cell inside the study polygon (total fixed), weights per objective, low-pop mask.
    Synthetic types get n_draws draws (fresh seeds); a type in population.real is a single deterministic draw read from a file."""
    pc = cfg["population"]
    total_raw = int(pc["grid_people_per_cell"]) * grid.nx * grid.ny  # divisible by the cell count so 'uniform' is constant
    area_w = grid.cell_area * sf
    units = []
    for t in types or pc["types"]:
        if t in pc.get("real", {}):
            raw = np.load(ROOT / pc["real"][t].format(res=f"{res:g}"))
            if raw.shape != grid.shape:
                raise ValueError(f"real field {t} has shape {raw.shape}, grid {grid.shape}")
            ks = [0]
            fields = {0: P.mask_to_study(raw, sf, float((raw * sf).sum()))}  # real study total is kept
        else:
            ks = [k for k in (draws if draws is not None else range(int(pc["n_draws"])))]
            fields = {k: P.mask_to_study(P.FIXTURES[t](grid, total_raw, int(pc["fixture_draw_seeds"][k])), sf, float(pc["study_total"]))
                      for k in ks}
        for k in ks:
            N = fields[k]
            units.append({"type": t, "draw": k, "counts": N, "w": objective_weights(N, area_w, sf, pc["betas"], float(pc["cap_c"])),
                          "low": PD.lowpop_mask(N, float(pc["lowpop_quantile"]), float(pc["support_min"]))})
    return units


def build_columns(units: list[dict[str, Any]], objs: list[str]) -> tuple[np.ndarray, dict[tuple[str, int, str], int]]:
    """Distinct weight maps (bitwise) as evaluator columns; column 0 is the area weight; identical maps (area, uniform draws) share one."""
    seen: dict[bytes, int] = {}
    ws: list[np.ndarray] = []
    col_of = {}
    for u in units:
        for o in objs:
            w = u["w"][o]
            j = seen.setdefault(w.tobytes(), len(ws))
            if j == len(ws):
                ws.append(w)
            col_of[(u["type"], u["draw"], o)] = j
    return np.stack(ws), col_of


def check_population(units: list[dict[str, Any]], sf: np.ndarray, total_expect: Callable[[str], float | None],
                     near: np.ndarray) -> None:
    """Counts sum to the declared total, are >= 0 and zero outside the study polygon; weights >= 0 sum to 1; population weight > 0 within
    one candidate spacing of some candidate (`near` = boolean map of such cells), else STOP."""
    for u in units:
        N = u["counts"]
        te = total_expect(u["type"])
        if N.min() < 0 or N[sf == 0].any() or (te is not None and not np.isclose(N.sum(), te, rtol=1e-9)):
            raise RuntimeError(f"population mask check failed for {u['type']} draw {u['draw']}")
        for o, w in u["w"].items():
            if w.min() < 0 or not np.isclose(w.sum(), 1.0):
                raise RuntimeError("objective weights must be non-negative and sum to 1")
        if not (u["w"]["population"][near] > 0).any():
            raise RuntimeError(f"STOP: population weight of {u['type']} draw {u['draw']} is zero near every candidate")


# ------------------------------------------------------------------------------------------------ batched bounded fit, evaluator
def nnls_p(H: np.ndarray, r: np.ndarray) -> np.ndarray:
    """min over theta >= 0 of 0.5 theta'H theta - r.theta for SPD p x p H.  H (N,p,p), r (N,R,p) -> (N,R,p).  Unconstrained solution if
    non-negative; otherwise the best feasible solution over the proper active sets (exact for a strictly convex problem)."""
    p = H.shape[-1]
    th = np.einsum("nij,nrj->nri", np.linalg.inv(H), r)
    bad = np.argwhere((th < 0).any(-1))
    if len(bad):
        n_i, r_i = bad[:, 0], bad[:, 1]
        Hp, rp = H[n_i], r[n_i, r_i]
        best_f, best = np.zeros(len(bad)), np.zeros((len(bad), p))
        for k in range(1, p):
            for F in itertools.combinations(range(p), k):
                F = list(F)
                t = np.einsum("nij,nj->ni", np.linalg.inv(Hp[:, F][:, :, F]), rp[:, F])
                f = -0.5 * (rp[:, F] * t).sum(1)
                ok = (t >= 0).all(1) & (f < best_f)
                full = np.zeros((len(bad), p))
                full[:, F] = t
                best_f, best = np.where(ok, f, best_f), np.where(ok[:, None], full, best)
        th[n_i, r_i] = best
    return th


class Ev(PDE.Evaluator):
    """PDE.Evaluator generalised to p parameters of any kind (here 3 emissions + bg inflow) and a subset of candidate sites.
    Design objective of any layout for K weight columns at once: mean over cases and noise reps of the weighted MSE of the bounded fit."""

    def __init__(self, cases: list[Any], forms: list[Any], weights: np.ndarray, alpha: float, sigma: float, sites: np.ndarray,
                 scale: np.ndarray):
        if not (alpha > 0 and sigma > 0):
            raise ValueError("exact enumeration needs alpha > 0 and noise_std > 0")
        p, nc, ns, R = len(scale), len(cases), len(sites), cases[0].noise.shape[0]
        self.p, self.scale, self.nc, self.ns, self.R, self.alpha, self.K = p, np.asarray(scale, float), nc, ns, R, alpha, weights.shape[0]
        self.Hs, self.rs = np.zeros((ns, nc, p, p)), np.zeros((ns, nc, R, p))
        coef = np.zeros((nc, p + p * p, self.K))
        w = weights / weights.sum(axis=(1, 2), keepdims=True)
        for ci, (c, cf) in enumerate(zip(cases, forms)):
            Aw = c.A_site[sites] * self.scale / sigma
            rhs = (c.truth_readings[sites][None] + sigma * c.noise[:, sites] - c.b_site[sites][None]) / sigma
            self.Hs[:, ci] = np.einsum("snp,snq->spq", Aw, Aw)
            self.rs[:, ci] = np.einsum("snp,rsn->srp", Aw, rhs)
            coef[ci, :p] = 2 * np.einsum("kyx,jyx->jk", w, cf.h)
            coef[ci, p:] = np.einsum("kyx,jlyx->jlk", w, cf.G).reshape(p * p, self.K)
        self.const = sum(np.einsum("kyx,yx->k", w, cf.c0) for cf in forms) / nc
        self.coef = coef.reshape(nc * (p + p * p), self.K) / nc

    def values(self, subsets: np.ndarray, cols: list[int] | None = None, chunk: int = 20000) -> np.ndarray:
        subsets = np.asarray(subsets)
        p = self.p
        co = self.coef if cols is None else self.coef[:, cols]
        cs = self.const if cols is None else self.const[cols]
        out = np.empty((len(subsets), co.shape[1]))
        eye = self.alpha * np.eye(p)
        for a in range(0, len(subsets), chunk):
            s = subsets[a:a + chunk]
            L = len(s)
            H, r = self.Hs[s[:, 0]].copy(), self.rs[s[:, 0]].copy()
            for j in range(1, s.shape[1]):
                H += self.Hs[s[:, j]]
                r += self.rs[s[:, j]]
            q = nnls_p((H + eye).reshape(L * self.nc, p, p), r.reshape(L * self.nc, self.R, p)) * self.scale
            M = np.einsum("nri,nrj->nij", q, q).reshape(-1, p * p) / self.R
            out[a:a + chunk] = np.concatenate([q.mean(1), M], 1).reshape(L, self.nc * (p + p * p)) @ co + cs
        return out


def enum_range(ev: Ev, b: int, a: int, z: int, keep: int, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """Combinations with lexicographic ordinal in [a, z): per column the `keep` smallest (value, ordinal), unsorted (keep, K) each."""
    it = itertools.islice(itertools.combinations(range(ev.ns), b), a, z)
    start, vk, ok = a, None, None
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
    return vk, ok


def merge_enum(parts: list[tuple[np.ndarray, np.ndarray]], keep: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge range results into the sorted global top-`keep` per column (ties by lexicographic ordinal), as PDE.exact_enum returns."""
    v, o = np.concatenate([x[0] for x in parts]), np.concatenate([x[1] for x in parts])
    K = v.shape[1]
    order = [np.lexsort((o[:, k], v[:, k]))[:keep] for k in range(K)]
    return np.stack([v[order[k], k] for k in range(K)], 1), np.stack([o[order[k], k] for k in range(K)], 1)


# ------------------------------------------------------------------------------------------------ worker tasks
_W: dict[str, Any] = {}


def task_case(arg: tuple[dict[str, Any], str, str, int]) -> str:
    """Build one case (bases, direct truth, components) + its per-cell error form; pickled in <work>/cases (resumable)."""
    cfg, work, role, i = arg
    path = Path(work) / "cases" / f"{role}_{i}.pkl"
    if path.exists():
        return str(path)
    if _W.get("cache") != work:
        _W["st"] = G.Setup(cfg, ResponseCache(Path(work) / "responses"))
        _W["cache"] = work
    st = _W["st"]
    e = cfg["ensembles"][role]
    d = G.draw_cases(cfg, int(e["n_cases"]), int(e["seed"]), role[0])[i]
    t0 = time.perf_counter()
    case, info = st.build(d, role, int(e["noise_seed"]) + i)
    form = None
    if role != "validation":  # the same direct truth solve Setup.build used (deterministic), so the cell form matches the case exactly
        bc = st.scenario(d)
        truth = run_forward(FIPY, bc.scenario, bc.theta, st.solver).concentration.transpose("time", "y", "x").values
        form = PD.cell_form(st.cache.get(bc.scenario, st.solver), np.asarray(truth, float), st.t_exclude)
    info["build_total_s"] = time.perf_counter() - t0
    atomic_pickle(path, (case, info, form))
    return str(path)


def build_role(cfg: dict[str, Any], work: Path, role: str, pool: Pool) -> list[tuple[Any, dict[str, Any], Any]]:
    n = int(cfg["ensembles"][role]["n_cases"])
    todo = [(cfg, str(work), role, i) for i in range(n)]
    for _ in pool.imap(task_case, [t for t in todo if not (work / "cases" / f"{role}_{t[3]}.pkl").exists()]):
        pass
    return [pickle.loads((work / "cases" / f"{role}_{i}.pkl").read_bytes()) for i in range(n)]


def sel_init(evs: dict[Any, Ev]) -> None:
    _W["evs"] = evs


def task_enum(arg: tuple[int, int, int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    fi, b, a, z, keep, chunk = arg
    return enum_range(_W["evs"][("full", fi)], b, a, z, keep, chunk)


def task_search(arg: tuple[Any, ...]) -> dict[str, Any]:
    """('full', fi, b, col, greedy_layout, starts) -> best-found layout; ('split', fi, b, s, col, starts) -> half-design search."""
    if arg[0] == "full":
        _, fi, b, col, gl, starts = arg
        return PDE.search_layout(_W["evs"][("full", fi)], col, gl, starts)
    _, fi, b, s, col, starts = arg
    eh, eo = _W["evs"][("h", fi, s)], _W["evs"][("o", fi, s)]
    gl = tuple(sorted(PDE.greedy_order(eh, col, b)[0]))
    r = PDE.search_layout(eh, col, gl, starts)
    lay = np.array([r["layout"]])
    return {**r, "own": float(eh.values(lay, [col])[0, 0]), "other": float(eo.values(lay, [col])[0, 0])}


# ------------------------------------------------------------------------------------------------ evaluation of layouts on test
def lkey(lay: Any) -> str:
    return "-".join(str(int(i)) for i in lay)


def metric_row(case: Any, form: Any, lay: tuple[int, ...], rep: int, opts: D.DesignOptions, ctx: dict[str, Any]) -> dict[str, Any]:
    f = D.fit_layout(case, lay, rep, opts)
    e2 = form.e2(f["q_hat"]).ravel()
    area = math.sqrt(float(ctx["aw"] @ e2 / ctx["aw"].sum()))
    if abs(area ** 2 - f["mse_w"]) > 1e-8 * f["mse_w"] + 1e-18:
        raise RuntimeError("per-cell error form disagrees with design.fit_layout mse_w")
    pop = np.sqrt(ctx["N"] @ e2 / ctx["N"].sum(1))
    low = np.sqrt(ctx["L"] @ e2 / ctx["L"].sum(1))
    nan = float("nan")
    fe = f.get("external_fraction_error")
    row = {"case": case.name, "rep": rep, "area_rmse": area, "worst_cell_rmse": math.sqrt(float(e2[ctx["inside"]].max())),
           "heldout_rmse": f["heldout_rmse"], "heldout_random_rmse": f["heldout_by_group"].get("random", nan),
           "heldout_block_rmse": f["heldout_by_group"].get("block", nan), "rmse_local": math.sqrt(f["mse_local"]),
           "rmse_external": math.sqrt(f["mse_external"]), "attr_err": math.sqrt(f["mse_attribution"]),
           "ext_frac_abs_err": nan if fe is None else abs(fe)}
    row.update({f"pop__{t}__{k}": float(v) for (t, k), v in zip(ctx["labels"], pop)})
    row.update({f"lowq__{t}__{k}": float(v) for (t, k), v in zip(ctx["labels"], low)})
    return row


def eval_init(cases: list[Any], forms: list[Any], opts: dict[int, D.DesignOptions], ctx: dict[str, Any]) -> None:
    _W.update(ecases=cases, eforms=forms, eopts=opts, ectx=ctx)


def task_eval(arg: tuple[int, float, int, list[tuple[int, ...]]]) -> pd.DataFrame:
    fi, f, b, lays = arg
    rows = []
    for lay in lays:
        for c, form in zip(_W["ecases"], _W["eforms"]):
            for rep in range(c.noise.shape[0]):
                rows.append({"fraction": f, "budget": b, "layout": lkey(lay),
                             **metric_row(c, form, lay, rep, _W["eopts"][fi], _W["ectx"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ checks
def check_uniform_equals_area(rec: pd.DataFrame, objs: list[str], rtol: float = 1e-9) -> list[str]:
    """STOP check (registered): on the uniform fixture every objective's layout equals the area layout.  A different layout is a bug unless
    its area-design value is within rtol of the area optimum (a rounding-level tie, reported).  Raises RuntimeError; returns tie notes."""
    notes = []
    u = rec[rec["type"] == "uniform"]
    if u.empty:
        raise RuntimeError("uniform fixture is not part of this run: the registered sanity check cannot run")
    for (f, b, k), g in u.groupby(["fraction", "budget", "draw"]):
        g = g.set_index("objective")
        for o in objs:
            if g.loc[o, "layout"] != g.loc["area", "layout"]:
                rel = g.loc[o, "cross_area"] / g.loc["area", "value"] - 1
                if rel > rtol:
                    raise RuntimeError(f"SANITY FAILED (stop condition): uniform layout for {o} (fraction {f}, budget {b}) "
                                       f"{g.loc[o, 'layout']} != area layout {g.loc['area', 'layout']}")
                notes.append(f"uniform {o} f={f} b={b}: layout differs but design value within {rel:.1e} (rounding tie)")
    return notes


def verify_ev(ev: Ev, cases: list[Any], forms: list[Any], W: np.ndarray, opts: D.DesignOptions, cands: Any, cons: Any,
              n_layouts: int, nb: int, sites: np.ndarray) -> float:
    """Ev against design.score_layout (area column), against the per-cell route (a population column), and the vectorised greedy against
    design.greedy_trace.  Returns the largest relative discrepancy."""
    worst = 0.0
    for lay in PDE.random_starts(ev.ns, nb, n_layouts, 1):
        ref = D.score_layout(lay, cases, opts)["value"]
        mine = float(ev.values(np.array([lay]), [0])[0, 0])
        worst = max(worst, abs(ref - mine) / abs(ref))
        for col in sorted({0, ev.K - 1, ev.K // 2}):
            w = W[col] / W[col].sum()
            v = np.mean([np.mean([float((w.ravel() * form.e2(D.fit_layout(c, lay, r, opts)["q_hat"]).ravel()).sum())
                                  for r in range(c.noise.shape[0])]) for c, form in zip(cases, forms)])
            mine = float(ev.values(np.array([lay]), [col])[0, 0])
            worst = max(worst, abs(v - mine) / abs(v))
    if worst > 1e-8:
        raise RuntimeError(f"vectorised objective differs from design.score_layout / the per-cell route (rel {worst:.2e})")
    tr = D.greedy_trace(cands, nb, cases, cons, opts, "total")
    go, gs = PDE.greedy_order(ev, 0, nb)
    if go != tr.order and not np.allclose(gs, tr.scores, rtol=1e-8):
        raise RuntimeError("vectorised greedy differs from design.greedy_trace")
    return worst


# ------------------------------------------------------------------------------------------------ statistics
def cluster_bootstrap(M: np.ndarray, n_boot: int = 10000, seed: int = 0, level: float = 0.95) -> dict[str, tuple[float, float]]:
    """M (draws, cases).  Percentile intervals of the grand mean: 'both' resamples draws and cases independently with replacement
    (primary), 'draws' only the rows (cases fixed), 'cases' only the columns (draws fixed)."""
    M = np.asarray(M, float)
    nd, nc = M.shape
    if nd < 2 or nc < 2:
        return {k: (float("nan"), float("nan")) for k in ("both", "draws", "cases")}
    rng = np.random.default_rng(seed)
    idd, idc = rng.integers(0, nd, (n_boot, nd)), rng.integers(0, nc, (n_boot, nc))
    q = [(1 - level) / 2, 1 - (1 - level) / 2]
    sims = {"both": M[idd[:, :, None], idc[:, None, :]].mean((1, 2)), "draws": M.mean(1)[idd].mean(1), "cases": M.mean(0)[idc].mean(1)}
    return {k: (float(np.quantile(v, q[0])), float(np.quantile(v, q[1]))) for k, v in sims.items()}


def cluster_stats(M: np.ndarray, stats: dict[str, Any]) -> dict[str, Any]:
    """Summary of a paired-difference matrix M (draws, cases), negative = first layout better.  frac_lower/frac_higher: fraction of the
    test scenarios (case means over draws) below / above 0; *_draws: the same over the per-draw means.  resolved = primary interval
    excludes 0 AND >= agree_frac of the scenarios have the sign of the mean."""
    M = np.asarray(M, float)
    ci = cluster_bootstrap(M, int(stats["n_boot"]), int(stats["boot_seed"]), float(stats["level"]))
    dm, cm = M.mean(1), M.mean(0)
    mean = float(M.mean())
    lo, hi = ci["both"]
    agree = float((cm < 0).mean() if mean < 0 else (cm > 0).mean())
    return {"n_draws": M.shape[0], "n_cases": M.shape[1], "mean_diff": mean, "sd_draws": float(dm.std(ddof=1)) if len(dm) > 1 else float("nan"),
            "sd_cases": float(cm.std(ddof=1)) if len(cm) > 1 else float("nan"), "boot_lo": lo, "boot_hi": hi,
            "boot_draws_lo": ci["draws"][0], "boot_draws_hi": ci["draws"][1], "boot_cases_lo": ci["cases"][0], "boot_cases_hi": ci["cases"][1],
            "frac_lower": float((cm < 0).mean()), "frac_higher": float((cm > 0).mean()),
            "frac_lower_draws": float((dm < 0).mean()), "frac_higher_draws": float((dm > 0).mean()),
            "resolved": bool((lo > 0 or hi < 0) and agree >= float(stats["agree_frac"])), "interval": BOOT_LABEL}


def _matrix(g: pd.DataFrame, col: str) -> np.ndarray:
    return g.pivot(index="draw", columns="case", values=col).to_numpy()


def paired_tables(tm: pd.DataFrame, ties: pd.DataFrame, stats: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """tm rows: fraction, budget, type, draw, objective, kind (best|greedy), case, METRICS.  Returns (objective-minus-area-layout table,
    per-draw means of the same differences, greedy-minus-best table).  Ties are flagged on design values only: the second block drops the
    draws whose (objective, area) pair is a tie."""
    key = ["fraction", "budget", "type", "draw", "case"]
    best = tm[tm["kind"] == "best"]
    ref = best[best["objective"] == "area"].set_index(key)
    tied = set()
    if len(ties):
        tt = ties[(ties["a"] == "area") & ties["tie"]]
        tied = {(r.fraction, r.type, r.draw, r.budget, r.b) for r in tt.itertuples()}
    rows, bydraw = [], []
    for (f, b, t, o), g in best[best["objective"] != "area"].groupby(["fraction", "budget", "type", "objective"]):
        for subset in ("all", "excluding_ties"):
            gg = g if subset == "all" else g[[(f, t, d, b, o) not in tied for d in g["draw"]]]
            if gg.empty:
                continue
            d = gg.set_index(key)[list(METRICS)] - ref.loc[gg.set_index(key).index, list(METRICS)]
            d = d.reset_index()
            for m in METRICS:
                if d[m].isna().all():
                    continue
                rows.append({"fraction": f, "budget": b, "type": t, "objective": o, "metric": m, "subset": subset,
                             "mean_area_layout": float(ref.loc[gg.set_index(key).index, m].mean()), **cluster_stats(_matrix(d, m), stats)})
                if subset == "all":
                    for dr, v in d.groupby("draw")[m].mean().items():
                        bydraw.append({"fraction": f, "budget": b, "type": t, "objective": o, "metric": m, "draw": dr, "mean_diff": float(v)})
    gr = []
    gv = tm[tm["kind"] == "greedy"].set_index(key + ["objective"])
    bs = best.set_index(key + ["objective"])
    for (f, b, t, o), g in tm[tm["kind"] == "greedy"].groupby(["fraction", "budget", "type", "objective"]):
        idx = g.set_index(key + ["objective"]).index
        d = (gv.loc[idx, list(METRICS)] - bs.loc[idx, list(METRICS)]).reset_index()
        for m in METRICS:
            if d[m].isna().all():
                continue
            gr.append({"fraction": f, "budget": b, "type": t, "objective": o, "metric": m, "mean_best": float(bs.loc[idx, m].mean()),
                       **cluster_stats(_matrix(d, m), stats)})
    return pd.DataFrame(rows), pd.DataFrame(bydraw), pd.DataFrame(gr)


# ------------------------------------------------------------------------------------------------ context and stages
@dataclass
class Ctx:
    cfg: dict[str, Any]
    st: Any
    units: list[dict[str, Any]]
    objs: list[str]
    W: np.ndarray
    col_of: dict[tuple[str, int, str], int]
    sites: np.ndarray
    scale: np.ndarray
    fracs: list[float]

    @property
    def labels(self) -> list[tuple[str, int]]:
        return [(u["type"], u["draw"]) for u in self.units]

    def col_draw(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for (t, k, o), j in self.col_of.items():
            out[j] = min(k, out.get(j, k))
        return out


def make_ctx(cfg: dict[str, Any]) -> Ctx:
    st = G.Setup(cfg)
    res = st.res
    lv = st.geo.region.level(res)
    sf = np.asarray(lv.arrays["study_fraction"], float)
    units = make_units(cfg, lv.grid, sf, res, cfg["run"]["types"], cfg["run"]["draws"])
    objs = PD.objective_names(cfg["population"]["betas"])
    W, col_of = build_columns(units, objs)
    cx, cy = np.meshgrid(lv.grid.x_centres(), lv.grid.y_centres())
    sp = float(cfg["candidates"]["interior_spacing_m"])
    near = np.zeros(sf.shape, bool)
    inter = np.flatnonzero(np.asarray(st.cands.categories) == G.INTERIOR)
    for x, y in zip(st.cands.x[inter], st.cands.y[inter]):  # grid centres and candidates are both in local metres
        near |= np.hypot(cx - x, cy - y) <= sp
    pcfg = cfg["population"]
    check_population(units, sf, lambda t: None if t in pcfg.get("real", {}) else float(pcfg["study_total"]), near)
    sites = np.flatnonzero(np.asarray(st.cands.categories) == G.INTERIOR)
    if not (sites == np.arange(len(sites))).all():
        raise RuntimeError("interior candidates must come first in the candidate table")
    return Ctx(cfg, st, units, objs, W, col_of, sites, np.zeros(0), [float(f) for f in cfg["noise"]["fractions"]])


def frozen_inputs(cfg: dict[str, Any], validation: list[Any] | None, st: Any, infos: Any = None) -> tuple[float, dict[float, float]]:
    """(noise scale, alpha per fraction).  With frozen files configured the file values are used and the validation ensemble must
    reproduce them (scale within 1e-6 relative; alpha re-selected by geodesign.select_alpha on the validation cases, area objective)."""
    fr = cfg["frozen"]
    fracs = [float(f) for f in cfg["noise"]["fractions"]]
    scale_f = None
    if fr["noise_scale_file"]:
        t = pd.read_csv(ROOT / fr["noise_scale_file"])
        scale_f = float(t.loc[t["ensemble"] == "validation", "signal_scale_ug_m3"].iloc[0])
    alpha_f = None
    if fr["alpha_file"]:
        sel = json.loads((ROOT / fr["alpha_file"]).read_text())["alpha_by_noise_fraction"]
        alpha_f = {f: float(sel[f"{f:g}"]) for f in fracs}
    if validation is None:
        if scale_f is None or alpha_f is None:
            raise ValueError("a run without validation needs frozen alpha and noise-scale files")
        return scale_f, alpha_f
    scale = float(np.median([c.truth_readings.max() for c, _, _ in validation]))
    if scale_f is not None and abs(scale - scale_f) > 1e-6 * scale_f:
        raise RuntimeError(f"validation signal scale {scale} differs from the frozen {scale_f}")
    ens = G.Ensemble("validation", [c for c, _, _ in validation], [i for _, i, _ in validation], scale,
                     {f: f * (scale_f or scale) for f in fracs})
    sel = G.select_alpha(st, ens)["alpha_by_noise_fraction"]
    chosen = {f: float(sel[f"{f:g}"]) for f in fracs}
    if alpha_f is not None and chosen != alpha_f:
        raise RuntimeError(f"alpha selected on validation {chosen} differs from the frozen {alpha_f}")
    return scale_f or scale, alpha_f or chosen


def selection_stage(ctx: Ctx, design: list[Any], alphas: dict[float, float], sigmas: dict[float, float], work: Path, workers: int,
                    timing: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg, st = ctx.cfg, ctx.st
    sc, ds = cfg["search"], cfg["design_split"]
    cases = [c for c, _, _ in design]
    forms = [f for _, _, f in design]
    budgets = sorted(int(b) for b in cfg["methods"]["budgets"])
    bmax = max(budgets)
    ns, K = len(ctx.sites), ctx.W.shape[0]
    cdr = ctx.col_draw()
    scales = D.attribution_scales(cases)
    ev: dict[Any, Ev] = {}
    scale = np.array([SCALE_BY_KIND[k] for k in cases[0].param_kinds])
    for fi, f in enumerate(ctx.fracs):
        ev[("full", fi)] = Ev(cases, forms, ctx.W, alphas[f], sigmas[f], ctx.sites, scale)
        for s in (0, 1):
            ev[("h", fi, s)] = Ev(cases[s::2], forms[s::2], ctx.W, alphas[f], sigmas[f], ctx.sites, scale)
            ev[("o", fi, s)] = Ev(cases[1 - s::2], forms[1 - s::2], ctx.W, alphas[f], sigmas[f], ctx.sites, scale)
        opts = st.opts(sigmas[f], alphas[f], scales)
        timing[f"verify_rel_f{f:g}"] = verify_ev(ev[("full", fi)], cases, forms, ctx.W, opts, st.cands, st.constraints(G.SETTING_A, G.STANDARD),
                                                 int(sc["verify_layouts"]), min(3, budgets[0] + 1), ctx.sites)
    say(f"evaluators built and verified against design.score_layout / greedy_trace (K = {K} columns, {ns} interior sites)")
    rec_rows, split_rows = [], []
    exact_b = [b for b in budgets if b <= int(sc["exact_max_budget"])]
    for b in exact_b:
        if math.comb(ns, b) > int(sc["exact_max_combinations"]):
            raise RuntimeError(f"C({ns},{b}) = {math.comb(ns, b)} exceeds search.exact_max_combinations; lower exact_max_budget "
                               "(budgets above it are best-found by swap search)")
    with Pool(workers, sel_init, (ev,), float(cfg["run"]["rss_limit_gb"])) as pool:
        for fi, f in enumerate(ctx.fracs):
            t0 = time.perf_counter()
            evf = ev[("full", fi)]
            greedy = {c: PDE.greedy_order(evf, c, bmax) for c in range(K)}
            timing[f"greedy_f{f:g}_s"] = time.perf_counter() - t0
            # exact enumeration, ranges in the pool
            exact: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for b in exact_b:
                tot = math.comb(ns, b)
                nr = max(1, min(int(sc["enum_ranges"]), tot // int(sc["min_range"])))
                edges = np.linspace(0, tot, nr + 1).astype(int)
                units = [(work / "enum" / f"f{fi}_b{b}_{i}.pkl", (fi, b, int(edges[i]), int(edges[i + 1]), int(sc["keep"]), int(sc["chunk"])))
                         for i in range(nr)]
                t1 = time.perf_counter()
                res = run_units(pool, task_enum, units)
                exact[b] = merge_enum([res[p] for p, _ in units], int(sc["keep"]))
                timing[f"enum_f{f:g}_b{b}_wall_s"] = time.perf_counter() - t1
            # swap searches: best-found budgets for every column; exact budgets for the check draws
            su, meta = [], {}
            for b in budgets:
                starts = PDE.random_starts(ns, b, int(sc["n_restarts"]), int(sc["restart_seed"]) + 1000 * fi + b)
                for c in range(K):
                    if b in exact and cdr[c] >= int(sc["search_check_draws"]):
                        continue
                    gl = tuple(sorted(greedy[c][0][:b]))
                    su.append((work / "search" / f"f{fi}_b{b}_c{c}.pkl", ("full", fi, b, c, gl, starts)))
            for b in [int(x) for x in ds["budgets"]]:
                starts = PDE.random_starts(ns, b, int(sc["n_restarts"]), int(sc["restart_seed"]) + 1000 * fi + b)
                for c in sorted({j for (t, k, o), j in ctx.col_of.items() if k in ds["draws"]}):
                    for s in (0, 1):
                        su.append((work / "search" / f"f{fi}_b{b}_split{s}_c{c}.pkl", ("split", fi, b, s, c, starts)))
            t1 = time.perf_counter()
            sres = run_units(pool, task_search, su)
            timing[f"search_f{f:g}_wall_s"] = time.perf_counter() - t1
            for (p, arg) in su:
                meta[(arg[0], arg[2], arg[3] if arg[0] == "full" else (arg[3], arg[4]))] = sres[p]
            for b in budgets:
                is_exact = b in exact
                for c in range(K):
                    if is_exact:
                        v, o = exact[b]
                        near = int((v[:, c] <= v[0, c] * (1 + float(sc["near_tie_frac"]))).sum())
                        r = {"layout": PDE.unrank(int(o[0, c]), ns, b), "value": float(v[0, c]), "margin_rel": float(v[1, c] / v[0, c] - 1),
                             "n_within": near, "n_within_censored": near >= len(v)}
                        s_ = meta.get(("full", b, c))
                        if s_ is not None:
                            r.update(search_value=s_["value"], search_hit_exact=bool(s_["value"] <= r["value"] * (1 + 1e-9)))
                    else:
                        s_ = meta[("full", b, c)]
                        r = {**s_, "margin_rel": np.nan, "n_within": -1, "n_within_censored": False}
                    gl = tuple(sorted(greedy[c][0][:b]))
                    gv = float(evf.values(np.array([gl]), [c])[0, 0])
                    r.update(greedy_layout=gl, greedy_value=gv)
                    meta[("sel", b, c)] = r
            for b in budgets:
                for (t, k, o), c in ctx.col_of.items():
                    r = meta[("sel", b, c)]
                    row = {"fraction": f, "budget": b, "type": t, "draw": k, "objective": o, "col": c,
                           "method": "exact" if b in exact else "best_found", "layout": lkey(r["layout"]),
                           "layout_ids": json.dumps([st.cands.ids[i] for i in r["layout"]]), "value": r["value"],
                           "greedy_layout": lkey(r["greedy_layout"]), "greedy_value": r["greedy_value"], "gap_rel": r["greedy_value"] / r["value"] - 1}
                    row.update({x: r.get(x, np.nan) for x in ("margin_rel", "n_within", "n_within_censored", "n_starts", "hit_starts",
                                                              "search_value", "search_hit_exact")})
                    cc = [ctx.col_of[(t, k, oo)] for oo in ctx.objs]
                    row.update({f"cross_{oo}": float(v) for oo, v in zip(ctx.objs, evf.values(np.array([r["layout"]]), cc)[0])})
                    rec_rows.append(row)
            for b in [int(x) for x in ds["budgets"]]:
                for (t, k, o), c in ctx.col_of.items():
                    if k not in ds["draws"]:
                        continue
                    for s in (0, 1):
                        r = meta[("split", b, (s, c))]
                        full = meta[("sel", b, c)]
                        split_rows.append({"fraction": f, "budget": b, "type": t, "draw": k, "objective": o, "split": s,
                                           "layout": lkey(r["layout"]), "own_design_value": r["own"], "other_half_value": r["other"],
                                           "optimism_gap": r["other"] / r["own"] - 1, "full_design_layout": lkey(full["layout"]),
                                           "full_design_value": full["value"]})
            say(f"selection done for noise fraction {f:g}")
    return pd.DataFrame(rec_rows), pd.DataFrame(split_rows)


def regression_check(cfg: dict[str, Any], rec: pd.DataFrame, st: Any) -> str:
    """Area greedy layout (design cases, frozen alpha/sigma) must equal the Phase 2 greedy_total / interior_only / standard layout."""
    f = cfg["frozen"]["regression_layouts_file"]
    if not f:
        return "skipped (no regression file configured)"
    ph2 = pd.read_csv(ROOT / f)
    ph2 = ph2[(ph2["method"] == "greedy_total") & (ph2["setting"] == G.SETTING_A) & (ph2["regime"] == G.STANDARD)]
    n = 0
    for r in rec[(rec["objective"] == "area")].drop_duplicates(["fraction", "budget"]).itertuples():
        m = ph2[(ph2["budget"] == r.budget) & np.isclose(ph2["noise_fraction"], r.fraction)]
        if len(m):
            n += 1
            mine = {st.cands.ids[int(i)] for i in r.greedy_layout.split("-")}
            if mine != set(m["sites"].iloc[0].split(",")):
                raise RuntimeError(f"area greedy layout (fraction {r.fraction}, budget {r.budget}) differs from Phase 2: {sorted(mine)} vs {m['sites'].iloc[0]}")
    return f"{n} area greedy layouts equal results/geodesign/layouts.csv"


def freeze_hash(cfg: dict[str, Any], alphas: dict[float, float], sigmas: dict[float, float], rec: pd.DataFrame, split: pd.DataFrame) -> str:
    lay = rec[["fraction", "budget", "type", "draw", "objective", "layout", "greedy_layout"]].astype(str).values.tolist()
    sp = split[["fraction", "budget", "type", "draw", "objective", "split", "layout"]].astype(str).values.tolist() if len(split) else []
    return hashlib.sha256(json.dumps({"config": cfg_hash(cfg), "alphas": {str(k): v for k, v in alphas.items()},
                                      "sigmas": {str(k): v for k, v in sigmas.items()}, "layouts": lay, "split": sp}, sort_keys=True).encode()).hexdigest()


def assert_frozen(out: Path, now: str) -> None:
    check_frozen(out, now)


def eval_layout_sets(ctx: Ctx, rec: pd.DataFrame, split: pd.DataFrame) -> dict[tuple[int, int], list[tuple[int, ...]]]:
    """Distinct layouts to evaluate on test per (fraction index, budget): best, greedy, half-design (split), random and spread baselines."""
    sets: dict[tuple[int, int], set[tuple[int, ...]]] = {}
    fidx = {f: i for i, f in enumerate(ctx.fracs)}
    parse = lambda s: tuple(int(x) for x in s.split("-"))  # noqa: E731
    for df, cols in ((rec, ("layout", "greedy_layout")), (split, ("layout", "full_design_layout"))):
        for c in cols:
            for f, b, s in zip(df["fraction"], df["budget"], df[c]):
                sets.setdefault((fidx[f], int(b)), set()).add(parse(s))
    base, _ = G.alpha_independent_layouts(ctx.st, G.SETTING_A, G.STANDARD)
    for fi in range(len(ctx.fracs)):
        for r in base:
            sets.setdefault((fi, r.budget), set()).add(tuple(sorted(r.layout)))
    return {k: sorted(v) for k, v in sets.items()}


def eval_stage(ctx: Ctx, rec: pd.DataFrame, split: pd.DataFrame, test: list[Any], design: list[Any], alphas: dict[float, float],
               sigmas: dict[float, float], work: Path, workers: int) -> pd.DataFrame:
    cfg, st = ctx.cfg, ctx.st
    scales = D.attribution_scales([c for c, _, _ in design])
    opts = {fi: st.opts(sigmas[f], alphas[f], scales) for fi, f in enumerate(ctx.fracs)}
    sf = st.weight
    aw = (st.geo.region.level(st.res).grid.cell_area * sf).ravel()
    ctxw = {"aw": aw, "N": np.stack([u["counts"].ravel() for u in ctx.units]),
            "L": np.stack([u["low"].ravel().astype(float) for u in ctx.units]), "inside": (sf > 0).ravel(), "labels": ctx.labels}
    sets = eval_layout_sets(ctx, rec, split)
    ch = int(cfg["search"]["eval_chunk"])
    units = []
    for (fi, b), lays in sorted(sets.items()):
        for i in range(0, len(lays), ch):
            units.append((work / "eval" / f"f{fi}_b{b}_{i // ch}.pkl", (fi, ctx.fracs[fi], b, lays[i:i + ch])))
    with Pool(workers, eval_init, ([c for c, _, _ in test], [f for _, _, f in test], opts, ctxw), float(cfg["run"]["rss_limit_gb"])) as pool:
        res = run_units(pool, task_eval, units)
    return pd.concat([res[p] for p, _ in units], ignore_index=True)


# ------------------------------------------------------------------------------------------------ analysis
def case_means(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.drop(columns="rep").groupby(["fraction", "budget", "layout", "case"], as_index=False).mean(numeric_only=True)


def long_metrics(cm: pd.DataFrame, rec: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (fraction, budget, type, draw, objective, kind in best|greedy, case): the unit's own population metrics."""
    parts = []
    for kind, col in (("best", "layout"), ("greedy", "greedy_layout")):
        r = rec[["fraction", "budget", "type", "draw", "objective", col]].rename(columns={col: "layout"})
        parts.append(r.assign(kind=kind))
    sel = pd.concat(parts, ignore_index=True)
    m = sel.merge(cm, on=["fraction", "budget", "layout"])
    for src, dst in (("pop", "pop_rmse"), ("lowq", "lowq_rmse")):
        m[dst] = np.nan
        for (t, k), idx in m.groupby(["type", "draw"]).groups.items():
            m.loc[idx, dst] = m.loc[idx, f"{src}__{t}__{k}"]
    return m[["fraction", "budget", "type", "draw", "objective", "kind", "layout", "case", *METRICS]]


def cross_fixture(cm: pd.DataFrame, rec: pd.DataFrame, obj: str) -> pd.DataFrame:
    """Layouts designed for fixture (X, draw) with objective `obj`, evaluated under the population of (Y, draw'): ratio of the case-mean
    population RMSE to that of the area layout.  matched = same (type, draw) as the design population."""
    lk = {k: g.set_index("case") for k, g in cm.groupby(["fraction", "budget", "layout"])}
    rows = []
    area = rec[rec["objective"] == "area"].drop_duplicates(["fraction", "budget"]).set_index(["fraction", "budget"])["layout"]
    pcols = [c for c in cm.columns if c.startswith("pop__")]
    for r in rec[rec["objective"] == obj].itertuples():
        a = lk[(r.fraction, r.budget, area[(r.fraction, r.budget)])]
        g = lk[(r.fraction, r.budget, r.layout)]
        for c in pcols:
            _, ty, ky = c.split("__")
            rows.append({"fraction": r.fraction, "budget": r.budget, "design_type": r.type, "design_draw": r.draw, "eval_type": ty,
                         "eval_draw": int(ky), "matched": (ty == r.type and int(ky) == r.draw), "ratio": float((g[c] / a[c]).mean())})
    d = pd.DataFrame(rows)
    return d.groupby(["fraction", "budget", "design_type", "eval_type", "matched"], as_index=False)["ratio"].mean()


def tables(ctx: Ctx, rec: pd.DataFrame, split: pd.DataFrame, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cfg = ctx.cfg
    cm = case_means(raw)
    tm = long_metrics(cm, rec)
    r2 = rec.assign(placement=rec["fraction"].map({f: i for i, f in enumerate(ctx.fracs)}))
    ties = PDE.tie_pairs(r2, float(cfg["search"]["near_tie_frac"]), ctx.objs)
    ties["fraction"] = ties["placement"].map(dict(enumerate(ctx.fracs)))
    paired, bydraw, greedy = paired_tables(tm, ties, cfg["stats"])
    gap = rec.groupby(["fraction", "type", "objective", "budget", "method"], as_index=False)["gap_rel"].agg(
        mean_gap="mean", median_gap="median", max_gap="max", frac_greedy_suboptimal=lambda x: float((x > 1e-9).mean()), n="size")
    out = {"selection": rec, "near_ties": ties, "test_case_means": cm, "test_metrics": tm, "paired": paired, "paired_by_draw": bydraw,
           "greedy_vs_best": greedy, "design_gap": gap,
           "cross_fixture": cross_fixture(cm, rec, cfg["figures"]["cross_objective"]),
           "pareto": tm[tm["kind"] == "best"].groupby(["fraction", "type", "objective", "budget"], as_index=False)[list(METRICS)].mean()}
    if len(split):
        s = split.copy()
        tmean = cm.groupby(["fraction", "budget", "layout"]).mean(numeric_only=True)
        for tag, col in (("split", "layout"), ("full", "full_design_layout")):
            s[f"test_pop_rmse_{tag}"] = [float(tmean.loc[(f, b, lay), f"pop__{t}__{k}"]) for f, b, lay, t, k in
                                         zip(s["fraction"], s["budget"], s[col], s["type"], s["draw"])]
        out["design_split"] = s
    return out


# ------------------------------------------------------------------------------------------------ figures
def figures(out: Path, ctx: Ctx, tabs: dict[str, pd.DataFrame], profile: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pravah.belauri import _km, draw_map
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    tag = f"{STAMP} ({profile}; {LABEL})"
    st, cfg = ctx.st, ctx.cfg
    lv = st.geo.region.level(st.res)
    sf = np.asarray(lv.arrays["study_fraction"], float)
    types = list(dict.fromkeys(u["type"] for u in ctx.units))
    colors = dict(zip(types, plt.get_cmap("tab10").colors))
    objs = ctx.objs
    mk = dict(zip(objs, ["o", "s", "^", "*", "D", "v"]))

    def finish(fig: Any, name: str) -> None:
        fig.text(0.995, 0.005, tag, ha="right", va="bottom", fontsize=6, color="0.4")
        fig.savefig(fd / name, dpi=130)
        plt.close(fig)

    ld, lb, f0 = int(cfg["figures"]["layout_draw"]), int(cfg["figures"]["layout_budget"]), ctx.fracs[0]
    fig, ax = plt.subplots(3, len(types), figsize=(3.2 * len(types), 9), squeeze=False)
    for j, t in enumerate(types):
        us = [x for x in ctx.units if x["type"] == t]
        u = next((x for x in us if x["draw"] == ld), us[0])
        for i in range(3):
            draw_map(ax[i, j], st.geo.region, st.res)
        ext = _km(lv)
        im = ax[0, j].imshow(np.ma.masked_where(sf == 0, u["counts"]), origin="lower", extent=ext, alpha=0.7, cmap="viridis")
        fig.colorbar(im, ax=ax[0, j], fraction=0.04, label="people / cell")
        w = u["w"]["mixed_0.5"] if "mixed_0.5" in u["w"] else u["w"]["population"]
        im = ax[1, j].imshow(np.ma.masked_where(sf == 0, w / u["w"]["area"]), origin="lower", extent=ext, alpha=0.7, cmap="magma")
        fig.colorbar(im, ax=ax[1, j], fraction=0.04, label="weight / area weight")
        r = tabs["selection"]
        for o in objs:
            g = r[(r["fraction"] == f0) & (r["budget"] == lb) & (r["type"] == t) & (r["draw"] == u["draw"]) & (r["objective"] == o)]
            if len(g):
                ix = [int(i) for i in g["layout"].iloc[0].split("-")]
                ax[2, j].plot(st.cands.x[ix] / 1000, st.cands.y[ix] / 1000, mk[o], ms=7, mfc="none", label=o)
        ax[0, j].set_title(f"{t} (draw {u['draw']}): population", fontsize=7)
        ax[1, j].set_title("mixed weight / area weight", fontsize=7)
        ax[2, j].set_title(f"layouts at budget {lb}", fontsize=7)
    ax[2, 0].legend(fontsize=6)
    fig.tight_layout()
    finish(fig, "fig1_population_weights_layouts.png")

    tm = tabs["pareto"]
    budgets = sorted(tm["budget"].unique())
    fig, ax = plt.subplots(1, len(budgets), figsize=(3.2 * len(budgets), 3.6), squeeze=False)
    for j, b in enumerate(budgets):
        for t in types:
            for o in objs:
                g = tm[(tm["fraction"] == f0) & (tm["budget"] == b) & (tm["type"] == t) & (tm["objective"] == o)]
                ax[0, j].plot(g["area_rmse"], g["pop_rmse"], mk[o], color=colors[t], ms=6)
        ax[0, j].set(title=f"budget {b}", xlabel="test area RMSE", ylabel="test pop RMSE" if j == 0 else "")
        ax[0, j].title.set_fontsize(8)
        ax[0, j].grid(alpha=0.3)
    ax[0, 0].legend(handles=[plt.Line2D([], [], marker=mk[o], ls="", color="k", label=o) for o in objs] +
                    [plt.Line2D([], [], marker="o", ls="", color=colors[t], label=t) for t in types], fontsize=5)
    fig.suptitle(f"Pareto: mean over draws and test scenarios, noise fraction {f0:g}", fontsize=8)
    fig.tight_layout()
    finish(fig, "fig2_pareto_pop_vs_area.png")

    cf = tabs["cross_fixture"]
    fig, ax = plt.subplots(1, len(budgets), figsize=(3.4 * len(budgets), 3.6), squeeze=False)
    for j, b in enumerate(budgets):
        g = cf[(cf["fraction"] == f0) & (cf["budget"] == b) & (~cf["matched"])]
        mat = g.pivot_table(index="design_type", columns="eval_type", values="ratio").reindex(index=types, columns=types)
        im = ax[0, j].imshow(mat.to_numpy(), cmap="RdBu_r", vmin=0, vmax=2)
        ax[0, j].set(xticks=range(len(types)), yticks=range(len(types)), title=f"budget {b}")
        ax[0, j].set_xticklabels(types, rotation=90, fontsize=5)
        ax[0, j].set_yticklabels(types if j == 0 else [], fontsize=5)
        ax[0, j].set_xlabel("evaluated on population of", fontsize=6)
    fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.02, label="pop RMSE / area-layout pop RMSE")
    fig.suptitle(f"layouts designed for row fixture ({cfg['figures']['cross_objective']}), other draw of the column fixture; <1 helps", fontsize=8)
    finish(fig, "fig3_cross_fixture_matrix.png")

    gv = tabs["greedy_vs_best"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.8))
    for i, o in enumerate(objs):
        m = "area_rmse" if o == "area" else "pop_rmse"
        g = gv[(gv["fraction"] == f0) & (gv["objective"] == o) & (gv["metric"] == m)]
        for t in types:
            gg = g[g["type"] == t].sort_values("budget")
            ax.plot(np.arange(len(gg)) + 0.08 * i, gg["mean_diff"] / gg["mean_best"] * 100, mk[o], color=colors[t], ms=5)
        ax.set_xticks(range(len(budgets)), [str(b) for b in budgets])
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set(xlabel="budget", ylabel="test RMSE of greedy layout vs best-found layout (%)")
    ax.legend(handles=[plt.Line2D([], [], marker=mk[o], ls="", color="k", label=o + (" (area RMSE)" if o == "area" else " (pop RMSE)")) for o in objs] +
              [plt.Line2D([], [], marker="o", ls="", color=colors[t], label=t) for t in types], fontsize=5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    finish(fig, "fig4_greedy_vs_exact_test_gap.png")


# ------------------------------------------------------------------------------------------------ pilot timing
def comb_ratio(ns: int, b: int) -> float:
    return math.comb(ns, b) / math.comb(ns, 4)


def estimate(m: dict[str, Any], prod: dict[str, Any], ns: int, K: int, K_split: int, W: int) -> dict[str, float]:
    """Production wall estimate (seconds) for W workers from pilot measurements m (all per single process)."""
    F = len(prod["noise"]["fractions"])
    budgets = prod["methods"]["budgets"]
    sc = prod["search"]
    exact_b = [b for b in budgets if b <= sc["exact_max_budget"]]
    nc = sum(int(prod["ensembles"][r]["n_cases"]) for r in ("design", "validation", "test"))
    ncase_test = int(prod["ensembles"]["test"]["n_cases"])
    reps = int(prod["noise"]["reps"])
    ts = m["t_search"]
    chk = max(1, K // int(prod["population"]["n_draws"]))
    searches = []  # (unit seconds, count)
    for b in budgets:
        searches.append((ts[str(b)] if str(b) in ts else ts["4"], (chk if b in exact_b else K) * F))
    for b in prod["design_split"]["budgets"]:
        searches.append((0.6 * ts[str(b)], 2 * K_split * F))
    enum = [m["t_enum4"] * comb_ratio(ns, b) * F for b in exact_b]
    L = 2 * K + int(prod["methods"]["n_random"]) + 1 + 2 * K_split
    fits = F * len(budgets) * L * ncase_test * reps
    out = {}
    for w in (1, W):
        t = m["t_setup"] + sum(math.ceil(int(prod["ensembles"][r]["n_cases"]) / w) for r in ("validation", "design", "test")) * m["t_case"] + m["t_alpha"] + F * m["t_greedy"] + 60 * (w > 1)
        t += sum(e / min(w, sc["enum_ranges"]) for e in enum)
        t += sum(math.ceil(n / w) * u if n >= w else u * (n > 0) for u, n in searches)
        t += fits * m["t_fit"] / w + (m["t_fit"] * 12 * reps * sc["eval_chunk"] if w > 1 else 0)
        out[str(w)] = t
    out["serial_parts"] = {"cases": nc * m["t_case"], "enum": sum(enum), "search": sum(u * n for u, n in searches), "eval": fits * m["t_fit"]}
    return out


def pilot(cfg: dict[str, Any], out: Path, work: Path, workers: int) -> None:
    """Pilot (EXPLORATORY, own seeds, no test cases): one full-size design case set, one exact enumeration, searches and eval timings,
    then a production estimate."""
    prod = load_config(ROOT / CONFIGS["production"])
    m: dict[str, Any] = {}
    t0 = time.perf_counter()
    ctx = make_ctx(cfg)
    m["t_setup"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    with Pool(workers, None, (), float(cfg["run"]["rss_limit_gb"])) as pool:
        design = build_role(cfg, work, "design", pool)
    wall = time.perf_counter() - t0
    m["t_case"] = float(np.mean([i["build_total_s"] for _, i, _ in design]))
    m["cases_wall_s"], m["workers"] = wall, workers
    cases, forms = [c for c, _, _ in design], [f for _, _, f in design]
    scale_f, alpha_f = frozen_inputs(cfg, None, ctx.st)
    f = ctx.fracs[0]
    sigma = f * scale_f
    scale = np.array([SCALE_BY_KIND[k] for k in cases[0].param_kinds])
    ns, K = len(ctx.sites), ctx.W.shape[0]
    ev = Ev(cases, forms, ctx.W, alpha_f[f], sigma, ctx.sites, scale)
    opts = ctx.st.opts(sigma, alpha_f[f], D.attribution_scales(cases))
    m["verify_rel"] = verify_ev(ev, cases, forms, ctx.W, opts, ctx.st.cands, ctx.st.constraints(G.SETTING_A, G.STANDARD), 3, 4, ctx.sites)
    t0 = time.perf_counter()
    greedy = {c: PDE.greedy_order(ev, c, 8) for c in range(K)}
    m["t_greedy"] = time.perf_counter() - t0
    sc = cfg["search"]
    t0 = time.perf_counter()
    vals, ords = enum_range(ev, 4, 0, math.comb(ns, 4), int(sc["keep"]), int(sc["chunk"]))
    m["t_enum4"] = time.perf_counter() - t0
    v, o = merge_enum([(vals, ords)], int(sc["keep"]))
    best4 = {tuple(PDE.unrank(int(o[0, c]), ns, 4)) for c in range(K)}
    m["distinct_best_layouts_b4"] = len(best4)
    ts = {}
    for b in (4, 5, 8):
        starts = PDE.random_starts(ns, b, int(sc["n_restarts"]), 1)
        t0 = time.perf_counter()
        for c in (0, K // 2):
            PDE.search_layout(ev, c, tuple(sorted(greedy[c][0][:b])), starts)
        ts[str(b)] = (time.perf_counter() - t0) / 2
    m["t_search"] = ts
    ec = {"aw": (ctx.st.geo.region.level(ctx.st.res).grid.cell_area * ctx.st.weight).ravel(), "N": np.stack([u["counts"].ravel() for u in ctx.units]),
          "L": np.stack([u["low"].ravel().astype(float) for u in ctx.units]), "inside": (ctx.st.weight > 0).ravel(), "labels": ctx.labels}
    t0 = time.perf_counter()
    nfit = 0
    for lay in sorted(best4)[:3]:
        for c, fm in zip(cases, forms):
            for rep in range(c.noise.shape[0]):
                metric_row(c, fm, lay, rep, opts, ec)
                nfit += 1
    m["t_fit"] = (time.perf_counter() - t0) / nfit
    nalpha = len(prod["alpha"]["grid"]) * len(prod["noise"]["fractions"]) * (prod["methods"]["n_random"] * len(prod["methods"]["budgets"]) + len(prod["methods"]["budgets"]))
    m["t_alpha"] = nalpha * 6 * int(prod["noise"]["reps"]) * m["t_fit"] * 0.5
    kd = ctx.col_draw()
    ds = set(prod["design_split"]["draws"])
    k_split = len({j for (t, k, oo), j in ctx.col_of.items() if k in ds})
    est = estimate(m, prod, ns, K, k_split, 64)
    m.update(n_interior=ns, n_columns=K, n_columns_split=k_split, estimate_s=est, estimate_h={k: v / 3600 for k, v in est.items() if k != "serial_parts"},
             peak_rss_gb=rss_gb(), note="EXPLORATORY pilot: production estimate extrapolated from single-process timings; K here includes this pilot's fixtures")
    write_json(out / "timing.json", m)
    say(f"PILOT t_case {m['t_case']:.1f}s (wall {wall:.0f}s at {workers} workers), t_enum(b=4) {m['t_enum4']:.1f}s, t_greedy {m['t_greedy']:.1f}s, "
        f"t_search {ts}, t_fit {m['t_fit'] * 1e3:.1f} ms")
    say(f"PRODUCTION ESTIMATE: 1 worker {est['1'] / 3600:.1f} h, 64 workers {est['64'] / 3600:.2f} h ({json.dumps(est['serial_parts'])})")


# ------------------------------------------------------------------------------------------------ main
def run(profile: str, cfg_path: str | None, out_arg: str | None, work_arg: str | None) -> int:
    cfg = load_cfg(profile, cfg_path)
    out = ROOT / (out_arg or OUTS[profile])
    work = Path(work_arg) if work_arg else out / "work"
    check_seeds(cfg)
    check_ensemble_draws(cfg)
    guard_out(out, cfg, profile)
    work.mkdir(parents=True, exist_ok=True)
    for d in ("tables", "figures", "raw"):
        (out / d).mkdir(exist_ok=True)
    workers = n_workers(cfg)
    write_json(out / "config.json", cfg)
    if profile == "pilot":
        pilot(cfg, out, work, workers)
        return 0
    timing: dict[str, Any] = {"profile": profile, "workers": workers}
    ctx = make_ctx(cfg)
    ctx.scale = np.zeros(0)
    say(f"{len(ctx.units)} fixture units, {ctx.W.shape[0]} distinct weight columns, {len(ctx.sites)} interior sites, {workers} workers")
    t0 = time.perf_counter()
    with Pool(workers, None, (), float(cfg["run"]["rss_limit_gb"])) as pool:
        val = None if cfg["run"]["skip_validation"] else build_role(cfg, work, "validation", pool)
        design = build_role(cfg, work, "design", pool)
    timing["design_validation_cases_s"] = time.perf_counter() - t0
    scale, alphas = frozen_inputs(cfg, val, ctx.st)
    sigmas = {f: f * scale for f in ctx.fracs}
    say(f"alpha {alphas}, sigma {sigmas}")
    rec, split = selection_stage(ctx, design, alphas, sigmas, work, workers, timing)
    objs = ctx.objs
    notes = check_uniform_equals_area(rec, [o for o in objs if o != "area"]) if "uniform" in set(rec["type"]) else []
    reg = regression_check(cfg, rec, ctx.st)
    say(f"uniform==area check passed ({len(notes)} rounding ties); regression: {reg}")
    fh = freeze_hash(cfg, alphas, sigmas, rec, split)
    pp = out / "provenance.json"
    if pp.exists():
        assert_frozen(out, fh)
    else:
        write_json(pp, {**provenance(cfg, {r: cfg["ensembles"][r] for r in cfg["ensembles"]}, {"experiment": STAMP, "profile": profile, "label": LABEL}),
                        "frozen_config_hash": fh, "alphas": alphas, "sigmas": sigmas, "regression": reg, "uniform_ties": notes,
                        "frozen_before_test_utc": pd.Timestamp.now("UTC").isoformat()})
    rec.to_csv(out / "layouts.csv", index=False)
    # test cases only now (layouts frozen on disk)
    assert pp.exists() and (out / "layouts.csv").exists()
    t0 = time.perf_counter()
    with Pool(workers, None, (), float(cfg["run"]["rss_limit_gb"])) as pool:
        test = build_role(cfg, work, "test", pool)
    timing["test_cases_s"] = time.perf_counter() - t0
    D.assert_disjoint([c for c, _, _ in design], [c for c, _, _ in test], None if val is None else [c for c, _, _ in val])
    t0 = time.perf_counter()
    raw = eval_stage(ctx, rec, split, test, design, alphas, sigmas, work, workers)
    timing["eval_s"] = time.perf_counter() - t0
    assert_frozen(out, freeze_hash(cfg, alphas, sigmas, rec, split))
    raw.to_csv(out / "raw" / "test_rows.csv", index=False)
    pd.DataFrame([i for _, i, _ in (val or []) + design + test]).to_csv(out / "cases.csv", index=False)
    tabs = tables(ctx, rec, split, raw)
    for k, t in tabs.items():
        t.to_csv(out / "tables" / f"{k}.csv", index=False)
    figures(out, ctx, tabs, profile)
    timing.update(total_s=time.perf_counter() - T0, peak_rss_gb=rss_gb(), freeze_hash_verified=fh)
    write_json(out / "timing.json", timing)
    write_json(pp, {**json.loads(pp.read_text()), "freeze_guard_passed_after_test": True})
    say(f"done -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=list(CONFIGS), default="smoke")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default=None, help="response-cache / unit-file directory (default OUT/work)")
    a = ap.parse_args(argv)
    return run(a.profile, a.config, a.out, a.work)


if __name__ == "__main__":
    sys.exit(main())

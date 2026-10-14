"""DYN-04 (EXP-032): dynamic population-weighted sensor design vs random / spread / high-population / static-optimal / source-info layouts.

    python -m pravah.studies.toy.dynamic_design_study --stage design --out DIR            # surrogate search, writes layouts.json (hashed)
    python -m pravah.studies.toy.dynamic_design_study --stage tune   --out DIR            # validation seeds 6000-6047, spread layout per budget
    python -m pravah.studies.toy.dynamic_design_study --stage test   --out DIR            # eval seeds 7800-7895; needs frozen_hparams.json

Design uses the Gaussian linear-Bayes surrogate of ``pravah.dynamic_design`` in x = theta/REF units.  The true wind is one of the 6
dynamic_study winds, so the ROBUST design risk is the mean over winds of the per-wind surrogate risk (``SumProblem``).  Only M^T M enters
the surrogate, so the (n_out = 37*432) x 30 operator M is replaced by its 30 x 30 symmetric square root (Gram trick; exact for the risk).
The surrogate ignores the nonnegativity constraint; the held-out evaluation uses the real constrained estimator M2 (dynamic_study).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from pravah.io import write_json
from pravah.dynamic_design import DesignProblem
from pravah.studies.toy import dynamic_study as ds

CAND_IX, CAND_IY = tuple(range(2, 22, 2)), tuple(range(2, 16, 2))
CANDS = tuple((ix, iy) for ix in CAND_IX for iy in CAND_IY)  # 70 cells (ix, iy); candidate index = position in this tuple
KS = (1, 2, 3, 5)
N_RANDOM = 10
EXH_MAX = 3  # exhaustive search is the optimiser up to this budget (C(70,3) = 54740); beyond it, multi-start swap
N_EXTRA_STARTS = 6
MIN_SEP = 3  # high-population layout: minimum Chebyshev separation (cells)
SEED_EVAL, N_SEEDS_EVAL = 7800, 2  # 6 winds x 8 families x 2 = 96 episodes
BASELINES = ("random", "spread", "highpop", "static_opt", "src_opt")
FAMILY_ORDER = ("random", "spread", "highpop", "static_opt", "dyn_opt", "src_opt")
ALPHA_ALL = 0.05 / (len(KS) * len(ds.NOISE) * len(BASELINES))  # stricter Bonferroni over every reported contrast
_RESP: dict = {}


def eval_episodes() -> list[dict]:
    return [{"id": i, "wind": w, "family": f, "seed": SEED_EVAL + i, "stage": "eval"}
            for i, (w, f, _) in enumerate(itertools.product(ds.WINDS, ds.FAMILIES, range(N_SEEDS_EVAL)))]


# ---------------------------------------------------------------- surrogate operators
def _resp(wind: str):
    if wind not in _RESP:
        _RESP[wind] = ds.build_responses(wind)
    return _RESP[wind]


def wind_operators(wind: str, noise: float, mode: str = "dyn", prior_var: float = 1.0) -> DesignProblem:
    """Per-wind surrogate.  mode 'dyn': 30 params (5 x 6 bins); 'static': bins collapsed to 5 params; 'src': identifiable dyn columns, M = I.
    x = theta / REF, so A = response * REF and prior_var is in x units."""
    R = _resp(wind)
    g = np.asarray(R.g)
    iy, ix = np.array([c[1] for c in CANDS]), np.array([c[0] for c in CANDS])
    ref = np.repeat(ds.REF, ds.N_INV)
    A = np.moveaxis(g[:, 1:, iy, ix], (0, 2), (2, 0)).reshape(len(CANDS), -1, g.shape[0]) * ref  # (cand, 36 obs, 30)
    sigma = noise * ds.reference_scale(R)
    w = np.sqrt(np.outer(ds._time_weights(np.asarray(R.times)), ds.population().reshape(-1) / ds.population().sum()).reshape(-1))
    gw = g.reshape(g.shape[0], -1) * w[None] * ref[:, None]
    Q = gw @ gw.T  # = M^T M, n_theta x n_theta
    if mode == "static":
        S = np.kron(np.eye(ds.NP), np.ones((ds.N_INV, 1)))  # (30, 5)
        A, Q = A @ S, S.T @ Q @ S
    elif mode == "src":
        nrm = np.sqrt((A ** 2).sum(axis=(0, 1)))
        keep = nrm > 1e-6 * nrm.max()
        A, Q = A[:, :, keep], np.eye(int(keep.sum()))
    elif mode != "dyn":
        raise ValueError(mode)
    lam, V = np.linalg.eigh((Q + Q.T) / 2)
    M = (V * np.sqrt(np.maximum(lam, 0.0))).T  # M^T M = Q
    return DesignProblem(A, M, np.full(A.shape[2], float(prior_var)), sigma)


class SumProblem:
    """Mean over winds of per-wind surrogate risks (kind 'risk' = tr(M Sigma M^T), 'info' = tr Sigma).  Search methods are DesignProblem's."""

    def __init__(self, problems: list[DesignProblem], kind: str = "risk"):
        if kind not in ("risk", "info"):
            raise ValueError(kind)
        self.problems, self.kind = list(problems), kind

    @property
    def n_cand(self) -> int:
        return self.problems[0].n_cand

    def _check_k(self, k: int) -> None:
        self.problems[0]._check_k(k)

    def risk_batch(self, subsets, chunk: int = 4096) -> np.ndarray:
        s = np.asarray(subsets, dtype=np.intp)
        if self.kind == "risk":
            return np.mean([p.risk_batch(s, chunk=chunk) for p in self.problems], axis=0)
        vals = []
        for p in self.problems:
            s2 = p._check_subsets(s)
            vals.append(np.concatenate([p._rows(s2[i:i + chunk], p._gram_p, p._cross_i, p._trace_p) for i in range(0, len(s2), chunk)]))
        return np.mean(vals, axis=0)

    def risk(self, subset) -> float:
        return float(self.risk_batch(np.asarray(subset, dtype=np.intp).reshape(1, -1))[0])

    exhaustive = DesignProblem.exhaustive
    greedy = DesignProblem.greedy
    swap_refine = DesignProblem.swap_refine


# ---------------------------------------------------------------- layout families (return sorted candidate-index tuples)
def random_layout(k: int, r: int, n: int = len(CANDS)) -> tuple[int, ...]:
    return tuple(sorted(int(i) for i in np.random.default_rng([k, r]).choice(n, k, replace=False)))


def spread_layout(k: int) -> tuple[int, ...]:
    """Farthest-point: start at the candidate nearest the domain centre, then add the candidate farthest from the chosen set (ties: lowest index)."""
    xy = np.array(CANDS, dtype=float)
    centre = np.array([(ds.scenario("steady_a").domain.nx - 1) / 2, (ds.scenario("steady_a").domain.ny - 1) / 2])
    chosen = [int(np.argmin(((xy - centre) ** 2).sum(1)))]
    while len(chosen) < k:
        d = np.min(np.linalg.norm(xy[:, None] - xy[chosen][None], axis=2), axis=1)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return tuple(sorted(chosen))


def highpop_layout(k: int, counts: np.ndarray | None = None, sep: int = MIN_SEP) -> tuple[int, ...]:
    """Top-k candidate cells by population count with pairwise Chebyshev separation >= sep cells (ties: lowest index)."""
    counts = ds.population() if counts is None else counts
    pop = np.array([counts[iy, ix] for ix, iy in CANDS])
    chosen: list[int] = []
    for i in np.argsort(-pop, kind="stable"):
        if all(max(abs(CANDS[i][0] - CANDS[j][0]), abs(CANDS[i][1] - CANDS[j][1])) >= sep for j in chosen):
            chosen.append(int(i))
        if len(chosen) == k:
            return tuple(sorted(chosen))
    raise ValueError("not enough separated candidates")


def search(prob: SumProblem, k: int, starts: tuple[tuple[int, ...], ...] = ()) -> dict:
    """Optimise one problem at budget k.  k <= EXH_MAX: exhaustive optimum, with greedy and greedy+swap gaps.
    Larger k: best of swap_refine from greedy, the given starts and N_EXTRA_STARTS deterministic random starts (no optimality certificate)."""
    t0 = time.time()
    g, rg = prob.greedy(k)
    gs, rgs = prob.swap_refine(g)
    out = {"k": k, "greedy": list(g), "greedy_risk": rg, "greedy_swap": list(gs), "greedy_swap_risk": rgs}
    if k <= EXH_MAX or k in GAP_EXTRA:
        ex = prob.exhaustive(k, chunk=2048)
        out.update(exhaustive=list(ex.best_subset), exhaustive_risk=ex.best_risk, median_risk=ex.median_risk, worst_risk=ex.worst_risk,
                   n_evaluated=ex.n_evaluated)
    if k <= EXH_MAX:
        best, risk, method = ex.best_subset, ex.best_risk, "exhaustive"
    else:
        ss = [gs] + [prob.swap_refine(s)[0] for s in starts]
        ss += [prob.swap_refine(random_layout(k, 100 + j, prob.n_cand))[0] for j in range(N_EXTRA_STARTS)]
        rs = [prob.risk(s) for s in ss]
        j = int(np.argmin(rs))
        best, risk, method = ss[j], rs[j], f"swap_multistart({len(ss)})"
        out["start_risks"] = rs
    out.update(subset=list(best), risk=float(risk), method=method, seconds=time.time() - t0)
    return out


GAP_EXTRA: tuple[int, ...] = ()


def _cells(idx) -> list[list[int]]:
    return [list(CANDS[i]) for i in idx]


def design(noises=ds.NOISE, ks=KS, gap_ks: tuple[int, ...] = (), prior_var: float = 1.0, winds=tuple(ds.WINDS)) -> dict:
    """Surrogate design for every (noise, budget).  Peak memory is one problem set (6 winds) at a time."""
    global GAP_EXTRA
    GAP_EXTRA = tuple(gap_ks)
    counts = ds.population()
    doc: dict = {"candidates": [list(c) for c in CANDS], "prior_var": prior_var, "winds": list(winds), "layouts": {}, "search": {}, "timings_s": {}}
    for noise in noises:
        lay = {k: {"random": [_cells(random_layout(k, r)) for r in range(N_RANDOM)], "spread": _cells(spread_layout(k)),
                   "highpop": _cells(highpop_layout(k, counts))} for k in ks}
        idx = {k: {"spread": spread_layout(k), "highpop": highpop_layout(k, counts)} for k in ks}
        srch: dict = {}
        for fam, mode, kind in (("static_opt", "static", "risk"), ("src_opt", "src", "info"), ("dyn_opt", "dyn", "risk")):
            t0 = time.time()
            prob = SumProblem([wind_operators(w, noise, mode, prior_var) for w in winds], kind)
            doc["timings_s"][f"{noise}|{fam}|build"] = time.time() - t0
            for k in sorted(set(ks) | set(gap_ks)):
                res = search(prob, k, tuple(idx[k].values()) if k in idx else ())
                srch[f"{fam}|{k}"] = res
                if k in ks:
                    lay[k][fam] = _cells(res["subset"])
                    idx[k][fam] = tuple(res["subset"])
                doc["timings_s"][f"{noise}|{fam}|{k}"] = res["seconds"]
            if fam == "dyn_opt":  # surrogate risk of every layout under the dynamic (robust) surrogate
                for k in ks:
                    lay[k]["surrogate_risk"] = {"random": [prob.risk(random_layout(k, r)) for r in range(N_RANDOM)],
                                                **{f: prob.risk(idx[k][f]) for f in FAMILY_ORDER if f != "random"}}
            del prob
        doc["layouts"][str(noise)] = {str(k): v for k, v in lay.items()}
        doc["search"][str(noise)] = srch
    doc["hash"] = layouts_hash(doc)
    return doc


def layouts_hash(doc: dict) -> str:
    return hashlib.sha256(json.dumps({"layouts": doc["layouts"], "candidates": doc["candidates"]}, sort_keys=True).encode()).hexdigest()[:16]


def load_layouts(path) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc["hash"] != layouts_hash(doc):
        raise RuntimeError("freeze guard: layouts file does not match its hash")
    return doc


def layout_dict(doc: dict, noise: float) -> dict[str, tuple[tuple[int, int], ...]]:
    """Named layouts for one design noise: 'k{k}_{family}' (random replicates 'k{k}_random_{rr}')."""
    out = {}
    for k, v in doc["layouts"][str(noise)].items():
        for fam in FAMILY_ORDER:
            if fam == "random":
                out.update({f"k{k}_random_{r:02d}": tuple(map(tuple, c)) for r, c in enumerate(v["random"])})
            else:
                out[f"k{k}_{fam}"] = tuple(map(tuple, v[fam]))
    return out


# ---------------------------------------------------------------- tuning (validation seeds only) and evaluation
def tune_hparams(workers: int, doc: dict, ks=KS) -> dict:
    """Per budget: dynamic_study.tune on the 'spread' layout (validation episodes, both noises); M0 and M2 are kept.  Reuses the ds grids."""
    frozen, edge = {}, []
    for k in ks:
        r = ds.tune(workers, {f"spread{k}": tuple(map(tuple, doc["layouts"][str(ds.NOISE[0])][str(k)]["spread"]))})
        for fr in ds.NOISE:
            d = r["frozen"][f"spread{k}|{fr}"]
            frozen[f"{k}|{fr}"] = {"M0": d["M0"], "M2": d["M2"]}
        edge += [f"k{k}:{e}" for e in r["edge_selections"] if ":M0:" in e or ":M2:" in e]
    return {"frozen": frozen, "edge_selections": edge, "layouts_hash": doc["hash"], "hash": ds._hash(frozen)}


def _eval_task(ep: dict, noise: float, layouts: dict, hp: dict) -> list[dict]:
    old = ds.NOISE
    ds.NOISE = (noise,)  # evaluate each layout only at the noise it was designed for
    try:
        rows = ds.run_episode(ep, hp, ("M2",), layouts)
    finally:
        ds.NOISE = old
    out = []
    for r in rows:
        if r["model"] == "floor":
            continue
        kk, fam = r["layout"].split("_", 1)
        out.append({**r, "k": int(kk[1:]), "design_family": fam, "group": "random" if fam.startswith("random") else fam})
    return out


def evaluate(doc: dict, frozen: dict, eps: list[dict], workers: int, ks=KS) -> list[dict]:
    """Real constrained estimator M2 with the frozen per-(budget, noise) hyperparameters on every layout of that budget."""
    tasks = []
    for noise in ds.NOISE:
        lay = {n: c for n, c in layout_dict(doc, noise).items() if int(n.split("_", 1)[0][1:]) in ks}
        hp = {f"{n}|{noise}": {m: [tuple(c) for c in frozen[f"{n.split('_', 1)[0][1:]}|{noise}"][m]] for m in ("M0", "M2")} for n in lay}
        tasks += [(ep, noise, lay, hp) for ep in eps]
    if workers <= 1:
        return [r for t in tasks for r in _eval_task(*t)]
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    ds._prebuild(workers)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_eval_task, *t) for t in tasks]
        return [r for f in futs for r in f.result()]


# ---------------------------------------------------------------- summary
def summarise(rows: list[dict], doc: dict) -> dict:
    ok_ep = {r["episode"] for r in rows if r["valid"]}
    wind_of = {r["episode"]: r["wind"] for r in rows}
    per: dict = {}  # (k, noise, group, metric) -> episode -> list of values
    for r in rows:
        if r["episode"] in ok_ep:
            for m in ("jpop", "area", "src_local", "src_bnd"):
                per.setdefault((r["k"], r["noise"], r["group"], m), {}).setdefault(r["episode"], []).append(r[m])

    def vec(k, nz, grp, m):  # episode -> value (random = mean of its replicates; any non-finite replicate -> nan)
        return {e: float(np.mean(v)) if np.all(np.isfinite(v)) else float("nan") for e, v in per.get((k, nz, grp, m), {}).items()}

    contrasts, tradeoff, spear, per_layout = [], [], [], []
    for k, nz in itertools.product(sorted({r["k"] for r in rows}), ds.NOISE):
        for m in ("jpop", "area", "src_local", "src_bnd"):
            a = vec(k, nz, "dyn_opt", m)
            for b_name in BASELINES:
                b = vec(k, nz, b_name, m)
                eps_ok = sorted(e for e in a if e in b and np.isfinite(a[e]) and np.isfinite(b[e]))
                if not eps_ok:
                    continue
                d = np.array([a[e] - b[e] for e in eps_ok])
                bm = float(np.mean([b[e] for e in eps_ok]))
                lo, hi = ds.boot_ci(d, ds.ALPHA_FWE)
                lo2, hi2 = ds.boot_ci(d, ALPHA_ALL)
                leave = {w: float(np.mean([a[e] - b[e] for e in eps_ok if wind_of[e] != w])) for w in ds.WINDS}
                rel = float(d.mean() / bm) if bm else float("nan")
                contrasts.append({"k": k, "noise": nz, "metric": m, "contrast": f"dyn_opt-{b_name}", "n": len(d), "mean_diff": float(d.mean()),
                                  "ci_lo": lo, "ci_hi": hi, "ci_lo_all": lo2, "ci_hi_all": hi2, "rel_change": rel, "frac_dyn_better": float((d < 0).mean()),
                                  "mean_dyn": float(np.mean([a[e] for e in eps_ok])), "mean_base": bm, "leave_one_wind_out": leave,
                                  "effect_criterion_met": bool(m == "jpop" and hi < 0 and rel <= -ds.MIN_REL_GAIN and all(v < 0 for v in leave.values()))})
        for grp in FAMILY_ORDER:
            v = {m: vec(k, nz, grp, m) for m in ("jpop", "area", "src_local", "src_bnd")}
            tradeoff.append({"k": k, "noise": nz, "family": grp, **{m: float(np.nanmean(list(x.values()))) if x else float("nan") for m, x in v.items()}})
        # surrogate vs actual over all evaluated layouts of this budget/noise
        sur = doc["layouts"][str(nz)][str(k)]["surrogate_risk"]
        act, sv = [], []
        for fam in sorted({r["design_family"] for r in rows if r["k"] == k and r["noise"] == nz}):
            vals = [r["jpop"] for r in rows if r["k"] == k and r["noise"] == nz and r["design_family"] == fam and r["episode"] in ok_ep]
            if not vals or not np.all(np.isfinite(vals)):
                continue
            s = sur["random"][int(fam.split("_")[1])] if fam.startswith("random_") else sur[fam]
            act.append(float(np.mean(vals)))
            sv.append(s)
            per_layout.append({"k": k, "noise": nz, "layout": fam, "surrogate_risk": s, "mean_jpop": act[-1]})
        rho = spearmanr(sv, act) if len(act) > 2 else None
        spear.append({"k": k, "noise": nz, "n_layouts": len(act), "spearman": float(rho.statistic) if rho is not None else float("nan"),
                      "p": float(rho.pvalue) if rho is not None else float("nan")})
    gaps = []
    for nz, s in doc["search"].items():
        for key, r in s.items():
            fam, k = key.split("|")
            gaps.append({"noise": float(nz), "problem": fam, "k": int(k), "method": r["method"], "best_risk": r["risk"], "greedy_risk": r["greedy_risk"],
                         "greedy_swap_risk": r["greedy_swap_risk"], "exhaustive_risk": r.get("exhaustive_risk"), "median_risk": r.get("median_risk"),
                         "greedy_gap": (r["greedy_risk"] / r["risk"] - 1) if r["risk"] else None,
                         "swap_gap": (r["greedy_swap_risk"] / r["risk"] - 1) if r["risk"] else None,
                         "start_risk_range": [min(r["start_risks"]), max(r["start_risks"])] if "start_risks" in r else None})
    return {"contrasts": contrasts, "tradeoff": tradeoff, "surrogate_vs_actual": spear, "per_layout": per_layout, "search_gaps": gaps,
            "primary_alpha": ds.ALPHA_FWE, "all_contrast_alpha": ALPHA_ALL, "min_relative_gain": ds.MIN_REL_GAIN,
            "n_nonconverged_fits": int(sum(1 for r in rows if not np.isfinite(r["jpop"]))),
            "n_invalid_episodes": len({r["episode"] for r in rows if not r["valid"]})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("design", "tune", "test"), required=True)
    ap.add_argument("--out", default=os.environ.get("RESEARCH_RUN_DIR", "results/dyn04"))
    ap.add_argument("--freeze", default=None, help="frozen_hparams.json (test stage; default OUT/frozen_hparams.json)")
    ap.add_argument("--layouts", default=None, help="layouts.json (default OUT/layouts.json)")
    ap.add_argument("--gap-ks", type=int, nargs="*", default=[], help="extra budgets for an exhaustive-vs-heuristic gap (k=4 is slow)")
    ap.add_argument("--prior-var", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lp = Path(a.layouts or out / "layouts.json")
    if a.stage == "design":
        doc = design(gap_ks=tuple(a.gap_ks), prior_var=a.prior_var)
        write_json(lp, doc)
        print("layouts", doc["hash"])
        return
    doc = load_layouts(lp)
    if a.stage == "tune":
        r = tune_hparams(a.workers, doc)
        write_json(out / "frozen_hparams.json", r)
        print("frozen", r["hash"], "layouts", doc["hash"], "edge selections:", r["edge_selections"])
        return
    fz = json.loads(Path(a.freeze or out / "frozen_hparams.json").read_text())
    if fz["hash"] != ds._hash(fz["frozen"]):
        raise RuntimeError("freeze guard: hyperparameter file does not match its hash")
    if fz["layouts_hash"] != doc["hash"]:
        raise RuntimeError("freeze guard: layouts changed after tuning")
    rows = evaluate(doc, fz["frozen"], eval_episodes(), a.workers)
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = summarise(rows, doc)
    write_json(out / "summary.json", {"frozen_hash": fz["hash"], "layouts_hash": doc["hash"], "n_episodes": len(eval_episodes()), **res,
                                      "note": "synthetic only; seeds 7800-7895; wind known; model error not included"})
    for c in res["contrasts"]:
        if c["metric"] == "jpop":
            print(c["k"], c["noise"], c["contrast"], f"{c['mean_diff']:+.5f} [{c['ci_lo']:+.5f},{c['ci_hi']:+.5f}] rel {c['rel_change']:+.2f} met={c['effect_criterion_met']}")


if __name__ == "__main__":
    main()

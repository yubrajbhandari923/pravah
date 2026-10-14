"""DYN-02 v2 (EXP-030): historical source inversion M0 / M1 / M2 on synthetic dynamic episodes (toy geometry).

Supersedes the F-037 pipeline after its independent review (solver never converged, null-sensitivity parameters in the truth and
the metrics, layout-dependent prior scale, under-tuned zero-prior M1, one-wind effect, oracle that was not a floor).

    python -m pravah.studies.toy.dynamic_study --stage tune --out DIR [--workers 8]   # validation seeds 6000-6047 only
    python -m pravah.studies.toy.dynamic_study --stage test --freeze FILE --out DIR   # test seeds 7200-7295, hyperparameters frozen

Everything is synthetic.  Truth: 12 fine bins with jittered edges, simulated at dt 60 s; inverse: 6 bins, dt 120 s.  Same grid, K and
(exact) wind: model error is the subject of DYN-07, not of this study.  The statistical unit is the episode; wind is a blocking factor
(6 winds) and every claim is also checked leaving each wind out.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from pravah import dynamic as dyn
from pravah.config import Boundary, Domain, GaussianSource, InflowBasis, Scenario, Times, Transport, WindSegment
from pravah.forward import simulate
from pravah.geo.regionio import GridSpec
from pravah.inverse import ResponseCache, Responses
from pravah.io import write_json
from pravah.population import separated_settlements

DUR = 21600.0
N_INV, N_FINE, NP = 6, 12, 5
SRC_XY = ((3000.0, 6500.0), (6500.0, 3000.0), (9000.0, 6500.0))
REF = np.array([1e5, 1e5, 1e5, 0.1, 0.1])  # fixed physical reference amplitudes (ug/s, ug/m^3): layout-independent prior scale
WINDS = {"steady_a": ((0.0, 1.0, 0.3),), "steady_b": ((0.0, 0.4, -0.9),),
         "rotation_a": ((0.0, 1.0, 0.3), (10800.0, 0.0, 1.0)), "rotation_b": ((0.0, 0.3, -1.0), (10800.0, 1.0, 0.4)),
         "reversal": ((0.0, 1.0, 0.2), (10800.0, -1.0, -0.2)), "slow_reversal": ((0.0, 0.5, 0.1), (10800.0, -0.5, -0.1))}
FAMILIES = ("constant", "activation", "shutdown", "ramp", "opposing", "pulse", "influx_end", "compensating")
LAYOUTS = {"L4": ((6, 5), (6, 12), (17, 5), (17, 12)),  # (ix, iy) cell indices
           "L9": tuple((x, y) for x in (4, 12, 20) for y in (3, 9, 15))}
NOISE = (0.02, 0.10)  # fraction of the wind's reference mean concentration S0 (independent of the truth field)
ALPHA_M = tuple(10.0 ** e for e in np.arange(-2.0, 9.01, 0.5))  # M0, M1z, M1 (exact solves): half-decade
ALPHA_2 = tuple(10.0 ** e for e in np.arange(-2.0, 9.01, 1.0))  # M2 (ADMM)
BETAS = tuple(10.0 ** e for e in np.arange(-1.0, 7.01, 1.0))
SEEDS = {"tune": 6000, "test": 7200}
N_SEEDS = {"tune": 1, "test": 2}  # seeds per (wind, family)
MIN_REL_GAIN = 0.10  # registered: M2 must also improve J_pop by >= 10% of M1's mean to count as an effect
ALPHA_FWE = 0.05 / 4  # Bonferroni over the 4 primary cells (layout x noise)


def scenario(wind: str, dt: float = 120.0) -> Scenario:
    srcs = tuple(GaussianSource(f"s{i}", x, y, 600.0) for i, (x, y) in enumerate(SRC_XY))
    bases = (InflowBasis("B_wsw", sides=("west", "south")), InflowBasis("B_ene", sides=("east", "north")))
    return Scenario(domain=Domain(lx=12000.0, ly=9000.0, nx=24, ny=18), sources=srcs,
                    transport=Transport(wind=tuple(WindSegment(*w) for w in WINDS[wind])),
                    boundary=Boundary(mode="open", bases=bases), times=Times(duration=DUR, dt=dt, output_interval=600.0))


def population() -> np.ndarray:
    return separated_settlements(GridSpec(0.0, 0.0, 24, 18, 500.0, 500.0, 32645), 100_000, seed=0).astype(float)


_CACHE = ResponseCache(os.environ.get("PRAVAH_RESPONSE_CACHE", ".cache/responses"))  # key covers scenario, solver, params


def build_responses(wind: str) -> Responses:
    return _CACHE.get(dyn.expand(scenario(wind), N_INV))


def live_mask(R: Responses) -> np.ndarray:
    """(NP, N_FINE) bool: the parameter's response field is non-zero in that fine bin (outflow-face bases are dead)."""
    nrm = np.sqrt((R.g.reshape(NP, N_INV, -1) ** 2).sum(axis=2))
    return np.repeat(nrm > 1e-9 * nrm.max(), N_FINE // N_INV, axis=1)


def reference_scale(R: Responses) -> float:
    """S0: mean concentration of the reference history (all parameters at REF, averaged over bins) - independent of any truth."""
    ref_field = np.tensordot(np.repeat(REF, N_INV) / N_INV, R.g, axes=1)
    return float(ref_field.mean())


def make_history(family: str, rng: np.random.Generator, live: np.ndarray) -> tuple[np.ndarray, bool]:
    """(NP, N_FINE) true history (ug/s rows 0-2, ug/m^3 rows 3-4) with boundary activity only where the basis is live.
    Returns (history, valid); valid is False when the wind offers no live window for the requested boundary event."""
    h = np.zeros((NP, N_FINE))
    h[:3] = rng.uniform(0.6, 1.6, 3)[:, None] * 1e5
    bamp = rng.uniform(0.5, 1.5) * 0.1
    k = int(rng.integers(3, 9))
    if family == "activation":
        h[0, :k] = 0.0
    elif family == "shutdown":
        h[1, k:] = 0.0
    elif family == "ramp":
        h[2] = h[2, 0] * np.linspace(1.0, 3.0, N_FINE)
    elif family == "opposing":
        h[0, k:] *= 3.0
        h[1, k:] *= 0.3
    elif family in ("pulse", "influx_end", "compensating"):
        order = [3, 4] if rng.random() < 0.5 else [4, 3]
        for b in order:
            lv = live[b]
            if family == "pulse":
                w = int(rng.integers(2, 4))
                starts = [i for i in range(N_FINE - w + 1) if lv[i:i + w].all()]
                if starts:
                    i = int(rng.choice(starts))
                    h[b, i:i + w] = bamp
                    return h, True
            elif family == "influx_end":
                ks = [kk for kk in range(2, 10) if lv[:kk].sum() >= 2 and lv[0]]
                if ks:
                    kk = int(rng.choice(ks))
                    h[b, :kk] = bamp
                    return h, True
            else:
                ks = [kk for kk in range(2, 9) if lv[kk:].sum() >= 2]
                if ks:
                    kk = int(rng.choice(ks))
                    h[0, kk:] *= 0.3
                    h[b, kk:] = bamp
                    return h, True
        return h, False
    elif family != "constant":
        raise ValueError(family)
    return h, True


def episodes(stage: str) -> list[dict]:
    return [{"id": i, "wind": w, "family": f, "seed": SEEDS[stage] + i, "stage": stage}
            for i, (w, f, _) in enumerate(itertools.product(WINDS, FAMILIES, range(N_SEEDS[stage])))]


def truth_field(wind: str, hist: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Truth simulated at dt 60 s with change times jittered by up to +-35% of a fine bin (never nested in the inverse bins)."""
    edges = dyn.bin_edges(DUR, N_FINE)
    edges[1:-1] += rng.uniform(-0.35, 0.35, N_FINE - 1) * (DUR / N_FINE)
    sc = dyn.truth_scenario(scenario(wind, dt=60.0), hist, edges)
    return np.asarray(simulate(sc, np.ones(NP)).concentration.values)


def _time_weights(t: np.ndarray) -> np.ndarray:
    w = np.zeros(len(t))
    w[1:] += np.diff(t) / 2
    w[:-1] += np.diff(t) / 2
    return w / (t[-1] - t[0])


def projection_floor(R: Responses, tf: np.ndarray, counts: np.ndarray) -> float:
    """Smallest attainable J_pop RMSE on the 6-bin nonnegative basis: population-time weighted NNLS of the true field."""
    t = np.asarray(R.times)
    w = np.sqrt(np.outer(_time_weights(t), counts.reshape(-1) / counts.sum())).reshape(-1)
    cols = (R.g.reshape(R.g.shape[0], -1) * w[None, :] * np.repeat(REF, N_INV)[:, None]).T
    x, _ = nnls(cols, tf.reshape(-1) * w, maxiter=5000)
    return float(np.linalg.norm(cols @ x - tf.reshape(-1) * w))


def jpop(R: Responses, theta: np.ndarray, tf: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    err = np.tensordot(theta.reshape(-1), R.g, axes=1) - tf
    wt = _time_weights(np.asarray(R.times))
    pop = (wt[:, None, None] * err ** 2 * (counts / counts.sum())[None]).sum()
    area = (wt[:, None, None] * err ** 2).mean(axis=(1, 2)).sum()
    return float(np.sqrt(pop)), float(np.sqrt(area))


def _cells(layout: str) -> tuple[np.ndarray, np.ndarray]:
    ix, iy = zip(*LAYOUTS[layout])
    return np.array(iy), np.array(ix)


def _fit(model: str, A, y, sig, hp, prior):
    a, b = hp
    try:
        if model == "M0":
            return dyn.fit_history(A, y, sig, NP, N_INV, "M0", alpha=a, ref=REF).theta
        if model == "M1z":
            return dyn.fit_history(A, y, sig, NP, N_INV, "M1", alpha=a, ref=REF).theta
        return dyn.fit_history(A, y, sig, NP, N_INV, "M1" if model == "M1" else "M2", alpha=a, beta=b, theta0=prior, ref=REF).theta
    except dyn.NotConverged:
        return None


def run_episode(ep: dict, hp: dict[str, dict[str, list[tuple[float, float]]]], models: tuple[str, ...]) -> list[dict]:
    """One row per (layout, noise, model, hyperparameter candidate); truth, floor and noise are shared by all of them.
    ``hp[key]['M0']`` must hold exactly one (frozen) entry when 'M1' / 'M2' are evaluated: it defines their static prior mean."""
    R = build_responses(ep["wind"])
    counts = population()
    rng = np.random.default_rng([ep["seed"], 0])
    hist, valid = make_history(ep["family"], rng, live_mask(R))
    tf = truth_field(ep["wind"], hist, np.random.default_rng([ep["seed"], 1]))
    t = np.asarray(R.times)
    s0 = reference_scale(R)
    binned = hist.reshape(NP, N_INV, N_FINE // N_INV).mean(axis=2)
    base = {"episode": ep["id"], "wind": ep["wind"], "family": ep["family"], "valid": valid, "stage": ep["stage"]}
    nan = float("nan")
    rows = [{**base, "layout": "-", "noise": 0.0, "model": "floor", "alpha": nan, "beta": nan, "jpop": projection_floor(R, tf, counts),
             "area": nan, "src_local": nan, "src_bnd": nan, "ident_frac": nan}]
    for layout, (ni, frac) in itertools.product(LAYOUTS, enumerate(NOISE)):
        iy, ix = _cells(layout)
        sd = frac * s0
        y = tf[1:, iy, ix] + np.random.default_rng([ep["seed"], 2, ni]).standard_normal((len(t) - 1, len(iy))) * sd
        A = np.moveaxis(R.g[:, 1:, iy, ix], 0, -1).reshape(-1, R.g.shape[0])
        sig = np.full(y.size, sd)
        sens = dyn.sensitivity(A * np.repeat(REF, N_INV), sig, NP, N_INV)
        ident = sens > 1e-6 * sens.max()
        cfg = hp[f"{layout}|{frac}"]
        prior = _fit("M0", A, y.reshape(-1), sig, cfg["M0"][0], None) if ("M1" in models or "M2" in models) else None
        for model in models:
            for h in cfg[model]:
                th = _fit(model, A, y.reshape(-1), sig, h, prior)
                if th is None:
                    rows.append({**base, "layout": layout, "noise": frac, "model": model, "alpha": h[0], "beta": h[1], "jpop": np.inf,
                                 "area": nan, "src_local": nan, "src_bnd": nan, "ident_frac": float(ident.mean())})
                    continue
                jp, ar = jpop(R, th, tf, counts)
                e = (th - binned) / REF[:, None]
                rows.append({**base, "layout": layout, "noise": frac, "model": model, "alpha": h[0], "beta": h[1], "jpop": jp, "area": ar,
                             "src_local": float(np.sqrt(np.mean(e[:3][ident[:3]] ** 2))) if ident[:3].any() else nan,
                             "src_bnd": float(np.sqrt(np.mean(e[3:][ident[3:]] ** 2))) if ident[3:].any() else nan,
                             "ident_frac": float(ident.mean())})
    return rows


def _pool_rows(eps: list[dict], hp, models, workers: int) -> list[dict]:
    if workers <= 1:
        return [r for ep in eps for r in run_episode(ep, hp, models)]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(run_episode, ep, hp, models) for ep in eps]
        return [r for f in futs for r in f.result()]


def _prebuild(workers: int) -> None:
    """Fill the response cache (6 independent winds) before the episode pool starts."""
    if workers <= 1:
        list(map(build_responses, WINDS))
        return
    with ProcessPoolExecutor(min(workers, len(WINDS)), mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(build_responses, WINDS))


def _score_table(rows: list[dict]) -> dict:
    sc: dict = {}
    for r in rows:
        if r["model"] != "floor":
            sc.setdefault((r["layout"], r["noise"], r["model"]), {}).setdefault((r["alpha"], r["beta"]), []).append(r["jpop"])
    return {k: {c: float(np.mean(v)) for c, v in d.items()} for k, d in sc.items()}


def tune(workers: int) -> dict:
    """Validation only.  Pass 1 picks M0's alpha; pass 2 picks M1z / M1 / M2 with M0@alpha* as the static prior of M1 / M2.
    Non-converged candidates score +inf.  Edge selections are flagged, never silently accepted."""
    _prebuild(workers)
    eps = episodes("tune")
    combos = [(lay, fr) for lay in LAYOUTS for fr in NOISE]
    g0 = [(a, 0.0) for a in ALPHA_M]
    s1 = _score_table(_pool_rows(eps, {f"{lay}|{fr}": {"M0": g0} for lay, fr in combos}, ("M0",), workers))
    m0 = {(lay, fr): min(s1[(lay, fr, "M0")], key=lambda c, d=s1[(lay, fr, "M0")]: d[c]) for lay, fr in combos}
    grid = {"M1z": [(a, 0.0) for a in ALPHA_M], "M1": [(a, 0.0) for a in ALPHA_M], "M2": list(itertools.product(ALPHA_2, BETAS))}
    s2 = _score_table(_pool_rows(eps, {f"{lay}|{fr}": {"M0": [m0[(lay, fr)]], **grid} for lay, fr in combos}, ("M1z", "M1", "M2"), workers))
    frozen: dict = {f"{lay}|{fr}": {"M0": [list(m0[(lay, fr)])]} for lay, fr in combos}
    edge, scores = [], {}
    for (lay, fr, m), d in s2.items():
        k = f"{lay}|{fr}"
        best = min(d, key=lambda c, d=d: d[c])
        frozen[k][m] = [list(best)]
        scores.setdefault(k, {})[m] = {f"{a:g}|{b:g}": v for (a, b), v in d.items()}
        al = ALPHA_M if m != "M2" else ALPHA_2
        if best[0] in (al[0], al[-1]) or (m == "M2" and best[1] in (BETAS[0], BETAS[-1])):
            edge.append(f"{k}:{m}:{best}")
    for (lay, fr, _), d in s1.items():
        if m0[(lay, fr)][0] in (ALPHA_M[0], ALPHA_M[-1]):
            edge.append(f"{lay}|{fr}:M0:{m0[(lay, fr)]}")
    scores_m0 = {f"{lay}|{fr}": {f"{a:g}|{b:g}": v for (a, b), v in d.items()} for (lay, fr, _), d in s1.items()}
    return {"frozen": frozen, "edge_selections": edge, "validation_jpop": scores, "validation_jpop_M0": scores_m0}


def _hash(frozen: dict) -> str:
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:16]


def boot_ci(d: np.ndarray, level: float, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(np.percentile(m, 100 * level / 2)), float(np.percentile(m, 100 * (1 - level / 2)))


def summarise(rows: list[dict]) -> dict:
    out, by_wind = [], []
    valid_ids = {r["episode"] for r in rows if r["valid"]}

    def vec(lay, fr, m, metric, winds=None):
        sel = sorted((r for r in rows if r["layout"] == lay and r["noise"] == fr and r["model"] == m and r["episode"] in valid_ids
                      and (winds is None or r["wind"] in winds)), key=lambda r: r["episode"])
        return np.array([r[metric] for r in sel])

    for lay, fr, metric in itertools.product(LAYOUTS, NOISE, ("jpop", "area", "src_local", "src_bnd")):
        for a, b in (("M2", "M1"), ("M2", "M1z"), ("M1", "M1z"), ("M2", "M0"), ("M1", "M0")):
            va, vb = vec(lay, fr, a, metric), vec(lay, fr, b, metric)
            ok = np.isfinite(va) & np.isfinite(vb)
            d = (va - vb)[ok]
            if not len(d):
                continue
            lo, hi = boot_ci(d, ALPHA_FWE)
            rel = float(d.mean() / vb[ok].mean()) if vb[ok].mean() else float("nan")
            leave = {}
            for w in WINDS:
                keep = [x for x in WINDS if x != w]
                v2a, v2b = vec(lay, fr, a, metric, keep), vec(lay, fr, b, metric, keep)
                ok2 = np.isfinite(v2a) & np.isfinite(v2b)
                leave[w] = float((v2a - v2b)[ok2].mean())
            out.append({"layout": lay, "noise": fr, "metric": metric, "contrast": f"{a}-{b}", "n": int(ok.sum()), "mean_diff": float(d.mean()),
                        "ci_lo": lo, "ci_hi": hi, "rel_change": rel, "frac_a_better": float((d < 0).mean()),
                        "mean_a": float(va[ok].mean()), "mean_b": float(vb[ok].mean()), "leave_one_wind_out": leave,
                        "effect_criterion_met": bool(metric == "jpop" and hi < 0 and rel <= -MIN_REL_GAIN and all(v < 0 for v in leave.values()))})
    for lay, fr, w in itertools.product(LAYOUTS, NOISE, WINDS):
        d = vec(lay, fr, "M2", "jpop", [w]) - vec(lay, fr, "M1", "jpop", [w])
        by_wind.append({"layout": lay, "noise": fr, "wind": w, "n": len(d), "M2_minus_M1_jpop": float(d.mean())})
    floor = float(np.mean([r["jpop"] for r in rows if r["model"] == "floor" and r["valid"]]))
    return {"summary": out, "by_wind": by_wind, "mean_projection_floor": floor,
            "n_invalid_episodes": len({r["episode"] for r in rows if not r["valid"]}),
            "primary_family_wise_alpha": ALPHA_FWE, "min_relative_gain": MIN_REL_GAIN,
            "n_nonconverged_fits": int(sum(1 for r in rows if not np.isfinite(r["jpop"])))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("tune", "test"), required=True)
    ap.add_argument("--out", default=os.environ.get("RESEARCH_RUN_DIR", "results/dyn02v2"))
    ap.add_argument("--freeze", default=None, help="path to frozen_hparams.json (test stage)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "tune":
        r = tune(a.workers)
        write_json(out / "frozen_hparams.json", {"hash": _hash(r["frozen"]), **r})
        print("frozen", _hash(r["frozen"]), "edge selections:", r["edge_selections"])
        return
    fz = json.loads(Path(a.freeze or out / "frozen_hparams.json").read_text())
    if fz["hash"] != _hash(fz["frozen"]):
        raise RuntimeError("freeze guard: hyperparameter file does not match its hash")
    hp = {k: {m: [tuple(c) for c in v] for m, v in d.items()} for k, d in fz["frozen"].items()}
    _prebuild(a.workers)
    rows = _pool_rows(episodes("test"), hp, ("M0", "M1z", "M1", "M2"), a.workers)
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = summarise(rows)
    write_json(out / "summary.json", {"frozen_hash": fz["hash"], "n_episodes": len(episodes("test")), **res,
                                      "note": "synthetic only; test seeds 7200-7295; wind known; model error not included"})
    for s in res["summary"]:
        if s["metric"] == "jpop" and s["contrast"] in ("M2-M1", "M2-M1z", "M1-M0"):
            print(s["layout"], s["noise"], s["contrast"], f"{s['mean_diff']:+.5f} [{s['ci_lo']:+.5f},{s['ci_hi']:+.5f}] rel {s['rel_change']:+.2f} met={s['effect_criterion_met']}")


if __name__ == "__main__":
    main()

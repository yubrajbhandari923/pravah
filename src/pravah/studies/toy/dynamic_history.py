"""DYN-03 (EXP-031): value of measurement history.  Current-time-only (CT) vs causal filter (CAUSAL) vs retrospective
smoother (RETRO) vs full-history independent bins (M1) vs oracle, on the DYN-02 v2 synthetic episodes (same data, sensors, truth).

    python -m pravah.studies.toy.dynamic_history --stage tune --out DIR [--workers 8]            # validation seeds 6000-6047
    python -m pravah.studies.toy.dynamic_history --stage test --freeze FILE --out DIR            # 96 episodes, seeds 8600-8695 (D-024)

Estimators (all share the frozen M0 static fit as prior mean, y, A, sigma and the REF scaling of dynamic_study):
  CT      bin b fitted from observations in (t_b, t_{b+1}] and bin-b columns only (ridge, exact), independently per bin.
  CAUSAL  for each b: M2 on observations up to t_{b+1} and columns of bins <= b; only the bin-b estimate is kept.
  RETRO   full-window M2 (= dynamic_study 'M2').
  M1      full-window independent bins (= dynamic_study 'M1').
  oracle  true 6-bin average.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path

import numpy as np

from pravah import dynamic as dyn
from pravah.io import write_json
from pravah.studies.toy import dynamic_study as ds
from pravah.studies.toy.dynamic_study import (ALPHA_2, ALPHA_FWE, ALPHA_M, BETAS, LAYOUTS, N_FINE, N_INV, NOISE, NP, REF, WINDS,
                                              _cells, _hash, _prebuild, boot_ci)

TEST_SEED0 = 8600  # v2 (D-024); 7600-7695 consumed by RUN-0092  # 96 test episodes, seeds 8600-8695 (disjoint from DYN-02's 7200-7295)
ESTIMATORS = ("CT", "CAUSAL", "RETRO", "M1", "oracle")
FIT_MODELS = ("M0", "CT", "CAUSAL", "RETRO", "M1")
CHANGE_FAMILIES = ("activation", "shutdown", "opposing", "pulse", "influx_end", "compensating")
STEADY = ("steady_a", "steady_b")
VARIABLE = tuple(w for w in WINDS if w not in STEADY)
CONTRASTS = (("CAUSAL", "CT"), ("RETRO", "CAUSAL"), ("M1", "CAUSAL"))  # the only primary contrasts (J_pop)
_TOL = 1e-6


def episodes_test() -> list[dict]:
    return [{"id": i, "wind": w, "family": f, "seed": TEST_SEED0 + i, "stage": "test"}
            for i, (w, f, _) in enumerate(itertools.product(WINDS, ds.FAMILIES, range(ds.N_SEEDS["test"])))]


def bin_index(obs_t: np.ndarray) -> np.ndarray:
    """Inverse bin (0..N_INV-1) of each observation time; bin b owns t in (t_b, t_{b+1}]."""
    e = dyn.bin_edges(ds.DUR, N_INV)
    return np.clip(np.searchsorted(e, np.asarray(obs_t) - _TOL, side="left") - 1, 0, N_INV - 1)


def estimate(name: str, A, y, sig, obs_t, a: float, b: float, prior: np.ndarray, a0: float = 1.0) -> np.ndarray | None:
    """(NP, N_INV) history of estimator ``name`` in {CT, CAUSAL, RETRO, M1}, or None if a fit did not converge."""
    bi = bin_index(obs_t)
    A3 = A.reshape(len(y), NP, N_INV)
    try:
        if name == "RETRO":
            return dyn.fit_history(A, y, sig, NP, N_INV, "M2", alpha=a, beta=b, theta0=prior, ref=REF).theta
        if name == "M1":
            return dyn.fit_history(A, y, sig, NP, N_INV, "M1", alpha=a, theta0=prior, ref=REF).theta
        th = np.zeros((NP, N_INV))
        for k in range(N_INV):
            # CAUSAL PRIOR (review F-053/R1): the static prior for bin k comes from data up to the end of bin k only
            mp = bi <= k
            pk = dyn.fit_history(A3[mp][:, :, :k + 1].reshape(mp.sum(), -1), y[mp], sig[mp], NP, k + 1, "M0", alpha=a0, ref=REF).theta
            if name == "CT":
                m = bi == k
                th[:, k] = dyn.fit_history(A3[m][:, :, k], y[m], sig[m], NP, 1, "M1", alpha=a, theta0=pk[:, k:k + 1], ref=REF).theta[:, 0]
            elif name == "CAUSAL":
                m = bi <= k
                Ak = A3[m][:, :, :k + 1].reshape(m.sum(), -1)
                th[:, k] = dyn.fit_history(Ak, y[m], sig[m], NP, k + 1, "M2", alpha=a, beta=b, theta0=pk, ref=REF).theta[:, k]
            else:
                raise ValueError(name)
        return th
    except dyn.NotConverged:
        return None


_AFFECTED = {"activation": (0,), "opposing": (0,), "shutdown": (1,), "pulse": (3, 4), "influx_end": (3, 4), "compensating": (3, 4)}


def change_spec(family: str, hist: np.ndarray) -> tuple[int, int, int, float, float] | None:
    """First step of the affected parameter of a fine truth history: (param, k, k2, pre, post) with k the fine bin of the step and
    k2 the fine bin of its next step (N_FINE if none).  Boundary families: the basis with a step; compensating's source drop is secondary."""
    best = None
    for p in _AFFECTED.get(family, ()):
        r = hist[p]
        ch = [i for i in range(1, N_FINE) if abs(r[i] - r[i - 1]) > 1e-9 * max(abs(r).max(), 1e-300)]
        if ch and (best is None or abs(r[ch[0]] - r[ch[0] - 1]) / REF[p] > best[0]):
            best = (abs(r[ch[0]] - r[ch[0] - 1]) / REF[p], p, ch[0], ch[1] if len(ch) > 1 else N_FINE, float(r[ch[0] - 1]), float(r[ch[0]]))
    return None if best is None else best[1:]


def detect(series: np.ndarray, pre: float, post: float, c: int, e: int) -> tuple[float, bool]:
    """Detection delay (bins after the true change bin c) = first b in [c, e) from which the estimate stays across the pre/post
    midpoint (towards post) through bin e-1; (nan, True) if never."""
    crossed = np.sign(post - pre) * (np.asarray(series, dtype=float) - (pre + post) / 2) > 0
    if c >= 1 and crossed[c - 1]:  # already on the post side before the change: a constant estimator, not a detection (review F-053/R1)
        return float("nan"), True
    for b in range(c, e):
        if crossed[b:e].all():
            return float(b - c), False
    return float("nan"), True


def run_episode(ep: dict, hp: dict, models: tuple[str, ...] = FIT_MODELS, layouts=None) -> list[dict]:
    """One row per (layout, noise, estimator, candidate) plus an oracle row per (layout, noise) and a floor row.  hp[key][model] lists
    (alpha, beta) candidates; hp[key]['M0'] must hold exactly one (frozen) entry (the prior mean of CT/CAUSAL/RETRO/M1)."""
    layouts = LAYOUTS if layouts is None else layouts
    R = ds.build_responses(ep["wind"])
    counts = ds.population()
    hist, valid = ds.make_history(ep["family"], np.random.default_rng([ep["seed"], 0]), ds.live_mask(R))
    tf = ds.truth_field(ep["wind"], hist, np.random.default_rng([ep["seed"], 1]))
    t = np.asarray(R.times)
    s0 = ds.reference_scale(R)
    binned = hist.reshape(NP, N_INV, N_FINE // N_INV).mean(axis=2)
    spec = change_spec(ep["family"], hist) if valid else None
    base = {"episode": ep["id"], "wind": ep["wind"], "family": ep["family"], "valid": valid, "stage": ep["stage"]}
    nan = float("nan")
    blank = {"alpha": nan, "beta": nan, "area": nan, "det_applicable": False, "det_delay": nan, "det_miss": nan}
    rows = [{**base, **blank, "layout": "-", "noise": 0.0, "model": "floor", "jpop": ds.projection_floor(R, tf, counts)}]
    for layout, (ni, frac) in itertools.product(layouts, enumerate(NOISE)):
        iy, ix = _cells(layouts[layout])
        sd = frac * s0
        y = tf[1:, iy, ix] + np.random.default_rng([ep["seed"], 2, ni]).standard_normal((len(t) - 1, len(iy))) * sd
        A = np.moveaxis(R.g[:, 1:, iy, ix], 0, -1).reshape(-1, R.g.shape[0])
        y = y.reshape(-1)
        sig = np.full(y.size, sd)
        obs_t = np.repeat(t[1:], len(iy))
        sens = dyn.sensitivity(A * np.repeat(REF, N_INV), sig, NP, N_INV)
        ident = sens > 1e-6 * sens.max()
        cfg = hp[f"{layout}|{frac}"]
        a0, b0 = cfg["M0"][0]
        prior = dyn.fit_history(A, y, sig, NP, N_INV, "M0", alpha=a0, ref=REF).theta
        jp, ar = ds.jpop(R, binned, tf, counts)
        rows.append({**base, **blank, "layout": layout, "noise": frac, "model": "oracle", "jpop": jp, "area": ar})
        for model in models:
            for a, b in cfg[model]:
                if model == "M0":
                    th = prior if (a, b) == (a0, b0) else dyn.fit_history(A, y, sig, NP, N_INV, "M0", alpha=a, ref=REF).theta
                else:
                    th = estimate(model, A, y, sig, obs_t, a, b, prior, a0)
                row = {**base, **blank, "layout": layout, "noise": frac, "model": model, "alpha": a, "beta": b}
                if th is None:
                    rows.append({**row, "jpop": np.inf})
                    continue
                row["jpop"], row["area"] = ds.jpop(R, th, tf, counts)
                if spec is not None and model != "M0":
                    p, k, k2, pre, post = spec
                    c = k // 2
                    e = min(max(k2 // 2, c + 1), N_INV)
                    if c >= 1 and ident[p, c - 1:e].all():
                        d, miss = detect(th[p], pre, post, c, e)
                        row.update(det_applicable=True, det_delay=d, det_miss=float(miss))
                rows.append(row)
    return rows


def _pool(eps, hp, models, workers, layouts=None):
    """Same pool as dynamic_study._pool_rows but runs this module's run_episode."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    if workers <= 1:
        return [r for ep in eps for r in run_episode(ep, hp, models, layouts)]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(run_episode, ep, hp, models, layouts) for ep in eps]
        return [r for f in futs for r in f.result()]


ALPHA_C = (0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4)  # D-024: reduced M2 grid for CAUSAL/RETRO (review: alpha <= 1 is a flat plateau)
BETAS_C = (0.0, 10.0, 100.0, 1e3, 1e4, 1e5)


def tune(workers: int, layouts=None) -> dict:
    """Validation only (dynamic_study.episodes('tune')).  M0 first; then CT / M1 (ALPHA_M) and CAUSAL / RETRO (ALPHA_2 x BETAS) with
    M0@alpha* as prior.  Selection = lowest mean J_pop per (layout, noise); non-converged = +inf; edge selections flagged."""
    _prebuild(workers)
    eps = ds.episodes("tune")
    layouts = LAYOUTS if layouts is None else layouts
    combos = [(lay, fr) for lay in layouts for fr in NOISE]
    s1 = ds._score_table(_pool(eps, {f"{lay}|{fr}": {"M0": [(a, 0.0) for a in ALPHA_M]} for lay, fr in combos}, ("M0",), workers, layouts))
    m0 = {(lay, fr): min(s1[(lay, fr, "M0")], key=lambda c, d=s1[(lay, fr, "M0")]: d[c]) for lay, fr in combos}
    g2 = list(itertools.product(ALPHA_C, BETAS_C))
    grid = {"CT": [(a, 0.0) for a in ALPHA_M], "M1": [(a, 0.0) for a in ALPHA_M], "CAUSAL": g2, "RETRO": g2}
    s2 = ds._score_table(_pool(eps, {f"{lay}|{fr}": {"M0": [m0[(lay, fr)]], **grid} for lay, fr in combos},
                               ("CT", "M1", "CAUSAL", "RETRO"), workers, layouts))
    frozen: dict = {f"{lay}|{fr}": {"M0": [list(m0[(lay, fr)])]} for lay, fr in combos}
    edge, scores = [], {}
    for (lay, fr, m), d in s2.items():
        k = f"{lay}|{fr}"
        best = min(d, key=lambda c, d=d: d[c])
        if m == "oracle":
            continue
        frozen[k][m] = [list(best)]
        scores.setdefault(k, {})[m] = {f"{a:g}|{b:g}": v for (a, b), v in d.items()}
        if m == "oracle":
            continue
        al = ALPHA_C if m in ("CAUSAL", "RETRO") else ALPHA_M
        if best[0] in (al[0], al[-1]) or (m in ("CAUSAL", "RETRO") and best[1] in (BETAS_C[1], BETAS_C[-1])):
            edge.append(f"{k}:{m}:{best}")
    for (lay, fr), c in m0.items():
        if c[0] in (ALPHA_M[0], ALPHA_M[-1]):
            edge.append(f"{lay}|{fr}:M0:{c}")
    return {"frozen": frozen, "edge_selections": edge, "validation_jpop": scores}


def load_frozen(path: str | Path) -> dict:
    """Freeze guard: the file must exist and its hash must match its content."""
    fz = json.loads(Path(path).read_text())
    if fz.get("hash") != _hash(fz["frozen"]):
        raise RuntimeError("freeze guard: hyperparameter file does not match its hash")
    return {k: {m: [tuple(c) for c in v] for m, v in d.items()} for k, d in fz["frozen"].items()}


def test(freeze: str | Path, out: str | Path, workers: int) -> dict:
    """Test stage (seeds 8600+).  Refuses to run without a valid frozen hyperparameter file."""
    if not freeze:
        raise RuntimeError("freeze guard: test stage requires --freeze FILE")
    hp = load_frozen(freeze)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _prebuild(workers)
    eps = episodes_test()
    rows = _pool(eps, hp, FIT_MODELS, workers)
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = summarise(rows)
    write_json(out / "summary.json", {"frozen_hash": json.loads(Path(freeze).read_text())["hash"], "n_episodes": len(eps), **res,
                                      "note": "synthetic only; test seeds 8600-8695 (v2, D-024); wind known; model error not included"})
    return res


def _vec(rows, lay, fr, m, winds=None, metric="jpop"):
    valid = {r["episode"] for r in rows if r["valid"]}
    sel = sorted((r for r in rows if r["layout"] == lay and r["noise"] == fr and r["model"] == m and r["episode"] in valid
                  and (winds is None or r["wind"] in winds)), key=lambda r: r["episode"])
    return np.array([r[metric] for r in sel], dtype=float)


def _contrast(rows, lay, fr, a, b, winds=None, level=ALPHA_FWE) -> dict | None:
    va, vb = _vec(rows, lay, fr, a, winds), _vec(rows, lay, fr, b, winds)
    ok = np.isfinite(va) & np.isfinite(vb)
    d = (va - vb)[ok]
    if not len(d):
        return None
    lo, hi = boot_ci(d, level)
    return {"n": int(ok.sum()), "mean_diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
            "rel_change": float(d.mean() / vb[ok].mean()), "frac_a_better": float((d < 0).mean())}


def summarise(rows: list[dict]) -> dict:
    prim, by_wind, groups, det = [], [], [], []
    for lay, fr in itertools.product(LAYOUTS, NOISE):
        for a, b in CONTRASTS:
            s = _contrast(rows, lay, fr, a, b)
            if s is None:
                continue
            s["leave_one_wind_out"] = {w: (_contrast(rows, lay, fr, a, b, [x for x in WINDS if x != w]) or {}).get("mean_diff") for w in WINDS}
            prim.append({"layout": lay, "noise": fr, "contrast": f"{a}-{b}", **s})
            for w in WINDS:
                g = _contrast(rows, lay, fr, a, b, [w])
                if g:
                    by_wind.append({"layout": lay, "noise": fr, "contrast": f"{a}-{b}", "wind": w, "n": g["n"], "mean_diff": g["mean_diff"]})
            for name, ws in (("steady", STEADY), ("variable", VARIABLE)):
                g = _contrast(rows, lay, fr, a, b, ws)
                if g:  # registered prediction: history helps most (most negative) in 'variable'; 'steady' is the negative control
                    groups.append({"layout": lay, "noise": fr, "contrast": f"{a}-{b}", "group": name, **g})
    for lay, fr, m in itertools.product(LAYOUTS, NOISE, ("CT", "CAUSAL", "RETRO", "M1")):
        for name, ws in (("all", None), ("steady", STEADY), ("variable", VARIABLE)):
            sel = [r for r in rows if r["layout"] == lay and r["noise"] == fr and r["model"] == m and r["det_applicable"]
                   and (ws is None or r["wind"] in ws)]
            if sel:
                dl = np.array([r["det_delay"] for r in sel])
                det.append({"layout": lay, "noise": fr, "model": m, "group": name, "n": len(sel), "miss_rate": float(np.mean([r["det_miss"] for r in sel])),
                            "mean_delay_detected": float(np.nanmean(dl)) if np.isfinite(dl).any() else float("nan")})
    viol = [g for g in groups if g["group"] == "steady" and g["contrast"] in ("CAUSAL-CT", "RETRO-CAUSAL") and g["ci_hi"] < 0]
    return {"primary": prim, "by_wind": by_wind, "wind_groups": groups, "change_detection": det,
            "negative_control_gain_ci_excluding_zero": viol, "primary_family_wise_alpha": ALPHA_FWE, "min_relative_gain": ds.MIN_REL_GAIN,
            "n_nonconverged_fits": int(sum(1 for r in rows if not np.isfinite(r["jpop"])))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("tune", "test"), required=True)
    ap.add_argument("--out", default=os.environ.get("RESEARCH_RUN_DIR", "results/dyn03"))
    ap.add_argument("--freeze", default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "tune":
        r = tune(a.workers)
        write_json(out / "frozen_hparams.json", {"hash": _hash(r["frozen"]), **r})
        print("FROZEN_JSON " + json.dumps({"hash": _hash(r["frozen"]), "frozen": r["frozen"], "edge_selections": r["edge_selections"]}), flush=True)
        print("frozen", _hash(r["frozen"]), "edge selections:", r["edge_selections"])
        return
    res = test(a.freeze or out / "frozen_hparams.json", out, a.workers)
    for s in res["primary"]:
        print(s["layout"], s["noise"], s["contrast"], f"{s['mean_diff']:+.5f} [{s['ci_lo']:+.5f},{s['ci_hi']:+.5f}]")


if __name__ == "__main__":
    main()

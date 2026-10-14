"""DYN-06 threshold study (EXP-034, protocol D-019): attribution confusion as a function of the pair's distinguishability d.

The change of each scenario is scaled, x(m) = x_pre + m (x_a - x_pre) with x_pre the pre-change state carried forward, so that the pair's
d_after_rescale equals a target in TARGETS; confusion is the rate at which the estimator attributes the change to the wrong group.

    python -m pravah.studies.toy.dynamic_separability_threshold --freeze results/dyn02v2/frozen_hparams.json --out DIR [--workers N]
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pravah.io import write_json
from pravah.studies.toy import dynamic_separability as sp
from pravah.studies.toy import dynamic_study as ds

TARGETS = (2.0, 4.0, 8.0, 16.0)
REPS = 100
SCEN = ("S1", "S2", "S4", "S5", "S6")


def x_pre(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    out[:, sp.SPLIT:] = x[:, [sp.SPLIT - 1]]
    return out


def scaled(sc: dict, target: float) -> tuple[np.ndarray, float, dict]:
    """(x(m), m, analytic).  Distinguishability d is defined on the CHANGE signatures (x - x_pre, each about its own pre-change state),
    which scale exactly linearly with m, so m = target / d(1) is exact (the RUN-0082 version mixed pre-states and missed the target
    by up to 10x for S2/S4/S5 - review F-043/R1).  The scenario simulated for attribution is a's own pre-state + m * (a's change)."""
    pre_a, pre_b = x_pre(sc["xa"]), x_pre(sc["xb"])
    ca, cb = sc["xa"] - pre_a, sc["xb"] - pre_b
    d1 = sp.analytic(sc["op_a"], sc["op_b"], ca, cb)["d_after_rescale"]
    if d1 < 1e-9:
        raise ZeroDivisionError("change signatures are perfectly confounded (d == 0 at every amplitude)")
    m = target / d1
    an = sp.analytic(sc["op_a"], sc["op_b"], m * ca, m * cb)
    if abs(an["d_after_rescale"] - target) > 0.02 * target:
        raise AssertionError(f"achieved d {an['d_after_rescale']:.3f} != target {target}")
    return np.maximum(pre_a + m * ca, 0.0), m, an


def attribute_calibrated(op, x, groups, alpha, reps=REPS):
    """Null-calibrated attribution (D-020/v2).  Returns dict(truth, detect, mis, ambiguous, thr)."""
    names = list(groups)
    pre = x_pre(x)

    def contribs(hist, seed0):
        out = []
        y0 = op.Ax @ hist.ravel()
        for r in range(reps):
            y = y0 + np.random.default_rng(seed0 + r).standard_normal(len(y0)) * op.sigma
            try:
                th = sp.dyn.fit_history(op.A, y, np.full(len(y), op.sigma), sp.NP, sp.N_INV, "M2", alpha=alpha, beta=0.0, ref=sp.REF).theta
            except sp.dyn.NotConverged:
                continue
            out.append(np.abs(sp._contrib(op, th / sp.REF[:, None], groups)))
        return np.array(out)

    null = contribs(pre, 9500)
    thr = np.percentile(null, 95, axis=0) + 1e-30
    sig = contribs(x, 9600)
    ct = np.abs(sp._contrib(op, x, groups))
    truth = names[int(ct.argmax())] if ct.max() > 0 else "none"
    ratio = sig / thr
    det = (ratio > 1).any(axis=1)
    top = np.sort(ratio, axis=1)[:, ::-1]
    amb = det & (top[:, 0] - top[:, 1] < 0.1 * top[:, 0]) if ratio.shape[1] > 1 else np.zeros(len(ratio), bool)
    named = ratio.argmax(axis=1)
    wrong = det & ~amb & (np.array(names)[named] != truth)
    nd = int(det.sum())
    # D-025 (review F-044/R1): misattribution is scored only where detection is clearly above the false-alarm level
    # (one-sided binomial test, p < 0.01 against 0.10 = two groups at 5% each); elsewhere the cell is 'undetected'.
    from scipy.stats import binomtest
    scored = binomtest(nd, len(det), 0.10, alternative="greater").pvalue < 0.01 if len(det) else False
    return {"truth": truth, "n_rep": len(sig), "detect": float(det.mean()), "n_det": nd, "n_wrong": int(wrong.sum()), "n_amb": int(amb.sum()),
            "scored": bool(scored), "mis": float(wrong.sum() / nd) if scored else float("nan"), "ambiguous": float(amb.sum() / nd) if scored else float("nan"),
            "false_alarm_rate_null": 0.05}


def wind_task(args) -> list[dict]:
    wind, frozen = args
    rows = []
    R = ds.build_responses(wind)
    lays = {"L4": ds.LAYOUTS["L4"], "L9": ds.LAYOUTS["L9"], "L4_ref": sp.l4_ref_cells(wind)}
    for fr in ds.NOISE:
        for lay, cells in lays.items():
            alpha = frozen[f"{'L4' if lay.startswith('L4') else 'L9'}|{fr}"]["M1z"][0][0]
            op = sp.make_op(wind, cells, fr, R)
            for s in SCEN:
                sc = sp.build(s, op)
                if not sc["valid"]:
                    rows.append({"wind": wind, "layout": lay, "noise": fr, "scenario": s, "valid": False})
                    continue
                if not sc["match_ok"]:
                    rows.append({"wind": wind, "layout": lay, "noise": fr, "scenario": s, "valid": False, "reason": "match_failed"})
                    continue
                cb = sc["xb"] - x_pre(sc["xb"])
                lookalike_signal = float(np.linalg.norm(sc["op_b"].Ax @ cb.ravel()) / sc["op_b"].sigma)
                for t in TARGETS:
                    try:
                        xa, m, an = scaled(sc, t)
                    except ZeroDivisionError:
                        rows.append({"wind": wind, "layout": lay, "noise": fr, "scenario": s, "valid": False, "reason": "confounded", "target_d": t})
                        continue
                    att = attribute_calibrated(op, xa, sc["groups"], alpha)
                    rows.append({"wind": wind, "layout": lay, "noise": fr, "scenario": s, "valid": True, "target_d": t, "m": m,
                                 "lookalike_signal_sd": lookalike_signal, "detectability_only": bool(lookalike_signal < 1.0 or an.get("c_star", 0) > 1e3),
                                 "d_after_rescale": an["d_after_rescale"], "cos_max": an["cos_max"], "alpha": alpha, **att})
    return rows


def controls(frozen: dict) -> dict:
    """Registered controls under the CALIBRATED rule: (i) perfectly confounded operator: change-signature distance 0 and the named
    group is arbitrary (misattribution + ambiguous high); (ii) no variation: the detection rate equals the false-alarm level (~5% per group)."""
    out = {}
    xa, xb, groups = sp.confounded_pair()
    op = sp.fixture_confounded()
    an = sp.analytic(op, op, xa, xb)
    res = {}
    for amp in (1.0, 4.0, 16.0):
        res[f"amp{amp:g}"] = attribute_calibrated(op, xa * amp / 3.0 + 0.0, groups, 1e-2)
    out["confounded"] = {"d_after_rescale": an["d_after_rescale"], **res}
    real = sp.make_op("steady_a", ds.LAYOUTS["L4"], 0.10)
    pre = sp._base()
    out["no_variation_steady_a_L4"] = attribute_calibrated(real, pre, sp.LOCAL_BOUNDARY, frozen["L4|0.1"]["M1z"][0][0])
    return out


def summarise(rows: list[dict]) -> list[dict]:
    """Per (scenario, layout, noise, d): detection across winds, POOLED misattribution over the winds where detection is clearly above
    the false-alarm level (sum wrong / sum detected), and the per-wind detail stays in rows.  Cells flagged detectability_only (the
    look-alike has no observable change signature) are reported separately from look-alike distinguishability cells."""
    out = []
    ok = [r for r in rows if r["valid"]]
    key = lambda r: (r["scenario"], r["layout"], r["noise"], r["target_d"])  # noqa: E731
    for (s, lay, fr, t), grp in itertools.groupby(sorted(ok, key=key), key=key):
        g = list(grp)
        sc = [r for r in g if r["scored"]]
        nd = sum(r["n_det"] for r in sc)
        out.append({"scenario": s, "layout": lay, "noise": fr, "target_d": t, "n_winds": len(g), "n_winds_scored": len(sc),
                    "mean_detect": float(np.mean([r["detect"] for r in g])), "pooled_mis": float(sum(r["n_wrong"] for r in sc) / nd) if nd else float("nan"),
                    "pooled_ambiguous": float(sum(r["n_amb"] for r in sc) / nd) if nd else float("nan"),
                    "n_detectability_only": int(sum(r["detectability_only"] for r in g)),
                    "mean_cos_max": float(np.nanmean([r["cos_max"] for r in g])) if not all(np.isnan(r["cos_max"]) for r in g) else float("nan")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--out", default=os.environ.get("RESEARCH_RUN_DIR", "results/dyn06"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    a = ap.parse_args()
    fz = json.loads(Path(a.freeze).read_text())
    if fz["hash"] != ds._hash(fz["frozen"]):
        raise RuntimeError("freeze guard: hyperparameter file does not match its hash")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds._prebuild(a.workers)
    tasks = [(w, fz["frozen"]) for w in ds.WINDS]
    with ProcessPoolExecutor(min(a.workers, len(tasks)), mp_context=mp.get_context("spawn")) as ex:
        rows = [r for rs in ex.map(wind_task, tasks) for r in rs]
    summ = summarise(rows)
    write_json(out / "summary.json", {"frozen_hash": fz["hash"], "reps": REPS, "targets": list(TARGETS), "summary": summ,
                                      "controls": controls(fz["frozen"]), "rows": rows})
    for s in summ:
        if s["noise"] == 0.1:
            print(f"{s['scenario']} {s['layout']:6s} d={s['target_d']:<4g} detect {s['mean_detect']:.2f} pooled-mis {s['pooled_mis']:.2f} winds scored {s['n_winds_scored']}/{s['n_winds']} detectability-only {s['n_detectability_only']}")


if __name__ == "__main__":
    main()

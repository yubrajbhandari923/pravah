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
    """(x(m), m, analytic at m) with m chosen so d_after_rescale == target (the distance is linear in the shared change)."""
    pre = x_pre(sc["xa"])
    xa1, xb1 = sc["xa"], sc["xb"]
    d1 = sp.analytic(sc["op_a"], sc["op_b"], xa1, xb1)["d_after_rescale"]
    m = target / d1
    xa = np.maximum(pre + m * (xa1 - pre), 0.0)
    xb = np.maximum(pre + m * (xb1 - pre), 0.0)
    return xa, m, sp.analytic(sc["op_a"], sc["op_b"], xa, xb)


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

    null = contribs(pre, 9100)
    thr = np.percentile(null, 95, axis=0) + 1e-30
    sig = contribs(x, 9200)
    ct = np.abs(sp._contrib(op, x, groups))
    truth = names[int(ct.argmax())] if ct.max() > 0 else "none"
    ratio = sig / thr
    det = (ratio > 1).any(axis=1)
    top = np.sort(ratio, axis=1)[:, ::-1]
    amb = det & (top[:, 0] - top[:, 1] < 0.1 * top[:, 0]) if ratio.shape[1] > 1 else np.zeros(len(ratio), bool)
    named = ratio.argmax(axis=1)
    wrong = det & ~amb & (np.array(names)[named] != truth)
    nd = max(int(det.sum()), 1)
    return {"truth": truth, "n_rep": len(sig), "detect": float(det.mean()), "mis": float(wrong.sum() / nd), "ambiguous": float(amb.sum() / nd),
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
                for t in TARGETS:
                    xa, m, an = scaled(sc, t)
                    att = attribute_calibrated(op, xa, sc["groups"], alpha)
                    rows.append({"wind": wind, "layout": lay, "noise": fr, "scenario": s, "valid": True, "target_d": t, "m": m,
                                 "d_after_rescale": an["d_after_rescale"], "cos_max": an["cos_max"], "alpha": alpha, **att})
    return rows


def controls(frozen: dict) -> dict:
    out = {}
    xa, xb, groups = sp.confounded_pair()
    op = sp.fixture_confounded()
    an = sp.analytic(op, op, xa, xb)
    out["confounded"] = {"d_after_rescale": an["d_after_rescale"],
                         **{f"d{t:g}": sp.attribute(op, xa * (t / max(an['d_raw'], 1e-9)), groups, (1e-2, 0.0), REPS)["conf_rate"] for t in TARGETS}}
    real = sp.make_op("steady_a", ds.LAYOUTS["L4"], 0.10)
    out["no_variation_steady_a_L4"] = sp.fixture_no_variation(real, (frozen["L4|0.1"]["M1z"][0][0], 0.0), REPS)  # legacy rule, kept for continuity
    return out


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    ok = [r for r in rows if r["valid"]]
    for (s, lay, fr, t), grp in itertools.groupby(sorted(ok, key=lambda r: (r["scenario"], r["layout"], r["noise"], r["target_d"])),
                                                  key=lambda r: (r["scenario"], r["layout"], r["noise"], r["target_d"])):
        g = list(grp)
        out.append({"scenario": s, "layout": lay, "noise": fr, "target_d": t, "n_winds": len(g), "mean_mis": float(np.nanmean([r["mis"] for r in g])),
                    "max_mis": float(np.nanmax([r["mis"] for r in g])), "mean_detect": float(np.nanmean([r["detect"] for r in g])),
                    "mean_cos_max": float(np.mean([r["cos_max"] for r in g]))})
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
            print(f"{s['scenario']} {s['layout']:6s} d={s['target_d']:<4g} detect {s['mean_detect']:.2f} mis {s['mean_mis']:.2f} (max {s['max_mis']:.2f}) cos {s['mean_cos_max']:.2f}")


if __name__ == "__main__":
    main()

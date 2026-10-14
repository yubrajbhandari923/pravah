"""DYN-07 (EXP-035): the DYN-02 estimators under model error.  The truth is simulated with a perturbed wind / diffusion / an
unmodelled source while the inverse keeps the NOMINAL model and the FROZEN nominal hyperparameters of DYN-02 v2 (nothing is
re-tuned per condition).  The same 96 episodes (seeds 8200-8295) are used in every condition, so conditions are paired.

    python -m pravah.studies.toy.dynamic_robustness --freeze results/dyn02v2/frozen_hparams.json --out DIR [--workers N]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path

import numpy as np

from pravah.io import write_json
from pravah.studies.toy import dynamic_study as ds

SEED0 = 8200
CONDITIONS: dict[str, dict] = {
    "nominal": {}, "rot5": {"rot_deg": 5.0}, "rot10": {"rot_deg": 10.0}, "rot20": {"rot_deg": 20.0},
    "speed0.8": {"speed_scale": 0.8}, "speed1.2": {"speed_scale": 1.2}, "K2": {"K_scale": 2.0},
    "extra0.1": {"extra_frac": 0.1}, "extra0.3": {"extra_frac": 0.3},
    "combined": {"rot_deg": 10.0, "speed_scale": 1.2, "extra_frac": 0.1}}
MODELS = ("M0", "M1", "M2")


def episodes(cond: str) -> list[dict]:
    return [{**e, "seed": SEED0 + e["id"], "stage": "robust", "perturb": CONDITIONS[cond] or None}
            for e in ds.episodes("test")]


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for lay, fr, cond in itertools.product(ds.LAYOUTS, ds.NOISE, CONDITIONS):
        def v(m, c=cond):
            sel = sorted((r for r in rows if r["layout"] == lay and r["noise"] == fr and r["model"] == m and r["condition"] == c),
                         key=lambda r: r["episode"])
            return np.array([r["jpop"] for r in sel])
        m0, m1, m2, n1, n2 = v("M0"), v("M1"), v("M2"), v("M1", "nominal"), v("M2", "nominal")
        fin = np.isfinite(m1) & np.isfinite(m2) & np.isfinite(n1) & np.isfinite(n2)
        d = (m2 - m1)[fin]
        lo, hi = ds.boot_ci(d, ds.ALPHA_FWE)
        out.append({"layout": lay, "noise": fr, "condition": cond, "n": int(fin.sum()), "M0": float(m0.mean()), "M1": float(m1[fin].mean()),
                    "M2": float(m2[fin].mean()), "M2_minus_M1": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                    "rel_M2_vs_M1": float(d.mean() / m1[fin].mean()), "M1_degradation_vs_nominal": float(m1[fin].mean() / n1[fin].mean()),
                    "M2_degradation_vs_nominal": float(m2[fin].mean() / n2[fin].mean()),
                    "M2_better_than_M0": float((m2[fin] < m0[fin]).mean()), "n_nonconverged_M2": int((~np.isfinite(m2)).sum())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--out", default=os.environ.get("RESEARCH_RUN_DIR", "results/dyn07"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    a = ap.parse_args()
    fz = json.loads(Path(a.freeze).read_text())
    if fz["hash"] != ds._hash(fz["frozen"]):
        raise RuntimeError("freeze guard: hyperparameter file does not match its hash")
    hp = {k: {m: [tuple(c) for c in v] for m, v in d.items()} for k, d in fz["frozen"].items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds._prebuild(a.workers)
    rows = []
    for cond in a.conditions:
        for r in ds._pool_rows(episodes(cond), hp, MODELS, a.workers):
            rows.append({**r, "condition": cond})
        print("done", cond, flush=True)
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = summarise(rows)
    write_json(out / "summary.json", {"frozen_hash": fz["hash"], "seed0": SEED0, "conditions": CONDITIONS, "summary": res,
                                      "note": "synthetic; model error only in the truth; nominal hyperparameters, never re-tuned"})
    for s in res:
        if s["layout"] == "L4" and s["noise"] == 0.1:
            print(f"{s['condition']:9s} M0 {s['M0']:.5f} M1 {s['M1']:.5f} M2 {s['M2']:.5f}  M2-M1 {s['M2_minus_M1']:+.5f} rel {s['rel_M2_vs_M1']:+.2f}")


if __name__ == "__main__":
    main()

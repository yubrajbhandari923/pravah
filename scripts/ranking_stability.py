"""Do the method rankings survive the robustness conditions?  Offline recomputation with a noise baseline.

    python scripts/ranking_stability.py [--robust results/geodesign-robust] [--out results/geodesign-robust]
                                        [--setting interior_only] [--noise-fraction 0.05] [--regime standard]
                                        [--alpha-label selected]

Reads ``<robust>/test_raw.csv`` only (no solver).  Writes ``<out>/ranking_stability.csv`` (one row per contrast) and
``<out>/ranking_stability.md`` (markdown tables for docs/phase2-validation.md).  Existing result files are never touched.

Subset, stated explicitly (the defaults are the ones the documentation quotes): setting ``interior_only``, regime
``standard``, noise fraction 0.05, alpha label ``selected``; contrasts = mean over cases x noise reps of the difference of the
layout-averaged metric (random: mean over its layouts) for greedy_total vs spread, greedy_total vs random_mean,
greedy_attribution vs greedy_total and greedy_attribution vs spread, on rmse and attr_err, at every budget; a = first method,
negative = a better.

Rules:
  * contrasts whose MATCHED difference is exactly zero (identical layouts, e.g. budget 1) are excluded: they have no sign;
  * a contrast "survives" a condition when its mean difference has the same sign as the matched one (rows of the condition
    are paired with the matched rows by case and rep);
  * NOISE BASELINE: the matched contrast recomputed on two disjoint, equally large sets of noise repetitions (every such pair
    of halves, or a seeded sample of 200 pairs when there are more); the share of contrasts whose sign agrees between the halves is what noise alone does to the sign.  With R
    reps the halves hold floor(R/2) reps each.  Halves use fewer data than the full contrast, so the baseline is a
    pessimistic (lower) reference for the full-data agreement; the cases are the same in both halves, so only the NOISE
    component of the sign instability is measured, not case-to-case variation.  Agreement near 50 % means the sign is noise.
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (("greedy_total", "spread"), ("greedy_total", "random"), ("greedy_attribution", "greedy_total"),
         ("greedy_attribution", "spread"))
METRICS = ("rmse", "attr_err")
KEY = ["condition", "budget", "case", "rep"]
MAX_SPLITS = 200  # all disjoint half pairs are enumerated when there are at most this many, else a seeded sample


def layout_means(raw: pd.DataFrame) -> pd.DataFrame:
    """Per (condition, method, budget, case, rep): the metric averaged over the layouts of the method."""
    return raw.groupby(["condition", "method", "budget", "case", "rep"], as_index=False)[list(METRICS)].mean()


def contrasts(lm: pd.DataFrame, reps: list[int] | None = None) -> pd.DataFrame:
    """Mean difference a - b per (condition, comparison, metric, budget), over the given noise reps (default all)."""
    d = lm if reps is None else lm[lm["rep"].isin(reps)]
    parts = []
    for a, b in PAIRS:
        wa, wb = d[d["method"] == a].drop(columns="method"), d[d["method"] == b].drop(columns="method")
        m = wa.merge(wb, on=KEY, suffixes=("_a", "_b"))
        for met in METRICS:
            diff = m.assign(diff=m[f"{met}_a"] - m[f"{met}_b"]).groupby(["condition", "budget"], as_index=False)["diff"].mean()
            diff.insert(1, "comparison", f"{a} vs {'random_mean' if b == 'random' else b}")
            diff.insert(2, "metric", met)
            parts.append(diff)
    return pd.concat(parts, ignore_index=True)


def noise_baseline(lm: pd.DataFrame) -> pd.DataFrame:
    """Per matched contrast: the share of disjoint equal rep-halves pairs on which the sign agrees, and the number of pairs."""
    reps = sorted(lm["rep"].unique())
    h = len(reps) // 2
    if h < 1:
        raise ValueError("need at least 2 noise reps for the noise baseline")
    matched = lm[lm["condition"] == "matched"]
    if math.comb(len(reps), h) * math.comb(len(reps) - h, h) // 2 <= MAX_SPLITS:
        halves = [(set(a), set(b)) for a in itertools.combinations(reps, h) for b in itertools.combinations(reps, h)
                  if not set(a) & set(b) and min(a) < min(b)]
    else:  # too many splits to enumerate: a seeded random sample of disjoint half pairs
        rng = np.random.default_rng(0)
        halves = [(set(p[:h]), set(p[h:2 * h])) for p in (rng.permutation(reps).tolist() for _ in range(MAX_SPLITS))]
    keys = ["comparison", "metric", "budget"]
    agree = []
    for a, b in halves:
        ca = contrasts(matched, sorted(a)).set_index(keys)["diff"]
        cb = contrasts(matched, sorted(b)).set_index(keys)["diff"]
        agree.append((np.sign(ca) == np.sign(cb)).astype(float).where((ca != 0) & (cb != 0)).rename(f"{sorted(a)}|{sorted(b)}"))
    t = pd.concat(agree, axis=1)
    return pd.DataFrame({"noise_baseline_agreement": t.mean(axis=1), "n_half_pairs": len(halves)}).reset_index()


def analyse(raw: pd.DataFrame, setting: str, regime: str, f: float, alpha_label: str) -> tuple[pd.DataFrame, str]:
    """Row table (one row per contrast and condition) and the markdown text."""
    sub = raw[(raw["setting"] == setting) & (raw["regime"] == regime) & (raw["noise_fraction"] == f)
              & (raw["alpha_label"] == alpha_label)]
    if sub.empty:
        raise ValueError("the requested subset has no rows")
    lm = layout_means(sub)
    full = contrasts(lm)
    base = full[full["condition"] == "matched"].drop(columns="condition").rename(columns={"diff": "matched_diff"})
    oth = full[full["condition"] != "matched"].rename(columns={"diff": "condition_diff"})
    t = oth.merge(base, on=["comparison", "metric", "budget"])
    t["matched_diff_is_zero"] = t["matched_diff"] == 0.0
    t["same_sign_as_matched"] = np.sign(t["condition_diff"]) == np.sign(t["matched_diff"])
    t = t.merge(noise_baseline(lm), on=["comparison", "metric", "budget"], how="left")
    t.insert(0, "subset", f"{setting}|{regime}|noise_fraction={f:g}|alpha={alpha_label}")
    nz = t[~t["matched_diff_is_zero"]]
    n_reps = int(lm["rep"].nunique())
    rows = [{"condition": c, "n_contrasts": len(g), "n_same_sign": int(g["same_sign_as_matched"].sum()),
             "share_same_sign": float(g["same_sign_as_matched"].mean())} for c, g in nz.groupby("condition", sort=False)]
    s = pd.DataFrame(rows)
    nb = nz.drop_duplicates(["comparison", "metric", "budget"])["noise_baseline_agreement"]
    lines = [f"Subset: {setting}, regime {regime}, noise fraction {f:g}, alpha '{alpha_label}'; contrasts = "
             f"{len(PAIRS)} method pairs x {len(METRICS)} metrics x budgets {[int(b) for b in sorted(sub['budget'].unique())]}; "
             f"{int((base['matched_diff'] == 0.0).sum())} of "
             f"{len(base)} contrasts excluded because the matched difference is exactly zero (identical layouts).", "",
             "| condition | contrasts | same sign as matched | share |", "|---|---|---|---|"]
    lines += [f"| {r.condition} | {r.n_contrasts} | {r.n_same_sign} | {100 * r.share_same_sign:.0f} % |" for r in s.itertuples()]
    lines += ["", f"Noise baseline (matched contrast, sign agreement between disjoint halves of the {n_reps} noise reps, "
                  f"{int(nz['n_half_pairs'].iloc[0]) if len(nz) else 0} half pair(s)): "
                  f"{100 * float(nb.mean()):.0f} % of {int(nb.notna().sum())} contrasts agree on average "
                  "(around 50 % = the sign is noise; halves hold fewer data than the full contrasts, so this is a "
                  "pessimistic reference; same cases in both halves, so only noise is measured)."]
    return t, "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robust", default=str(ROOT / "results" / "geodesign-robust"))
    ap.add_argument("--out", default=None, help="output directory (default: --robust)")
    ap.add_argument("--setting", default="interior_only")
    ap.add_argument("--regime", default="standard")
    ap.add_argument("--noise-fraction", type=float, default=0.05)
    ap.add_argument("--alpha-label", default="selected")
    a = ap.parse_args()
    rd = Path(a.robust)
    out = Path(a.out) if a.out else rd
    out.mkdir(parents=True, exist_ok=True)
    cols = ["condition", "setting", "regime", "noise_fraction", "alpha_label", "method", "budget", "case", "rep", *METRICS]
    raw = pd.read_csv(rd / "test_raw.csv", usecols=cols)
    table, text = analyse(raw, a.setting, a.regime, a.noise_fraction, a.alpha_label)
    table.to_csv(out / "ranking_stability.csv", index=False)
    (out / "ranking_stability.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

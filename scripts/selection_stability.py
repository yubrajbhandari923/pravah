"""Resampling baseline for greedy site selection: how unstable is the selected layout when only the DESIGN cases change?

    python scripts/selection_stability.py [--config configs/geodesign.json] [--alpha-json results/geodesign/alpha_selected.json]
                                          [--out results/geodesign-robust] [--robust results/geodesign-robust]
                                          [--n-subsets 10] [--subset-size 3] [--budgets 3 5] [--seed 12345]

Why: the 250 m re-selection comparison of the robustness study (refinement_overlap.csv) used 3 fine design cases.  Greedy
selection is a heuristic that already moves when the design cases are resampled at the SAME mesh; this script measures that
movement so the 250 m vs 500 m site changes can be read against it.

What it does (the solver is used only to build the design cases; nothing is evaluated on test data):
  1. builds the DESIGN ensemble of the config once (production: 6 cases, a few minutes) on a fresh in-memory response cache;
  2. sigma and alpha are those of the main run: sigma = noise fraction x the VALIDATION signal scale and the frozen alpha per
     noise fraction, both read from ``--alpha-json`` (alpha_selected.json of the main run; no validation ensemble is built);
  3. runs greedy_total and greedy_attribution on the full design ensemble and on ``--n-subsets`` seeded random subsets of
     ``--subset-size`` design cases (attribution scales from the cases used), one greedy trace per run, truncated to each budget;
  4. reports Jaccard index of the site sets and the mean nearest-site distance (``geodesign_robust.layout_overlap``) between
     subset layouts (all pairs) and between each subset layout and the full-ensemble layout, per setting, noise fraction,
     method and budget.
Writes ``selection_stability.csv`` (summary), ``selection_stability_layouts.csv`` (the layouts) and
``selection_stability.md`` (a markdown snippet, with the 250 m vs 500 m refinement overlap beside it when
``<robust>/refinement_overlap.csv`` exists).  Existing result files are never touched.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pravah import design as D
from pravah import geodesign as G
from pravah.config import load_config
from pravah.geodesign_robust import layout_overlap
from pravah.inverse import ResponseCache

ROOT = Path(__file__).resolve().parents[1]
METHODS = (("greedy_total", "total"), ("greedy_attribution", "attribution"))


def draw_subsets(n_cases: int, size: int, n: int, seed: int) -> list[tuple[int, ...]]:
    """``n`` distinct subsets of ``size`` case indices, seeded (all of them when fewer exist)."""
    if not 1 <= size <= n_cases:
        raise ValueError("subset size must be between 1 and the number of design cases")
    allsub = list(itertools.combinations(range(n_cases), size))
    if len(allsub) <= n:
        return allsub
    rng = np.random.default_rng(seed)
    return [allsub[i] for i in sorted(rng.choice(len(allsub), size=n, replace=False))]


def overlap_stats(st: Any, layouts: dict[str, G.LayoutRec], full: G.LayoutRec) -> list[dict[str, Any]]:
    """Subset-vs-subset (all pairs) and subset-vs-full statistics of the layouts of ONE (setting, fraction, method, budget).
    ``st`` needs ``cands`` only."""
    def ov(a: G.LayoutRec, b: G.LayoutRec) -> dict[str, Any]:
        return layout_overlap(st, a, b)
    rows = []
    pairs = [ov(layouts[a], layouts[b]) for a, b in itertools.combinations(sorted(layouts), 2)]
    vs_full = [ov(r, full) for r in layouts.values()]
    for kind, items in (("subset_vs_subset", pairs), ("subset_vs_full", vs_full)):
        if not items:
            continue
        j = np.array([i["jaccard"] for i in items])
        d = np.array([i["mean_nearest_site_distance_m"] for i in items])
        rows.append({"comparison": kind, "n_pairs": len(items), "jaccard_mean": float(j.mean()), "jaccard_min": float(j.min()),
                     "jaccard_max": float(j.max()), "share_identical": float((j == 1.0).mean()),
                     "mean_nearest_site_distance_m": float(d.mean()), "max_pair_mean_distance_m": float(d.max())})
    return rows


def run(cfg: dict[str, Any], payload: dict[str, Any], subset_size: int, n_subsets: int, budgets: list[int], seed: int,
        settings: list[str] | None = None, progress: G.Progress = None) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(summary, layouts, number of design cases).  ``payload`` is an alpha_selected.json dict."""
    st = G.Setup(cfg, ResponseCache())
    design = st.ensemble("design", progress, float(payload["validation"]["signal_scale_ug_m3"]))
    alphas = {f: float(payload["alpha_by_noise_fraction"][f"{f:g}"]) for f in st.fractions}
    n_cases = len(design.cases)
    subsets = draw_subsets(n_cases, subset_size, n_subsets, seed)
    runs: dict[str, list[D.DesignCase]] = {"full": design.cases}
    runs.update({"s" + "".join(map(str, s)): [design.cases[i] for i in s] for s in subsets})
    srows, lrows = [], []
    for setting in settings or [G.SETTING_A, G.SETTING_B]:
        cons = st.constraints(setting, G.STANDARD)
        feas = [b for b in budgets if b not in D.infeasible_budgets(st.cands, cons, budgets)]
        if not feas:
            continue
        for f in st.fractions:
            for method, obj in METHODS:
                recs: dict[str, dict[int, G.LayoutRec]] = {}
                for name, cases in runs.items():
                    opts = st.opts(design.sigma[f], alphas[f], D.attribution_scales(cases))
                    tr = D.greedy_trace(st.cands, max(feas), cases, cons, opts, obj)
                    recs[name] = {b: G.LayoutRec(setting, G.STANDARD, method, b, 0, tuple(sorted(tr.order[:b])), f, obj,
                                                 alphas[f], design.sigma[f], None, tr.order[:b]) for b in feas}
                    for b in feas:
                        lrows.append({"setting": setting, "noise_fraction": f, "method": method, "budget": b, "design_cases": name,
                                      "sites": ",".join(st.cands.ids[i] for i in recs[name][b].order)})
                    _say(progress, f"{setting} f={f:g} {method} {name} done")
                for b in feas:
                    sub = {n: r[b] for n, r in recs.items() if n != "full"}
                    for row in overlap_stats(st, sub, recs["full"][b]):
                        srows.append({"setting": setting, "noise_fraction": f, "method": method, "budget": b, **row,
                                      "n_subsets": len(sub), "subset_size": subset_size, "n_design_cases": n_cases, "seed": seed})
    return pd.DataFrame(srows), pd.DataFrame(lrows), n_cases


def _say(progress: G.Progress, msg: str) -> None:
    if progress is not None:
        progress(msg)


def markdown(summ: pd.DataFrame, overlap_250: pd.DataFrame | None) -> str:
    """Markdown snippet: resampling baseline table and, when given, the 250 m vs 500 m refinement overlap beside it."""
    if summ.empty:
        return "No feasible budgets: nothing to report.\n"
    n, k, m = int(summ["n_subsets"].iloc[0]), int(summ["subset_size"].iloc[0]), int(summ["n_design_cases"].iloc[0])
    g = summ.groupby(["setting", "method", "budget", "comparison"], as_index=False).agg(
        jaccard=("jaccard_mean", "mean"), identical=("share_identical", "mean"), dist=("mean_nearest_site_distance_m", "mean"))
    lines = [f"Resampling baseline at the 500 m mesh: greedy sites selected on {n} random subsets of {k} of the {m} design cases "
             "versus each other and versus the full-ensemble layout (mean over the noise fractions; Jaccard of the site sets, "
             "mean nearest-site distance in metres).", "",
             "| setting | method | budget | comparison | Jaccard | share identical | mean nearest-site distance (m) |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {r.setting} | {r.method} | {r.budget} | {r.comparison} | {r.jaccard:.2f} | {100 * r.identical:.0f} % | {r.dist:.0f} |"
              for r in g.itertuples()]
    if overlap_250 is not None and len(overlap_250):
        o = overlap_250.groupby(["setting", "method", "budget"], as_index=False).agg(
            jaccard=("jaccard", "mean"), dist=("mean_nearest_site_distance_m", "mean"), n=("jaccard", "size"))
        lines += ["", "For comparison, the 250 m vs 500 m re-selection (refinement_overlap.csv, 3 fine design cases, mean over noise "
                  "fractions):", "", "| setting | method | budget | Jaccard | mean nearest-site distance (m) |", "|---|---|---|---|---|"]
        lines += [f"| {r.setting} | {r.method} | {r.budget} | {r.jaccard:.2f} | {r.dist:.0f} |" for r in o.itertuples()]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "geodesign.json"))
    ap.add_argument("--alpha-json", default=str(ROOT / "results" / "geodesign" / "alpha_selected.json"))
    ap.add_argument("--robust", default=str(ROOT / "results" / "geodesign-robust"))
    ap.add_argument("--out", default=None, help="output directory (default: --robust)")
    ap.add_argument("--n-subsets", type=int, default=10)
    ap.add_argument("--subset-size", type=int, default=3)
    ap.add_argument("--budgets", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if not Path(cfg["region"]["manifest"]).is_absolute():
        cfg["region"]["manifest"] = str(ROOT / cfg["region"]["manifest"])
    payload = json.loads(Path(a.alpha_json).read_text())
    out = Path(a.out) if a.out else Path(a.robust)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    summ, lays, _ = run(cfg, payload, a.subset_size, a.n_subsets, a.budgets, a.seed,
                        progress=lambda m: print(f"[{time.perf_counter() - t0:6.0f} s] {m}", flush=True))
    ro = Path(a.robust) / "refinement_overlap.csv"
    o250 = pd.read_csv(ro) if ro.is_file() else None
    if o250 is not None:
        o250 = o250[o250["budget"].isin(a.budgets)]
    text = markdown(summ, o250)
    summ.to_csv(out / "selection_stability.csv", index=False)
    lays.to_csv(out / "selection_stability_layouts.csv", index=False)
    (out / "selection_stability.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

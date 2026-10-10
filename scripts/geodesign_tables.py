"""Fill the result slots of docs/geodesign.md and docs/phase2-validation.md from the saved result files.

    python scripts/geodesign_tables.py [--geodesign results/geodesign] [--robust results/geodesign-robust]

Offline and deterministic (reads CSV/JSON only). Replaces the lines
GEODESIGN_RESULTS: ..., GEODESIGN_ROBUST_RESULTS: ..., ROBUSTNESS_RESULTS: ... (placeholders) or the text between
<!-- GEODESIGN_BEGIN --> / <!-- GEODESIGN_END --> and <!-- ROBUST_BEGIN --> / <!-- ROBUST_END --> markers.
sd values are standard deviations, never confidence intervals; cases (6) are the more honest unit than rows.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fmt(x: float) -> str:
    return "n/a" if pd.isna(x) else (f"{x:.2e}" if abs(x) < 1e-3 else f"{x:.4f}" if abs(x) < 1 else f"{x:.3f}")


def strs(df: pd.DataFrame, cols: list[str], f=fmt) -> pd.DataFrame:  # noqa: ANN001
    """Copy of df with the named columns formatted as strings (other columns untouched)."""
    return pd.DataFrame({c: ([f(v) for v in df[c]] if c in cols else list(df[c])) for c in df.columns})


def frame(x: object) -> pd.DataFrame:
    """Narrow pandas' loose return types (boolean-mask / column-subset results are DataFrames)."""
    assert isinstance(x, pd.DataFrame)
    return x


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main_tables(d: Path) -> str:
    s = pd.read_csv(d / "summary.csv")
    a = json.loads((d / "alpha_selected.json").read_text())
    out = [f"Run `{d.name}`; alpha selected on validation cases only: {a['alpha_by_noise_fraction']} (frozen before the test ensemble); "
           "interior-only setting, standard regime, selected alpha; means over 6 test scenarios x 3 noise draws; budgets in sensors."]
    for nf in sorted(s.noise_fraction.unique()):
        q = s[(s.alpha_label == "selected") & (s.regime == "standard") & (s.noise_fraction == nf) & (s.setting == "interior_only")]
        for metric, label in (("mean_rmse", "total regional RMSE (ug/m3)"), ("mean_attr_err", "attribution error (dimensionless)")):
            t = q.pivot(index="method", columns="budget", values=metric).reset_index()
            t = t.rename(columns={c: f"{c} sensors" for c in t.columns if c != "method"})
            t = strs(frame(t), [c for c in t.columns if c != "method"])
            out += [f"\n**{label}, noise {nf:.0%} of the signal scale**\n", md(t)]
    p = pd.read_csv(d / "paired.csv")
    for nf in sorted(p.noise_fraction.unique()):
        q = p[(p.regime == "standard") & (p.setting == "interior_only") & (p.noise_fraction == nf) & (p.metric == "rmse")
              & p.comparison.isin(["greedy_total vs spread", "greedy_attribution vs greedy_total", "greedy_total vs random_mean"])]
        t = q[["comparison", "budget", "mean_diff", "sd_diff", "a_wins_cases", "n_cases"]].copy()
        t = strs(frame(t), ["mean_diff", "sd_diff"])
        t = t.rename(columns={"a_wins_cases": "cases where A is lower", "mean_diff": "mean (A minus B), total RMSE", "sd_diff": "sd"})
        out += [f"\n**Paired total-RMSE comparisons, noise {nf:.0%} (A is the first-named method)**\n", md(t)]
    return "\n".join(out)


def robust_tables(d: Path) -> str:
    s = pd.read_csv(d / "summary.csv")
    out = [f"Run `{d.name}` (fixed design-stage layouts re-evaluated, never reselected); interior-only, 5% noise, selected alpha, mean total RMSE (ug/m3) of `greedy_total`."]
    q = s[(s.noise_fraction == 0.05) & (s.setting == "interior_only") & (s.alpha_label == "selected") & (s.regime == "standard") & (s.method == "greedy_total")]
    t = q.pivot(index="condition", columns="budget", values="mean_rmse").reset_index()
    t = t.rename(columns={c: f"{c} sensors" for c in t.columns if c != "condition"})
    t = strs(frame(t), [c for c in t.columns if c != "condition"])
    out += ["", md(t)]
    f = pd.read_csv(d / "flags_vs_error.csv")
    g = f.groupby("condition").agg(frac_flags_equal_matched=("frac_flags_equal_matched", "mean"), rmse_ratio=("rmse_ratio_to_matched", "median")).reset_index()
    g = strs(g, ["frac_flags_equal_matched", "rmse_ratio"])
    out += ["\n**Identifiability flags versus error (all methods and budgets)**: fraction of rows whose flags equal the matched rows, and the median error ratio to matched.\n", md(g)]
    r = pd.read_csv(d / "ranking_changes.csv")
    rr = r.groupby("condition").same_sign_as_matched.mean().reset_index().rename(columns={"same_sign_as_matched": "fraction of method contrasts keeping their matched sign"})
    rr = strs(rr, [c for c in rr.columns if c != "condition"])
    out += ["\n**Do method rankings survive the condition?**\n", md(rr)]
    nd = pd.read_csv(d / "numerical_diffusion.csv")[["mesh", "speed_label", "speed_m_s", "peclet", "courant", "K_eff_over_K"]]
    nd = strs(frame(nd), ["speed_m_s", "peclet", "courant", "K_eff_over_K"], lambda v: f"{v:.3g}")
    out += ["\n**Numerical diffusion at the design mesh (K = 100 m2/s is the nominal physical value; not tuned)**\n", md(nd)]
    ov = d / "refinement_overlap.csv"
    if ov.exists():
        o = pd.read_csv(ov)
        o = o[(o.setting == "interior_only") & (o.noise_fraction == 0.05)][["method", "budget", "jaccard", "n_shared_sites", "mean_nearest_site_distance_m"]]
        o = strs(strs(frame(o), ["jaccard"]), ["mean_nearest_site_distance_m"], lambda v: f"{v:.0f}")
        out += ["\n**250 m re-selection versus 500 m layouts (3 design cases, 5% noise)**\n", md(o)]
    return "\n".join(out)


def put(path: Path, begin: str, end: str, placeholder: str, text: str) -> None:
    s = path.read_text()
    block = f"<!-- {begin} -->\n{text}\n<!-- {end} -->"
    if f"<!-- {begin} -->" in s:
        s = re.sub(rf"<!-- {begin} -->.*?<!-- {end} -->", lambda _m: block, s, flags=re.S)
    elif placeholder in s:
        s = s.replace(placeholder, block, 1)
    else:
        return
    path.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geodesign", default=str(ROOT / "results/geodesign"))
    ap.add_argument("--robust", default=str(ROOT / "results/geodesign-robust"))
    a = ap.parse_args()
    main_t, rob_t = main_tables(Path(a.geodesign)), robust_tables(Path(a.robust))
    g = ROOT / "docs/geodesign.md"
    put(g, "GEODESIGN_BEGIN", "GEODESIGN_END", "GEODESIGN_RESULTS: <to be filled by coordinator>", main_t)
    put(g, "ROBUST_BEGIN", "ROBUST_END", "GEODESIGN_ROBUST_RESULTS: <to be filled by coordinator>", rob_t)
    v = ROOT / "docs/phase2-validation.md"
    put(v, "ROBUST_BEGIN", "ROBUST_END", "ROBUSTNESS_RESULTS: <to be filled by coordinator>", rob_t)


if __name__ == "__main__":
    main()

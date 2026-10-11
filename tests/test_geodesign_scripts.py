"""Tests for the offline analysis scripts scripts/ranking_stability.py and scripts/selection_stability.py."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pravah import geodesign as G
from pravah.config import load_config
from pravah.design import Candidates
from scripts import ranking_stability as RS
from scripts import selection_stability as SS

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "belauri" / "manifest.json"


# ---------------------------------------------------------------------------------------------- ranking stability
def synthetic_raw() -> pd.DataFrame:
    """matched + two conditions; 4 cases, 4 reps.  greedy_total beats spread clearly at budget 2 (everywhere), loses at budget 3
    in matched but wins under 'flip'; budget 1 has IDENTICAL greedy_total / spread layouts (matched difference exactly 0)."""
    rows = []
    rng = np.random.default_rng(0)
    for cond in ("matched", "same", "flip"):
        for b in (1, 2, 3):
            for case in range(4):
                for rep in range(4):
                    noise = 0.001 * rng.standard_normal()
                    base = {"greedy_total": 1.0, "spread": 1.0, "random": 1.1, "greedy_attribution": 1.0}
                    if b == 2:
                        base["greedy_total"] = 0.8
                    if b == 3:
                        base["greedy_total"] = 1.2 if cond != "flip" else 0.8
                    for m, v in base.items():
                        # budget 1: exactly identical numbers for greedy_total and spread and greedy_attribution
                        val = v if b == 1 else v + (noise if m == "greedy_total" else 0.0)
                        for lay in range(2 if m == "random" else 1):
                            rows.append({"condition": cond, "setting": "interior_only", "regime": "standard", "noise_fraction": 0.05,
                                         "alpha_label": "selected", "method": m, "layout_id": f"{m}{lay}", "budget": b,
                                         "case": f"c{case}", "rep": rep, "rmse": val, "attr_err": val})
    return pd.DataFrame(rows)


def test_ranking_stability_rules() -> None:
    t, text = RS.analyse(synthetic_raw(), "interior_only", "standard", 0.05, "selected")
    assert set(t["condition"]) == {"same", "flip"} and "matched" not in set(t["condition"])
    b1 = t[(t["budget"] == 1) & (t["comparison"] == "greedy_total vs spread")]
    assert b1["matched_diff_is_zero"].all()  # identical layouts: no sign
    assert "excluded because the matched difference is exactly zero" in text
    gt = t[(t["comparison"] == "greedy_total vs spread") & (t["metric"] == "rmse") & ~t["matched_diff_is_zero"]]
    same = gt[gt["condition"] == "same"].set_index("budget")["same_sign_as_matched"]
    flip = gt[gt["condition"] == "flip"].set_index("budget")["same_sign_as_matched"]
    assert same.all() and flip[2] and not flip[3]
    # the noise baseline: clear contrasts keep their sign in every pair of disjoint rep halves
    nb = gt[gt["condition"] == "same"].set_index("budget")["noise_baseline_agreement"]
    assert nb[2] == 1.0 and nb[3] == 1.0 and (gt["n_half_pairs"] == 3).all()  # 4 reps -> halves of 2 reps: 3 splits
    assert "Noise baseline" in text and "around 50 %" in text


def test_noise_baseline_is_about_half_for_a_pure_noise_contrast() -> None:
    rng = np.random.default_rng(1)
    rows = [{"condition": "matched", "method": m, "budget": 2, "case": f"c{c}", "rep": r, "rmse": float(rng.standard_normal()),
             "attr_err": float(rng.standard_normal())} for m in ("greedy_total", "spread", "random", "greedy_attribution")
            for c in range(3) for r in range(40)]
    nb = RS.noise_baseline(RS.layout_means(pd.DataFrame(rows)))
    assert len(nb) == 8 and nb["noise_baseline_agreement"].between(0, 1).all()
    assert abs(nb["noise_baseline_agreement"].mean() - 0.5) < 0.3  # signs are noise: nothing like 100 %


# ---------------------------------------------------------------------------------------------- selection stability
def test_draw_subsets_seeded_and_distinct() -> None:
    a = SS.draw_subsets(6, 3, 10, 5)
    assert a == SS.draw_subsets(6, 3, 10, 5) and len(set(a)) == 10 and all(len(s) == 3 and len(set(s)) == 3 for s in a)
    assert a != SS.draw_subsets(6, 3, 10, 6)
    assert len(SS.draw_subsets(4, 3, 10, 1)) == 4  # fewer than requested exist
    with pytest.raises(ValueError):
        SS.draw_subsets(2, 3, 1, 1)


def test_overlap_stats_values() -> None:
    cands = Candidates(tuple(f"s{i}" for i in range(4)), np.array([0.0, 1000.0, 2000.0, 3000.0]), np.zeros(4), ("interior",) * 4)
    st = SimpleNamespace(cands=cands)
    mk = lambda *i: G.LayoutRec("interior_only", "standard", "greedy_total", len(i), 0, tuple(i))  # noqa: E731
    rows = SS.overlap_stats(st, {"a": mk(0, 1), "b": mk(0, 1), "c": mk(1, 2)}, mk(0, 1))
    pp = next(r for r in rows if r["comparison"] == "subset_vs_subset")
    assert pp["n_pairs"] == 3 and pp["jaccard_max"] == 1.0 and pp["jaccard_min"] == pytest.approx(1 / 3)
    fu = next(r for r in rows if r["comparison"] == "subset_vs_full")
    assert fu["n_pairs"] == 3 and fu["share_identical"] == pytest.approx(2 / 3)
    # (0,1) vs (1,2): nearest distances 1000 (site 0 -> 1), 0, 0, 1000 (site 2 -> 1): mean 500
    assert fu["mean_nearest_site_distance_m"] == pytest.approx(500.0 / 3)


@pytest.mark.skipif(not MANIFEST.exists(), reason="data/belauri/manifest.json is missing")
def test_selection_stability_smoke_run() -> None:
    cfg: dict[str, Any] = load_config(ROOT / "configs" / "geodesign_smoke.json")
    cfg["region"]["manifest"] = str(MANIFEST)
    payload = {"validation": {"signal_scale_ug_m3": 1.0}, "alpha_by_noise_fraction": {"0.05": 0.01, "0.15": 0.1}}
    summ, lays, n = SS.run(cfg, payload, subset_size=1, n_subsets=2, budgets=[1, 3], seed=3)
    assert n == 2 and set(summ["comparison"]) == {"subset_vs_subset", "subset_vs_full"}
    assert set(summ["budget"]) == {1, 3} and summ["jaccard_mean"].between(0, 1).all()
    assert set(lays["design_cases"]) == {"full", "s0", "s1"} and (lays["sites"].str.count(",") + 1 == lays["budget"]).all()
    assert "Resampling baseline" in SS.markdown(summ, None)

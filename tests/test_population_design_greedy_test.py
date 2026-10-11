"""EXP-022 script: uniform stop check, paired helper, reference-hash assertion (no FiPy solves)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pravah.config import load_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("population_design_greedy_test", ROOT / "scripts/population_design_greedy_test.py")
gt = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["population_design_greedy_test"] = gt
spec.loader.exec_module(gt)  # type: ignore[union-attr]


def uniform_sel(corrupt: bool = False) -> pd.DataFrame:
    rows = [{"placement": p, "objective": o, "budget": b, "layout": json.dumps([0, b])}
            for p in (0, 1) for o in ("area", "population", "mixed_0.5", "capped") for b in (2, 3)]
    if corrupt:
        rows[5]["layout"] = json.dumps([1, 2])
    return pd.DataFrame(rows)


def test_uniform_check_passes_and_writes(tmp_path):
    r = gt.uniform_check(uniform_sel(), tmp_path / "u.json")
    assert r["passed"] and r["n_compared"] == 2 * 3 * 2
    assert json.loads((tmp_path / "u.json").read_text())["passed"]


def test_uniform_check_raises_on_corrupted_layout(tmp_path):
    with pytest.raises(RuntimeError, match="uniform check failed"):
        gt.uniform_check(uniform_sel(corrupt=True), tmp_path / "u.json")
    assert json.loads((tmp_path / "u.json").read_text())["passed"] is False


def test_paired_greedy_exact_per_placement_means():
    rows = [{"type": "t", "placement": p, "draw": k, "objective": "area", "budget": 2, "same_layout": False,
             **{f"{m}_exact": 1.0 for m in gt.METRICS}, **{f"{m}_greedy": 1.0 + (0.1 if k else -0.1) for m in gt.METRICS}}
            for p in range(4) for k in range(2)]
    w = pd.DataFrame(rows)
    pg = gt.paired_greedy_exact(w, {"n_boot": 200, "boot_seed": 0, "level": 0.95})
    r = pg[pg["metric"] == "pop_rmse"].iloc[0]
    # per-placement mean of (+d, -d) is 0 for every placement; settings: half worse
    assert r["n_placements"] == 4 and r["n_settings"] == 8
    assert np.allclose(json.loads(r["placement_means"]), 0.0) and r["mean_diff"] == pytest.approx(0.0)
    assert r["frac_greedy_worse"] == 0.5


def test_verify_reference_hash():
    cfg = load_config(ROOT / gt.CONFIGS["smoke"])
    alphas = {"0": 0.1}
    guard = {"config_hash_without_alpha": gt.PE.frozen_hash(cfg, None)}
    prov = {"frozen_config_hash": gt.PE.frozen_hash(cfg, alphas)}
    assert gt.verify_reference(cfg, guard, prov, alphas) == prov["frozen_config_hash"]
    with pytest.raises(RuntimeError, match="alphas differ"):
        gt.verify_reference(cfg, guard, prov, {"0": 1.0})
    with pytest.raises(RuntimeError, match="run_guard"):
        gt.verify_reference(cfg, {"config_hash_without_alpha": "x"}, prov, alphas)

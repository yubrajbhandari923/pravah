"""EXP-017 script: weights, bootstrap, freeze guard, sanity check, seed disjointness (no FiPy solves)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pravah import population as P
from pravah.config import load_config
from pravah.geo.regionio import GridSpec

spec = importlib.util.spec_from_file_location("population_design", Path(__file__).resolve().parents[1] / "scripts/population_design.py")
pdz = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["population_design"] = pdz  # dataclasses need the module registered
spec.loader.exec_module(pdz)  # type: ignore[union-attr]

BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
GRID = GridSpec(0.0, 0.0, 20, 20, 500.0, 500.0, 32645)


def test_weights_sum_to_one_and_uniform_equals_area():
    for f, fn in P.FIXTURES.items():
        w = pdz.objective_weights(fn(GRID, 40000, 1), BETAS, 3.0)
        assert set(w) == set(pdz.objective_names(BETAS))
        for o, x in w.items():
            assert x.shape == GRID.shape and np.isclose(x.sum(), 1.0, atol=1e-12) and (x >= 0).all(), (f, o)
    wu = pdz.objective_weights(P.uniform(GRID, 40000), BETAS, 3.0)
    for x in wu.values():
        assert np.allclose(x, wu["area"], rtol=0, atol=1e-15)


def test_capped_weight_caps():
    n = P.one_town(GRID, 40000, 1)
    w = pdz.objective_weights(n, BETAS, 3.0)["capped"]
    assert w.max() <= 3 * n.mean() / np.minimum(n, 3 * n.mean()).sum() + 1e-15


def test_bootstrap_ci():
    x = np.full(16, 2.0)
    assert pdz.bootstrap_ci(x, 200) == (2.0, 2.0)
    lo, hi = pdz.bootstrap_ci(np.arange(16.0), 2000, seed=1)
    assert lo < 7.5 < hi and pdz.bootstrap_ci(np.arange(16.0), 2000, seed=1) == (lo, hi)
    assert np.isnan(pdz.bootstrap_ci(np.array([1.0]))[0])


def test_disjoint_seeds():
    d = load_config(Path(pdz.ROOT) / "configs/population_toy.json")["design"]
    pdz.check_ensemble_seeds(d)
    with pytest.raises(ValueError):
        pdz.check_ensemble_seeds({**d, "test_seed": d["design_seed"]})
    with pytest.raises(ValueError):
        pdz.check_ensemble_seeds({**d, "test_noise_seed": d["design_noise_seed"] + 1})


def test_freeze_guard(tmp_path):
    cfg = load_config(Path(pdz.ROOT) / "configs/population_toy.json")
    (tmp_path / "provenance.json").write_text(json.dumps({"frozen_config_hash": pdz.frozen_hash(cfg, 0.1)}))
    pdz.assert_frozen(tmp_path, cfg, 0.1)
    with pytest.raises(RuntimeError):
        pdz.assert_frozen(tmp_path, cfg, 0.2)
    cfg["population"]["cap_c"] = 4.0
    with pytest.raises(RuntimeError):
        pdz.assert_frozen(tmp_path, cfg, 0.1)
    cfg2 = load_config(Path(pdz.ROOT) / "configs/population_toy.json")
    cfg2["run"]["fixtures"] = ["one_town"]  # run selection is not frozen
    assert pdz.frozen_hash(cfg2, 0.1) == pdz.frozen_hash(load_config(Path(pdz.ROOT) / "configs/population_toy.json"), 0.1)


def test_uniform_sanity_raises():
    objs, budgets = ["area", "population"], [2]
    lay = {("uniform", "area", 2): (1, 2), ("uniform", "population", 2): (1, 2)}
    pdz.check_uniform_equals_area(lay, budgets, objs)
    lay[("uniform", "population", 2)] = (1, 3)
    with pytest.raises(RuntimeError):
        pdz.check_uniform_equals_area(lay, budgets, objs)


def test_lowpop_mask():
    n = np.array([[0.0, 1.0], [5.0, 9.0]])
    assert pdz.lowpop_mask(n, 0.25, 1.0).tolist() == [[False, True], [False, False]]

import copy
import json

import numpy as np
import pytest

from pravah.dynamic_design import DesignProblem
from pravah.studies.toy import dynamic_design_study as dd


def _sum(kind="risk", n_cand=9, seed=0):
    rng = np.random.default_rng(seed)
    probs = [DesignProblem(rng.normal(size=(n_cand, 4, 5)), rng.normal(size=(5, 5)), np.ones(5), 0.7) for _ in range(3)]
    return dd.SumProblem(probs, kind), probs


def test_sum_problem_is_mean_of_winds():
    sp, probs = _sum()
    s = [[0, 3], [1, 8], [2, 5]]
    np.testing.assert_allclose(sp.risk_batch(s), np.mean([p.risk_batch(s) for p in probs], axis=0))
    spi, _ = _sum("info")
    np.testing.assert_allclose(spi.risk([1, 4]), np.mean([p.information_risk([1, 4]) for p in probs]))


def test_exhaustive_not_worse_than_greedy_and_swap_deterministic():
    sp, _ = _sum()
    ex = sp.exhaustive(3, chunk=16)
    g, rg = sp.greedy(3)
    gs, rgs = sp.swap_refine(g)
    assert ex.best_risk <= rgs + 1e-12 <= rg + 1e-12
    assert sp.swap_refine(g) == (gs, rgs) and sp.exhaustive(3, chunk=16).best_subset == ex.best_subset


@pytest.mark.parametrize("k", dd.KS)
def test_layout_generators_valid(k):
    lays = [dd.spread_layout(k), dd.highpop_layout(k)] + [dd.random_layout(k, r) for r in range(dd.N_RANDOM)]
    for s in lays:
        assert len(s) == k == len(set(s)) and all(0 <= i < len(dd.CANDS) for i in s)
    assert len(dd.CANDS) == 70 and len({dd.random_layout(k, r) for r in range(dd.N_RANDOM)}) > 1
    assert dd.random_layout(k, 3) == dd.random_layout(k, 3)


def test_highpop_separation():
    s = dd.highpop_layout(5)
    for i in s:
        for j in s:
            if i != j:
                assert max(abs(dd.CANDS[i][0] - dd.CANDS[j][0]), abs(dd.CANDS[i][1] - dd.CANDS[j][1])) >= dd.MIN_SEP


def test_search_deterministic():
    sp, _ = _sum(n_cand=12)
    dd.GAP_EXTRA = ()
    a, b = dd.search(sp, 5), dd.search(sp, 5)
    assert a["subset"] == b["subset"] and a["risk"] == b["risk"] and a["risk"] <= a["greedy_swap_risk"] + 1e-12


def test_hash_guard(tmp_path):
    doc = {"candidates": [[2, 2]], "layouts": {"0.1": {"1": {"spread": [[2, 2]]}}}}
    doc["hash"] = dd.layouts_hash(doc)
    p = tmp_path / "layouts.json"
    p.write_text(json.dumps(doc))
    assert dd.load_layouts(p)["hash"] == doc["hash"]
    bad = copy.deepcopy(doc)
    bad["layouts"]["0.1"]["1"]["spread"] = [[4, 4]]
    p.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="freeze guard"):
        dd.load_layouts(p)


def test_eval_episodes_seed_range():
    eps = dd.eval_episodes()
    assert len(eps) == 96 and min(e["seed"] for e in eps) == 7800 and max(e["seed"] for e in eps) == 7895

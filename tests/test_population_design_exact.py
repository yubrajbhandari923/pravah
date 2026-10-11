"""EXP-020 script: batched bounded fit, exact enumeration, search, ties, seeds, out-dir guard (no FiPy solves)."""
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from pravah.config import load_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("population_design_exact", ROOT / "scripts/population_design_exact.py")
pe = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["population_design_exact"] = pe
spec.loader.exec_module(pe)  # type: ignore[union-attr]


def fake_evaluator(seed=0, ns=7, nc=3, R=2, K=2):
    """Evaluator on synthetic arrays (no FiPy): random operators, noise and quadratic forms."""
    rng = np.random.default_rng(seed)
    ev = pe.Evaluator.__new__(pe.Evaluator)
    ev.nc, ev.ns, ev.R, ev.alpha, ev.K = nc, ns, R, 0.5, K
    A = rng.normal(size=(ns, nc, 5, 3))
    ev.Hs = np.einsum("sctp,sctq->scpq", A, A)
    ev.rs = np.einsum("sctp,srct->scrp", A, rng.normal(size=(ns, R, nc, 5)))
    ev.coef = rng.normal(size=(nc * 12, K)) * 1e-12
    ev.const = rng.uniform(50, 60, K)  # objectives must be positive (the greedy tie rule is relative)
    return ev, A


def test_nnls3_matches_scipy():
    rng = np.random.default_rng(1)
    for _ in range(200):
        B = rng.normal(size=(6, 3))
        H = B.T @ B + 0.3 * np.eye(3)
        r = rng.normal(size=3) * 3
        got = pe.nnls3(H[None], r[None, None])[0, 0]
        L = np.linalg.cholesky(H)
        ref = optimize.lsq_linear(L.T, np.linalg.solve(L, r), bounds=(0, np.inf), method="bvls", tol=1e-14).x
        assert np.allclose(got, ref, atol=1e-9), (got, ref)
        assert (got >= 0).all()


def test_values_match_direct_fit():
    ev, _ = fake_evaluator()
    lay = (1, 4, 5)
    H = ev.Hs[list(lay)].sum(0) + ev.alpha * np.eye(3)
    r = ev.rs[list(lay)].sum(0)
    feats = []
    for c in range(ev.nc):
        q = np.array([pe.nnls3(H[c][None], r[c][None])[0, k] * pe.QS for k in range(ev.R)])
        feats.append(np.concatenate([q.mean(0), (np.einsum("ri,rj->ij", q, q) / ev.R).ravel()]))
    want = np.concatenate(feats) @ ev.coef + ev.const
    assert np.allclose(ev.values(np.array([lay]))[0], want, rtol=1e-12)


def test_enum_is_global_min_and_unrank():
    ev, _ = fake_evaluator()
    allv = np.array([ev.values(np.array([c]))[0] for c in itertools.combinations(range(ev.ns), 3)])
    vals, ords = pe.exact_enum(ev, 3, keep=5, chunk=7)  # chunk not dividing C(7,3) on purpose
    assert np.allclose(vals[0], allv.min(0))
    assert np.allclose(vals[1], np.sort(allv, 0)[1])
    combos = list(itertools.combinations(range(ev.ns), 3))
    for k in range(ev.K):
        assert combos[ords[0, k]] == pe.unrank(int(ords[0, k]), ev.ns, 3)
        assert np.isclose(allv[ords[0, k], k], vals[0, k])


def test_search_never_beats_exact_and_greedy_is_upper_bound():
    ev, _ = fake_evaluator(seed=3)
    vals, ords = pe.exact_enum(ev, 3, keep=3, chunk=50)
    for k in range(ev.K):
        order, scores = pe.greedy_order(ev, k, 3)
        s = pe.search_layout(ev, k, tuple(sorted(order)), pe.random_starts(ev.ns, 3, 5, 0))
        assert s["value"] >= vals[0, k] * (1 - 1e-12)
        assert s["value"] <= scores[-1] * (1 + 1e-12)  # search starts from the greedy layout and only improves


def test_tie_pairs():
    rows = []
    for o, lay, cross in (("area", "[1, 2]", {"area": 1.0, "population": 1.005}), ("population", "[1, 3]", {"area": 1.5, "population": 2.0})):
        rows.append({"type": "t", "placement": 0, "draw": 0, "budget": 2, "objective": o, "layout": lay, "method": "exact",
                     "value": cross[o], **{f"cross_{k}": v for k, v in cross.items()}})
    # area layout is within 0.5% under population (1.005 vs 2.0?) -> no: compare against the population optimum
    t = pe.tie_pairs(pd.DataFrame(rows), 0.01, ["area", "population"])
    assert len(t) == 1 and not t.iloc[0]["tie"] and not t.iloc[0]["same_layout"]
    rows[0]["cross_population"], rows[1]["cross_area"] = 2.01, 1.001
    assert bool(pe.tie_pairs(pd.DataFrame(rows), 0.01, ["area", "population"]).iloc[0]["tie"])


def test_sources_and_seeds():
    pts = pe.draw_sources(10000.0, 10000.0, 3, 5, 0.15, 2000.0)
    assert pts.shape == (3, 2) and (pts >= 1500).all() and (pts <= 8500).all()
    assert np.array_equal(pts, pe.draw_sources(10000.0, 10000.0, 3, 5, 0.15, 2000.0))
    assert not np.array_equal(pts, pe.draw_sources(10000.0, 10000.0, 3, 6, 0.15, 2000.0))
    prod = load_config(ROOT / "configs/population_toy_exact.json")
    smoke = load_config(ROOT / "configs/population_toy_exact_smoke.json")
    pil = pe._merge(prod, prod["pilot"])
    for c in (prod, smoke, pil):
        pe.check_seeds(c)
    ids = lambda c: {(lab.split()[0], a + i) for lab, a, n in pe.seed_blocks(c) for i in range(n)}  # noqa: E731
    assert not ids(smoke) & ids(prod) and not ids(pil) & ids(prod) and not ids(smoke) & ids(pil)  # smoke/pilot never reuse production seeds
    bad = json.loads(json.dumps(prod))
    bad["design"]["test_seed"] = bad["design"]["design_seed"] + bad["replicates"]["seed_stride"]  # placement 1 design == placement 0 test
    with pytest.raises(ValueError):
        pe.check_seeds(bad)
    bad = json.loads(json.dumps(prod))
    bad["design"]["design_seed"] = 4101  # EXP-017 seed
    with pytest.raises(ValueError):
        pe.check_seeds(bad)


def test_placement_cfg_moves_sources_and_seeds():
    prod = load_config(ROOT / "configs/population_toy_exact.json")
    a, b = pe.placement_cfg(prod, 0), pe.placement_cfg(prod, 1)
    assert a["scenario"]["sources"] != b["scenario"]["sources"] and a["design"]["test_seed"] != b["design"]["test_seed"]
    assert prod["scenario"]["sources"][0]["x"] == 2500.0  # input untouched


def test_freeze_and_out_guard(tmp_path):
    cfg = load_config(ROOT / "configs/population_toy_exact.json")
    h = pe.frozen_hash(cfg, {"0": 0.1})
    (tmp_path / "provenance.json").write_text(json.dumps({"frozen_config_hash": h}))
    pe.assert_frozen(tmp_path, cfg, {"0": 0.1})
    with pytest.raises(RuntimeError):
        pe.assert_frozen(tmp_path, cfg, {"0": 0.2})
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["run"]["workers"] = 1  # run selection / workers are not frozen
    assert pe.frozen_hash(cfg2, {"0": 0.1}) == h
    cfg2["search"]["n_restarts"] = 3
    assert pe.frozen_hash(cfg2, {"0": 0.1}) != h
    sel = {"types": ["one_town"], "placements": [0], "draws": [0]}
    out = tmp_path / "o"
    pe.guard_out(out, cfg, sel)  # empty -> new
    pe.guard_out(out, cfg, sel)  # same hash -> resume
    with pytest.raises(RuntimeError):
        pe.guard_out(out, cfg2, sel)
    with pytest.raises(RuntimeError):
        pe.guard_out(out, cfg, {**sel, "draws": [0, 1]})
    stray = tmp_path / "s"
    stray.mkdir()
    (stray / "x.csv").write_text("a")
    with pytest.raises(RuntimeError):
        pe.guard_out(stray, cfg, sel)  # non-empty dir without guard file


def test_paired_over_settings_and_moderator():
    rows = []
    for p in range(2):
        for d in range(3):
            for o, off in (("area", 0.0), ("population", -0.1 * (d + 1))):
                rows.append({"type": "t", "placement": p, "draw": d, "objective": o, "budget": 2,
                             **{m: 1.0 + off for m in pe.METRICS}})
    sm = pd.DataFrame(rows)
    pr = pe.paired_tables(sm, pd.DataFrame(), {"n_boot": 200, "boot_seed": 0, "level": 0.95})
    r = pr[(pr["metric"] == "pop_rmse") & (pr["subset"] == "all")].iloc[0]
    assert r["n_settings"] == 6 and np.isclose(r["mean_diff"], -0.2) and r["boot_hi"] < 0 and r["frac_lower"] == 1.0
    units = pd.DataFrame([{"type": "t", "placement": p, "draw": d, "dist_mean_km": float(d), "dist_centroid_km": float(d)} for p in range(2) for d in range(3)])
    mod, cor = pe.moderator_tables(sm, units)
    assert np.allclose(mod["benefit"], 0.1 * (mod["draw"] + 1)) and cor.iloc[0]["spearman_dist_mean"] > 0.8


def test_moderator_distance():
    cnt = np.zeros((10, 10))
    cnt[0, 0] = 1.0
    m = pe.moderator(cnt, np.array([[500.0, 500.0]]), 10000.0, 10000.0)  # cell (0,0) centre = (500, 500)
    assert m["dist_mean_km"] == pytest.approx(0.0) and m["dist_centroid_km"] == pytest.approx(0.0)

"""EXP-016 script: masks, weights, p-parameter bounded fit, evaluator, enumeration ranges, seeds, guards, resume, cluster bootstrap,
uniform==area check (no FiPy solves; the end-to-end smoke profile is run by hand, see the script docstring)."""
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from pravah import population as P
from pravah.geo.regionio import GridSpec

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("population_belauri", ROOT / "scripts/population_belauri.py")
pb = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["population_belauri"] = pb
spec.loader.exec_module(pb)  # type: ignore[union-attr]
PROD = pb.load_cfg("production", None)


def grid_sf(nx=12, ny=10):
    g = GridSpec(0.0, 0.0, nx, ny, 500.0, 500.0, 32644)
    sf = np.zeros((ny, nx))
    sf[2:8, 3:10] = 1.0
    sf[2, 3:10] = 0.5
    return g, sf


def test_mask_to_study_total_zero_outside_nonneg():
    g, sf = grid_sf()
    raw = P.FIXTURES["one_town"](g, 20 * g.nx * g.ny, 3)
    N = P.mask_to_study(raw, sf, 25000.0)
    assert N.sum() == pytest.approx(25000.0) and (N >= 0).all() and not N[sf == 0].any()
    with pytest.raises(ValueError):
        P.mask_to_study(np.zeros(sf.shape), sf, 10.0)


def test_units_masks_sum_weights_and_uniform_equals_area_weight():
    g, sf = grid_sf()
    cfg = json.loads(json.dumps(PROD))
    cfg["population"].update(types=["uniform", "one_town", "mixed_rural_urban"], n_draws=2, real={})
    units = pb.make_units(cfg, g, sf, 500.0)
    assert len(units) == 6
    for u in units:
        assert u["counts"].sum() == pytest.approx(25000.0) and not u["counts"][sf == 0].any()
        assert all(w.min() >= 0 and w.sum() == pytest.approx(1.0) for w in u["w"].values())
    uni = next(u for u in units if u["type"] == "uniform")
    for o, w in uni["w"].items():  # uniform population: every objective is the area weight (cap never binds)
        assert np.allclose(w, uni["w"]["area"], rtol=1e-12, atol=0), o
    W, col_of = pb.build_columns(units, ["area", "population", "capped"])
    assert col_of[("one_town", 0, "area")] == 0 == col_of[("uniform", 1, "area")]
    assert col_of[("uniform", 0, "population")] == col_of[("uniform", 1, "population")]  # identical uniform draws share a column
    # real field: single draw, masked by the study, real total kept
    cfg["population"]["real"] = {"real_x": "unused"}
    raw = np.random.default_rng(0).random(sf.shape) * 40
    np.save(ROOT / "results" / ".tmp_real_test.npy", raw)
    try:
        cfg["population"].update(types=["real_x"], real={"real_x": "results/.tmp_real_test.npy"})
        (u,) = pb.make_units(cfg, g, sf, 500.0)
    finally:
        (ROOT / "results" / ".tmp_real_test.npy").unlink()
    assert u["draw"] == 0 and u["counts"].sum() == pytest.approx((raw * sf).sum()) and not u["counts"][sf == 0].any()


def test_population_check_stops():
    g, sf = grid_sf()
    cfg = json.loads(json.dumps(PROD))
    cfg["population"].update(types=["one_town"], n_draws=1, real={})
    units = pb.make_units(cfg, g, sf, 500.0)
    near = np.zeros(sf.shape, bool)
    with pytest.raises(RuntimeError, match="STOP"):
        pb.check_population(units, sf, lambda t: 25000.0, near)
    pb.check_population(units, sf, lambda t: 25000.0, sf > 0)


def test_nnls_p_matches_scipy_and_nnls3():
    rng = np.random.default_rng(1)
    for p in (3, 4):
        for _ in range(150):
            B = rng.normal(size=(7, p))
            H = B.T @ B + 0.3 * np.eye(p)
            r = rng.normal(size=p) * 3
            got = pb.nnls_p(H[None], r[None, None])[0, 0]
            L = np.linalg.cholesky(H)
            ref = optimize.lsq_linear(L.T, np.linalg.solve(L, r), bounds=(0, np.inf), method="bvls", tol=1e-14).x
            assert np.allclose(got, ref, atol=1e-9) and (got >= 0).all()
            if p == 3:
                assert np.allclose(got, pb.PDE.nnls3(H[None], r[None, None])[0, 0], atol=1e-12)


def fake_problem(seed=0, ns=7, nc=3, R=2, K=3, p=4, ny=4, nx=5, nt=6):
    rng = np.random.default_rng(seed)
    cases = [SimpleNamespace(A_site=rng.normal(size=(ns, nt, p)) * 1e-6, b_site=rng.normal(size=(ns, nt)) * 0.01,
                             truth_readings=rng.random((ns, nt)) * 0.1, noise=rng.normal(size=(R, ns, nt))) for _ in range(nc)]
    forms = [SimpleNamespace(c0=rng.random((ny, nx)), h=rng.normal(size=(p, ny, nx)) * 1e-7, G=None) for _ in range(nc)]
    for f in forms:
        B = rng.normal(size=(p, ny, nx))
        f.G = np.einsum("jyx,kyx->jkyx", B, B) * 1e-12
    W = rng.random((K, ny, nx)) + 0.1
    scale = np.array([1e5, 1e5, 1e5, 0.05])
    return cases, forms, W, scale, 0.01, 0.02


def test_ev_matches_direct_fit_and_ranges_merge_to_exact():
    cases, forms, W, scale, alpha, sigma = fake_problem()
    sites = np.arange(7)
    ev = pb.Ev(cases, forms, W, alpha, sigma, sites, scale)
    lay = (1, 4, 5)
    for k in range(3):
        w = W[k] / W[k].sum()
        tot = 0.0
        for c, f in zip(cases, forms):
            A, b = c.A_site[list(lay)].transpose(1, 0, 2).reshape(-1, 4), c.b_site[list(lay)].T.reshape(-1)
            for rep in range(2):
                y = (c.truth_readings[list(lay)] + sigma * c.noise[rep, list(lay)]).T.reshape(-1)
                Aw = np.vstack([A * scale / sigma, np.sqrt(alpha) * np.eye(4)])
                u = optimize.lsq_linear(Aw, np.concatenate([(y - b) / sigma, np.zeros(4)]), bounds=(0, np.inf), method="bvls", tol=1e-14).x
                q = u * scale
                e2 = f.c0 + 2 * np.tensordot(q, f.h, 1) + np.einsum("j,k,jkyx->yx", q, q, f.G)
                tot += (w * e2).sum() / 2
        assert ev.values(np.array([lay]), [k])[0, 0] == pytest.approx(tot / len(cases), rel=1e-7)
    # range-wise enumeration merged == one pass
    v1, o1 = pb.PDE.exact_enum(ev, 3, 10, 7)
    parts = [pb.enum_range(ev, 3, a, z, 10, 7) for a, z in ((0, 12), (12, 20), (20, 35))]
    v2, o2 = pb.merge_enum(parts, 10)
    assert np.allclose(v1, v2) and (o1 == o2).all()
    best = min(itertools.combinations(range(7), 3), key=lambda s: ev.values(np.array([s]), [0])[0, 0])
    assert pb.PDE.unrank(int(o2[0, 0]), 7, 3) == best


def test_uniform_equals_area_check_executes():
    cases, forms, W, scale, alpha, sigma = fake_problem(K=1)
    w = W[0]
    # two columns: the area weight and a rounding-level perturbation of it (the uniform fixture's population weight)
    ev = pb.Ev(cases, forms, np.stack([w, w * (1 + 1e-15)]), alpha, sigma, np.arange(7), scale)
    v, o = pb.PDE.exact_enum(ev, 3, 5, 100)
    lay = [pb.lkey(pb.PDE.unrank(int(o[0, c]), 7, 3)) for c in range(2)]
    rows = []
    for obj, c in (("area", 0), ("population", 1)):
        rows.append({"fraction": 0.05, "budget": 3, "type": "uniform", "draw": 0, "objective": obj, "layout": lay[c], "value": v[0, c],
                     "cross_area": float(ev.values(np.array([[int(x) for x in lay[c].split("-")]]), [0])[0, 0])})
    rec = pd.DataFrame(rows)
    assert pb.check_uniform_equals_area(rec, ["population"]) == []
    bad = rec.copy()
    bad.loc[1, "layout"] = "0-1-2"
    bad.loc[1, "cross_area"] = rec.loc[0, "value"] * 1.2
    with pytest.raises(RuntimeError, match="SANITY FAILED"):
        pb.check_uniform_equals_area(bad, ["population"])
    tie = bad.copy()
    tie.loc[1, "cross_area"] = rec.loc[0, "value"] * (1 + 1e-12)
    assert len(pb.check_uniform_equals_area(tie, ["population"])) == 1
    with pytest.raises(RuntimeError):  # no uniform fixture: the registered check cannot run
        pb.check_uniform_equals_area(rec.assign(type="one_town"), ["population"])


def test_seeds_disjoint_incl_fixture_seeds():
    pb.check_seeds(PROD)  # production seeds are fresh (101/202 declared reuse)
    pb.check_seeds(pb.load_cfg("smoke", None))
    pb.check_seeds(pb.load_cfg("pilot", None))
    hist = ({1, 2}, [(500, 10)])

    def mod(f):
        c = json.loads(json.dumps(PROD))
        f(c)
        return c
    with pytest.raises(ValueError, match="fixture"):
        pb.check_seeds(mod(lambda c: c["population"].update(fixture_draw_seeds=[6501, 6501, 6503, 6504, 6505, 6506])))
    with pytest.raises(ValueError, match="fixture draw seeds"):
        pb.check_seeds(mod(lambda c: c["population"].update(fixture_draw_seeds=[1, 6502, 6503, 6504, 6505, 6506])), hist)
    with pytest.raises(ValueError, match="distinct"):
        pb.check_seeds(mod(lambda c: c["population"].update(fixture_draw_seeds=[305, 6502, 6503, 6504, 6505, 6506])))
    with pytest.raises(ValueError, match="noise"):
        pb.check_seeds(mod(lambda c: c["ensembles"]["test"].update(noise_seed=2003)))
    with pytest.raises(ValueError, match="used by another"):  # design seed reuse is only allowed when declared
        pb.check_seeds(mod(lambda c: c.update(seeds={"allowed_reuse": []})), ({101, 202}, []))
    with pytest.raises(ValueError, match="fresh"):
        pb.check_seeds(mod(lambda c: c.update(seeds={"allowed_reuse": [101, 202, 305]})))
    with pytest.raises(ValueError, match="noise"):
        pb.check_seeds(mod(lambda c: c["ensembles"]["test"].update(noise_seed=498)), hist)


def test_run_units_resume(tmp_path):
    calls = []

    def fn(x):
        calls.append(x)
        return x * 2
    units = [(tmp_path / f"u{i}.pkl", i) for i in range(4)]
    with pb.Pool(1) as pool:
        r1 = pb.run_units(pool, fn, units)
        assert len(calls) == 4
        (tmp_path / "u2.pkl").unlink()
        r2 = pb.run_units(pool, fn, units)
    assert len(calls) == 5 and calls[-1] == 2 and r1 == r2


def test_run_guard_and_freeze_guard(tmp_path):
    cfg = json.loads(json.dumps(PROD))
    out = tmp_path / "o"
    pb.guard_out(out, cfg, "production")
    pb.guard_out(out, cfg, "production")  # resume with the same hash
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["population"]["cap_c"] = 2.0
    with pytest.raises(RuntimeError, match="different frozen config"):
        pb.guard_out(out, cfg2, "production")
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "x").write_text("1")
    with pytest.raises(RuntimeError, match="no run_guard"):
        pb.guard_out(tmp_path / "p", cfg, "production")
    pb.guard_out(out, {**cfg, "run": {**cfg["run"], "workers": 3}}, "production")  # 'run' keys are not part of the hash
    rec = pd.DataFrame([{"fraction": 0.05, "budget": 2, "type": "t", "draw": 0, "objective": "area", "layout": "1-2", "greedy_layout": "1-3"}])
    h1 = pb.freeze_hash(cfg, {0.05: 0.01}, {0.05: 0.1}, rec, pd.DataFrame())
    (out / "provenance.json").write_text(json.dumps({"frozen_config_hash": h1}))
    pb.assert_frozen(out, h1)
    with pytest.raises(RuntimeError, match="freeze guard"):
        pb.assert_frozen(out, pb.freeze_hash(cfg, {0.05: 0.01}, {0.05: 0.1}, rec.assign(layout="1-4"), pd.DataFrame()))


def test_cluster_bootstrap_and_stats():
    stats = {"n_boot": 2000, "boot_seed": 0, "level": 0.95, "agree_frac": 0.75}
    rng = np.random.default_rng(0)
    M = -1.0 + 0.1 * rng.normal(size=(6, 12))
    s = pb.cluster_stats(M, stats)
    assert s["boot_hi"] < 0 and s["resolved"] and s["frac_lower"] == 1.0 and s["n_draws"] == 6 and s["n_cases"] == 12
    assert s == pb.cluster_stats(M, stats)  # deterministic in the seed
    N = rng.normal(size=(6, 12))
    assert not pb.cluster_stats(N, stats)["resolved"]
    # draw-level heterogeneity: the two-way interval is at least as wide as the narrower one-way interval, mean inside it
    H = np.repeat(rng.normal(size=(6, 1)), 12, 1) + 0.01 * rng.normal(size=(6, 12))
    ci = pb.cluster_bootstrap(H, 4000)
    wid = {k: v[1] - v[0] for k, v in ci.items()}
    assert wid["both"] >= 0.9 * wid["draws"] and wid["draws"] > 5 * wid["cases"]
    assert ci["both"][0] < H.mean() < ci["both"][1]
    assert np.isnan(pb.cluster_bootstrap(M[:1], 100)["both"][0])
    # an interval that excludes 0 but with < 75% agreement across scenarios is not resolved
    Q = np.full((6, 12), -0.1)
    Q[:, 8:] = 0.01
    sq = pb.cluster_stats(Q, stats)
    assert sq["boot_hi"] < 0 and sq["frac_lower"] < 0.75 and not sq["resolved"]


def test_metric_vectorisation_matches_population_functions():
    g, sf = grid_sf()
    N = P.mask_to_study(P.FIXTURES["one_town"](g, 20 * g.nx * g.ny, 2), sf, 25000.0)
    e2 = np.random.default_rng(0).random(sf.shape)
    vec = np.sqrt(N.ravel()[None] @ e2.ravel() / N.sum())
    assert vec[0] == pytest.approx(P.pop_weighted_rmse(np.sqrt(e2), N))
    aw = (g.cell_area * sf).ravel()
    assert np.sqrt(aw @ e2.ravel() / aw.sum()) == pytest.approx(P.area_weighted_rmse(np.sqrt(e2), g.cell_area, sf))

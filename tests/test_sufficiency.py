"""E08 sufficiency: schedules, the fast exact scorer, masked estimator causality, breakpoints, order-statistic CIs, cache
reuse of E07 fields, and one tiny end-to-end scenario."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pravah import benchmarks as B
from pravah.experiments import cross_objective as X
from pravah.experiments import sufficiency as S

_p = Path(__file__).resolve().parents[1] / "scripts" / "e08_sufficiency.py"
_spec = importlib.util.spec_from_file_location("e08_driver", _p)
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


@pytest.mark.parametrize("W,n", list(S.SCHEDULES.values()))
def test_schedule_has_n_readings_in_every_window(W, n):
    m = S.schedule_mask(400, W, n)
    c = np.convolve(m, np.ones(W, int), "valid")
    assert (c == n).all()


def test_schedule_rejects_uneven():
    with pytest.raises(ValueError):
        S.schedule_mask(10, 24, 5)


def test_gram_scorer_equals_score():
    rng = np.random.default_rng(1)
    K, H, nx = 3, 40, 6
    F, tf = rng.random((K, H, nx, nx)), 5 + rng.random((H, nx, nx))
    issue, a_true = np.arange(10, 30), rng.random(K) * 1e6
    u = rng.normal(size=(len(issue), K))
    groups = {"a": issue < 20, "b": issue >= 20}
    g = S.GramScorer(F, tf, issue, a_true, 1e6, 5.0, groups)(u)
    ref = X.score(u, F, tf, issue, a_true, 1e6, 5.0, np.full((nx, nx), 1 / nx ** 2))
    for c in ("forecast_rmse", "analysis_rmse", "source_rmse"):
        assert g[c] == pytest.approx(ref[c], rel=1e-10)
    sub = X.score(u[:10], F, tf, issue[:10], a_true, 1e6, 5.0, np.ones((nx, nx)))
    assert g["fc_rmse_a"] == pytest.approx(sub["forecast_rmse"], rel=1e-10)


def test_masked_estimator_is_causal_and_ignores_unsampled_hours():
    rng = np.random.default_rng(2)
    A, z = rng.normal(size=(4, 100, 2)), rng.normal(size=(4, 100))
    issue = np.arange(40, 60)
    assert np.allclose(X.estimate_series(A, z, issue, 24), X.estimate_series(A, z, issue, 24, mask=np.ones(100, bool)))
    m = S.schedule_mask(100, 24, 6)
    u = X.estimate_series(A, z, issue, 24, mask=m)
    z2 = z.copy()
    z2[:, ~m] += 100.0          # unsampled hours
    z2[:, 60:] += 100.0         # after the last issue time
    assert np.array_equal(u, X.estimate_series(A, z2, issue, 24, mask=m))
    A2 = A.copy()
    A2[:, ~m] *= 7.0
    assert np.array_equal(X.window_info(A, issue, 24, m), X.window_info(A2, issue, 24, m))


def test_breakpoint():
    ks = [2, 4, 8, 16, 32]
    assert S.breakpoint(ks, [.9, .5, .15, .1, .05], .2) == 8
    assert S.breakpoint(ks, [.1, .5, .15, .1, .05], .2) == 8      # dip at k=2 does not count
    assert S.breakpoint(ks, [.1, .1, .1, .1, .3], .2) == S.K_NONE  # must hold up to the largest k
    assert S.breakpoint(ks, [.1, .1, np.inf, .1, .1], .2) == 16   # undefined = not sufficient
    assert S.breakpoint(ks, [.1] * 5, .2) == 2


def test_df_ci():
    lo, hi, cov = S.df_ci(np.arange(1, 21))
    assert (lo, hi) == (6, 15) and cov == pytest.approx(0.9586, abs=1e-4)
    lo, hi, cov = S.df_ci([3, 1, 4, 2])
    assert (lo, hi) == (1, 4) and cov == pytest.approx(0.875)
    assert S.df_ci([1, 2, np.inf, -np.inf, 5])[:2] == (-np.inf, np.inf)


def test_wilson():
    lo, hi = S.wilson(10, 20)
    assert lo < .5 < hi and lo == pytest.approx(1 - hi)


def test_sweep_reuses_e07_field_caches():
    for K in (2, 4):
        assert [n for n, _ in D.E7.scenario_fields("e08sweep", "mixed", K, 3)] == \
               [n for n, _ in D.E7.scenario_fields("sweep", "mixed", K, 3)]


def test_tiny_scenario_end_to_end():
    D.E7.SCALES["tiny8"] = dict(grid=(16, 16), dt=300, scheme="upwind", spinup=6, split=((1, 1), (2, 2), (3, 4)), Ks=(2,),
                                ks=(2, 4), seeds=(0,), R=2)
    sc = B.generate_scenario(D.E7.make_cfg("tiny8", "mixed"), 0, 2)
    F, G3 = X.responses(sc)
    sch = {"S1": (1, 1), "S2": (6, 6), "S4": (24, 24)}
    rows, drows, _ = S.evaluate(sc, F, G3, (2, 4), 2, schedules=sch)
    assert len(rows) == 1 + len(S.ARMS) * 2 * len(S.STRATEGIES) * len(sch)
    ok = [r for r in rows[1:] if r["status"] == "ok"]
    assert ok and all(np.isfinite(r["forecast_rmse"]) for r in ok)
    assert {r["n_net_ok"] for r in rows[1:] if r["strategy"] == "random"} <= {0, X.N_RANDOM}
    # a reference row equals the E07 machinery (MAP over 24 h, X.score) on the same draws
    sigma = X.site_sigma(sc, False)
    Sd = [int(i) for i in next(d["sites"] for d in drows if d["strategy"] == "source_near" and d["k"] == 4).split()]
    a_ref, c_bg = sc.cfg.sources.reference_rate_ug_s, sc.cfg.sources.background_ug_m3
    A = a_ref * G3[Sd] / sigma[Sd][:, None, None]
    issue, n_h = X.issue_times(sc), len(sc.truth.winds)
    w = np.ones(sc.truth.field.shape[1:])
    m = [X.score(X.estimate_series(A, (X.readings(sc, sigma, "white", r, n_h)[Sd] - c_bg) / sigma[Sd][:, None], issue),
                 F, sc.truth.field, issue, sc.truth.strengths_ug_s, a_ref, c_bg, w)["forecast_rmse"] for r in range(2)]
    row = next(r for r in rows if r["strategy"] == "source_near" and r["k"] == 4 and r["arm"] == "w1" and r["schedule"] == "S4")
    assert row["forecast_rmse"] == pytest.approx(np.mean(m), rel=1e-9)
    # common random numbers: w5 noise is exactly 5x the w1 noise
    y1, y5 = (X.readings(sc, sigma * f, "white", 0, 10) for f in (1, 5))
    st = sc.truth.site_truth[:, :10]
    assert np.allclose(y5 - st, 5 * (y1 - st))


def test_growing_window_equals_windowed_over_same_span():
    rng = np.random.default_rng(3)
    A, z = rng.normal(size=(3, 60, 2)), rng.normal(size=(3, 60))
    m = S.schedule_mask(60, 6, 3)
    full = X.estimate_series(A, z, np.array([23, 40]), None, mask=m)
    assert np.allclose(full[0], X.estimate_series(A, z, np.array([23]), 24, mask=m)[0])  # span = all past data at hour 23
    assert np.allclose(full[1], X.estimate_series(A, z, np.array([40]), 41, mask=m)[0])
    assert not np.allclose(full[1], X.estimate_series(A, z, np.array([40]), 24, mask=m)[0])


def test_exclusion_and_full_window_variants_run():
    D.E7.SCALES["tiny8"] = dict(grid=(16, 16), dt=300, scheme="upwind", spinup=6, split=((1, 1), (2, 2), (3, 4)), Ks=(2,),
                                ks=(2, 4), seeds=(0,), R=2)
    sc = B.generate_scenario(D.E7.make_cfg("tiny8", "mixed"), 0, 2)
    F, G3 = X.responses(sc)
    r = 1200.0
    keep = X.exclude_near_sources(sc.site_xy, sc.admissible_sites(), sc.truth.source_xy, r)
    assert 0 < len(keep) < len(sc.admissible_sites())
    rows, drows, _ = S.evaluate(sc, F, G3, (2, 4), 2, schedules={"S4": (24, 24)}, exclude_m=r, full_window=True)
    used = {int(i) for d in drows if d.get("sites") for i in d["sites"].split()}
    assert used and used <= set(keep.tolist())
    assert all(np.isfinite(x["forecast_rmse"]) for x in rows if x["status"] == "ok")

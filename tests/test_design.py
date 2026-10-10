"""Tests for design.py.  Real forward solves on a tiny 20x20 problem (T=3600, output 600, dt 120, 3 sources)."""
import itertools

import numpy as np
import pandas as pd
import pytest

from pravah import design
from pravah.config import Domain, GaussianSource, Scenario, Times, Transport, WindSegment
from pravah.design import (Candidates, Constraints, DesignOptions, InfeasibleError, assert_disjoint, build_case,
                           candidate_grid, evaluate_layouts, feasible_sites, fit_layout, greedy_trace,
                           paired_comparison, random_layouts, score_layout, select_layout, spread_layout)
from pravah.forward import simulate
from pravah.inverse import InferenceOptions, ResponseCache, fit_sources, reconstruct_linear
from pravah.metrics import field_mse, population_weight
from pravah.observations import Observations, SensorLayout, _wrap

SRC = (GaussianSource("S1", 2500.0, 2500.0, 600.0), GaussianSource("S2", 4000.0, 6000.0, 600.0),
       GaussianSource("S3", 6500.0, 3500.0, 600.0))
DOM = Domain(10_000.0, 10_000.0, 20, 20)


def scen(u=1.0, v=0.3, sources=SRC) -> Scenario:
    return Scenario(domain=DOM, sources=sources, transport=Transport(wind=(WindSegment(0.0, u, v),)),
                    times=Times(duration=3600.0, dt=120.0, output_interval=600.0))


INF = scen()
CANDS = candidate_grid(DOM, 3)
OPT = DesignOptions(noise_std=0.05, alpha=0.0)
Q1, Q2, Q3, Q4 = (np.array(q) for q in ([1e5, 1.5e5, 8e4], [1.4e5, 0.9e5, 1.2e5], [0.7e5, 1.8e5, 1.0e5],
                                         [1.1e5, 1.1e5, 0.6e5]))


@pytest.fixture(scope="module")
def cache():
    return ResponseCache()


@pytest.fixture(scope="module")
def cases(cache):
    mk = lambda name, role, sc, q, seed: build_case(name, role, sc, q, INF, CANDS, OPT, seed, 3, cache)  # noqa: E731
    return {
        "d1": mk("d1", "design", INF, Q1, 11),
        "d2": mk("d2", "design", scen(0.7, 0.6), Q2, 12),  # truth wind differs from the inference wind
        "t1": mk("t1", "test", scen(1.2, 0.0), Q3, 101),
        "t2": mk("t2", "test", INF, Q4, 102),
    }


# ---------------------------------------------------------------- candidates / constraints
def test_candidate_grid_default_and_hull():
    c = candidate_grid(Domain())
    assert len(c) == 100 and c.x.min() == 500.0 and c.x.max() == 9500.0 and c.y.min() == 500.0
    assert len(set(c.ids)) == 100
    with pytest.raises(ValueError):
        candidate_grid(Domain(nx=50, ny=50), 60)  # first site 83 m < dx/2


def test_feasible_sites_and_min_separation():
    k = Constraints(allowed=np.array([True] * 8 + [False]), min_separation=3000.0)
    assert 8 not in feasible_sites(CANDS, k)
    f = feasible_sites(CANDS, k, (0,))
    assert 0 not in f and 1 in f and 4 in f  # spacing 3333 m >= 3000: neighbours stay feasible
    # min_separation 3500 excludes the direct neighbours (3333 m) but not the diagonal (4714 m)
    f2 = feasible_sites(CANDS, Constraints(min_separation=3500.0), (0,))
    assert set(f2) == {2, 4, 5, 6, 7, 8}


def test_layouts_cardinality_uniqueness_separation():
    k = Constraints(allowed=np.array([True] * 8 + [False]), min_separation=3500.0)
    for lay in random_layouts(CANDS, 3, 30, k, seed=1) + [spread_layout(CANDS, 3, k)]:
        assert lay == tuple(sorted(set(lay))) and len(lay) == 3 and 8 not in lay
        for i, j in itertools.combinations(lay, 2):
            assert np.hypot(CANDS.x[i] - CANDS.x[j], CANDS.y[i] - CANDS.y[j]) >= 3500.0


def test_infeasible_raises():
    k = Constraints(allowed=np.array([True] * 4 + [False] * 5))
    with pytest.raises(InfeasibleError):
        random_layouts(CANDS, 5, 2, k, 0)
    with pytest.raises(InfeasibleError):
        spread_layout(CANDS, 5, k)
    with pytest.raises(InfeasibleError):  # at most 4 sites (the corners) are 6000 m apart... 5 is impossible
        spread_layout(CANDS, 5, Constraints(min_separation=6000.0))
    with pytest.raises(InfeasibleError):
        random_layouts(CANDS, 2, 2, Constraints(min_separation=1e5), 0)
    assert issubclass(InfeasibleError, ValueError)


def test_random_deterministic_and_seed_dependent():
    a = random_layouts(CANDS, 3, 10, Constraints(), seed=5)
    assert a == random_layouts(CANDS, 3, 10, Constraints(), seed=5)
    assert a != random_layouts(CANDS, 3, 10, Constraints(), seed=6)


def test_spread_deterministic_and_first_site_is_centre():
    lay = spread_layout(CANDS, 1)
    assert lay == (4,)  # centre of the 3x3 grid
    assert spread_layout(CANDS, 5) == spread_layout(CANDS, 5)
    # second point: farthest from the centre = a corner, ties -> lowest index (0)
    assert spread_layout(CANDS, 2) == (0, 4)
    # when the centre is not allowed the start is the lowest-index among the equidistant nearest sites
    k = Constraints(allowed=np.array([i != 4 for i in range(9)]))
    assert spread_layout(CANDS, 1, k) == (1,)


# ---------------------------------------------------------------- fast score vs brute force
@pytest.mark.parametrize("name", ["d1", "d2", "t1"])
@pytest.mark.parametrize("layout", [(4,), (0, 8), (1, 3, 5), (0, 2, 6, 8)])
@pytest.mark.parametrize("weighted", [False, True])
def test_fast_fit_matches_brute_force(cache, cases, name, layout, weighted):
    opt = DesignOptions(noise_std=0.05, alpha=0.01, prior=(5e4, 5e4, 5e4)) if not weighted else \
        DesignOptions(noise_std=0.05, weight=population_weight(DOM), t_exclude=1200.0)
    sc_truth = {"d1": INF, "d2": scen(0.7, 0.6), "t1": scen(1.2, 0.0)}[name]
    q = {"d1": Q1, "d2": Q2, "t1": Q3}[name]
    case = build_case(name, "design", sc_truth, q, INF, CANDS, opt, {"d1": 11, "d2": 12, "t1": 101}[name], 3, cache)
    rep = 1
    fast = fit_layout(case, layout, rep, opt)
    # brute force: Observations with exactly the same noise, fit_sources, direct field_mse
    y = case.truth_readings[list(layout)] + opt.noise_std * case.noise[rep, list(layout)]
    sl = CANDS.sensor_layout(layout)
    obs = Observations(_wrap(y.T, sl, np.array(case.obs_times)), np.full(len(layout), opt.noise_std), sl,
                       case.obs_times)
    fit = fit_sources(INF, obs, InferenceOptions(alpha=opt.alpha, prior=opt.prior), cache)
    truth = simulate(sc_truth, q).concentration
    brute = field_mse(reconstruct_linear(cache.get(INF), fit), truth, opt.weight, opt.t_exclude)
    assert np.allclose(fast["q_hat"], fit.q_hat, rtol=1e-8, atol=1e-6)
    assert abs(fast["mse_w"] - brute) <= 1e-8 * brute
    assert fast["rank"] == fit.rank and fast["effective_rank"] == fit.effective_rank
    # and the exact quadratic form agrees with the brute-force metric for an arbitrary q
    qq = np.array([3e4, 2e5, 1e4])
    assert abs(case.field_error(qq) - field_mse(cache.get(INF).field(qq), truth, opt.weight, opt.t_exclude)) \
        <= 1e-10 * case.field_error(qq)


def test_unmodelled_source_and_heldout(cache):
    extra = SRC + (GaussianSource("X", 7500.0, 7500.0, 600.0),)
    opt = DesignOptions(noise_std=0.05, eval_sites=Candidates(("e0", "e1"), np.array([2000.0, 7000.0]),
                                                              np.array([8000.0, 2000.0])))
    q = np.array([1e5, 1e5, 1e5, 5e4])
    case = build_case("x", "test", scen(sources=extra), q, INF, CANDS, opt, 7, 2, cache)
    f = fit_layout(case, (0, 4, 8), 0, opt)
    assert f["unmodelled_source_mass"] == pytest.approx(5e4 / 3.5e5)
    assert np.isfinite(f["heldout_rmse"]) and f["heldout_rmse"] > 0
    assert len(f["q_hat"]) == 3 and f["source_rel_err"] > 0
    with pytest.raises(ValueError):
        fit_layout(case, (0,), 0, DesignOptions(noise_std=0.05))  # build-time options differ
    with pytest.raises(ValueError):  # truth with fewer sources than the model
        build_case("y", "test", scen(sources=SRC[:2]), q[:2], INF, CANDS, opt, 7, 2, cache)


def test_time_grid_mismatch_raises(cache):
    other = Scenario(domain=DOM, sources=SRC, transport=Transport(),
                     times=Times(duration=3600.0, dt=120.0, output_interval=300.0))
    with pytest.raises(ValueError, match="time"):
        build_case("z", "design", other, Q1, INF, CANDS, OPT, 1, 1, cache)


# ---------------------------------------------------------------- selection
def mean_mse(lay, dcs):
    return np.mean([np.mean([fit_layout(c, lay, r, OPT)["mse_w"] for r in range(c.n_noise)]) for c in dcs])


def test_greedy_matches_exhaustive_steps(cases):
    dcs = [cases["d1"], cases["d2"]]
    tr = greedy_trace(CANDS, 3, dcs, Constraints(), OPT)
    singles = [mean_mse((i,), dcs) for i in range(9)]
    assert tr.order[0] == int(np.argmin(singles))
    assert tr.scores[0] == pytest.approx(min(singles), rel=1e-12)
    ext = [mean_mse(tuple(sorted((tr.order[0], i))), dcs) if i != tr.order[0] else np.inf for i in range(9)]
    assert tr.order[1] == int(np.argmin(ext))
    assert tr.layout == tuple(sorted(tr.order)) and len(tr.scores) == 3
    assert select_layout(CANDS, 3, dcs, Constraints(), OPT) == tr.layout
    # the optimum by enumeration: only REPORT the gap
    best = min(itertools.combinations(range(9), 3), key=lambda l: mean_mse(l, dcs))
    print(f"\ngreedy {tr.layout} score {tr.scores[-1]:.6g}; exhaustive optimum {best} {mean_mse(best, dcs):.6g}")
    assert tr.scores[-1] >= mean_mse(best, dcs) * (1 - 1e-12)


def test_greedy_respects_constraints(cases):
    k = Constraints(allowed=np.array([True] * 8 + [False]), min_separation=3500.0)
    lay = select_layout(CANDS, 3, [cases["d1"]], k, OPT)
    assert 8 not in lay and len(lay) == 3
    for i, j in itertools.combinations(lay, 2):
        assert np.hypot(CANDS.x[i] - CANDS.x[j], CANDS.y[i] - CANDS.y[j]) >= 3500.0


def test_role_guard_and_disjointness(cases, cache):
    with pytest.raises(ValueError, match="leakage"):
        select_layout(CANDS, 2, [cases["d1"], cases["t1"]], Constraints(), OPT)
    with pytest.raises(ValueError, match="leakage"):
        score_layout((0,), [cases["t1"]], OPT)
    with pytest.raises(ValueError, match="leakage"):
        evaluate_layouts({"m": [(0,)]}, [cases["d1"]], OPT)
    assert_disjoint([cases["d1"], cases["d2"]], [cases["t1"], cases["t2"]])
    clone = build_case("t_clone", "test", INF, Q1, INF, CANDS, OPT, 999, 1, cache)  # same truth as d1
    with pytest.raises(ValueError, match="fingerprint"):
        assert_disjoint([cases["d1"]], [clone])
    clone = build_case("t_seed", "test", INF, Q4, INF, CANDS, OPT, 11, 1, cache)
    with pytest.raises(ValueError, match="seed"):
        assert_disjoint([cases["d1"]], [clone])
    clone = build_case("d1", "test", INF, Q4, INF, CANDS, OPT, 55, 1, cache)
    with pytest.raises(ValueError, match="name"):
        assert_disjoint([cases["d1"]], [clone])


def test_selection_invariant_to_test_truth(cases, cache):
    dcs = [cases["d1"], cases["d2"]]
    lay = select_layout(CANDS, 2, dcs, Constraints(), OPT)
    # a very different test truth and other test cases do not enter selection at all
    other = build_case("t9", "test", scen(0.5, -0.8), 3 * Q3, INF, CANDS, OPT, 777, 2, cache)
    assert_disjoint(dcs, [other])
    evaluate_layouts({"g": [lay]}, [other], OPT)
    assert select_layout(CANDS, 2, dcs, Constraints(), OPT) == lay


def test_weighted_objective_changes_scores(cache, cases):
    w_opt = DesignOptions(noise_std=0.05, weight=population_weight(DOM))
    wc = build_case("d1", "design", INF, Q1, INF, CANDS, w_opt, 11, 3, cache)
    s0 = score_layout((0, 4), [cases["d1"]], OPT)["mse_w"]
    s1 = score_layout((0, 4), [wc], w_opt)["mse_w"]
    assert s0 != s1 and np.isfinite(s1)
    assert len(select_layout(CANDS, 2, [wc], Constraints(), w_opt)) == 2


def test_score_layout_structure(cases):
    s = score_layout((1, 4), [cases["d1"], cases["d2"]], OPT)
    assert set(s["per_case"]) == {"d1", "d2"}
    assert s["mse_w"] == pytest.approx(np.mean(list(s["per_case"].values())))
    assert s["rmse"] == pytest.approx(np.sqrt(s["mse_w"]))


# ---------------------------------------------------------------- evaluation
def test_evaluate_and_paired(cases):
    lays = {"spread": [spread_layout(CANDS, 2), spread_layout(CANDS, 3)],
            "random": random_layouts(CANDS, 2, 4, Constraints(), 3)}
    df = evaluate_layouts(lays, [cases["t1"], cases["t2"]], OPT, n_noise=2)
    assert list(df.columns) == design.COLUMNS
    assert set(df["method"]) == {"spread", "random"} and set(df["budget"]) == {2, 3}
    assert len(df) == (2 + 4) * 2 * 2
    assert df["sensors"].iloc[0].startswith("c")
    diffs, summ = paired_comparison(df, "spread", "random", "mse_w")
    assert list(summ.columns) == ["budget", "mean_diff", "sd_diff", "n", "frac_a_better"]
    assert list(summ["budget"]) == [2] and int(summ["n"].iloc[0]) == 1 * 4 * 2 * 2  # 1 spread layout x 4 random
    assert np.allclose(diffs["diff"], diffs["mse_w_a"] - diffs["mse_w_b"])
    dist = design.random_distribution(df, "random")
    assert int(dist["n_layouts"].iloc[0]) == 4 and dist["min"].iloc[0] <= dist["median"].iloc[0] <= dist["max"].iloc[0]
    assert isinstance(dist, pd.DataFrame)

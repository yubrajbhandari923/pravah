"""Contract for pravah.experiments.cross_objective (E07; docs/research/cross-objective-design.md)."""
import dataclasses

import numpy as np
import pytest

from pravah import benchmarks as B
from pravah import placement as P
from pravah.experiment_config import ExperimentConfig, Split, Winds
from pravah.experiments import cross_objective as X


@pytest.fixture(scope="module")
def tiny():
    c = ExperimentConfig.default("EXP-E07-test")
    c.model.grid, c.model.spinup_h = (16, 16), 6
    c.design.n_sources = (2,)
    c.split = Split(train=(1, 1), validation=(2, 2), test=(3, 4))
    c.winds = Winds(train=["veering"], heldout=c.winds.heldout)
    sc = B.generate_scenario(c, 0, 2)
    F, G3 = X.responses(sc)
    return sc, F, G3


def _all_designs(sc, G3, sigma, ks=(2, 4)):
    p = X.problem(X.design_input(sc, G3, sigma))
    T, seed, out = len(sc.split_hours("train")), sc.cfg.seeds.streams(0)["placement"], {}
    for k in ks:
        for name, net in [("random", 0), ("random", 1)] + [(n, 0) for n in X.STRATEGIES]:
            try:
                out[(k, name, net)] = X.make_design(name, p, k, T, seed, net)[0]
            except ValueError as e:
                out[(k, name, net)] = str(e)
    return p, out


def test_responses_superpose_to_truth(tiny):
    sc, F, G3 = tiny
    rec = sc.cfg.sources.background_ug_m3 + np.einsum("k,khxy->hxy", sc.truth.strengths_ug_s, F)
    assert np.abs(rec - sc.truth.field).max() < 1e-3 * np.abs(sc.truth.field - 5).max()
    assert G3.shape == (sc.n_candidate_sites, len(sc.truth.winds), 2)


@pytest.mark.parametrize("per_site", [False, True])
def test_designs_k_distinct_admissible_spaced(tiny, per_site):
    sc, F, G3 = tiny
    p, des = _all_designs(sc, G3, X.site_sigma(sc, per_site))
    adm = set(sc.admissible_sites().tolist())
    for (k, name, _), S in des.items():
        if isinstance(S, str):
            assert name == "podqr" and "undefined" in S  # only podqr may be undefined, and says why
            continue
        assert len(S) == k == len(set(S)) and set(S) <= adm
        d = np.hypot(*(sc.site_xy[S][:, None] - sc.site_xy[S][None]).transpose(2, 0, 1)) + np.eye(k) * 1e9
        assert d.min() >= sc.cfg.design.min_sensor_separation_m
    assert des[(4, "random", 0)] != des[(4, "random", 1)]  # distinct random networks


def test_design_input_holds_no_truth_or_heldout(tiny):
    sc, F, G3 = tiny
    di = X.design_input(sc, G3, X.site_sigma(sc, False))
    assert {f.name for f in dataclasses.fields(di)} == {"xy", "cand", "min_sep", "a_ref", "sigma", "G3_train", "source_xy",
                                                        "placement_seed", "prior_loguniform"}
    assert di.G3_train.shape[1] == len(sc.split_hours("train"))


def test_designs_unchanged_when_truth_and_heldout_poisoned(tiny):
    sc, F, G3 = tiny
    sigma = X.site_sigma(sc, False)
    _, ref = _all_designs(sc, G3, sigma)
    G3p = G3.copy()
    G3p[:, len(sc.split_hours("train")):] = np.nan  # validation/test-hour sensitivities
    nan = lambda a: np.full_like(np.asarray(a, float), np.nan)
    sc2 = B.Scenario(sc.cfg, sc.replicate, sc.n_sources,
                     dataclasses.replace(sc.truth, field=nan(sc.truth.field), strengths_ug_s=nan(sc.truth.strengths_ug_s),
                                         site_truth=nan(sc.truth.site_truth)),
                     sc.site_ij, sc.site_xy, sc.noise_sigma, {k: nan(v) for k, v in sc._noise.items()}, sc.lattice_n)
    _, got = _all_designs(sc2, G3p, sigma)
    assert got == ref


def test_estimator_ignores_future_and_matches_closed_form(tiny):
    sc, F, G3 = tiny
    sigma, a_ref = X.site_sigma(sc, False), sc.cfg.sources.reference_rate_ug_s
    S = [int(i) for i in sc.admissible_sites()[:3]]
    A = a_ref * G3[S] / sigma[S][:, None, None]
    y = X.readings(sc, sigma, "white", 0, G3.shape[1])
    z = (y[S] - 5.0) / sigma[S][:, None]
    issue = X.issue_times(sc)
    u = X.estimate_series(A, z, issue)
    t, i0 = issue[3], 3
    z2 = z.copy()
    z2[:, t + 1:] = 1e6  # future readings: must not matter
    assert np.allclose(X.estimate_series(A, z2, issue[:i0 + 1])[i0], u[i0])
    h = np.arange(t - X.WINDOW_H + 1, t + 1)
    M = np.vstack([A[:, h].reshape(-1, 2), np.eye(2)])
    rhs = np.concatenate([z[:, h].ravel(), np.zeros(2)])
    assert np.allclose(u[i0], np.linalg.lstsq(M, rhs, rcond=None)[0])
    un = X.estimate_series(A, z, issue, nonneg=True)
    assert (un >= 0).all()
    from scipy.optimize import nnls
    assert np.allclose(un[i0], nnls(M, rhs)[0], atol=1e-8)


def test_paired_draws_and_reproducible_scores(tiny):
    sc, F, G3 = tiny
    sigma = X.site_sigma(sc, True)
    y1, y2 = X.readings(sc, sigma, "ar1", 2, G3.shape[1]), X.readings(sc, sigma, "ar1", 2, G3.shape[1])
    assert np.array_equal(y1, y2) and not np.array_equal(y1, X.readings(sc, sigma, "ar1", 3, G3.shape[1]))
    a_ref, issue, w = sc.cfg.sources.reference_rate_ug_s, X.issue_times(sc), X.pop_weight((16, 16), (1e4, 1e4))

    def run(S):
        A = a_ref * G3[S] / sigma[S][:, None, None]
        u = X.estimate_series(A, (y1[S] - 5.0) / sigma[S][:, None], issue)
        return X.score(u, F, sc.truth.field, issue, sc.truth.strengths_ug_s, a_ref, 5.0, w)

    S = [int(i) for i in sc.admissible_sites()[[0, 5, 9]]]
    assert run(S) == run(S)
    assert np.isclose(w.sum(), 1)
    # with the true strengths the forecast error is only the superposition residual
    u_true = np.tile(sc.truth.strengths_ug_s / a_ref, (len(issue), 1))
    assert X.score(u_true, F, sc.truth.field, issue, sc.truth.strengths_ug_s, a_ref, 5.0, w)["forecast_rmse"] < 1e-3


def test_issue_times_inside_test_after_warmup(tiny):
    sc, F, G3 = tiny
    te, it = sc.split_hours("test"), X.issue_times(sc)
    assert it[0] == te[0] + X.WARMUP_H and it[-1] + 1 == te[-1]


def test_pod_leading_run_and_swap_and_fieldvar():
    rng = np.random.default_rng(0)
    n, T, K = 30, 12, 3
    G = rng.normal(size=(n, T, K))
    xy = np.c_[np.arange(n) * 1000.0, np.zeros(n)]
    di = X.DesignInput(xy, np.arange(n), 1500.0, 1.0, np.full(n, 0.5), G, np.zeros((K, 2)), 7, (0.5, 5.0))
    p = X.problem(di)
    r = X.pod_leading_modes(p, T)
    assert 0 < r <= K * T
    S = P.greedy(p, 4, P.dopt)
    S2, v, _ = P.swap_refine(p, S, P.dopt)
    assert v >= P.dopt(p, S) - 1e-12 and len(set(S2)) == 4
    assert all(p.feasible(i, [j for j in S2 if j != i]) for i in S2)
    M = np.eye(K) + sum(p.F[i] for i in S)
    assert np.isclose(P.fieldvar(p, S), -np.trace(np.linalg.inv(M) @ p.Q))
    assert P.fieldvar(p, S) >= -np.trace(p.Q)  # data never increase posterior variance


def test_stats_helpers():
    assert np.allclose(X.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert X.sign_test([1, 2, -1, 0])[:3] == (2, 1, 1)
    lo, hi = X.boot_median_ci([-np.inf, 1, 2, 3, 4])
    assert lo <= 2 <= hi

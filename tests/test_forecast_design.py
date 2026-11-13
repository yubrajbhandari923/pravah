"""Contract for pravah.experiments.forecast_design (E11; docs/research/forecast-design-placement.md)."""
import dataclasses

import numpy as np
import pytest

from pravah import benchmarks as B
from pravah import filter as KF
from pravah import placement as P
from pravah.experiment_config import ExperimentConfig, Split, Winds
from pravah.experiments import cross_objective as X
from pravah.experiments import forecast_design as FD


@pytest.fixture(scope="module")
def tiny():
    c = ExperimentConfig.default("EXP-E11-test")
    c.model.grid, c.model.spinup_h = (16, 16), 6
    c.design.n_sources = (2,)
    c.split = Split(train=(1, 2), validation=(3, 3), test=(4, 5))
    c.winds = Winds(train=["veering"], heldout=c.winds.heldout)
    sc = B.generate_scenario(c, 0, 2)
    F, G3 = X.responses(sc)
    dw = FD.design_winds()
    Frob = np.stack([X.source_field(c, xy, dw, len(dw)) for xy in sc.truth.source_xy])
    return sc, F, G3, Frob, X.site_responses(sc, Frob)


def _setup(sc, F, G3, Frob, G3rob):
    di = X.design_input(sc, G3, X.site_sigma(sc, False))
    p = X.problem(di)
    return p, FD.Objective(p, FD.forecast_input(di, F, G3rob, Frob))


def _designs(sc, F, G3, Frob, G3rob, k=3):
    p, obj = _setup(sc, F, G3, Frob, G3rob)
    return {n: FD.make_fc_design(n, p, obj, k)[0] for n in FD.FC}


def test_design_winds_directions():
    w = FD.design_winds()
    assert w.shape == (FD.N_DIRS * FD.BLOCK_H, 2) and np.allclose(np.hypot(*w.T), FD.PRIOR_SPEED)
    assert np.allclose(w[0], [0, -3]) and np.allclose(w[4 * FD.BLOCK_H], [0, 3])  # from N blows south; from S blows north


def test_expected_mse_equals_kalman_covariance_and_monte_carlo():
    rng = np.random.default_rng(0)
    k, T, K, N = 3, 30, 2, 5
    A = rng.normal(0, .7, (k, T, K))
    Fr = rng.normal(0, 1, (K, T, N))
    Q = np.einsum("khn,lhn->hkl", Fr, Fr) / N
    C = FD._cum(A).sum(0)
    issue, hs = np.arange(10, 25), (1, 4)
    # data-free KF recursion on the static state a (P0 = I, R = 1): Sigma_t after assimilating hours 0..t
    Pt, Sig = np.eye(K), []
    for h in range(T):
        _, Pt, *_ = KF.update(np.zeros((K, 1)), Pt, np.zeros((k, 1)), A[:, h], 1.0)
        Sig.append(Pt)
    ref = np.mean([np.mean([np.trace(Q[t + h] @ Sig[t]) for t in issue if t + h <= T - 1]) for h in hs])
    assert np.isclose(FD.expected_mse(C, Q, issue, hs), ref, rtol=1e-10)
    # Monte Carlo: a ~ N(0, I), white unit noise, growing-window MAP forecast error
    n = 4000
    a = rng.normal(size=(K, n))
    z = np.einsum("shk,kn->shn", A, a) + rng.normal(size=(k, T, n))
    u = FD.estimate(A, z, issue)  # (n_issue, K, n)
    mse = np.mean([np.mean([np.mean((np.einsum("kn,kc->nc", Fr[:, t + h], u[i] - a)) ** 2) for i, t in enumerate(issue)
                            if t + h <= T - 1]) for h in hs])
    assert abs(mse / ref - 1) < 0.05


def test_objectives_match_explicit_formulas(tiny):
    sc, F, G3, Frob, G3rob = tiny
    p, obj = _setup(sc, F, G3, Frob, G3rob)
    S = [int(i) for i in p.cand[[0, 7, 20]]]
    A = p.A3[S]
    T, K = obj.T, obj.K
    Q = FD.field_weights(F[:, :T], p.a_ref)
    Sig = lambda h0, h1: np.linalg.inv(np.eye(K) + np.einsum("shk,shl->kl", A[:, h0:h1], A[:, h0:h1]))
    grow = np.mean([np.mean([np.trace(Q[t + h] @ Sig(0, T)) for t in range(T - h)]) for h in FD.HORIZONS])
    assert np.isclose(-obj.grow(p, S), grow)
    w24 = np.mean([np.mean([np.trace(Q[t + h] @ Sig(t - 23, t + 1)) for t in range(23, T - h)]) for h in FD.HORIZONS])
    assert np.isclose(-obj.w24(p, S), w24)
    Ar = G3rob[S] * p.a_ref / X.site_sigma(sc, False)[S][:, None, None]
    Qr = FD.field_weights(Frob, p.a_ref)
    blk = [slice(d * 24, d * 24 + 24) for d in range(FD.N_DIRS)]
    Md = [np.einsum("shk,shl->kl", Ar[:, b], Ar[:, b]) for b in blk]
    Mtr = np.einsum("shk,shl->kl", A, A)
    rob = np.mean([np.trace(Qr[b].mean(0) @ np.linalg.inv(np.eye(K) + Mtr + M)) for b, M in zip(blk, Md)])
    assert np.isclose(-obj.grow_rob(p, S), rob)
    rob24 = np.mean([np.trace(Qr[b].mean(0) @ np.linalg.inv(np.eye(K) + M)) for b, M in zip(blk, Md)])
    assert np.isclose(-obj.w24_rob(p, S), rob24)


def test_greedy_matches_brute_force_on_small_set(tiny):
    sc, F, G3, Frob, G3rob = tiny
    p, obj = _setup(sc, F, G3, Frob, G3rob)
    ps = dataclasses.replace(p, cand=p.cand[::3][:14], _pod={})
    for name in FD.FC:
        v = obj.value(name)
        best1 = P.brute_force(ps, 1, v)[0]
        assert np.isclose(v(ps, P.greedy(ps, 1, v)), best1)
        best2 = P.brute_force(ps, 2, v)[0]
        S, info = FD.make_fc_design(name, ps, obj, 2)
        assert info["refined_value"] >= info["greedy_value"] - 1e-12
        assert np.isclose(v(ps, S), best2, rtol=1e-9), name


def test_designs_unchanged_when_nontraining_info_poisoned(tiny):
    sc, F, G3, Frob, G3rob = tiny
    ref = _designs(sc, F, G3, Frob, G3rob)
    T = len(sc.split_hours("train"))
    G3p, Fp = G3.copy(), F.copy()
    G3p[:, T:], Fp[:, T:] = np.nan, np.nan  # validation/test-hour sensitivities and fields
    nan = lambda a: np.full_like(np.asarray(a, float), np.nan)
    sc2 = B.Scenario(sc.cfg, sc.replicate, sc.n_sources,
                     dataclasses.replace(sc.truth, field=nan(sc.truth.field), strengths_ug_s=nan(sc.truth.strengths_ug_s),
                                         site_truth=nan(sc.truth.site_truth)),
                     sc.site_ij, sc.site_xy, sc.noise_sigma, {k: nan(v) for k, v in sc._noise.items()}, sc.lattice_n)
    sig = X.site_sigma(sc, False)  # sigma uses training truth (E07 protocol sigma); pass it as E07's test does
    di = X.design_input(sc2, G3p, sig)
    p = X.problem(di)
    obj = FD.Objective(p, FD.forecast_input(di, Fp, G3rob, Frob))
    assert {n: FD.make_fc_design(n, p, obj, 3)[0] for n in FD.FC} == ref
    assert len(set(map(tuple, ref.values()))) > 1  # the objectives are not all the same design


def test_estimator_memories_causal_and_window24_equals_e07(tiny):
    sc, F, G3, *_ = tiny
    sigma, a_ref = X.site_sigma(sc, False), sc.cfg.sources.reference_rate_ug_s
    S = [int(i) for i in sc.admissible_sites()[:3]]
    A = a_ref * G3[S] / sigma[S][:, None, None]
    z = (X.readings(sc, sigma, "white", 0, G3.shape[1])[S] - 5.0) / sigma[S][:, None]
    issue = X.issue_times(sc)
    assert np.allclose(FD.estimate(A, z, issue, 24), X.estimate_series(A, z, issue))
    g = FD.estimate(A, z, issue)
    i0, t = 4, issue[4]
    z2 = z.copy()
    z2[:, t + 1:] = 1e6  # future readings must not matter
    assert np.allclose(FD.estimate(A, z2, issue, None)[i0], g[i0])
    h = np.arange(t + 1)
    M = np.vstack([A[:, h].reshape(-1, 2), np.eye(2)])
    assert np.allclose(g[i0], np.linalg.lstsq(M, np.concatenate([z[:, h].ravel(), np.zeros(2)]), rcond=None)[0])


def test_score_true_strengths_and_horizon_bounds(tiny):
    sc, F, *_ = tiny
    a_ref, issue = sc.cfg.sources.reference_rate_ug_s, X.issue_times(sc)
    T = len(sc.truth.winds)
    Fr = a_ref * F.reshape(2, T, -1)
    tf = sc.truth.field.reshape(T, -1) - 5.0
    a = sc.truth.strengths_ug_s / a_ref
    r = FD.score(np.tile(a, (len(issue), 1)), Fr, tf, issue, a, T - 1)
    assert r["source_rmse"] == 0 and r["forecast_rmse_h1"] < 1e-3 * np.abs(tf).max()
    assert r["n_issue_h24"] == int((issue + 24 <= T - 1).sum()) < r["n_issue_h1"] == len(issue)


def test_paired_summary_verdicts():
    assert FD.paired_summary(np.linspace(.1, .3, 12))["verdict"] == "better"
    assert FD.paired_summary(np.linspace(-.3, -.1, 12))["verdict"] == "worse"
    assert FD.paired_summary(np.linspace(-.03, .02, 12))["verdict"] == "non-inferior"

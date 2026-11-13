"""Contract tests for src/pravah/sensitivity.py and src/pravah/inverse.py (T-014). Written before the code.

Contract
sensitivity.build_response(forward, K) -> G (n_obs, K): column k = forward(e_k), e_k the k-th unit source vector.
    forward: callable, source coefficients (K,) -> readings (n_obs,). Units follow forward (here ug m^-3 per ug s^-1).
sensitivity.superposition_error(forward, q1, q2, alpha, beta) -> relative error
    ||H(a q1 + b q2) - a H(q1) - b H(q2)|| / ||a H(q1) + b H(q2)||. Meaningful only for ONE fixed physics (same wind,
    K, lambda, grid, source shapes, release history); a forward that changes physics between calls must NOT pass.
sensitivity.scenario_response_matrix(sc, hours) -> G (n_candidate_sites*hours, K), row order (site, hour), built from
    the same solver, winds and spin-up as the truth's anomaly (background excluded; it is a separate nuisance).
inverse.rank_info(G, tol=1e-8) -> dict(rank, cond, singular_values); rank uses tol * s_max; cond = inf if rank deficient.
inverse.solve_nnls(G, y, *, noise_cov=None, ridge=0.0, nuisance=None) -> dict(coef, nuisance_coef, residual_norm, rank, cond)
    minimises 1/2||W(y-Ga-Nd)||^2 + 1/2 lam ||a||^2 with lam = ridge*||WG||_2^2 (ridge is relative), a >= 0 (source coefficients), d sign-free (nuisance, e.g. a
    constant or time-varying background basis, NOT penalised, never part of `coef`).
    noise_cov: None (identity), scalar sigma, (n,) sigmas, or (n,n) covariance; W = R^{-1/2}.
    Errors: ValueError on shape mismatch, non-finite input, sigma <= 0, ridge < 0, non-SPD covariance.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

from pravah.inverse import rank_info, solve_nnls
from pravah.sensitivity import build_response, scenario_response_matrix, superposition_error

root = pathlib.Path(__file__).resolve().parents[1]

# ---- tiny hand-computed linear inverse: G a = y with a = (3, 1)
G0 = np.array([[2., 0.], [0., 1.], [1., 1.]])


def test_hand_computed_exact_recovery():
    r = solve_nnls(G0, np.array([6., 1., 4.]))
    assert np.allclose(r["coef"], [3., 1.], atol=1e-12) and r["residual_norm"] < 1e-12 and r["rank"] == 2


def test_hand_computed_weighted_scalar():
    # one unknown, y=(1,3), sigma=(1,2): weights 1, 1/4 -> (1*1 + .25*3)/1.25 = 1.4
    r = solve_nnls(np.ones((2, 1)), np.array([1., 3.]), noise_cov=np.array([1., 2.]))
    assert r["coef"][0] == pytest.approx(1.4, abs=1e-12)
    r2 = solve_nnls(np.ones((2, 1)), np.array([1., 3.]), noise_cov=np.diag([1., 4.]))  # same, as covariance
    assert r2["coef"][0] == pytest.approx(1.4, abs=1e-12)
    assert solve_nnls(np.ones((2, 1)), np.array([1., 3.]), noise_cov=7.0)["coef"][0] == pytest.approx(2.0)


def test_hand_computed_ridge():
    # G=[[1]], y=2: lam = ridge*||G||^2 = 1 -> a = 2/(1+1) = 1
    assert solve_nnls(np.array([[1.]]), np.array([2.]), ridge=1.0)["coef"][0] == pytest.approx(1.0, abs=1e-12)


# ---- NNLS respects constraints
def test_nonnegativity_and_kkt():
    rng = np.random.default_rng(1)
    G = np.abs(rng.normal(size=(30, 6)))
    y = G @ np.array([1, 0, 2, 0, 0, 3.]) + rng.normal(scale=0.5, size=30)
    a = solve_nnls(G, y)["coef"]
    assert (a >= 0).all()
    g = G.T @ (G @ a - y)  # KKT: g >= 0 everywhere, g = 0 where a > 0
    assert (g >= -1e-9).all() and np.abs(g[a > 0]).max() < 1e-8


def test_constraint_binds_against_unconstrained_ls():
    G, y = np.eye(2), np.array([1., -1.])
    assert np.allclose(solve_nnls(G, y)["coef"], [1., 0.])
    assert (solve_nnls(G, -np.ones(2))["coef"] == 0).all()


# ---- coherent and rank-deficient matrices
def test_rank_deficient_is_reported_and_nonunique():
    G = np.array([[1., 1.], [2., 2.]])
    y = np.array([1., 2.])
    info = rank_info(G)
    assert info["rank"] == 1 and info["cond"] == np.inf
    r = solve_nnls(G, y)
    assert r["rank"] == 1 and r["residual_norm"] < 1e-10 and (r["coef"] >= 0).all()
    assert np.allclose(G @ r["coef"], y)
    assert np.allclose(G @ np.array([0.5, 0.5]), y) and np.allclose(G @ np.array([1., 0.]), y)  # data cannot tell apart


def test_coherent_columns_noise_amplification_and_ridge():
    d = 1e-4
    G = np.array([[1., 1.], [1., 1. + d], [1., 1. - d]])
    a = np.array([1., 1.])
    assert rank_info(G)["cond"] > 1e3
    assert np.allclose(solve_nnls(G, G @ a)["coef"], a, atol=1e-6)  # noiseless: still recoverable
    y = G @ a + np.array([1e-3, -1e-3, 1e-3])
    plain = np.linalg.norm(solve_nnls(G, y)["coef"] - a)
    ridge = np.linalg.norm(solve_nnls(G, y, ridge=1e-3)["coef"] - a)
    assert ridge < plain  # coherence amplifies noise; ridge damps it (fixed data, deterministic)


# ---- source coefficients vs time-varying background nuisance
def test_nuisance_is_separate_from_source_coefficients():
    t = np.arange(12.)
    G = np.abs(np.random.default_rng(2).normal(size=(12, 2))) + np.outer(np.sin(t), [1., 0.5])
    G = np.abs(G)
    a = np.array([2., 1.])
    N = np.column_stack([np.ones(12), t])  # constant + linear-in-time background
    y = G @ a + N @ np.array([5., 0.3])
    r = solve_nnls(G, y, nuisance=N)
    assert r["coef"].shape == (2,) and np.allclose(r["coef"], a, atol=1e-8)
    assert np.allclose(r["nuisance_coef"], [5., 0.3], atol=1e-8)
    assert np.linalg.norm(solve_nnls(G, y)["coef"] - a) > 1e-2  # ignoring the nuisance biases the sources
    neg = solve_nnls(G, G @ a - 2.0 * N[:, 0], nuisance=N[:, :1])  # nuisance sign-free
    assert neg["nuisance_coef"][0] == pytest.approx(-2.0, abs=1e-8)


# ---- superposition only under fixed linear physics
def test_superposition_fixed_linear_physics_and_failure_cases():
    M = np.random.default_rng(3).normal(size=(5, 3))
    lin = lambda q: M @ q
    q1, q2 = np.array([1., 0., 2.]), np.array([0., 3., 1.])
    assert superposition_error(lin, q1, q2, 2.0, -0.5) < 1e-14
    assert np.allclose(build_response(lin, 3), M)
    calls = {"n": 0}

    def changing_physics(q):  # "wind" differs between calls: not one fixed linear operator
        calls["n"] += 1
        return (1 + 0.1 * calls["n"]) * (M @ q)
    assert superposition_error(changing_physics, q1, q2, 2.0, -0.5) > 1e-2
    assert superposition_error(lambda q: M @ q + 1.0, q1, q2, 1.0, 1.0) > 1e-2  # affine offset breaks superposition
    assert superposition_error(lambda q: M @ q**2, q1, q2, 1.0, 1.0) > 1e-2


# ---- errors
@pytest.mark.parametrize("kw", [dict(noise_cov=-1.0), dict(noise_cov=0.0), dict(ridge=-1.0),
                                dict(noise_cov=np.ones(2)), dict(noise_cov=np.array([[1., 2.], [2., 1.]])),
                                dict(nuisance=np.ones((2, 1)))])
def test_errors(kw):
    with pytest.raises(ValueError):
        solve_nnls(G0, np.array([6., 1., 4.]), **kw)


def test_nonfinite_and_shape_errors():
    with pytest.raises(ValueError):
        solve_nnls(G0, np.array([6., np.nan, 4.]))
    with pytest.raises(ValueError):
        solve_nnls(G0, np.array([6., 1.]))
    with pytest.raises(ValueError):
        build_response(lambda q: np.array([np.nan]), 2)


# ---- real solver: operator reproduces hidden truth anomaly (smoke scale), background excluded
def test_scenario_response_matches_truth():
    spec = importlib.util.spec_from_file_location("e02", root / "scripts" / "e02_examples.py")
    e02 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e02)
    sc = e02.build_examples(["separated2"])["separated2"]
    hrs = 3
    G = scenario_response_matrix(sc, hrs)
    assert G.shape == (sc.n_candidate_sites * hrs, 2) and (G >= -1e-12).all()
    y = (sc.truth.site_truth[:, :hrs] - sc.cfg.sources.background_ug_m3).ravel()
    pred = G @ sc.truth.strengths_ug_s
    assert np.linalg.norm(pred - y) / np.linalg.norm(y) < 1e-4  # derivation check saw 3.5e-6 at 24 h

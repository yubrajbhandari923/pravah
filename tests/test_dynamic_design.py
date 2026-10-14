"""Tests for pravah.dynamic_design: Gaussian linear-Bayes surrogate, exhaustive/greedy/swap search."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pravah.dynamic_design import DesignProblem, population_weights


def _problem(seed: int = 0, n_cand: int = 7, n_obs: int = 2, n_theta: int = 4, n_out: int = 6) -> DesignProblem:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n_cand, n_obs, n_theta))
    M = rng.normal(size=(n_out, n_theta))
    prior_var = rng.uniform(0.5, 2.0, size=n_theta)
    return DesignProblem(A_cand=A, M=M, prior_var=prior_var, sigma=0.3)


def _brute_risk(p: DesignProblem, subset: tuple[int, ...]) -> float:
    """Explicit posterior covariance by matrix inverse; no Woodbury."""
    a_s = p.A_cand[list(subset)].reshape(-1, p.A_cand.shape[2])
    prec = a_s.T @ a_s / p.sigma**2 + np.diag(1.0 / p.prior_var)
    sigma_post = np.linalg.inv(prec)
    return float(np.trace(p.M @ sigma_post @ p.M.T))


def _brute_information(p: DesignProblem, subset: tuple[int, ...]) -> float:
    a_s = p.A_cand[list(subset)].reshape(-1, p.A_cand.shape[2])
    prec = a_s.T @ a_s / p.sigma**2 + np.diag(1.0 / p.prior_var)
    return float(np.trace(np.linalg.inv(prec)))


def test_surrogate_matches_brute_force_posterior():
    p = _problem()
    for subset in [(0,), (1, 4), (0, 2, 3), (0, 1, 5, 6)]:
        np.testing.assert_allclose(p.risk(subset), _brute_risk(p, subset), rtol=1e-10)


def test_information_risk_matches_brute_force_trace():
    p = _problem(seed=3)
    for subset in [(2,), (0, 3), (1, 4, 6)]:
        np.testing.assert_allclose(p.information_risk(subset), _brute_information(p, subset), rtol=1e-10)


def test_empty_subset_is_prior_risk():
    p = _problem()
    np.testing.assert_allclose(p.risk(()), float(np.trace(p.M @ np.diag(p.prior_var) @ p.M.T)), rtol=1e-12)


def test_risk_batch_equals_looped_risk():
    p = _problem(seed=1)
    subsets = np.array(list(itertools.combinations(range(p.n_cand), 3)), dtype=np.intp)
    batch = p.risk_batch(subsets)
    looped = np.array([p.risk(tuple(s)) for s in subsets])
    np.testing.assert_allclose(batch, looped, rtol=1e-12)


def test_exhaustive_equals_manual_enumeration_and_bounds_greedy():
    p = _problem(seed=2, n_cand=8)
    k = 3
    manual = {c: _brute_risk(p, c) for c in itertools.combinations(range(p.n_cand), k)}
    best_c = min(manual, key=manual.get)
    res = p.exhaustive(k, return_all=True)
    assert res.best_subset == best_c
    np.testing.assert_allclose(res.best_risk, manual[best_c], rtol=1e-10)
    np.testing.assert_allclose(res.median_risk, np.median(list(manual.values())), rtol=1e-10)
    np.testing.assert_allclose(res.worst_risk, max(manual.values()), rtol=1e-10)
    assert res.n_evaluated == len(manual)
    assert np.all(np.diff(res.risks_sorted) >= 0)
    _, greedy_risk = p.greedy(k)
    assert res.best_risk <= greedy_risk + 1e-12


def test_swap_refine_never_increases_and_terminates():
    p = _problem(seed=4, n_cand=10)
    start = (0, 1, 2)
    start_risk = p.risk(start)
    prev = start_risk
    for max_iter in range(0, 6):
        subset, val = p.swap_refine(start, max_iter=max_iter)
        assert len(subset) == len(start) and len(set(subset)) == len(start)
        assert val <= prev + 1e-12
        assert val <= start_risk + 1e-12
        np.testing.assert_allclose(val, p.risk(subset), rtol=1e-12)
        prev = val
    # Unbounded run must terminate (at a local minimum for single swaps).
    subset, val = p.swap_refine(start, max_iter=10_000)
    assert p.swap_refine(subset, max_iter=10_000)[0] == subset


def test_population_weights_sum_to_one_and_zero_population_gets_zero():
    counts = np.array([[1.0, 0.0, 2.0], [3.0, 0.0, 4.0], [0.0, 0.0, 5.0]])
    w = population_weights(counts, times=[0.0, 1.0, 3.0])
    np.testing.assert_allclose(w.sum(), 1.0, rtol=0, atol=1e-15)
    assert np.all(w[:, 1] == 0.0)
    assert np.all(w >= 0.0)


def test_trapezoid_time_weights_are_applied():
    w = population_weights(np.ones((3, 1)), times=[0.0, 1.0, 3.0])
    np.testing.assert_allclose(w[:, 0], np.array([0.5, 1.5, 1.0]) / 3.0, rtol=1e-14)
    np.testing.assert_allclose(population_weights(np.ones(4))[:], np.full(4, 0.25))


def test_population_weights_reject_all_zero_and_bad_times():
    with pytest.raises(ValueError):
        population_weights(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        population_weights(np.ones((2, 2)), times=[1.0, 1.0])


def test_adding_a_sensor_never_increases_risk():
    p = _problem(seed=5, n_cand=9)
    rng = np.random.default_rng(11)
    for _ in range(25):
        k = int(rng.integers(0, 5))
        subset = tuple(sorted(rng.choice(p.n_cand, size=k, replace=False).tolist()))
        for j in range(p.n_cand):
            if j in subset:
                continue
            assert p.risk(tuple(sorted(subset + (j,)))) <= p.risk(subset) + 1e-12
            assert p.information_risk(tuple(sorted(subset + (j,)))) <= p.information_risk(subset) + 1e-12


def test_no_global_rng_use():
    p = _problem(seed=6)
    np.random.seed(0)
    first = p.exhaustive(2).best_subset, p.risk((0, 1))
    np.random.seed(999)
    second = p.exhaustive(2).best_subset, p.risk((0, 1))
    assert first == second


def test_chunking_gives_identical_results():
    p = _problem(seed=7, n_cand=8)
    subsets = np.array(list(itertools.combinations(range(p.n_cand), 3)), dtype=np.intp)
    np.testing.assert_array_equal(p.risk_batch(subsets, chunk=1), p.risk_batch(subsets, chunk=10**6))
    a = p.exhaustive(3, chunk=5)
    b = p.exhaustive(3, chunk=100_000)
    assert a.best_subset == b.best_subset
    assert a.best_risk == b.best_risk
    assert a.median_risk == b.median_risk
    assert a.worst_risk == b.worst_risk


def test_invalid_inputs_are_rejected():
    p = _problem()
    with pytest.raises(ValueError):
        p.risk((0, 0))
    with pytest.raises(ValueError):
        p.risk((p.n_cand,))
    with pytest.raises(ValueError):
        p.exhaustive(0)
    with pytest.raises(ValueError):
        p.risk_batch(np.zeros((1, 2), dtype=np.intp), chunk=0)

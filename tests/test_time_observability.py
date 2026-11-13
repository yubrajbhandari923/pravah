"""Frozen-style oracles for src/pravah/temporal_basis.py (T-028). Toy linear-time-invariant kernels; no FiPy solve."""
import numpy as np
import pytest

from pravah import temporal_basis as T
from pravah.information import fisher


def delay_kernel(n_sites=2, n_t=30, K=2, delay=3):
    """Delta-pulse transport: a unit pulse released at r reaches site s, source k at o = r + delay + s with gain (k+1)."""
    H = np.zeros((n_sites, n_t, n_t, K))
    for s in range(n_sites):
        for k in range(K):
            for r in range(n_t):
                o = r + delay + s
                if o < n_t:
                    H[s, o, r, k] = k + 1.0
    return H


def test_delta_pulse_delay_and_no_lookahead():
    H = delay_kernel(delay=3)
    B = np.zeros((30, 1)); B[5, 0] = 1.0  # pulse at t=5 only
    G = T.kernel_to_response(H, B)  # (sites, T, K*J)
    assert G.shape == (2, 30, 2)
    assert np.flatnonzero(G[0, :, 0]) == [8] and np.flatnonzero(G[1, :, 1]) == [9]
    assert G[0, 8, 1] == 2.0
    assert not G[:, :8, :].any() and not G[0, 9:, :].any()  # nothing before arrival: causal


def test_noncausal_kernel_rejected():
    H = delay_kernel(); H[0, 2, 5, 0] = 1.0  # response before release
    with pytest.raises(ValueError, match="causal"):
        T.kernel_to_response(H, np.ones((30, 1)))


def test_basis_shapes_and_periodicity():
    t = np.arange(48)
    for kind, J in [("constant", 1), ("diurnal", 3), ("piecewise", 4)]:
        b = T.periodic_basis(kind, t)
        assert b.shape == (48, J)
        np.testing.assert_allclose(b[:24], b[24:], atol=1e-12)
    np.testing.assert_array_equal(T.periodic_basis("constant", t), np.ones((48, 1)))
    np.testing.assert_array_equal(T.periodic_basis("piecewise", t).sum(1), np.ones(48))  # partition of unity
    with pytest.raises(ValueError):
        T.periodic_basis("nope", t)


def test_J1_equals_branchA_and_diurnal_linearity():
    H = delay_kernel(K=2)
    GA = T.kernel_to_response(H, T.periodic_basis("constant", np.arange(30)))
    direct = np.einsum("sork->sok", H)  # constant unit source for all release times
    np.testing.assert_allclose(GA, direct, atol=1e-12)
    B = T.periodic_basis("diurnal", np.arange(30))
    G = T.kernel_to_response(H, B)
    assert G.shape == (2, 30, 6)
    beta = np.array([1.0, 0.5, -0.2, 2.0, 0.1, 0.3])  # (k, j) flattened k-major
    g = (B @ beta.reshape(2, 3).T)  # (t, k) source histories
    np.testing.assert_allclose(G @ beta, np.einsum("sork,rk->so", H, g), atol=1e-12)


def test_stack_window_dimensions_order_and_leakage():
    G3 = np.random.default_rng(0).normal(size=(4, 30, 3))
    A = T.stack_window(G3, [0, 2], [5, 6, 7], issue=7)
    assert A.shape == (6, 3)
    np.testing.assert_array_equal(A[:3], G3[0, [5, 6, 7]])  # site-major
    with pytest.raises(ValueError, match="leak"):
        T.stack_window(G3, [0], [5, 8], issue=7)
    with pytest.raises(ValueError):
        T.stack_window(G3, [0], [-1], issue=7)


def test_schedule_matched_budget_and_infeasible():
    assert T.reading_hours(issue=100, window=1, m=1).tolist() == [100]
    assert T.reading_hours(issue=100, window=168, m=1).tolist() == [100]  # snapshot is the issue hour, not the window start
    h = T.reading_hours(issue=100, window=24, m=4)
    assert len(set(h)) == 4 and h[-1] == 100 and h[0] == 77 and h.max() <= 100
    assert T.reading_hours(100, 6, 6).tolist() == list(range(95, 101))
    with pytest.raises(ValueError, match="infeasible"):
        T.reading_hours(100, 6, 8)
    with pytest.raises(ValueError):
        T.reading_hours(100, 6, 0)


def test_repeated_snapshot_adds_no_rank_information_scales_m():
    G3 = np.random.default_rng(1).normal(size=(3, 10, 2))
    A1 = T.stack_window(G3, [0, 1, 2], [9], issue=9)
    Am = T.stack_window(G3, [0, 1, 2], [9] * 4, issue=9)
    assert np.linalg.matrix_rank(Am) == np.linalg.matrix_rank(A1)
    np.testing.assert_allclose(fisher(Am, 1.0), 4 * fisher(A1, 1.0), rtol=1e-12)


def test_ar1_cov_matches_analytic_information():
    for n, rho in [(1, 0.8), (5, 0.8), (24, 0.8), (168, 0.8), (10, 0.0)]:
        R = T.ar1_cov(1, np.arange(n), rho, 1.0)
        info = np.ones(n) @ np.linalg.solve(R, np.ones(n))
        assert info == pytest.approx((n * (1 - rho) + 2 * rho) / (1 + rho), rel=1e-10)
    R = T.ar1_cov(2, [0, 3], 0.5, 2.0)  # site-major blocks, independent sites, lag-3 corr 0.5**3, var 4
    assert R.shape == (4, 4) and R[0, 1] == pytest.approx(4 * 0.5**3) and not R[:2, 2:].any()
    with pytest.raises(ValueError):
        T.ar1_cov(1, [0, 1], 1.0, 1.0)


def test_whitening_keeps_exact_rank():
    G3 = np.random.default_rng(2).normal(size=(2, 8, 3))
    A = T.stack_window(G3, [0, 1], list(range(8)), issue=7)
    R = T.ar1_cov(2, np.arange(8), 0.8, 1.0)
    assert np.linalg.matrix_rank(np.linalg.cholesky(np.linalg.inv(R)).T @ A) == np.linalg.matrix_rank(A)

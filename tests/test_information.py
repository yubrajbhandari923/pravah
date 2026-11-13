import numpy as np
import pytest

from pravah import information as I

rng = np.random.default_rng(0)


def test_fd_jacobian_matches_analytic_green1d():
    x, th = np.array([0.5, 1.0, 2.0]), np.array([2.0, 0.7])
    f = lambda t: t[0] * np.exp(-t[1] * x)
    np.testing.assert_allclose(I.fd_jacobian(f, th), I.green1d_jacobian(x, *th), rtol=1e-6)


def test_stack_jacobian_order_and_shape():
    G3 = rng.normal(size=(5, 4, 3))  # sites, hours, K
    G = I.stack_jacobian(G3, sites=[3, 1], hours=[0, 2])
    assert G.shape == (4, 3)
    np.testing.assert_array_equal(G[1], G3[3, 2]); np.testing.assert_array_equal(G[2], G3[1, 0])
    with pytest.raises(IndexError):
        I.stack_jacobian(G3, sites=[5])


def test_fisher_symmetric_psd_and_white_equals_inverse_variance():
    G = rng.normal(size=(8, 3)); R = np.diag(rng.uniform(.5, 2, 8))
    M = I.fisher(G, R)
    assert np.array_equal(M, M.T) and np.linalg.eigvalsh(M).min() > 0
    np.testing.assert_allclose(M, G.T @ np.linalg.inv(R) @ G, rtol=1e-10)
    np.testing.assert_allclose(I.fisher(G, 2.0), G.T @ G / 4)  # scalar = sigma


def test_fisher_rejects_bad_noise_and_nonfinite():
    G = np.ones((3, 2))
    with pytest.raises(ValueError):
        I.fisher(G, np.diag([1, -1, 1.0]))
    with pytest.raises(ValueError):
        I.fisher(np.array([[np.nan, 1.0]]))


def test_zero_information_in_null_direction():
    A = rng.normal(size=(6, 2)); G = np.column_stack([A, A @ [1.0, -2.0]])  # col3 dependent
    M = I.fisher(G); n = np.array([1.0, -2.0, -1.0])  # G n = 0
    assert abs(n @ M @ n) < 1e-12
    rc = I.rank_coherence(G)
    assert rc["rank"] == 2
    assert np.linalg.norm(rc["null_basis"].T @ n) == pytest.approx(np.linalg.norm(n))  # n lies in the null space


def test_fisher_additive_over_independent_times():
    Gs = [rng.normal(size=(3, 4)) for _ in range(5)]
    np.testing.assert_allclose(I.fisher(np.vstack(Gs)), sum(I.fisher(g) for g in Gs), rtol=1e-12)


def test_prior_adds_precision():
    G = rng.normal(size=(5, 2)); P = np.diag([4.0, 9.0])
    np.testing.assert_allclose(I.fisher(G, prior_cov=P), I.fisher(G) + np.linalg.inv(P))


def test_criteria_full_rank_matches_eigen_formulas():
    M = np.diag([4.0, 1.0, 0.25])
    c = I.criteria(M)
    assert c["defined"] and c["rank"] == 3
    assert c["d_opt"] == pytest.approx(np.log(1.0)) and c["a_opt"] == pytest.approx(0.25 + 1 + 4)
    assert c["e_opt"] == pytest.approx(0.25)


def test_criteria_rank_deficient_not_blind_logdet():
    M = np.diag([4.0, 1.0, 0.0])
    c = I.criteria(M)
    assert not c["defined"] and c["rank"] == 2
    assert c["d_opt"] == -np.inf and c["a_opt"] == np.inf and c["e_opt"] == 0.0
    c2 = I.criteria(np.diag([1.0, -1e-18]))  # rounding-negative eigenvalue
    assert not c2["defined"] and c2["e_opt"] == 0.0 and c2["d_opt"] == -np.inf


def test_criteria_scale_behaviour():
    M = np.diag([2.0, 3.0]); c = I.criteria(10 * M)
    assert c["d_opt"] == pytest.approx(I.criteria(M)["d_opt"] + 2 * np.log(10))


def test_confidence_ellipse_axes_and_orientation():
    M = np.diag([4.0, 1.0])  # cov = diag(.25, 1)
    e = I.confidence_ellipse(M, (0, 1), level=0.95)
    k = np.sqrt(5.991464547107979)
    np.testing.assert_allclose(e["semi_axes"], [1.0 * k, 0.5 * k], rtol=1e-9)  # descending
    assert abs(np.sin(e["angle"])) == pytest.approx(1.0)  # major axis along y
    with pytest.raises(ValueError):
        I.confidence_ellipse(np.diag([1.0, 0.0]), (0, 1))
    with pytest.raises(ValueError):
        I.confidence_ellipse(M, (0, 1), level=1.0)


def test_confidence_ellipse_marginal_is_inverse_then_subset():
    A = rng.normal(size=(6, 3)); M = A.T @ A
    e = I.confidence_ellipse(M, (0, 2), level=0.9)
    C = np.linalg.inv(M)[np.ix_([0, 2], [0, 2])]
    np.testing.assert_allclose(np.sort(e["semi_axes"] ** 2), np.sort(np.linalg.eigvalsh(C) * 4.605170185988092), rtol=1e-9)


def test_rank_coherence():
    G = np.array([[1.0, 1.0, 0], [0, 0, 1], [0, 0, 0]])  # cols 0,1 identical
    rc = I.rank_coherence(G)
    assert rc["rank"] == 2 and rc["coherence"] == pytest.approx(1.0)
    assert np.isinf(rc["cond"])  # rank < K
    assert I.rank_coherence(np.eye(3))["coherence"] == 0.0
    assert I.rank_coherence(np.zeros((3, 2)))["rank"] == 0  # zero columns do not raise


def test_green1d_d_opt_two_sensors_analytic_optimum():
    a, b = 3.0, 0.5; grid = np.linspace(0, 8, 161)
    best, arg = -np.inf, None
    for i, x1 in enumerate(grid):
        for x2 in grid[i + 1:]:
            c = I.criteria(I.fisher(I.green1d_jacobian(np.array([x1, x2]), a, b)))
            if c["defined"] and c["d_opt"] > best:
                best, arg = c["d_opt"], (x1, x2)
    assert arg == pytest.approx((0.0, 1 / b), abs=0.06)  # maximise (x2-x1)^2 e^{-2b(x1+x2)}
    assert best == pytest.approx(np.log(a**2 * np.exp(-2.0) * (1 / b) ** 2), abs=1e-6)


def test_green1d_one_sensor_is_singular():
    assert not I.criteria(I.fisher(I.green1d_jacobian(np.array([1.0]), 2.0, 0.3)))["defined"]

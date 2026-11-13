"""Independent reference tests for src/pravah/filter.py (design: docs/research/filter-design.md §4).

Reference values come from hand algebra, dense batch posteriors or the forward solver, never from filter.py itself."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from pravah import filter as KF
from pravah.forward import simulate
from pravah.observations import bilinear_sample


def test_scalar_predict_update_by_hand():
    m, P = KF.predict(np.array([[1.0]]), np.array([[2.0]]), np.array([[0.9]]), None, np.array([[0.5]]))
    assert m[0, 0] == pytest.approx(0.9) and P[0, 0] == pytest.approx(2.12)
    m, P, nu, w, nis, k = KF.update(m, P, np.array([[2.0]]), np.array([[1.0]]), 1.0)
    g = 2.12 / 3.12
    assert m[0, 0] == pytest.approx(0.9 + g * 1.1) and P[0, 0] == pytest.approx(2.12 - 2.12 ** 2 / 3.12)
    assert nis[0] == pytest.approx(1.1 ** 2 / 3.12) and k == 1


def _toy(seed=0, n=4, K=2, k=2, T=12):
    r = np.random.default_rng(seed)
    A = 0.6 * np.linalg.qr(r.standard_normal((n, n)))[0]
    B, F0, H = r.standard_normal((n, K)), r.standard_normal((n, K)), r.standard_normal((k, n))
    return A, B, F0, H, T


def test_augmented_kf_equals_batch_posterior_and_forecasts():
    """Static parameter a, x_h = A x_{h-1} + B a, x_0 = F_0 a, Q = 0: KF on [x; a] equals the batch Gaussian posterior."""
    A, B, F0, H, T = _toy()
    n, K = B.shape
    abar, s2, r2 = np.array([1.0, -0.5]), 0.7, 0.3
    a_true = np.array([0.4, 1.3])
    F = [F0]
    for _ in range(1, T):
        F.append(A @ F[-1] + B)
    rng = np.random.default_rng(1)
    Y = np.stack([H @ Fh @ a_true + np.sqrt(r2) * rng.standard_normal(len(H)) for Fh in F])[:, :, None]
    Aaug = np.block([[A, B], [np.zeros((K, n)), np.eye(K)]])
    Haug = np.hstack([H, np.zeros((len(H), K))])
    m0 = np.concatenate([F0 @ abar, abar])
    P0 = s2 * np.block([[F0 @ F0.T, F0], [F0.T, np.eye(K)]])
    out = KF.run(m0, P0, lambda h: (Aaug, None, None), Haug, r2, Y, issue=range(T - 3), horizons=(1, 3))
    for t in range(T):
        Gt = np.vstack([H @ F[h] for h in range(t + 1)])
        prec = np.eye(K) / s2 + Gt.T @ Gt / r2
        mean = np.linalg.solve(prec, abar / s2 + Gt.T @ Y[:t + 1, :, 0].ravel() / r2)
        np.testing.assert_allclose(out["am"][t, n:, 0], mean, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(out["av"][t, n:], np.diag(np.linalg.inv(prec)), rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(out["am"][t, :n, 0], F[t] @ mean, rtol=1e-8, atol=1e-10)
        if t < T - 3:
            np.testing.assert_allclose(out["fc"][3][t, :n, 0], F[t + 3] @ mean, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(out["fc"][1][t, :n, 0], F[t + 1] @ mean, rtol=1e-8, atol=1e-10)


def test_future_observation_cannot_change_earlier_forecasts():
    A, B, F0, H, T = _toy(2, n=5)
    Q = 0.1 * np.eye(5)
    Y = np.random.default_rng(3).standard_normal((T, 2, 3))
    run = lambda Y: KF.run(np.zeros(5), np.eye(5), lambda h: (A, B[:, 0], Q), H, 0.5, Y, issue=range(T), horizons=(1, 2, 4))
    tp = 7
    Y2 = Y.copy()
    Y2[tp] += 10.0
    o1, o2 = run(Y), run(Y2)
    assert np.array_equal(o1["fm"][:tp + 1], o2["fm"][:tp + 1])  # forecast for hour tp is issued at tp - 1
    assert np.array_equal(o1["am"][:tp], o2["am"][:tp])
    for hz in (1, 2, 4):
        assert np.array_equal(o1["fc"][hz][:tp], o2["fc"][hz][:tp], equal_nan=True)
        assert not np.allclose(o1["fc"][hz][tp], o2["fc"][hz][tp], equal_nan=True) or tp + hz >= T  # leakage is detectable
    assert not np.allclose(o1["am"][tp], o2["am"][tp])


def test_transition_indexing_and_no_update_limit():
    """u_h = h with negligible updates: forecast for hour h is m0 + sum_{j<=h} j, so transition(h) is used for h-1 -> h."""
    T, calls = 6, []
    def tr(h):
        calls.append(h)
        return np.eye(1), np.array([float(h)]), None
    Y = np.ones((T, 1, 1))
    out = KF.run(np.zeros(1), np.array([[1e-12]]), tr, np.ones((1, 1)), 1e20, Y, issue=[0, 2], horizons=(1, 3))
    assert calls == list(range(1, T))
    np.testing.assert_allclose(out["fm"][:, 0, 0], np.cumsum(np.arange(T)), atol=1e-6)
    assert out["fc"][3][0, 0, 0] == pytest.approx(6.0) and out["fc"][3][1, 0, 0] == pytest.approx(1 + 2 + 3 + 4 + 5)
    assert out["fc"][1][1, 0, 0] == pytest.approx(out["fm"][3, 0, 0])


def test_absent_sensor_no_update_and_partial_missing_rejected():
    A, B, F0, H, T = _toy(4)
    Y = np.random.default_rng(5).standard_normal((T, 2, 2))
    Y[3] = np.nan
    Y[5, 0] = np.nan
    out = KF.run(np.zeros(4), np.eye(4), lambda h: (A, None, 0.1 * np.eye(4)), H, 1.0, Y)
    assert np.array_equal(out["am"][3], out["fm"][3]) and np.array_equal(out["av"][3], out["fv"][3]) and out["kobs"][3] == 0
    assert out["kobs"][5] == 1 and np.isnan(out["white"][5, 0]).all()
    Y[6, 0, 0] = np.nan
    with pytest.raises(ValueError):
        KF.run(np.zeros(4), np.eye(4), lambda h: (A, None, None), H, 1.0, Y)


def test_q_zero_without_data_is_deterministic_propagation():
    A, B, F0, H, T = _toy(6)
    P0 = np.diag([1.0, 2, 3, 4])
    out = KF.run(np.ones(4), P0, lambda h: (A, None, None), H, 1.0, np.full((T, 2, 1), np.nan))
    P, m = P0, np.ones(4)
    for h in range(1, T):
        P, m = A @ P @ A.T, A @ m
    np.testing.assert_allclose(out["am"][-1, :, 0], m, atol=1e-12)
    np.testing.assert_allclose(out["av"][-1], np.diag(P), atol=1e-12)


def test_perfect_model_nees_nis_calibration_and_psd():
    r = np.random.default_rng(7)
    n, k, T, C = 6, 2, 60, 300
    A = 0.95 * np.linalg.qr(r.standard_normal((n, n)))[0]
    Lq = 0.3 * r.standard_normal((n, n))
    Q, H, R = Lq @ Lq.T + 0.01 * np.eye(n), r.standard_normal((k, n)), np.diag([0.2, 0.5])
    P0 = np.eye(n)
    x = np.linalg.cholesky(P0) @ r.standard_normal((n, C))
    X, Y = [], []
    for h in range(T):
        if h:
            x = A @ x + np.linalg.cholesky(Q) @ r.standard_normal((n, C))
        X.append(x)
        Y.append(H @ x + np.sqrt(np.diag(R))[:, None] * r.standard_normal((k, C)))
    out = KF.run(np.zeros(n), P0, lambda h: (A, None, Q), H, R, np.array(Y))
    nis = out["nis"].mean() / k
    assert abs(nis - 1) < 4 * np.sqrt(2 / (k * T * C))  # chi2_k / k has variance 2 / k
    # NEES needs the full analysis covariance: recompute the Riccati recursion by hand (independent of filter.py)
    P = P0
    for h in range(T):
        if h:
            P = A @ P @ A.T + Q
        S = H @ P @ H.T + R
        P = P - P @ H.T @ np.linalg.solve(S, H @ P)
    e = X[-1] - out["am"][-1]
    nees = np.einsum("nc,nc->c", e, np.linalg.solve(P, e)).mean() / n
    assert abs(nees - 1) < 4 * np.sqrt(2 / (n * C))
    np.testing.assert_allclose(out["av"][-1], np.diag(P), rtol=1e-8)
    w = out["white"][1:]
    lag1 = np.mean(w[1:] * w[:-1])
    assert abs(lag1) < 4 / np.sqrt(w[1:].size)
    assert np.linalg.eigvalsh(P).min() > 0


@pytest.mark.parametrize("bc", ["study", "periodic"])
def test_step_and_hour_operator_match_simulate(bc):
    kw = dict(nx=8, ny=6, Lx=8000.0, Ly=6000.0, dt=300, u=(2.0, -1.2), K=50.0, lam=1 / 43200)
    M = KF.step_operator(bc=bc, **kw)
    d0 = np.random.default_rng(11).standard_normal((8, 6))
    ref = simulate(n_steps=3, c0=d0, c_bg=0.0, bc=bc, **kw)
    got = (KF.hour_operator(M, 3) @ d0.ravel()).reshape(8, 6)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-11 * np.abs(ref).max())


def test_step_operator_refuses_nonlinear_vanleer():
    with pytest.raises(ValueError, match="not a linear"):
        KF.step_operator(nx=8, ny=8, Lx=8000.0, Ly=8000.0, dt=30, u=(2.0, 1.0), K=50.0, lam=1 / 43200, scheme="vanleer")


def test_observation_matrix_matches_bilinear_sample():
    r = np.random.default_rng(12)
    grid, dom = (7, 5), (7000.0, 5000.0)
    xy = r.uniform(0, 1, (4, 2)) * dom
    c = r.standard_normal(grid)
    np.testing.assert_allclose(KF.observation_matrix(xy, grid, dom) @ c.ravel(), bilinear_sample(c, xy, dom), rtol=1e-12)


def test_crps_and_coverage_against_numerical_reference():
    mu, sd, x = 0.3, 1.7, -0.9
    num = quad(lambda t: norm.cdf((t - mu) / sd) ** 2, -np.inf, x)[0] + quad(lambda t: (1 - norm.cdf((t - mu) / sd)) ** 2, x, np.inf)[0]
    assert KF.gaussian_crps(mu, sd, x) == pytest.approx(num, rel=1e-7)
    z = np.random.default_rng(13).standard_normal(200000)
    cov = KF.coverage(0.0, 1.0, z)
    assert all(abs(cov[lv] - lv) < 0.005 for lv in cov)

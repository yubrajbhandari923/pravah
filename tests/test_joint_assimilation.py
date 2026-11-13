"""E10 reference tests (design: docs/research/joint-assimilation-design.md §9).

The random-walk augmented filter (filter.run with Q on the source block) is checked against a dense batch Gaussian
posterior over (a_0, w_1..w_t) built here independently; the truncated mode against KKT and a generic optimiser."""
import numpy as np
import pytest
from scipy.optimize import minimize

from pravah import filter as KF


def _toy(seed=0, n=4, K=2, k=2, T=10):
    r = np.random.default_rng(seed)
    A = 0.6 * np.linalg.qr(r.standard_normal((n, n)))[0]
    return A, r.standard_normal((n, K)), r.standard_normal((n, K)), r.standard_normal((k, n)), T


def _batch(A, B, F0, H, T, q, r2):
    """Linear maps theta = (a_0, w_1..w_{T-1}) -> a_h (L[h]) and x_h (G[h]); prior covariance of theta."""
    n, K = B.shape
    L = [np.hstack([np.eye(K)] + [np.eye(K) if j <= h else np.zeros((K, K)) for j in range(1, T)]) for h in range(T)]
    G = [F0 @ L[0]]
    for h in range(1, T + 1):
        G.append(A @ G[-1] + B @ L[h - 1])
    S = np.diag(np.r_[np.ones(K), np.full(K * (T - 1), q)])
    return L, G, S


@pytest.mark.parametrize("q", [0.0, 0.05, 1.0])
def test_random_walk_filter_equals_batch_posterior(q):
    A, B, F0, H, T = _toy()
    n, K = B.shape
    r2 = 0.3
    L, G, S = _batch(A, B, F0, H, T, q, r2)
    rng = np.random.default_rng(2)
    theta = rng.multivariate_normal(np.zeros(S.shape[0]), S + 1e-12 * np.eye(len(S)))
    Y = np.stack([H @ G[h] @ theta + np.sqrt(r2) * rng.standard_normal(len(H)) for h in range(T)])[:, :, None]
    Z = np.zeros
    tr = lambda h: (np.block([[A, B], [Z((K, n)), np.eye(K)]]), None, np.diag(np.r_[Z(n), np.full(K, q)]))
    P0 = np.block([[F0 @ F0.T, F0], [F0.T, np.eye(K)]])
    issue = np.arange(T - 1)
    o = KF.run(Z(n + K), P0, tr, np.hstack([H, Z((len(H), K))]), r2, Y, issue=issue, horizons=(1, 2),
               keep=np.arange(n, n + K))
    for t in issue:
        Ht = np.vstack([H @ G[h] for h in range(t + 1)])           # covariance form: valid for singular prior (q = 0)
        Kg = S @ Ht.T @ np.linalg.inv(Ht @ S @ Ht.T + r2 * np.eye(len(Ht)))
        mu, C = Kg @ Y[:t + 1, :, 0].ravel(), S - Kg @ Ht @ S
        assert np.allclose(o["am"][t, n:, 0], L[t] @ mu, atol=1e-9)
        assert np.allclose(o["aPk"][t][n:], L[t] @ C @ L[t].T, atol=1e-9)
        assert np.allclose(o["aPk"][t][:n], G[t] @ C @ L[t].T, atol=1e-9)   # field-source cross covariance
        assert np.allclose(o["fc"][1][t, :n, 0], G[t + 1] @ mu, atol=1e-9)  # future w have prior mean 0
        if t + 2 < T:
            assert np.allclose(o["fc"][2][t, :n, 0], G[t + 2] @ mu, atol=1e-9)


def test_q0_equals_growing_window_map():
    """q = 0: source mean equals the information-form MAP of a constant a from all past data (prior N(0, I))."""
    A, B, F0, H, T = _toy(seed=3)
    n, K = B.shape
    r2 = 0.5
    F = [F0]
    for _ in range(1, T):
        F.append(A @ F[-1] + B)
    Y = np.random.default_rng(4).standard_normal((T, len(H), 1))
    Z = np.zeros
    o = KF.run(Z(n + K), np.block([[F0 @ F0.T, F0], [F0.T, np.eye(K)]]),
               lambda h: (np.block([[A, B], [Z((K, n)), np.eye(K)]]), None, None), np.hstack([H, Z((len(H), K))]), r2, Y)
    for t in range(T):
        Nt = np.eye(K) + sum((H @ F[h]).T @ (H @ F[h]) for h in range(t + 1)) / r2
        bt = sum((H @ F[h]).T @ Y[h, :, 0] for h in range(t + 1)) / r2
        assert np.allclose(o["am"][t, n:, 0], np.linalg.solve(Nt, bt), atol=1e-10)


def test_nonneg_mode_feasible_unchanged_and_kkt():
    r = np.random.default_rng(5)
    for _ in range(50):
        K = int(r.integers(1, 5))
        X = r.standard_normal((K, K))
        C = X @ X.T + 0.1 * np.eye(K)
        m = r.standard_normal((K, 3))
        m[:, 0] = np.abs(m[:, 0])
        a = KF.nonneg_mode(m, C)
        assert np.array_equal(a[:, 0], m[:, 0]) and (a >= 0).all()
        Ci = np.linalg.inv(C)
        for j in range(3):
            g = Ci @ (a[:, j] - m[:, j])                       # KKT: g = 0 where a > 0, g >= 0 where a = 0
            act = a[:, j] > 1e-9
            assert np.allclose(g[act], 0, atol=1e-7) and (g[~act] > -1e-7).all()
            ref = minimize(lambda x: (x - m[:, j]) @ Ci @ (x - m[:, j]), np.ones(K), bounds=[(0, None)] * K,
                           method="L-BFGS-B", options=dict(ftol=1e-15, gtol=1e-12)).x
            assert np.allclose(a[:, j], ref, atol=1e-4)
    assert KF.nonneg_mode(np.array([-1.0, 2.0]), np.eye(2)).tolist() == [0.0, 2.0]   # 1-D input shape kept

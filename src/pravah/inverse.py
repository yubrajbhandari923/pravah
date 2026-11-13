"""Weighted / ridge nonnegative least squares for source coefficients. See tests/test_source_inversion.py."""
import numpy as np
from scipy.optimize import nnls


def rank_info(G, tol=1e-8):
    s = np.linalg.svd(np.asarray(G, float), compute_uv=False)
    rank = int((s > tol * s[0]).sum()) if s[0] > 0 else 0
    return dict(rank=rank, cond=float(s[0] / s[-1]) if rank == np.shape(G)[1] else float("inf"), singular_values=s)


def _whiten(R, n):
    """Return W = R^{-1/2} (a Cholesky solve for a full covariance)."""
    if R is None:
        return np.eye(n)
    R = np.asarray(R, float)
    if R.ndim == 0 or R.ndim == 1:
        s = np.broadcast_to(R, (n,)) if R.ndim == 0 else R
        if s.shape != (n,) or not np.isfinite(s).all() or (s <= 0).any():
            raise ValueError("noise_cov must be positive finite sigma(s) of length n")
        return np.diag(1.0 / s)
    if R.shape != (n, n) or not np.isfinite(R).all():
        raise ValueError("noise_cov must be (n, n)")
    if not np.allclose(R, R.T, rtol=1e-10, atol=1e-12 * np.abs(R).max()):  # cholesky reads one triangle only
        raise ValueError("covariance is not symmetric positive definite")
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError as e:
        raise ValueError("covariance is not symmetric positive definite") from e
    return np.linalg.inv(L)  # W^T W = R^-1


def solve_nnls(G, y, *, noise_cov=None, ridge=0.0, nuisance=None):
    G, y = np.asarray(G, float), np.asarray(y, float)
    if G.ndim != 2 or y.shape != (G.shape[0],):
        raise ValueError("G must be (n, K) and y (n,)")
    if not (np.isfinite(G).all() and np.isfinite(y).all()):
        raise ValueError("non-finite input")
    if ridge < 0:
        raise ValueError("ridge must be >= 0")
    n, K = G.shape
    N = np.zeros((n, 0)) if nuisance is None else np.asarray(nuisance, float)
    if N.ndim != 2 or N.shape[0] != n or not np.isfinite(N).all():
        raise ValueError("nuisance must be (n, m) and finite")
    W = _whiten(noise_cov, n)
    WG, Wy, WN = W @ G, W @ y, W @ N
    A, b = np.hstack([WG, WN, -WN]), Wy  # sign-free nuisance d = d+ - d-, both >= 0, unpenalised
    if ridge > 0:
        lam = ridge * np.linalg.norm(WG, 2) ** 2
        pen = np.hstack([np.sqrt(lam) * np.eye(K), np.zeros((K, 2 * N.shape[1]))])
        A, b = np.vstack([A, pen]), np.concatenate([b, np.zeros(K)])
    z, _ = nnls(A, b, maxiter=50 * A.shape[1])
    a, d = z[:K], z[K:K + N.shape[1]] - z[K + N.shape[1]:]
    info = rank_info(WG)
    return dict(coef=a, nuisance_coef=d, residual_norm=float(np.linalg.norm(Wy - WG @ a - WN @ d)),
                rank=info["rank"], cond=info["cond"])

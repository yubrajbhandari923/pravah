"""Causal linear Kalman filtering of the concentration state (E09). Design: docs/research/filter-design.md.

Holds no truth. State vectors are row-major flattenings of [i, j] fields (x = d.ravel(), d = c - c_bg). Means are
(n, c) arrays: c columns are paired data sets (noise draws) sharing one covariance recursion, which does not depend
on the data. Tests: tests/test_filter.py.
"""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_solve, solve_triangular
from scipy.sparse.linalg import splu
from scipy.stats import norm

from .forward import simulate
from .observations import bilinear_sample


def _op(A):
    return A if callable(A) else (lambda X: A @ X)


def predict(m, P, A, u=None, Q=None):
    """m- = A m + u, P- = A P A^T + Q. A: (n, n) array or callable X -> A X; m (n, c); u (n,) or None."""
    f = _op(A)
    m = f(m) if u is None else f(m) + np.reshape(u, (-1, 1))
    P = f(f(P).T)  # A (A P)^T = A P^T A^T = A P A^T for symmetric P
    if Q is not None:
        P = P + Q
    return m, (P + P.T) / 2


def update(m, P, y, H, R):
    """Kalman update with readings y (k, c). A row that is NaN in every column is an absent sensor (dropped).
    Returns m, P, innovation nu (k_obs, c), whitened innovation L^-1 nu, NIS (c,) = nu^T S^-1 nu, k_obs."""
    y, H, R = np.asarray(y, float).reshape(len(H), -1), np.asarray(H, float), np.asarray(R, float)
    R = np.diag(np.broadcast_to(R, (len(H),))) if R.ndim < 2 else R
    ok = ~np.isnan(y).all(1)
    if np.isnan(y[ok]).any():
        raise ValueError("a sensor row is missing in some data columns only")
    if not ok.any():
        e = np.zeros((0, y.shape[1]))
        return m, P, e, e, np.zeros(y.shape[1]), 0
    H, R, y = H[ok], R[np.ix_(ok, ok)], y[ok]
    PHt = P @ H.T
    S = H @ PHt + R
    S = (S + S.T) / 2
    L = np.linalg.cholesky(S)
    K = cho_solve((L, True), PHt.T).T
    nu = y - H @ m
    w = solve_triangular(L, nu, lower=True)
    P = P - K @ PHt.T
    return m + K @ nu, (P + P.T) / 2, nu, w, (w ** 2).sum(0), int(ok.sum())


def run(m0, P0, transition, H, R, Y, *, issue=(), horizons=(1,)):
    """Causal forecast-observe-update loop over hours h = 0 .. T-1.

    (m0, P0): forecast (prior) for hour 0. transition(h) -> (A, u, Q) for hour h-1 -> h, h >= 1, called once per hour
    in order. Y (T, k, c): readings; Y[h] is read only after the forecast for hour h is stored. For each issue time t,
    the mean forecasts of hour t + hz (hz in horizons) are made from the analysis at t alone (NaN past T-1).
    Returns dict: fm/fv (forecast mean (T, n, c) / variance diag (T, n)), am/av (analysis), nis (T, c), kobs (T,),
    white (T, k, c) whitened innovations (NaN where absent), fc {hz: (len(issue), n, c)}.
    """
    Y = np.asarray(Y, float)
    T, (n, c) = len(Y), (len(m0), Y.shape[2])
    issue = [int(t) for t in issue]
    if any(t < 0 or t >= T for t in issue) or any(hz < 1 for hz in horizons):
        raise ValueError("issue times must lie in [0, T) and horizons be >= 1")
    out = dict(fm=np.empty((T, n, c)), fv=np.empty((T, n)), am=np.empty((T, n, c)), av=np.empty((T, n)),
               nis=np.full((T, c), np.nan), kobs=np.zeros(T, int), white=np.full((T,) + Y.shape[1:], np.nan),
               fc={hz: np.full((len(issue), n, c), np.nan) for hz in horizons})
    pos = {t: i for i, t in enumerate(issue)}
    m, P = np.repeat(np.reshape(m0, (n, 1)), c, 1).astype(float), np.array(P0, float)
    pending = []  # [issue index, target hour, mean] advanced with the mean transition each hour
    for h in range(T):
        if h:
            A, u, Q = transition(h)
            m, P = predict(m, P, A, u, Q)
            f = _op(A)
            for p in pending:
                p[2] = f(p[2]) if u is None else f(p[2]) + np.reshape(u, (-1, 1))
            for p in [p for p in pending if p[1] == h]:
                out["fc"][p[3]][p[0]] = p[2]
            pending = [p for p in pending if p[1] > h]
        out["fm"][h], out["fv"][h] = m, np.diag(P)
        if h - 1 in pos and 1 in out["fc"]:
            out["fc"][1][pos[h - 1]] = m
        m, P, _, w, nis, k = update(m, P, Y[h], H, R)  # first use of Y[h]
        ok = ~np.isnan(Y[h]).all(1)
        out["white"][h][ok], out["nis"][h], out["kobs"][h] = w, nis if k else np.nan, k
        out["am"][h], out["av"][h] = m, np.diag(P)
        if h in pos:
            pending += [[pos[h], h + hz, m.copy(), hz] for hz in horizons if hz > 1 and h + hz < T]
    return out


# ---------------------------------------------------------------- transport operators from the verified solver
_STENCIL = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))


def step_operator(*, nx, ny, Lx, Ly, dt, u, K, lam, bc="study", scheme="upwind", n_probe=8, seed=0, rtol=1e-8):
    """Sparse M_u with one forward.simulate step d1 = M_u^-1 d0 for the anomaly (c_bg = 0, no source).

    Recovered by probing simulate with random fields: each row of M_u has the 5-point stencil, fitted by least squares
    from n_probe probes, then checked on a fresh probe. ValueError if simulate is not linear with that stencil (e.g.
    VanLeer, whose limiter is non-linear: amendment A2)."""
    kw = dict(nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt, n_steps=1, u=tuple(u), K=K, lam=lam, c_bg=0.0, bc=bc, scheme=scheme)
    X = np.random.default_rng(seed).standard_normal((n_probe + 1, nx, ny))
    Y = np.stack([simulate(c0=x, **kw) for x in X]).reshape(n_probe + 1, -1)
    X = X.reshape(n_probe + 1, -1)
    i, j = np.divmod(np.arange(nx * ny), ny)
    cols, valid = [], []
    for di, dj in _STENCIL:
        ii, jj = i + di, j + dj
        v = np.ones(nx * ny, bool) if bc == "periodic" else (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        cols.append((ii % nx) * ny + jj % ny)
        valid.append(v)
    cols, valid = np.array(cols).T, np.array(valid).T  # (N, 5)
    Z = (Y[:n_probe][:, cols] * valid).transpose(1, 0, 2)  # (N, probes, 5)
    coef = np.einsum("nkp,np->nk", np.linalg.pinv(Z), X[:n_probe].T)
    rows = np.repeat(np.arange(nx * ny), 5).reshape(-1, 5)
    M = sp.csr_matrix((coef[valid], (rows[valid], cols[valid])), shape=(nx * ny,) * 2)
    scale = np.abs(X).max()
    fit = np.abs(M @ Y[:n_probe].T - X[:n_probe].T).max() / scale
    chk = np.abs(splu(M.tocsc()).solve(X[-1]) - Y[-1]).max() / np.abs(Y[-1]).max()
    if not (fit < rtol and chk < rtol):
        raise ValueError(f"simulate is not a linear 5-point step here (fit {fit:.1e}, check {chk:.1e}); "
                         "no exact transition matrix exists")
    return M


def hour_operator(M, n_steps):
    """Dense A = (M^-1)^n_steps, by one sparse LU solve with N right-hand sides and repeated squaring."""
    A1 = splu(sp.csc_matrix(M)).solve(np.eye(M.shape[0]))
    out, b = None, A1
    while n_steps:
        if n_steps & 1:
            out = b if out is None else b @ out
        n_steps >>= 1
        if n_steps:
            b = b @ b
    return np.eye(M.shape[0]) if out is None else out


def observation_matrix(xy, grid, domain):
    """H (k, N) with H x = observations.bilinear_sample(x.reshape(grid), xy, domain), built column by column."""
    N = grid[0] * grid[1]
    H = np.empty((len(np.atleast_2d(xy)), N))
    e = np.zeros(N)
    for col in range(N):
        e[col] = 1.0
        H[:, col] = bilinear_sample(e.reshape(grid), xy, domain)
        e[col] = 0.0
    return H


# ---------------------------------------------------------------- calibration scores
def gaussian_crps(mean, sd, x):
    """CRPS of N(mean, sd^2) at x (Gneiting & Raftery 2007, closed form); sd > 0."""
    z = (np.asarray(x) - mean) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def coverage(mean, sd, x, levels=(0.5, 0.8, 0.95)):
    """Fraction of x inside the central Gaussian intervals of each level."""
    z = np.abs((np.asarray(x) - mean) / sd)
    return {lv: float(np.mean(z <= norm.ppf((1 + lv) / 2))) for lv in levels}

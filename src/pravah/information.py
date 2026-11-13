"""Footprints, Fisher information and identifiability (E04/E05).

Units: G rows are sensitivities (concentration per unit source), so M = G^T R^-1 G has units of 1/(source unit)^2.
D/A values therefore depend on the scale of G; only ranks and eigenvalue ratios are scale-free.
Synthetic passive tracer only. The 1-D Green example is a pedagogical analytic comparator, not Green (2006)'s setup.
"""
import numpy as np
from scipy.stats import chi2

from pravah.inverse import _whiten, rank_info

RANK_TOL = 1e-8  # relative to the largest singular value; same default as inverse.rank_info


def fd_jacobian(f, theta, h=1e-6):
    """Central-difference Jacobian d f / d theta, shape (len(f(theta)), len(theta)); step is relative to |theta_j|."""
    theta = np.asarray(theta, float)
    cols = []
    for j in range(theta.size):
        d = h * max(1.0, abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += d; tm[j] -= d
        cols.append((np.asarray(f(tp), float) - np.asarray(f(tm), float)) / (2 * d))
    return np.column_stack(cols)


def stack_jacobian(G3, sites=None, hours=None):
    """Stack a (sites, hours, K) sensitivity array into (n_sites*n_hours, K), site-major (as scenario_response_matrix)."""
    G3 = np.asarray(G3, float)
    if G3.ndim != 3:
        raise ValueError("G3 must be (sites, hours, K)")
    sites = np.arange(G3.shape[0]) if sites is None else np.asarray(sites, int)
    hours = np.arange(G3.shape[1]) if hours is None else np.asarray(hours, int)
    if sites.size and (sites.max() >= G3.shape[0] or sites.min() < -G3.shape[0]):
        raise IndexError("site index out of range")
    if hours.size and (hours.max() >= G3.shape[1] or hours.min() < -G3.shape[1]):
        raise IndexError("hour index out of range")
    return G3[np.ix_(sites, hours)].reshape(-1, G3.shape[2])


def fisher(G, noise_cov=None, prior_cov=None):
    """M = G^T R^-1 G (+ P0^-1). noise_cov: None, sigma scalar/vector, or full covariance (as inverse._whiten)."""
    G = np.asarray(G, float)
    if G.ndim != 2 or not np.isfinite(G).all():
        raise ValueError("G must be a finite (n, K) matrix")
    WG = _whiten(noise_cov, G.shape[0]) @ G
    M = WG.T @ WG
    if prior_cov is not None:
        P = np.asarray(prior_cov, float)
        if P.shape != (G.shape[1],) * 2 or not np.allclose(P, P.T):
            raise ValueError("prior_cov must be symmetric (K, K)")
        try:
            np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            raise ValueError("prior_cov must be positive definite") from None
        M = M + np.linalg.inv(P)
    return (M + M.T) / 2  # exact symmetry


def criteria(M, tol=RANK_TOL):
    """D/A/E from the eigenvalues of M. Only defined if M has full numerical rank (sqrt(eig) > tol * sqrt(max eig), i.e. singular values of G as in rank_info);
    otherwise d_opt=-inf, a_opt=inf, e_opt=0 and defined=False (no logdet of a singular matrix)."""
    w = np.linalg.eigvalsh(np.asarray(M, float))
    rank = int((w > tol**2 * w[-1]).sum()) if w[-1] > 0 else 0
    if rank < w.size:
        return dict(defined=False, rank=rank, d_opt=-np.inf, a_opt=np.inf, e_opt=0.0)
    return dict(defined=True, rank=rank, d_opt=float(np.log(w).sum()), a_opt=float((1 / w).sum()), e_opt=float(w[0]))


def confidence_ellipse(M, idx, level=0.95):
    """Local (Cramer-Rao) confidence ellipse for two parameters idx of the full M: marginal covariance inv(M)[idx, idx].
    Returns semi_axes (descending) and angle (rad) of the major axis from the first parameter axis."""
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    M = np.asarray(M, float)
    if not criteria(M)["defined"]:
        raise ValueError("M is numerically singular: covariance undefined")
    C = np.linalg.inv(M)[np.ix_(idx, idx)]
    w, V = np.linalg.eigh((C + C.T) / 2)
    k2 = chi2.ppf(level, 2)
    return dict(semi_axes=np.sqrt(k2 * w[::-1]), angle=float(np.arctan2(V[1, -1], V[0, -1])))


def rank_coherence(G, tol=RANK_TOL):
    """Rank, condition (inf if rank < K), null-space basis (K, K-rank) and max absolute cosine between distinct columns
    (zero columns are ignored in the coherence)."""
    G = np.asarray(G, float)
    info = rank_info(G, tol)
    _, s, Vt = np.linalg.svd(G)
    s = np.concatenate([s, np.zeros(G.shape[1] - s.size)])
    nrm = np.linalg.norm(G, axis=0)
    ok = nrm > 0
    U = G[:, ok] / nrm[ok]
    C = np.abs(U.T @ U) - np.eye(U.shape[1])
    return dict(rank=info["rank"], cond=info["cond"], singular_values=info["singular_values"],
                null_basis=Vt[info["rank"]:].T, coherence=float(C.max()) if C.size else 0.0)


def green1d_jacobian(x, a, b):
    """Analytic sensitivities of c(x) = a exp(-b x) (steady 1-D decay profile, two parameters a, b); shape (len(x), 2)."""
    x = np.asarray(x, float)
    e = np.exp(-b * x)
    return np.column_stack([e, -a * x * e])

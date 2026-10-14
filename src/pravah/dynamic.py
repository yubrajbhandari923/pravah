"""Dynamic (time-varying) source histories on top of the existing linear forward model.

A history theta[j, b] (parameter j, temporal bin b) is realised by expanding every source / inflow basis into one
virtual parameter per bin whose piecewise-constant profile is that bin's indicator.  The existing ``compute_responses``
then gives the full source-history -> concentration operator, and everything downstream (sampling, caching, population
RMSE) is reused unchanged.  The forward solver itself is not touched.  Truth episodes use their own profiles (see
``truth_scenario``) so the inverse basis is never the generating basis.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from .config import Scenario

_EPS = 1e-30


def bin_edges(duration: float, n_bins: int) -> np.ndarray:
    return np.linspace(0.0, duration, n_bins + 1)


def _indicator(edges: np.ndarray, b: int) -> tuple[tuple[float, float], ...]:
    pts: list[tuple[float, float]] = []
    if b > 0:
        pts.append((0.0, 0.0))
    pts.append((float(edges[b]), 1.0))
    if b < len(edges) - 2:
        pts.append((float(edges[b + 1]), 0.0))
    return tuple(pts)


def expand(scenario: Scenario, n_bins: int) -> Scenario:
    """Scenario whose parameter vector is theta.reshape(-1) of shape (n_params * n_bins), parameter-major, bin-minor
    (emissions first, then inflow bases, as in ``Scenario.param_table``)."""
    edges = bin_edges(scenario.times.duration, n_bins)
    srcs = [dataclasses.replace(s, name=f"{s.name}@{b}", profile=_indicator(edges, b))
            for s in scenario.sources for b in range(n_bins)]
    bases = tuple(dataclasses.replace(k, name=f"{k.name}@{b}", profile=_indicator(edges, b))
                  for k in scenario.boundary.bases for b in range(n_bins))
    return dataclasses.replace(scenario, sources=tuple(srcs), boundary=dataclasses.replace(scenario.boundary, bases=bases))


def truth_scenario(scenario: Scenario, history: np.ndarray, edges: np.ndarray) -> Scenario:
    """Scenario whose sources/bases carry ``history`` (n_params, n_fine) as profiles; run it with q = ones.
    ``edges`` has n_fine + 1 entries on [0, duration]; amplitudes keep their native units (ug/s, ug/m^3)."""
    h = np.asarray(history, dtype=float)
    if h.shape != (scenario.n_params, len(edges) - 1) or np.any(h < 0):
        raise ValueError("history must be non-negative with shape (n_params, n_fine)")
    prof = [tuple((float(t), float(v)) for t, v in zip(edges[:-1], row)) for row in h]
    n = scenario.n_emission
    srcs = tuple(dataclasses.replace(s, profile=prof[i]) for i, s in enumerate(scenario.sources))
    bases = tuple(dataclasses.replace(k, profile=prof[n + i]) for i, k in enumerate(scenario.boundary.bases))
    return dataclasses.replace(scenario, sources=srcs, boundary=dataclasses.replace(scenario.boundary, bases=bases))


def _diff(n_params: int, n_bins: int) -> np.ndarray:
    d1 = np.diff(np.eye(n_bins), axis=0)
    return np.kron(np.eye(n_params), d1)


class NotConverged(RuntimeError):
    pass


def tv_nnls(A: np.ndarray, y: np.ndarray, sigma: np.ndarray, theta0: np.ndarray, alpha: float, beta: float,
            D: np.ndarray, max_iter: int = 20000, tol: float = 1e-8) -> tuple[np.ndarray, int]:
    """min_{theta>=0} 0.5||(A theta - y)/sigma||^2 + alpha||theta-theta0||^2 + beta||D theta||_1  by ADMM.
    rho is matched to the data curvature (sqrt of the largest eigenvalue of the whitened Gram matrix): a fixed rho = 1 stopped
    every production-conditioned fit at the iteration cap (review of F-037).  beta == 0 is solved exactly (bvls).  Stops on relative primal AND dual residuals
    below ``tol`` and raises NotConverged otherwise.  Returns (theta, iterations)."""
    n = A.shape[1]
    Aw, yw = A / sigma[:, None], y / sigma
    if beta == 0:  # no TV term: exact bounded least squares on the stacked (data; ridge) system
        r = lsq_linear(np.vstack([Aw, np.sqrt(2 * alpha) * np.eye(n)]), np.concatenate([yw, np.sqrt(2 * alpha) * theta0]),
                       bounds=(0.0, np.inf), method="bvls", tol=1e-14, max_iter=20 * n + 200)
        if not r.success:
            raise NotConverged(f"bounded least squares failed: {r.message}")
        return r.x, int(r.nit)
    G = Aw.T @ Aw
    rho = max(float(np.sqrt(np.linalg.eigvalsh(G)[-1] * 2 * max(alpha, 1e-12))), 1e-3)  # start between the curvature extremes; adapted below
    K = np.vstack([D, np.eye(n)]) if beta > 0 else np.eye(n)
    KtK = K.T @ K
    Hinv = np.linalg.inv(G + 2 * alpha * np.eye(n) + rho * KtK)  # n is tens: explicit inverse beats re-solving
    rhs0 = Aw.T @ yw + 2 * alpha * theta0
    v = np.zeros(K.shape[0]); u = np.zeros(K.shape[0])  # u: scaled dual
    m = D.shape[0] if beta > 0 else 0
    for it in range(1, max_iter + 1):
        th = Hinv @ (rhs0 + rho * K.T @ (v - u))
        Kt = 1.6 * (K @ th) - 0.6 * v if it > 1 else K @ th  # over-relaxation
        v_old = v
        w = Kt + u
        v = w.copy()
        if m:
            v[:m] = np.sign(w[:m]) * np.maximum(np.abs(w[:m]) - beta / rho, 0.0)
        v[m:] = np.maximum(w[m:], 0.0)
        u = u + Kt - v
        pr = np.linalg.norm(K @ th - v)
        du = rho * np.linalg.norm(K.T @ (v - v_old))
        scale = 1.0 + max(np.linalg.norm(K @ th), np.linalg.norm(v))
        if pr < tol * scale and du < tol * (scale + rho * np.linalg.norm(u)):
            return np.maximum(v[m:], 0.0), it
        if it % 25 == 0 and (pr > 10 * du or du > 10 * pr):  # residual balancing
            f = 2.0 if pr > 10 * du else 0.5
            rho *= f
            u = u / f
            Hinv = np.linalg.inv(G + 2 * alpha * np.eye(n) + rho * KtK)
    raise NotConverged(f"ADMM did not converge in {max_iter} iterations (alpha={alpha:g}, beta={beta:g}, rho={rho:.3g})")


@dataclass(frozen=True)
class HistoryFit:
    theta: np.ndarray  # (n_params, n_bins), native units
    model: str
    iterations: int = 0


def sensitivity(A: np.ndarray, sigma: np.ndarray, n_params: int, n_bins: int) -> np.ndarray:
    """(n_params, n_bins) whitened column norms of A: signal-to-noise a unit-reference amplitude produces.  Columns at
    (essentially) zero are unidentifiable (e.g. an inflow basis whose faces are outflow in that bin)."""
    return np.linalg.norm(A / sigma[:, None], axis=0).reshape(n_params, n_bins)


def fit_history(A: np.ndarray, y: np.ndarray, sigma: np.ndarray, n_params: int, n_bins: int, model: str,
                alpha: float = 1e-3, beta: float = 0.0, theta0: np.ndarray | None = None,
                ref: np.ndarray | None = None) -> HistoryFit:
    """Fit a source/inflow history.  A: (n_obs, n_params*n_bins) from the expanded scenario; y already has the base
    (zero-amplitude) reading subtracted.  model: M0 static (one constant per parameter), M1 independent bins
    (ridge only), M2 temporally regularised (ridge + TV).  Variables are x = theta/ref with ``ref`` (n_params,) FIXED physical
    reference amplitudes (default 1), so alpha and beta mean the same thing for every sensor layout and wind: a layout-dependent
    scale would change the prior with the design.  ``theta0`` (n_params, n_bins) or (n_params,) is the prior mean in native units."""
    r = np.ones(n_params) if ref is None else np.asarray(ref, dtype=float)
    rr = np.repeat(r, n_bins)
    As = A * rr
    if theta0 is None:
        t0 = np.zeros(A.shape[1])
    else:
        t0 = np.broadcast_to(np.asarray(theta0, dtype=float).reshape(n_params, -1), (n_params, n_bins)).reshape(-1) / rr
    if model == "M0":
        S = np.kron(np.eye(n_params), np.ones((n_bins, 1)))
        x, it = tv_nnls(As @ S, y, sigma, np.zeros(n_params), alpha, 0.0, np.zeros((0, n_params)))
        x = S @ x
    elif model in ("M1", "M2"):
        x, it = tv_nnls(As, y, sigma, t0, alpha, beta if model == "M2" else 0.0, _diff(n_params, n_bins))
    else:
        raise ValueError(f"unknown model {model!r}")
    return HistoryFit((x * rr).reshape(n_params, n_bins), model, it)

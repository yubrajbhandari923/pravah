"""Dynamic (time-varying) source histories on top of the existing linear forward model.

A history theta[j, b] (parameter j, temporal bin b) is realised by expanding every source / inflow basis into one
virtual parameter per bin whose piecewise-constant profile is that bin's indicator.  The existing ``compute_responses``
then gives the full source-history -> concentration operator, and everything downstream (sampling, caching, population
RMSE) is reused unchanged.  The forward solver itself is not touched.  Truth episodes use their own (finer, jittered)
profiles so the inverse basis is never the generating basis (no inverse crime).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

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


def tv_nnls(A: np.ndarray, y: np.ndarray, sigma: np.ndarray, theta0: np.ndarray, alpha: float, beta: float,
            D: np.ndarray, iters: int = 4000, rho: float = 1.0, tol: float = 1e-10) -> np.ndarray:
    """min_{theta>=0} 0.5||(A theta - y)/sigma||^2 + alpha||theta-theta0||^2 + beta||D theta||_1  (scaled ADMM).
    Variables should be O(1): the caller column-scales A.  Needs alpha > 0 or full column rank."""
    n = A.shape[1]
    Aw, yw = A / sigma[:, None], y / sigma
    K = np.vstack([D, np.eye(n)]) if beta > 0 else np.eye(n)
    H = Aw.T @ Aw + 2 * alpha * np.eye(n) + rho * K.T @ K
    L = np.linalg.cholesky(H + _EPS * np.eye(n))
    rhs0 = Aw.T @ yw + 2 * alpha * theta0
    v = np.zeros(K.shape[0]); u = np.zeros(K.shape[0]); th = np.zeros(n)
    m = D.shape[0] if beta > 0 else 0
    for _ in range(iters):
        th = np.linalg.solve(L.T, np.linalg.solve(L, rhs0 + rho * K.T @ (v - u)))
        Kt = K @ th
        v_old = v
        w = Kt + u
        v = w.copy()
        if m:
            v[:m] = np.sign(w[:m]) * np.maximum(np.abs(w[:m]) - beta / rho, 0.0)
        v[m:] = np.maximum(w[m:], 0.0)
        u = u + Kt - v
        if np.linalg.norm(Kt - v) < tol * (1 + np.linalg.norm(v)) and rho * np.linalg.norm(K.T @ (v - v_old)) < tol * (1 + np.linalg.norm(u)):
            break
    return np.maximum(v[m:], 0.0)  # the feasible copy


@dataclass(frozen=True)
class HistoryFit:
    theta: np.ndarray  # (n_params, n_bins), native units
    model: str


def fit_history(A: np.ndarray, y: np.ndarray, sigma: np.ndarray, n_params: int, n_bins: int, model: str,
                alpha: float = 1e-3, beta: float = 0.0, theta0: np.ndarray | None = None) -> HistoryFit:
    """Fit a source/inflow history.  A: (n_obs, n_params*n_bins) from the expanded scenario; y already has the base
    (zero-amplitude) reading subtracted.  model: M0 static (one constant per parameter), M1 independent bins
    (ridge only), M2 temporally regularised (ridge + TV).  alpha/beta act on column-scaled variables (theta*scale)."""
    scale = np.maximum(np.linalg.norm(A, axis=0), _EPS)
    scale = scale.reshape(n_params, n_bins).max(axis=1).repeat(n_bins)  # one scale per parameter (units differ across kinds)
    As = A / scale
    t0 = np.zeros(A.shape[1]) if theta0 is None else np.asarray(theta0, float).reshape(-1) * scale
    if model == "M0":
        S = np.kron(np.eye(n_params), np.ones((n_bins, 1)))
        th = tv_nnls(As @ S, y, sigma, np.zeros(n_params), alpha, 0.0, np.zeros((0, n_params)))
        theta = (S @ th)
    elif model in ("M1", "M2"):
        b = beta if model == "M2" else 0.0
        theta = tv_nnls(As, y, sigma, t0, alpha, b, _diff(n_params, n_bins))
    else:
        raise ValueError(f"unknown model {model!r}")
    return HistoryFit((theta / scale).reshape(n_params, n_bins), model)

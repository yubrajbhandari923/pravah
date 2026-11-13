"""Lagged time-stacked design matrices and a periodic temporal source basis (E05 / EXP-006, design: docs/research/time-observability-design.md).

Model: y(s, h) = c_bg + sum_{k,j} G3[s, h, (k, j)] beta_kj, with q_k(t) = sum_j beta_kj b_j(t) (Branch A: J=1, b=1).
Units: G in ug m^-3 per (ug s^-1); hours are integer reading hours. Column index is k*J + j. Rows are site-major, (site, hour),
as information.stack_jacobian. Only linear identifiability is addressed; no estimator, no positivity cone.
"""
import numpy as np

from .information import stack_jacobian


def periodic_basis(kind, hours, period=24):
    """(len(hours), J) basis b_j(t), period `period` h: constant J=1; diurnal {1, sin, cos} J=3; piecewise 4 indicators of the
    quarter-period, J=4 (partition of unity)."""
    t = np.asarray(hours, float)
    if kind == "constant":
        return np.ones((t.size, 1))
    if kind == "diurnal":
        w = 2 * np.pi * t / period
        return np.column_stack([np.ones_like(t), np.sin(w), np.cos(w)])
    if kind == "piecewise":
        q = (np.floor(4 * (t % period) / period)).astype(int)
        return np.eye(4)[q]
    raise ValueError(f"unknown basis {kind!r}")


def kernel_to_response(H, B):
    """Delta-pulse kernel H[s, obs_hour, release_hour, k] (response to a unit pulse released at one hour) and basis
    B[release_hour, J] -> G3[s, obs_hour, k*J + j] = sum_r H[s, o, r, k] B[r, j]. H must be causal (zero for release > obs)."""
    H, B = np.asarray(H, float), np.asarray(B, float)
    if H.ndim != 4 or B.ndim != 2 or H.shape[2] != B.shape[0] or H.shape[1] != H.shape[2]:
        raise ValueError("H must be (sites, T, T, K) and B (T, J)")
    if not (np.isfinite(H).all() and np.isfinite(B).all()):
        raise ValueError("non-finite input")
    if (np.triu(np.ones(H.shape[1:3]), 1).astype(bool) & (np.abs(H).sum((0, 3)) > 0)).any():  # release after obs
        raise ValueError("kernel is not causal (response before release)")
    G = np.einsum("sork,rj->sokj", H, B)
    return G.reshape(H.shape[0], H.shape[1], -1)


def stack_window(G3, sites, hours, issue):
    """Rows (site-major) for the given readings. Any reading hour > issue is look-ahead leakage and is rejected; repeats allowed."""
    hours = np.asarray(hours, int)
    if hours.size and (hours.max() > issue):
        raise ValueError(f"leakage: reading hour {hours.max()} is after issue time {issue}")
    if hours.size and hours.min() < 0:
        raise ValueError("negative reading hour")
    return stack_jacobian(G3, sites, hours)


def reading_hours(issue, window, m):
    """m distinct hourly reading times evenly spread over the `window` hours ending at `issue` (last = issue); window=1, m=1 is
    the snapshot. Raises if the window cannot hold m distinct hours (infeasible cell, never padded)."""
    if m < 1 or window < 1:
        raise ValueError("m and window must be >= 1")
    h = np.array([issue]) if m == 1 else np.round(np.linspace(issue - window + 1, issue, m)).astype(int)  # m=1: the issue hour (gate-05)
    if len(set(h)) < m:
        raise ValueError(f"infeasible: {m} distinct hourly readings do not fit in {window} h")
    return h


def ar1_cov(n_sites, hours, rho, sigma):
    """Site-major noise covariance: independent sites, stationary AR(1) in time with corr rho**|dh| and marginal sd sigma."""
    h = np.asarray(hours, float)
    if not 0 <= rho < 1 or sigma <= 0:
        raise ValueError("need 0 <= rho < 1 and sigma > 0")
    C = sigma**2 * rho ** np.abs(h[:, None] - h[None, :]) if rho > 0 else sigma**2 * np.eye(h.size)
    return np.kron(np.eye(n_sites), C)

"""Error metrics: weighted space-time field error, source error, held-out sensor error, synthetic weight map."""
from __future__ import annotations

import numpy as np
import xarray as xr

from .config import Domain
from .observations import SensorLayout, sample

_TOL = 1e-9


def _arr(da: xr.DataArray) -> np.ndarray:
    return np.asarray(da.transpose("time", "y", "x").values, dtype=float)


def quadrature_weights(times: np.ndarray, ny: int, nx: int, weight: np.ndarray | None, t_exclude: float
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(sel, dt, w, den) of the MSE_w definition: boolean mask of the used output times, trapezoid time weights dt over
    them, the (ny, nx) weight map w and den = sum(w) * sum(dt), so MSE_w = einsum(dt, diff2[sel], w) / den."""
    t = np.asarray(times, dtype=float)
    sel = t >= t_exclude - _TOL * max(1.0, abs(t[-1]))
    t_sel = t[sel]
    if t_sel.size < 2:
        raise ValueError("fewer than two output times at or after t_exclude")
    w = np.ones((ny, nx)) if weight is None else np.asarray(weight, dtype=float)
    if w.shape != (ny, nx) or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weight must be a finite non-negative (ny, nx) array")
    dt = np.zeros(t_sel.size)
    h = np.diff(t_sel)
    dt[:-1] += h / 2
    dt[1:] += h / 2
    den = float(w.sum() * dt.sum())
    if not den > 0:
        raise ValueError("zero weight denominator")
    return sel, dt, w, den


def field_mse(est: xr.DataArray, truth: xr.DataArray, weight: np.ndarray | None = None,
              t_exclude: float = 0.0) -> float:
    """MSE_w = sum_t sum_cells w dA dt_q (est-truth)^2 / sum w dA dt_q.

    Both fields are (time, y, x) on identical coordinates.  Only stored output times t >= t_exclude are used (nothing
    is interpolated); dt_q are trapezoid weights over those times, dA is the (uniform) cell area, w >= 0 an (ny, nx)
    map (default 1).  dA cancels for a uniform grid.  Zero denominator (w == 0, or fewer than 2 times) -> ValueError.
    """
    for dim in ("time", "x", "y"):
        if est.sizes[dim] != truth.sizes[dim] or not np.allclose(est[dim].values, truth[dim].values):
            raise ValueError(f"est and truth differ in coordinate {dim!r}")
    sel, dt, w, den = quadrature_weights(np.asarray(est["time"].values, dtype=float), est.sizes["y"], est.sizes["x"],
                                         weight, t_exclude)
    diff2 = (_arr(est) - _arr(truth))[sel] ** 2
    return float(np.einsum("t,tyx,yx->", dt, diff2, w) / den)


def field_rmse(est: xr.DataArray, truth: xr.DataArray, weight: np.ndarray | None = None,
               t_exclude: float = 0.0) -> float:
    return float(np.sqrt(field_mse(est, truth, weight, t_exclude)))


def source_error(q_hat: np.ndarray, q_true: np.ndarray) -> dict[str, float | np.ndarray]:
    """{'relative_l2': ||q_hat-q||/||q||, 'per_source_abs': q_hat-q, 'per_source_rel': (q_hat-q)/q (nan if q_j=0)}."""
    qh = np.asarray(q_hat, dtype=float)
    qt = np.asarray(q_true, dtype=float)
    if qh.shape != qt.shape or qt.ndim != 1:
        raise ValueError("q_hat and q_true must be 1-D of equal length")
    norm = float(np.linalg.norm(qt))
    if not norm > 0:
        raise ValueError("q_true is zero: relative error undefined")
    d = qh - qt
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(qt != 0, d / np.where(qt != 0, qt, 1.0), np.nan)
    return {"relative_l2": float(np.linalg.norm(d) / norm), "per_source_abs": d, "per_source_rel": rel}


def sensor_rmse(est_field: xr.DataArray, truth_field: xr.DataArray, layout: SensorLayout,
                times: list[float] | tuple[float, ...] | np.ndarray) -> float:
    """RMS difference of the noiseless readings of two fields at the given sensors and times (e.g. held-out ones)."""
    d = sample(est_field, layout, times).values - sample(truth_field, layout, times).values
    return float(np.sqrt(np.mean(np.asarray(d, dtype=float) ** 2)))


def population_weight(domain: Domain,
                      centres: tuple[tuple[float, float], ...] = ((0.3, 0.3), (0.65, 0.7), (0.75, 0.25)),
                      amplitudes: tuple[float, ...] = (1.0, 0.7, 0.5), width: float = 0.12,
                      floor: float = 0.05) -> np.ndarray:
    """SYNTHETIC population-like importance map (ny, nx), >= 0, normalised to mean 1.

    Sum of Gaussian blobs (centres and width as fractions of the domain lengths) plus a small floor.  It is NOT
    real population data; it is only an importance weight for the error metric.
    """
    if len(amplitudes) != len(centres):
        raise ValueError("one amplitude per centre")
    x = (np.arange(domain.nx) + 0.5) / domain.nx
    y = (np.arange(domain.ny) + 0.5) / domain.ny
    w = np.full((domain.ny, domain.nx), floor, dtype=float)
    for (cx, cy), a in zip(centres, amplitudes):
        w += a * np.exp(-((x[None, :] - cx) ** 2 + (y[:, None] - cy) ** 2) / (2 * width**2))
    if np.any(w < 0) or not w.sum() > 0:
        raise ValueError("amplitudes/floor must give a non-negative, non-zero map")
    return w / w.mean()

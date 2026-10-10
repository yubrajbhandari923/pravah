"""Synthetic sensor observations of a concentration field.

Interpolation convention (documented contract):
* Space: bilinear on the cell-centre grid (x = (i+0.5)*dx, y = (j+0.5)*dy). Exact for fields affine in x and y.
  Sensors must lie inside the cell-centre hull [dx/2, Lx-dx/2] x [dy/2, Ly-dy/2]; there is no extrapolation.
* Time: linear between stored output times. Requested times must lie in [t_first, t_last] of the field.
* Flat cell index is k = i + j*nx (see config.py), so weights index ``values.reshape(nt, ny*nx)``.
* ``sample`` is noiseless; ``observe`` adds iid Gaussian noise and keeps negative readings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from .config import Domain

_TOL = 1e-9  # relative tolerance for points exactly on the hull / time range


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@dataclass(frozen=True)
class SensorLayout:
    ids: tuple[str, ...]
    x: tuple[float, ...]
    y: tuple[float, ...]

    def __post_init__(self) -> None:
        _need(len(self.ids) == len(self.x) == len(self.y), "ids, x, y must have equal lengths")
        _need(len(set(self.ids)) == len(self.ids), "sensor ids must be unique")
        _need(all(math.isfinite(v) for v in (*self.x, *self.y)), "sensor coordinates must be finite")

    def __len__(self) -> int:
        return len(self.ids)

    def select(self, indices: list[int] | tuple[int, ...]) -> SensorLayout:
        return SensorLayout(tuple(self.ids[i] for i in indices), tuple(self.x[i] for i in indices),
                            tuple(self.y[i] for i in indices))

    def subset(self, ids: list[str] | tuple[str, ...]) -> SensorLayout:
        pos = {s: i for i, s in enumerate(self.ids)}
        missing = [s for s in ids if s not in pos]
        _need(not missing, f"unknown sensor ids: {missing}")
        return self.select([pos[s] for s in ids])


@dataclass(frozen=True)
class ObservationSpec:
    times: tuple[float, ...]
    noise_std: float | tuple[float, ...]
    seed: int

    def __post_init__(self) -> None:
        _need(len(self.times) >= 1, "need at least one observation time")
        _need(all(b > a for a, b in zip(self.times, self.times[1:])), "observation times must be strictly increasing")
        s = np.atleast_1d(np.asarray(self.noise_std, dtype=float))
        _need(bool(np.all(np.isfinite(s)) and np.all(s >= 0)), "noise_std must be finite and >= 0")


@dataclass(frozen=True)
class Observations:
    values: xr.DataArray  # (time, sensor); coords: sensor ids, per-sensor x and y
    noise_std: np.ndarray  # per sensor
    layout: SensorLayout
    times: tuple[float, ...] = field(default_factory=tuple)


def candidate_hull(domain: Domain) -> tuple[tuple[float, float], tuple[float, float]]:
    """Valid sensor-position intervals ((x_min, x_max), (y_min, y_max)): the cell-centre hull."""
    return ((domain.dx / 2, domain.lx - domain.dx / 2), (domain.dy / 2, domain.ly - domain.dy / 2))


def _axis_weights(coords: np.ndarray, p: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Lower cell index and fractional position for each p along a strictly increasing 1-D axis."""
    _need(coords.size >= 2, f"need at least 2 {name} cell centres")
    tol = _TOL * (coords[-1] - coords[0])
    bad = (p < coords[0] - tol) | (p > coords[-1] + tol)
    _need(not bad.any(), f"sensor {name} outside the cell-centre hull [{coords[0]}, {coords[-1]}]: {p[bad].tolist()}")
    p = np.clip(p, coords[0], coords[-1])
    i = np.clip(np.searchsorted(coords, p, side="right") - 1, 0, coords.size - 2)
    return i, (p - coords[i]) / (coords[i + 1] - coords[i])


def interp_weights(x_coords: np.ndarray, y_coords: np.ndarray, layout: SensorLayout) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear stencil per sensor: (flat cell indices (n,4), weights (n,4)); weights sum to 1.

    Order of the 4 cells: (i,j), (i+1,j), (i,j+1), (i+1,j+1), flat index k = i + j*nx.
    """
    xc = np.asarray(x_coords, dtype=float)
    yc = np.asarray(y_coords, dtype=float)
    i, fx = _axis_weights(xc, np.asarray(layout.x, dtype=float), "x")
    j, fy = _axis_weights(yc, np.asarray(layout.y, dtype=float), "y")
    nx = xc.size
    k = i + j * nx
    idx = np.stack([k, k + 1, k + nx, k + nx + 1], axis=1)
    w = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1)
    return idx, w


def sample(concentration: xr.DataArray, layout: SensorLayout, times: list[float] | tuple[float, ...] | np.ndarray
           ) -> xr.DataArray:
    """Noiseless readings, DataArray (time, sensor). See module docstring for the interpolation convention."""
    t = np.asarray(times, dtype=float)
    _need(t.ndim == 1 and t.size >= 1, "times must be a non-empty 1-D sequence")
    _need(bool(np.all(np.diff(t) > 0)), "observation times must be strictly increasing")
    ft = np.asarray(concentration["time"].values, dtype=float)
    tol = _TOL * max(ft[-1] - ft[0], 1.0)
    _need(bool(t[0] >= ft[0] - tol and t[-1] <= ft[-1] + tol),
          f"observation times [{t[0]}, {t[-1]}] outside field time range [{ft[0]}, {ft[-1]}]")
    idx, w = interp_weights(concentration["x"].values, concentration["y"].values, layout)
    v = np.asarray(concentration.transpose("time", "y", "x").values, dtype=float).reshape(ft.size, -1)
    t = np.clip(t, ft[0], ft[-1])
    if ft.size == 1:
        i0 = np.zeros(t.size, dtype=int)
        i1, ft_frac = i0, np.zeros(t.size)
    else:
        i0 = np.clip(np.searchsorted(ft, t, side="right") - 1, 0, ft.size - 2)
        i1 = i0 + 1
        ft_frac = (t - ft[i0]) / (ft[i1] - ft[i0])
    s0 = np.einsum("tsk,sk->ts", v[i0][:, idx], w)
    s1 = np.einsum("tsk,sk->ts", v[i1][:, idx], w)
    out = (1 - ft_frac)[:, None] * s0 + ft_frac[:, None] * s1
    return _wrap(out, layout, t)


def _wrap(values: np.ndarray, layout: SensorLayout, times: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values, dims=("time", "sensor"),
        coords={"time": times, "sensor": list(layout.ids),
                "x": ("sensor", np.asarray(layout.x, dtype=float)),
                "y": ("sensor", np.asarray(layout.y, dtype=float))},
        name="observation", attrs={"units": "microgram/m^3"})


def observe(concentration: xr.DataArray, layout: SensorLayout, spec: ObservationSpec) -> Observations:
    """Noisy observations: sample() plus iid N(0, noise_std^2), drawn from default_rng(spec.seed)."""
    clean = sample(concentration, layout, spec.times)
    std = np.broadcast_to(np.asarray(spec.noise_std, dtype=float), (len(layout),)).copy() \
        if np.ndim(spec.noise_std) == 0 else np.asarray(spec.noise_std, dtype=float)
    _need(std.shape == (len(layout),), "noise_std must be a scalar or have one entry per sensor")
    rng = np.random.default_rng(spec.seed)
    noisy = clean.values + rng.standard_normal(clean.shape) * std[None, :]
    return Observations(_wrap(noisy, layout, np.asarray(spec.times, dtype=float)), std, layout, tuple(spec.times))

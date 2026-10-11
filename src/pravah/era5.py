"""ERA5 hourly 10 m wind: load, validate, sample, and feed `winds.segments_from_series` as a spatially UNIFORM series.

ERA5 is ~28 km (0.25 deg) and cannot resolve the valley/slope flow around Belauri (a ~120 km2 municipality, ~12 km across):
use it only as a large-scale, point-or-box time series, never as a local wind field.  ``u10`` / ``v10`` are the flow TOWARD
east / north (the CF eastward / northward wind), which is the convention of ``config.WindSegment`` (u toward +x, v toward +y);
the meteorological 'from' bearing is ``from_deg(u, v)``.  Model axes are UTM 44N grid axes; grid north differs from true north by
the convergence angle (lon - 81 E) sin(lat) = -0.31 deg at Belauri (80.354 E, 28.700 N); IGNORED (0.31 deg is ~160 m over 30 km,
far below ERA5 direction error).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import xarray as xr

from .config import Domain, WindSegment
from .winds import segments_from_series


class Era5Wind(NamedTuple):
    times: np.ndarray  # (nt,) datetime64[ns], UTC, strictly increasing
    lon: np.ndarray    # (nx,) degrees E, increasing
    lat: np.ndarray    # (ny,) degrees N, increasing
    u: np.ndarray      # (nt, ny, nx) m/s toward east
    v: np.ndarray      # (nt, ny, nx) m/s toward north


def load_era5_wind(path: str) -> Era5Wind:
    ds = xr.open_dataset(path)
    try:
        for name in ("u10", "v10"):
            if ds[name].dims != ("valid_time", "latitude", "longitude"):
                raise ValueError(f"{name}: dims {ds[name].dims}, expected (valid_time, latitude, longitude)")
            if ds[name].attrs.get("units") not in ("m s**-1", "m s-1", "m/s"):
                raise ValueError(f"{name}: units {ds[name].attrs.get('units')!r}, expected m/s")
        ds = ds.sortby("latitude")  # CDS files list latitude north to south
        t, lon, lat = ds.valid_time.values, ds.longitude.values, ds.latitude.values
        u, v = ds.u10.values.astype(float), ds.v10.values.astype(float)
    finally:
        ds.close()
    if not (np.all(np.diff(t) > np.timedelta64(0, "ns")) and np.all(np.diff(lon) > 0) and np.all(np.diff(lat) > 0)):
        raise ValueError("time, longitude, latitude must be strictly increasing")
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        raise ValueError("u10/v10 contain non-finite values")
    return Era5Wind(t, lon, lat, u, v)


def sample_at(w: Era5Wind, lon: float, lat: float) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear (u, v) time series (nt,) at a point; ValueError outside the grid."""
    if not (w.lon[0] <= lon <= w.lon[-1] and w.lat[0] <= lat <= w.lat[-1]):
        raise ValueError("point outside the ERA5 grid")
    i, j = (min(int(np.searchsorted(g, p, side="right")) - 1, len(g) - 2) for g, p in ((w.lon, lon), (w.lat, lat)))
    fx, fy = (lon - w.lon[i]) / (w.lon[i + 1] - w.lon[i]), (lat - w.lat[j]) / (w.lat[j + 1] - w.lat[j])
    k = np.array([[(1 - fy) * (1 - fx), (1 - fy) * fx], [fy * (1 - fx), fy * fx]])
    return tuple(np.einsum("tyx,yx->t", a[:, j:j + 2, i:i + 2], k) for a in (w.u, w.v))  # type: ignore[return-value]


def box_mean(w: Era5Wind, lon0: float, lat0: float, lon1: float, lat1: float) -> tuple[np.ndarray, np.ndarray]:
    """Plain mean (u, v) over the grid points with lon0 <= lon <= lon1, lat0 <= lat <= lat1 (no area weights)."""
    mx, my = (w.lon >= lon0) & (w.lon <= lon1), (w.lat >= lat0) & (w.lat <= lat1)
    if not (mx.any() and my.any()):
        raise ValueError("no ERA5 grid point inside the box")
    return w.u[:, my][:, :, mx].mean(axis=(1, 2)), w.v[:, my][:, :, mx].mean(axis=(1, 2))


def from_deg(u: np.ndarray | float, v: np.ndarray | float) -> np.ndarray:
    """Meteorological bearing (deg clockwise from north) the wind blows FROM, given toward-components."""
    return np.degrees(np.arctan2(-np.asarray(u), -np.asarray(v))) % 360.0


def to_segments(domain: Domain, times_s: np.ndarray, u_t: np.ndarray, v_t: np.ndarray) -> tuple[WindSegment, ...]:
    """Spatially uniform hourly series -> hold-last segments (sample k holds from times_s[k] to times_s[k+1]; the last one
    holds to the end of the run).  ``times_s`` starts at 0.  u_t, v_t (nt,) m/s TOWARD +x / +y.  Uniform in space, so the
    divergent part dropped by `segments_from_series` is zero."""
    ones = np.ones((len(times_s), domain.ny, domain.nx))
    return segments_from_series(domain, np.asarray(times_s, float),
                                u=np.asarray(u_t, float)[:, None, None] * ones, v=np.asarray(v_t, float)[:, None, None] * ones)

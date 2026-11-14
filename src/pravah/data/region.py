"""E17 regional forcing: ERA5 10 m winds and WorldPop masks on a square model domain (design: regional-case-design.md).

Everything here is a deterministic transform of immutable raw files; no network. Output is synthetic-study forcing only
("synthetic tracer under ERA5 winds, not validation").

Conventions
- Time: ERA5 `valid_time` is UTC and stays UTC throughout; hour h of the model calendar uses the instantaneous ERA5 wind
  at UTC time t_h for [t_h, t_h + 1 h). Nepal time (UTC+05:45) is for display only (`to_npt`).
- Space: a local tangent-plane (equirectangular) projection about the box centre, x east, y north, metres; over 100 km
  its distortion is < 0.1 %, and ERA5 u (eastward), v (northward) map directly to the model's (ux, uy) with no grid
  rotation. shortcut: no pyproj/UTM; upgrade if boxes reach several hundred km.
- Wind units: m s^-1, kept from ERA5 (u10, v10). Model (ux, uy) = (u10, v10).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R_EARTH = 6371008.8  # m, mean radius
NPT = np.timedelta64(345, "m")
MAX_SPEED = 60.0  # m/s; anything faster at 10 m is a decoding error, not weather


@dataclass(frozen=True)
class Box:
    """Square model box of side `side_m` centred at (lat0, lon0); grid cells ((i+.5)dx, (j+.5)dy) from the SW corner."""
    lat0: float
    lon0: float
    side_m: float

    def latlon(self, x, y):
        """Model metres (from the SW corner) -> (lat, lon) degrees."""
        h = self.side_m / 2
        lat = self.lat0 + np.degrees((np.asarray(y) - h) / R_EARTH)
        lon = self.lon0 + np.degrees((np.asarray(x) - h) / (R_EARTH * np.cos(np.radians(self.lat0))))
        return lat, lon

    def xy(self, lat, lon):
        h = self.side_m / 2
        return (np.radians(np.asarray(lon) - self.lon0) * R_EARTH * np.cos(np.radians(self.lat0)) + h,
                np.radians(np.asarray(lat) - self.lat0) * R_EARTH + h)

    def cell_centres(self, grid):
        """(lat[nx, ny], lon[nx, ny]) of model cell centres."""
        nx, ny = grid
        x = (np.arange(nx) + .5) * self.side_m / nx
        y = (np.arange(ny) + .5) * self.side_m / ny
        return self.latlon(*np.meshgrid(x, y, indexing="ij"))


BELAURI = Box(28.79, 80.31, 100_000.0)  # municipal centroid (regional-feasibility.md §4); 79.80-80.82 E, 28.34-29.24 N


# ---------------------------------------------------------------- ERA5
def read_era5(paths):
    """Concatenate ERA5 single-level NetCDF files -> dict(time UTC datetime64[s] (T,), lat (ny,) ascending, lon (nx,),
    u10, v10 [, blh] as (T, nlat, nlon) float arrays, m s^-1 / m). Duplicated times raise."""
    import xarray as xr
    parts = []
    for p in sorted(map(str, paths)):
        with xr.open_dataset(p, engine="h5netcdf") as ds:
            tname = "valid_time" if "valid_time" in ds.variables else "time"
            ds = ds.rename({tname: "time"}).sortby("latitude")
            keep = [v for v in ("u10", "v10", "blh") if v in ds]
            for d in [d for d in ds[keep[0]].dims if d not in ("time", "latitude", "longitude")]:
                ds = ds.isel({d: 0})  # length-1 'number'/'expver' dims of the new CDS NetCDF
            parts.append({v: ds[v].transpose("time", "latitude", "longitude").values.astype(float) for v in keep}
                         | dict(time=ds.time.values.astype("datetime64[s]"), lat=ds.latitude.values.astype(float),
                                lon=ds.longitude.values.astype(float)))
    out = {k: (np.concatenate([p[k] for p in parts]) if k not in ("lat", "lon") else parts[0][k]) for k in parts[0]}
    for p in parts[1:]:
        if not (np.array_equal(p["lat"], out["lat"]) and np.array_equal(p["lon"], out["lon"])):
            raise ValueError("ERA5 files are on different grids")
    o = np.argsort(out["time"], kind="stable")
    out.update({k: out[k][o] for k in out if k not in ("lat", "lon")})
    if (np.diff(out["time"]) == np.timedelta64(0, "s")).any():
        raise ValueError("duplicated ERA5 times")
    return out


def quality(u, v):
    """Bool mask of valid wind values: finite and speed <= MAX_SPEED."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    return np.isfinite(u) & np.isfinite(v) & (np.hypot(u, v) <= MAX_SPEED)


def interp_to_grid(field, lat, lon, box: Box, grid):
    """Bilinear interpolation of field (T, nlat, nlon) on the ascending ERA5 lat/lon to model cell centres -> (T, nx, ny).
    Raises if a model cell centre lies outside the ERA5 grid (no extrapolation) or the field has invalid values."""
    from scipy.interpolate import RegularGridInterpolator
    f = np.asarray(field, float)
    if not np.isfinite(f).all():
        raise ValueError("non-finite ERA5 values; check quality() first")
    la, lo = box.cell_centres(grid)
    if la.min() < lat.min() or la.max() > lat.max() or lo.min() < lon.min() or lo.max() > lon.max():
        raise ValueError("model box not inside the ERA5 grid; pad the request")
    pts = np.stack([la.ravel(), lo.ravel()], 1)
    return np.stack([RegularGridInterpolator((lat, lon), ft)(pts).reshape(la.shape) for ft in f])


def model_winds(era5, box: Box, grid, times=None):
    """(winds (T, 2) box-mean (ux, uy), u_grid (T, nx, ny), v_grid) at `times` (UTC datetime64; default all).
    The box mean is the mean of the bilinearly interpolated cell-centre winds (the uniform-wind variant the solver
    supports, forward.py). Missing times or invalid values raise."""
    t = era5["time"]
    idx = np.arange(len(t)) if times is None else np.searchsorted(t, np.asarray(times, "datetime64[s]"))
    if times is not None and (np.any(idx >= len(t)) or np.any(t[np.minimum(idx, len(t) - 1)] != np.asarray(times, "datetime64[s]"))):
        raise ValueError("requested times missing from ERA5")
    u, v = era5["u10"][idx], era5["v10"][idx]
    if not quality(u, v).all():
        raise ValueError("invalid ERA5 wind values")
    ug, vg = (interp_to_grid(a, era5["lat"], era5["lon"], box, grid) for a in (u, v))
    return np.stack([ug.mean((1, 2)), vg.mean((1, 2))], 1), ug, vg


def hourly_times(start, days):
    """UTC hourly datetime64[s] from `start` (YYYY-MM-DD, 00 UTC) for `days` days."""
    return np.datetime64(start, "s") + np.arange(days * 24) * np.timedelta64(3600, "s")


def to_npt(t):
    """UTC datetime64 -> Nepal time (UTC+05:45), display only."""
    return np.asarray(t, "datetime64[m]") + NPT


def wind_rose(winds, sectors=16, speed_bins=(0, 1, 2, 3, 5, np.inf)):
    """Frequency table (sectors, len(speed_bins)-1) of FROM-direction (meteorological, 0 = from north) x speed."""
    w = np.asarray(winds, float)
    s, th = np.hypot(w[:, 0], w[:, 1]), np.degrees(np.arctan2(-w[:, 0], -w[:, 1])) % 360
    sec = np.floor((th + 180 / sectors) % 360 / (360 / sectors)).astype(int)
    sb = np.digitize(s, speed_bins[1:-1])
    out = np.zeros((sectors, len(speed_bins) - 1))
    np.add.at(out, (sec, sb), 1)
    return out / len(w)


# ---------------------------------------------------------------- WorldPop
def read_worldpop(path):
    """GeoTIFF (WGS84, north-up) population counts -> (counts (nrow, ncol) with NaN for NoData, lat (nrow,) centres
    descending, lon (ncol,) centres). Uses Pillow (already installed); only the pixel-scale/tiepoint tags are read."""
    from PIL import Image
    im = Image.open(path)
    sx, sy, _ = im.tag_v2[33550]
    _, _, _, x0, y0, _ = im.tag_v2[33922]
    nod = float(im.tag_v2.get(42113, "nan"))
    a = np.array(im, float)
    a[(a == nod) | (a < 0)] = np.nan
    return a, y0 - (np.arange(a.shape[0]) + .5) * sy, x0 + (np.arange(a.shape[1]) + .5) * sx


def aggregate_counts(counts, lat, lon, box: Box, grid):
    """Sum of pixel counts whose centre falls in each model cell -> (pop (nx, ny), valid_frac (nx, ny)): valid_frac is the
    share of the cell's pixels that are not NoData (outside the country raster)."""
    nx, ny = grid
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    x, y = box.xy(LA, LO)
    i, j = np.floor(x / box.side_m * nx).astype(int), np.floor(y / box.side_m * ny).astype(int)
    m = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    pop, n, nv = np.zeros(grid), np.zeros(grid), np.zeros(grid)
    ok = m & np.isfinite(counts)
    np.add.at(pop, (i[ok], j[ok]), counts[ok])
    np.add.at(n, (i[m], j[m]), 1)
    np.add.at(nv, (i[ok], j[ok]), 1)
    return pop, np.divide(nv, n, out=np.zeros(grid), where=n > 0)


def candidate_mask(valid_frac, site_ij, min_valid=0.5):
    """Bool per candidate site: its cell lies mostly inside the population raster (i.e. inside Nepal, deployable)."""
    ij = np.asarray(site_ij, int)
    return valid_frac[ij[:, 0], ij[:, 1]] >= min_valid


def source_mask(pop_lattice, quantile=0.5):
    """Bool (na, nb) from population aggregated on the source lattice (aggregate_counts with grid = lattice): blocks at or
    above the `quantile` of the populated blocks. A proxy for where synthetic sources may sit (settlement-linked
    emissions), NOT an emission inventory; blocks outside the raster (India) are never sources."""
    pos = pop_lattice[pop_lattice > 0]
    if pos.size == 0:
        raise ValueError("no populated lattice block")
    return pop_lattice >= np.quantile(pos, quantile)

"""Population fields as PEOPLE PER CELL (counts) on a ``regionio.GridSpec`` (UTM metres, arrays [y, x]).

Counts are extensive (sum = people); density (people/km^2) is intensive. Only ``counts_to_density`` /
``density_to_counts`` convert between them (physical cell area dx*dy); nothing else accepts a density.
"""
from __future__ import annotations

import datetime
from collections.abc import Callable
from typing import Any

import numpy as np

from pravah.geo.regionio import GridSpec

M2_PER_KM2 = 1.0e6


# ---------------------------------------------------------------- count <-> density
def counts_to_density(counts: np.ndarray, grid: GridSpec) -> np.ndarray:
    """people/cell -> people/km^2."""
    return np.asarray(counts, float) / (grid.cell_area / M2_PER_KM2)


def density_to_counts(density: np.ndarray, grid: GridSpec) -> np.ndarray:
    """people/km^2 -> people/cell."""
    return np.asarray(density, float) * (grid.cell_area / M2_PER_KM2)


# ---------------------------------------------------------------- regridding
def _overlap_1d(src_lo: float, src_d: float, n_src: int, dst_lo: float, dst_d: float, n_dst: int) -> np.ndarray:
    """(n_dst, n_src) matrix: fraction of source cell s lying inside destination cell d (columns sum to <= 1)."""
    s0 = src_lo + src_d * np.arange(n_src)
    d0 = dst_lo + dst_d * np.arange(n_dst)
    ov = np.minimum(d0[:, None] + dst_d, s0[None, :] + src_d) - np.maximum(d0[:, None], s0[None, :])
    return np.clip(ov, 0.0, None) / src_d


def _report(counts: np.ndarray, out: np.ndarray) -> dict[str, float]:
    nd = int(np.isnan(counts).sum())
    tin = float(np.nansum(counts))
    tout = float(out.sum())
    return {"total_in": tin, "total_out": tout, "lost_outside": tin - tout, "nodata_cells": nd}


def regrid_counts(counts: np.ndarray, src: GridSpec, dst: GridSpec) -> tuple[np.ndarray, dict[str, float]]:
    """Conservative (area-overlap) regrid of counts between axis-aligned grids in the same UTM CRS.

    Each source cell's people are split in proportion to overlap area (uniform within a cell). NaN cells (nodata) are
    excluded from every total and contribute nothing; their number is reported, they are never silently zero-filled
    into the sums. ``lost_outside`` = people in source cells (or parts) outside ``dst``. Returns (out, report).
    """
    c = np.asarray(counts, float)
    if c.shape != src.shape:
        raise ValueError(f"counts shape {c.shape} != src grid shape {src.shape}")
    if src.epsg != dst.epsg:
        raise ValueError("src and dst grids must share one CRS (reproject first)")
    if np.nanmin(c, initial=0.0) < 0:
        raise ValueError("counts must be non-negative")
    wx = _overlap_1d(src.x0, src.dx, src.nx, dst.x0, dst.dx, dst.nx)
    wy = _overlap_1d(src.y0, src.dy, src.ny, dst.y0, dst.dy, dst.ny)
    out = wy @ np.nan_to_num(c, nan=0.0) @ wx.T
    return out, _report(c, out)


def counts_from_lonlat(arr: np.ndarray, west: float, north: float, dlon: float, dlat: float, dst: GridSpec,
                       nodata: float | None = None) -> tuple[np.ndarray, dict[str, float]]:
    """Counts raster in EPSG:4326 (row 0 = north edge ``north``, cell size dlon x dlat degrees) -> ``dst`` counts.

    shortcut: each source cell's people go whole to the dst cell holding its centre (not area-split); exact for totals
    inside dst, but only accurate when source cells are much smaller than dst cells. Upgrade to polygon overlap if not.
    """
    from pyproj import Transformer
    a = np.array(arr, float)
    if nodata is not None:
        a[a == nodata] = np.nan
    rows, cols = np.indices(a.shape)
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{dst.epsg}", always_xy=True)
    x, y = t.transform(west + (cols.ravel() + 0.5) * dlon, north - (rows.ravel() + 0.5) * dlat)
    j, i = dst.cell_of(x, y)
    v = np.nan_to_num(a.ravel(), nan=0.0)
    ok = j >= 0
    out = np.zeros(dst.shape)
    np.add.at(out, (j[ok], i[ok]), v[ok])
    return out, _report(a, out)


def counts_from_geotiff(path: str, dst: GridSpec) -> tuple[np.ndarray, dict[str, float]]:
    """Read a north-up EPSG:4326 counts GeoTIFF (band 1) with rasterio (optional extra ``geo``) -> ``dst`` counts."""
    import rasterio
    with rasterio.open(path) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 4326:
            raise ValueError("only EPSG:4326 rasters are supported; reproject first")
        tr = ds.transform
        if tr.b != 0 or tr.d != 0 or tr.e >= 0:
            raise ValueError("raster must be north-up")
        return counts_from_lonlat(ds.read(1), tr.c, tr.f, tr.a, -tr.e, dst, ds.nodata)


# ---------------------------------------------------------------- synthetic fixtures
def _largest_remainder(w: np.ndarray, total: int) -> np.ndarray:
    """Integer allocation of ``total`` proportional to w >= 0 (sum exact, deterministic ties by flat index)."""
    f = w / w.sum() * total
    n = np.floor(f)
    short = int(total - n.sum())
    order = np.argsort(-(f - n).ravel(), kind="stable")[:short]
    n.ravel()[order] += 1
    return n


def _gauss(grid: GridSpec, cx: float, cy: float, s: float) -> np.ndarray:
    X, Y = np.meshgrid(grid.x_centres(), grid.y_centres())
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))


def _centres(grid: GridSpec, rng: np.random.Generator, fracs: list[tuple[float, float]], jitter: float = 0.04):
    return [((fx + rng.uniform(-jitter, jitter)) * grid.lx, (fy + rng.uniform(-jitter, jitter)) * grid.ly)
            for fx, fy in fracs]


def _finish(w: np.ndarray, total: int) -> np.ndarray:
    if total < 0 or int(total) != total:
        raise ValueError("total_population must be a non-negative integer")
    return _largest_remainder(w, int(total))


def uniform(grid: GridSpec, total_population: int, seed: int = 0) -> np.ndarray:
    return _finish(np.ones(grid.shape), total_population)


def one_town(grid: GridSpec, total_population: int, seed: int = 0) -> np.ndarray:
    (c,) = _centres(grid, np.random.default_rng(seed), [(0.5, 0.5)])
    return _finish(_gauss(grid, *c, 0.06 * min(grid.lx, grid.ly)) + 1e-9, total_population)


def separated_settlements(grid: GridSpec, total_population: int, seed: int = 0) -> np.ndarray:
    cs = _centres(grid, np.random.default_rng(seed), [(0.2, 0.25), (0.75, 0.3), (0.5, 0.8)])
    s = 0.05 * min(grid.lx, grid.ly)
    w = sum(a * _gauss(grid, *c, s) for a, c in zip((1.0, 0.6, 0.35), cs))
    return _finish(np.asarray(w) + 1e-9, total_population)


def dense_core_sparse_surround(grid: GridSpec, total_population: int, seed: int = 0) -> np.ndarray:
    (c,) = _centres(grid, np.random.default_rng(seed), [(0.5, 0.5)])
    core = _gauss(grid, *c, 0.05 * min(grid.lx, grid.ly))
    return _finish(0.85 * core / core.sum() + 0.15 / core.size, total_population)


def mixed_rural_urban(grid: GridSpec, total_population: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cs = _centres(grid, rng, [(0.3, 0.6), (0.7, 0.35)])
    s = 0.07 * min(grid.lx, grid.ly)
    urban = sum(_gauss(grid, *c, s) for c in cs)
    urban = np.asarray(urban)
    rural = rng.lognormal(0.0, 0.5, grid.shape)
    return _finish(0.5 * urban / urban.sum() + 0.5 * rural / rural.sum(), total_population)


FIXTURES: dict[str, Callable[[GridSpec, int, int], np.ndarray]] = {
    f.__name__: f for f in (uniform, one_town, separated_settlements, dense_core_sparse_surround, mixed_rural_urban)}


def mask_to_study(counts: np.ndarray, study_fraction: np.ndarray, total: float) -> np.ndarray:
    """People per cell inside the study polygon only: counts * study_fraction (partial border cells keep their inside
    share), rescaled so the study holds exactly ``total`` people. Fractional people per cell are kept (not rounded)."""
    c = np.asarray(counts, float) * np.asarray(study_fraction, float)
    if c.shape != np.shape(study_fraction) or np.any(c < 0) or not c.sum() > 0 or not total > 0:
        raise ValueError("need counts >= 0 with positive mass inside the study, matching shapes and total > 0")
    return c * (float(total) / c.sum())


# ---------------------------------------------------------------- weights and metrics
def _as_area(area: float | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.broadcast_to(np.asarray(area, float), shape)


def mixed_weight(counts: np.ndarray, area: float | np.ndarray, beta: float) -> np.ndarray:
    """w = (1-beta) A/sum(A) + beta N/sum(N): dimensionless (ny, nx), sums to 1; beta=0 area, beta=1 population.
    Pass as ``weight=`` to metrics.field_mse (it takes a non-negative (ny, nx) map and normalises it itself)."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    n = np.nan_to_num(np.asarray(counts, float), nan=0.0)
    a = _as_area(area, n.shape)
    if np.any(n < 0) or np.any(a <= 0) or not n.sum() > 0:
        raise ValueError("need counts >= 0 with positive total and area > 0")
    return (1 - beta) * a / a.sum() + beta * n / n.sum()


def _wrmse(err: np.ndarray, w: np.ndarray, times: np.ndarray | None) -> float:
    e = np.asarray(err, float)
    if e.ndim == 2:
        e = e[None]
        if times is not None:
            raise ValueError("times given but err has no time dimension")
    elif e.ndim == 3:
        if times is None:
            raise ValueError("err is (time, y, x): pass times")
    else:
        raise ValueError("err must be (y, x) or (time, y, x)")
    if e.shape[1:] != w.shape:
        raise ValueError(f"err spatial shape {e.shape[1:]} != weight shape {w.shape}")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    if not w.sum() > 0:
        raise ValueError("zero total weight")
    if np.any(np.isnan(e[:, w > 0])):
        raise ValueError("NaN error where weight > 0")
    num = np.einsum("tyx,yx->t", np.nan_to_num(e) ** 2, w)
    if times is None:
        return float(np.sqrt(num[0] / w.sum()))
    t = np.asarray(times, float)
    if t.shape != (e.shape[0],) or t.size < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing, one per slice, at least 2")
    return float(np.sqrt(np.trapezoid(num, t) / (w.sum() * (t[-1] - t[0]))))


def pop_weighted_rmse(err_field: np.ndarray, counts: np.ndarray, study_weights: np.ndarray | None = None,
                      times: np.ndarray | None = None) -> float:
    """sqrt(sum N_i e_i^2 / sum N_i), N = counts * study_weights (study mask fraction in [0,1]); trapezoid in time for
    (time, y, x) errors. NaN counts (nodata) get weight 0; NaN error where weight > 0 raises."""
    n = np.nan_to_num(np.asarray(counts, float), nan=0.0)
    if study_weights is not None:
        n = n * np.asarray(study_weights, float)
    return _wrmse(err_field, n, times)


def area_weighted_rmse(err_field: np.ndarray, area: float | np.ndarray, study_weights: np.ndarray | None = None,
                       times: np.ndarray | None = None) -> float:
    """sqrt(sum A_i e_i^2 / sum A_i) with A = cell area (scalar or map) * optional study mask fractions."""
    shape = np.asarray(err_field).shape[-2:]
    a = _as_area(area, shape)
    if study_weights is not None:
        a = a * np.asarray(study_weights, float)
    return _wrmse(err_field, np.array(a), times)


# ---------------------------------------------------------------- provenance
def dataset_record(*, source: str, url: str, version: str, license: str, attribution: str,
                   original_resolution: str, crs: str, units: str, sha256: str, resampling: str,
                   limitations: str, download_date: str | None = None) -> dict[str, Any]:
    """Provenance dict for a population dataset (no network access; the caller supplies every field)."""
    if units not in ("people/cell", "people/km2"):
        raise ValueError("units must be 'people/cell' or 'people/km2'")
    return {"source": source, "url": url, "version": version, "license": license, "attribution": attribution,
            "original_resolution": original_resolution, "crs": crs, "units": units,
            "download_date": download_date or datetime.date.today().isoformat(), "sha256": sha256,
            "resampling": resampling, "limitations": limitations}

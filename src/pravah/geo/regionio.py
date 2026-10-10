"""Offline geometry/grid functions. Everything metric is done in a projected (UTM) CRS, never in degrees.

Conventions
* Axis order: geographic coordinates are (lon, lat); projected are (x=easting, y=northing); pyproj is always used with
  ``always_xy=True``.
* A ``GridSpec`` is a rectangular mesh in projected metres. Its lower-left corner is (x0, y0) in the UTM CRS; *local*
  coordinates (the pravah simulation coordinates) are UTM minus (x0, y0). Cell (j, i) spans local
  [i*dx, (i+1)*dx] x [j*dy, (j+1)*dy]; arrays are ``[y, x]`` = ``[j, i]`` with j increasing northward.
* Geometry functions take shapely geometries in the grid's projected CRS (absolute UTM metres).
* shapely / pyproj / geopandas are imported lazily inside functions.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


# ---------------------------------------------------------------- CRS
def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing (lon, lat): 326xx north, 327xx south (zone = floor((lon+180)/6)+1).
    The Norway/Svalbard zone exceptions are not implemented (they are irrelevant here)."""
    if not (-180.0 <= lon <= 180.0 and -80.0 <= lat <= 84.0):
        raise ValueError(f"lon/lat out of UTM range: ({lon}, {lat})")
    zone = min(int((lon + 180.0) // 6.0) + 1, 60)
    return (32600 if lat >= 0 else 32700) + zone


def to_metric(gdf: Any, epsg: int) -> Any:
    """Re-project a GeoDataFrame/GeoSeries to ``epsg``; refuses data without a declared CRS."""
    if gdf.crs is None:
        raise ValueError("data has no CRS; refusing to guess one")
    return gdf.to_crs(epsg=epsg)


def project_geom(geom: Any, src: int | str, dst: int | str) -> Any:
    """Re-project one shapely geometry between CRSs (EPSG ints or 'EPSG:xxxx' strings), x/y = lon/lat order."""
    from pyproj import Transformer
    from shapely.ops import transform
    t = Transformer.from_crs(crs_str(src), crs_str(dst), always_xy=True)
    return transform(t.transform, geom)


def crs_str(c: int | str) -> str:
    return f"EPSG:{c}" if isinstance(c, int) else c


def roundtrip_error_m(lon: float, lat: float, epsg: int) -> float:
    """WGS84 -> UTM -> WGS84 -> UTM, distance (m) between the two UTM points (expected << 1e-6 m)."""
    from pyproj import Transformer
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    back = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    x, y = fwd.transform(lon, lat)
    lon2, lat2 = back.transform(x, y)
    x2, y2 = fwd.transform(lon2, lat2)
    return math.hypot(x2 - x, y2 - y)


# ---------------------------------------------------------------- grid
@dataclass(frozen=True)
class GridSpec:
    """Rectangular metric grid; (x0, y0) = lower-left corner in the projected CRS ``epsg``."""
    x0: float
    y0: float
    nx: int
    ny: int
    dx: float
    dy: float
    epsg: int

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1 or not (self.dx > 0 and self.dy > 0):
            raise ValueError("grid needs nx, ny >= 1 and dx, dy > 0")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    @property
    def lx(self) -> float:
        return self.nx * self.dx

    @property
    def ly(self) -> float:
        return self.ny * self.dy

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) in projected metres."""
        return (self.x0, self.y0, self.x0 + self.lx, self.y0 + self.ly)

    def x_centres(self) -> np.ndarray:
        """Local x of cell centres, shape (nx,)."""
        return (np.arange(self.nx) + 0.5) * self.dx

    def y_centres(self) -> np.ndarray:
        return (np.arange(self.ny) + 0.5) * self.dy

    def cell_of(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """(j, i) integer indices of projected points (x, y); -1 where the point is outside the grid."""
        i = np.floor((np.atleast_1d(np.asarray(x, float)) - self.x0) / self.dx).astype(int)
        j = np.floor((np.atleast_1d(np.asarray(y, float)) - self.y0) / self.dy).astype(int)
        out = (i < 0) | (i >= self.nx) | (j < 0) | (j >= self.ny)
        return np.where(out, -1, j), np.where(out, -1, i)

    def boxes(self) -> np.ndarray:
        """shapely boxes in LOCAL coordinates, flat index k = j*nx + i."""
        import shapely
        i, j = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        i, j = i.ravel(), j.ravel()
        return shapely.box(i * self.dx, j * self.dy, (i + 1) * self.dx, (j + 1) * self.dy)

    def to_dict(self) -> dict[str, Any]:
        return {"x0": self.x0, "y0": self.y0, "nx": self.nx, "ny": self.ny, "dx": self.dx, "dy": self.dy,
                "epsg": self.epsg}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GridSpec:
        return cls(float(d["x0"]), float(d["y0"]), int(d["nx"]), int(d["ny"]), float(d["dx"]), float(d["dy"]),
                   int(d["epsg"]))


def make_grid(bounds_m: Sequence[float], dx: float, pad_to: float | None = None, dy: float | None = None,
              epsg: int = 0) -> GridSpec:
    """Smallest grid of ``dx`` x ``dy`` cells covering ``bounds_m`` = (minx, miny, maxx, maxy).

    ``pad_to`` (metres, default dx) rounds each extent UP to a multiple of that length; the padding is split equally on
    both sides. Grids of different dx built from the same bounds and the same ``pad_to`` (which must be a multiple of
    every dx) therefore cover the identical domain and share (x0, y0) -- the two resolutions nest."""
    dy = dx if dy is None else dy
    minx, miny, maxx, maxy = (float(b) for b in bounds_m)
    if not (maxx > minx and maxy > miny):
        raise ValueError("bounds must have positive extent")
    unit = float(pad_to) if pad_to is not None else float(dx)
    for d in (dx, dy):
        if abs(unit / d - round(unit / d)) > 1e-9:
            raise ValueError(f"pad_to={unit} must be a multiple of the cell size {d}")
    w, h = maxx - minx, maxy - miny
    lx = math.ceil(w / unit - 1e-9) * unit
    ly = math.ceil(h / unit - 1e-9) * unit
    nx, ny = round(lx / dx), round(ly / dy)
    return GridSpec(minx - (lx - w) / 2.0, miny - (ly - h) / 2.0, nx, ny, float(dx), float(dy), epsg)


# ---------------------------------------------------------------- rasterisation (exact, shapely)
def _local(geoms: Any, grid: GridSpec) -> Any:
    import shapely
    return shapely.transform(geoms, lambda c: c - np.array([grid.x0, grid.y0]))


def as_geom_array(geoms: Any) -> np.ndarray:
    import shapely
    g = geoms if isinstance(geoms, np.ndarray) else np.array(list(geoms) if not hasattr(geoms, "geom_type") else [geoms],
                                                              dtype=object)
    g = np.asarray(g, dtype=object).ravel()
    return g[~shapely.is_missing(g) & ~shapely.is_empty(g)]


def explode_parts(geoms: Any) -> np.ndarray:
    """Flatten multi-part geometries and collections into simple parts (empty ones dropped)."""
    import shapely
    g = as_geom_array(geoms)
    return shapely.get_parts(g) if len(g) else g


def polygon_cell_fractions(polygon_m: Any, grid: GridSpec) -> np.ndarray:
    """Fraction (ny, nx) in [0, 1] of each cell's area covered by ``polygon_m`` (Polygon/MultiPolygon/collection of
    polygons, projected metres), by exact shapely intersection. Overlapping parts are unioned first so nothing is
    counted twice. Only cells found by an STRtree bounding-box query are intersected.
    Area outside the grid is not counted (sum*cell_area < polygon area then)."""
    import shapely
    out = np.zeros(grid.ny * grid.nx)
    geoms = as_geom_array(polygon_m)
    if len(geoms) == 0:
        return out.reshape(grid.shape)
    if not bool(np.all(np.isin(shapely.get_type_id(geoms), (3, 6, 7)))):
        raise ValueError("polygon_cell_fractions needs polygonal geometry")
    poly = shapely.union_all(_local(geoms, grid))
    parts = shapely.get_parts(poly)
    boxes = grid.boxes()
    tree = shapely.STRtree(boxes)
    qi, ti = tree.query(parts, predicate="intersects")
    area = shapely.area(shapely.intersection(parts[qi], boxes[ti]))
    np.add.at(out, ti, area)
    return np.clip(out / grid.cell_area, 0.0, 1.0).reshape(grid.shape)


def rasterize_polygon_area(polys_m: Any, grid: GridSpec) -> np.ndarray:
    """Area fraction (ny, nx) of cells covered by the UNION of the polygons (e.g. building footprints)."""
    return polygon_cell_fractions(polys_m, grid)


def rasterize_line_length(lines_m: Any, grid: GridSpec) -> np.ndarray:
    """Metres of line per cell (ny, nx), by exact intersection with each cell box. Length is conserved for lines
    inside the grid (a segment lying exactly on a shared cell edge would be counted in both cells; real data
    essentially never does). Inputs must be LineString/MultiLineString (use ``explode_parts`` + filter otherwise)."""
    import shapely
    out = np.zeros(grid.ny * grid.nx)
    g = as_geom_array(lines_m)
    if len(g) == 0:
        return out.reshape(grid.shape)
    if not bool(np.all(np.isin(shapely.get_type_id(g), (1, 2, 5)))):
        raise ValueError("rasterize_line_length needs line geometry")
    g = _local(g, grid)
    boxes = grid.boxes()
    tree = shapely.STRtree(boxes)
    qi, ti = tree.query(g, predicate="intersects")
    np.add.at(out, ti, shapely.length(shapely.intersection(g[qi], boxes[ti])))
    return out.reshape(grid.shape)


def rasterize_points(xy_m: Any, grid: GridSpec) -> np.ndarray:
    """Count of points per cell (ny, nx); xy_m is an (n, 2) array of projected metres. Points outside are ignored."""
    xy = np.asarray(xy_m, float).reshape(-1, 2)
    j, i = grid.cell_of(xy[:, 0], xy[:, 1])
    ok = j >= 0
    return np.bincount(j[ok] * grid.nx + i[ok], minlength=grid.nx * grid.ny).astype(float).reshape(grid.shape)


# ---------------------------------------------------------------- DEM
def aggregate_dem(array: Any, transform: Any, src_crs: int | str, grid: GridSpec, how: str = "mean",
                  nodata: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate a north-up DEM raster onto ``grid`` by source-pixel centres falling in each cell.

    ``transform`` is an affine (``a b c d e f`` attributes, e.g. ``rasterio.Affine``) mapping (col, row) of the source
    raster to its CRS ``src_crs``. Nodata = NaN, masked entries, or values equal to ``nodata``; nodata pixels are NEVER
    averaged in. Returns (values, valid_fraction), both (ny, nx): values is NaN where a cell has no valid pixel;
    valid_fraction = valid pixels / all source pixels whose centre lies in the cell (0 where no pixel falls in it).
    ``how``: 'mean' | 'min' | 'max'."""
    from pyproj import Transformer
    if how not in ("mean", "min", "max"):
        raise ValueError(f"unknown aggregation {how!r}")
    if not all(hasattr(transform, k) for k in "abcdef"):
        raise TypeError("transform must be an affine with attributes a..f")
    arr = np.ma.filled(np.ma.asarray(array).astype(float), np.nan)
    if arr.ndim != 2:
        raise ValueError("DEM array must be 2-D")
    if nodata is not None and math.isfinite(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    t = transform
    cols, rows = np.meshgrid(np.arange(arr.shape[1]) + 0.5, np.arange(arr.shape[0]) + 0.5)
    xs = t.a * cols + t.b * rows + t.c
    ys = t.d * cols + t.e * rows + t.f
    X, Y = Transformer.from_crs(crs_str(src_crs), f"EPSG:{grid.epsg}", always_xy=True).transform(xs, ys)
    j, i = grid.cell_of(X, Y)
    inside = j >= 0
    flat = (j * grid.nx + i)[inside]
    z = arr[inside]
    n = grid.nx * grid.ny
    total = np.bincount(flat, minlength=n)
    ok = np.isfinite(z)
    nvalid = np.bincount(flat[ok], minlength=n)
    if how == "mean":
        vals = np.bincount(flat[ok], weights=z[ok], minlength=n) / np.maximum(nvalid, 1)
    elif how == "min":
        vals = np.full(n, np.inf)
        np.minimum.at(vals, flat[ok], z[ok])
    else:
        vals = np.full(n, -np.inf)
        np.maximum.at(vals, flat[ok], z[ok])
    vals = np.where(nvalid > 0, vals, np.nan)
    valid_fraction = np.where(total > 0, nvalid / np.maximum(total, 1), 0.0)
    return vals.reshape(grid.shape), valid_fraction.reshape(grid.shape)


def _gradients(dem: np.ndarray, dx: float, dy: float | None) -> tuple[np.ndarray, np.ndarray]:
    dz_dy, dz_dx = np.gradient(np.asarray(dem, float), dx if dy is None else dy, dx)
    return dz_dx, dz_dy  # rows run south -> north, so dz_dy is the northward derivative


def slope_deg(dem: np.ndarray, dx: float, dy: float | None = None) -> np.ndarray:
    """Terrain slope in degrees from central differences (one-sided at the edges); NaN propagates to neighbours."""
    gx, gy = _gradients(dem, dx, dy)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def hillshade(dem: np.ndarray, dx: float, azimuth: float = 315.0, altitude: float = 45.0,
              dy: float | None = None) -> np.ndarray:
    """Lambertian hillshade in [0, 1]; azimuth in degrees clockwise from north (direction of the light), altitude
    above the horizon. Rows must run south -> north (the pravah grid order)."""
    gx, gy = _gradients(dem, dx, dy)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, -gy)  # direction the surface faces, clockwise from north
    zen = math.radians(90.0 - altitude)
    az = math.radians(azimuth)
    hs = math.cos(zen) * np.cos(slope) + math.sin(zen) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0.0, 1.0)


def profile(dem_grid: np.ndarray, axis: Literal["ns", "ew"], index: int) -> np.ndarray:
    """1-D elevation profile: 'ns' = along y (south -> north) at column ``index``; 'ew' = along x (west -> east) at
    row ``index``."""
    a = np.asarray(dem_grid)
    if axis == "ns":
        return a[:, index].copy()
    if axis == "ew":
        return a[index, :].copy()
    raise ValueError("axis must be 'ns' or 'ew'")


# ---------------------------------------------------------------- sources and sites
def normalise_source(weights: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Source shape f = w / (sum(w) * cell_area) in 1/m^2, so that sum(f * cell_area) = 1 (the pravah convention for
    ``RasterSource``). ``weights`` must be finite, >= 0 and not all zero (otherwise the source does not exist: fails)."""
    w = np.asarray(weights, float)
    if w.shape != grid.shape:
        raise ValueError(f"weights shape {w.shape} != grid shape {grid.shape}")
    if not (np.all(np.isfinite(w)) and np.all(w >= 0) and w.sum() > 0):
        raise ValueError("source weights must be finite, non-negative and have a positive sum")
    return w / (w.sum() * grid.cell_area)


@dataclass(frozen=True)
class CandidateSite:
    id: str
    x: float  # projected metres
    y: float
    row: int
    col: int


def candidate_sites(polygon_m: Any, spacing_m: float, grid_origin: tuple[float, float]) -> list[CandidateSite]:
    """Regular lattice (x = ox + col*s, y = oy + row*s) clipped to the polygon (points on its boundary are excluded).
    IDs are 'c{row:03d}{col:03d}' from the lattice indices, so they are stable if the polygon or spacing is
    unchanged and independent of the order of evaluation. Sorted row-major. ``grid_origin`` (projected metres) must lie
    south-west of the polygon."""
    import shapely
    if spacing_m <= 0:
        raise ValueError("spacing must be positive")
    ox, oy = grid_origin
    minx, miny, maxx, maxy = polygon_m.bounds
    if minx < ox or miny < oy:
        raise ValueError("grid_origin must be south-west of the polygon (lattice indices would be negative)")
    cols = np.arange(math.ceil((minx - ox) / spacing_m), math.floor((maxx - ox) / spacing_m) + 1)
    rows = np.arange(math.ceil((miny - oy) / spacing_m), math.floor((maxy - oy) / spacing_m) + 1)
    cc, rr = np.meshgrid(cols, rows)
    cc, rr = cc.ravel(), rr.ravel()
    x, y = ox + cc * spacing_m, oy + rr * spacing_m
    keep = shapely.contains_xy(polygon_m, x, y)
    return [CandidateSite(f"c{int(r):03d}{int(c):03d}", float(px), float(py), int(r), int(c))
            for r, c, px, py in zip(rr[keep], cc[keep], x[keep], y[keep])]


def domain_bounds_m(study_m: Any, buffer_m: float) -> tuple[float, float, float, float]:
    """Municipality bounding box expanded by ``buffer_m`` on every side (projected metres)."""
    minx, miny, maxx, maxy = study_m.bounds
    return (minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)


def utm_to_lonlat(x: float, y: float, epsg: int) -> tuple[float, float]:
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(x, y)
    return float(lon), float(lat)


def lonlat_box(bounds_m: Sequence[float], epsg: int) -> tuple[float, float, float, float]:
    """Lon/lat bounding box (west, south, east, north) of a projected rectangle (all four corners are projected)."""
    from pyproj import Transformer
    minx, miny, maxx, maxy = bounds_m
    t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = t.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
    return (float(min(lon)), float(min(lat)), float(max(lon)), float(max(lat)))

"""Geo tests. ALL data here is a SYNTHETIC fixture, not Belauri: obviously fake names, simple shapes, and a fake plane DEM.
The network is never touched (HTTP is exercised through an interface mock, not science)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("geopandas")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")
pytest.importorskip("rasterio")
pytest.importorskip("osmnx")
pytest.importorskip("requests")

import rasterio  # noqa: E402
import shapely  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import LineString, Point, Polygon, box  # noqa: E402

from pravah.geo import fetch, prepare, regionio as rio  # noqa: E402


# ---------------------------------------------------------------- CRS
def test_utm_epsg_zones_and_hemispheres() -> None:
    assert rio.utm_epsg(80.3, 28.7) == 32644  # Belauri-like lon/lat
    assert rio.utm_epsg(-70.5, -10.0) == 32719  # southern + western
    assert rio.utm_epsg(0.1, -33.0) == 32731
    assert rio.utm_epsg(-0.1, 51.0) == 32630
    assert rio.utm_epsg(179.99, 0.0) == 32660
    from pyproj import CRS
    assert "UTM zone 44N" in CRS.from_epsg(32644).name
    assert "UTM zone 19S" in CRS.from_epsg(32719).name
    with pytest.raises(ValueError):
        rio.utm_epsg(10.0, 89.0)


@pytest.mark.parametrize("lon,lat", [(80.3, 28.7), (-70.5, -10.0), (0.1, -33.0)])
def test_roundtrip_error_tiny(lon: float, lat: float) -> None:
    assert rio.roundtrip_error_m(lon, lat, rio.utm_epsg(lon, lat)) < 1e-6


def test_to_metric_requires_crs_and_is_metric() -> None:
    import geopandas as gpd
    g = gpd.GeoSeries([box(80.0, 28.0, 80.01, 28.01)])
    with pytest.raises(ValueError):
        rio.to_metric(g, 32644)
    m = rio.to_metric(g.set_crs(4326), 32644)
    assert 900 < m.iloc[0].bounds[2] - m.iloc[0].bounds[0] < 1100  # ~0.01 deg of longitude at 28 N is ~ 980 m


# ---------------------------------------------------------------- grid and rasterisation
def test_make_grid_nests_and_roundtrips() -> None:
    b = (1000.0, 2000.0, 8300.0, 6100.0)
    g1, g2 = rio.make_grid(b, 500, pad_to=500, epsg=32644), rio.make_grid(b, 250, pad_to=500, epsg=32644)
    assert (g1.x0, g1.y0, g1.lx, g1.ly) == (g2.x0, g2.y0, g2.lx, g2.ly)
    assert g1.lx >= 7300 and g1.x0 <= 1000 and g1.x0 + g1.lx >= 8300
    assert g2.nx == 2 * g1.nx and g2.ny == 2 * g1.ny
    assert rio.GridSpec.from_dict(json.loads(json.dumps(g1.to_dict()))) == g1
    with pytest.raises(ValueError):
        rio.make_grid(b, 300, pad_to=500)


def _grid() -> rio.GridSpec:
    return rio.GridSpec(500_000.0, 3_000_000.0, 20, 16, 500.0, 500.0, 32644)


def test_polygon_fractions_sum_equals_area() -> None:
    grid = _grid()
    t = np.linspace(0, 2 * math.pi, 1000, endpoint=False)
    poly = Polygon(np.column_stack([505_000 + 3_333 * np.cos(t), 3_004_000 + 2_717 * np.sin(1.3 * t) * 0.9 + 0 * t]))
    poly = poly if poly.is_valid else poly.buffer(0)
    f = rio.polygon_cell_fractions(poly, grid)
    assert f.shape == grid.shape and f.min() >= 0 and f.max() <= 1
    assert abs(f.sum() * grid.cell_area - poly.area) / poly.area <= 1e-9
    mp = shapely.MultiPolygon([box(500_100, 3_000_100, 501_300, 3_000_900), box(507_000, 3_005_000, 508_700, 3_006_222)])
    f2 = rio.polygon_cell_fractions(mp, grid)
    assert abs(f2.sum() * grid.cell_area - mp.area) / mp.area <= 1e-9


def test_polygon_inside_one_cell() -> None:
    grid = _grid()
    poly = box(500_600, 3_000_600, 500_800, 3_000_900)  # inside cell (j=1, i=1)
    f = rio.polygon_cell_fractions(poly, grid)
    assert np.count_nonzero(f) == 1
    assert f[1, 1] == pytest.approx(poly.area / grid.cell_area, rel=1e-12)
    assert rio.rasterize_polygon_area([poly, poly], grid).sum() == pytest.approx(f.sum(), rel=1e-12)  # union, no double count


def test_polygon_outside_grid_is_not_counted() -> None:
    grid = _grid()
    f = rio.polygon_cell_fractions(box(509_000, 3_007_000, 511_000, 3_009_000), grid)  # half beyond the east/north edge
    assert f.sum() * grid.cell_area == pytest.approx(1000 * 1000)


def test_line_length_conserved_and_exact() -> None:
    grid = _grid()
    lines = [LineString([(500_100, 3_000_200), (508_900, 3_006_900), (503_000, 3_007_500)]),
             shapely.MultiLineString([[(501_250, 3_001_000), (501_250, 3_005_000)]])]
    out = rio.rasterize_line_length(lines, grid)
    assert out.sum() == pytest.approx(sum(l.length for l in lines), rel=1e-12)
    vert = rio.rasterize_line_length([lines[1]], grid)
    assert vert[2:10, 2].sum() == pytest.approx(4000.0) and np.count_nonzero(vert) == 8  # column 2, rows 2..9
    one = rio.rasterize_line_length([LineString([(500_100, 3_000_250), (500_400, 3_000_250)])], grid)
    assert np.count_nonzero(one) == 1 and one[0, 0] == pytest.approx(300.0)
    with pytest.raises(ValueError):
        rio.rasterize_line_length([box(0, 0, 1, 1)], grid)


def test_rasterize_points_counts() -> None:
    grid = _grid()
    c = rio.rasterize_points([(500_100, 3_000_100), (500_200, 3_000_300), (509_900, 3_007_900), (1.0, 1.0)], grid)
    assert c[0, 0] == 2 and c[15, 19] == 1 and c.sum() == 3  # the point outside the grid is ignored


# ---------------------------------------------------------------- DEM
def test_aggregate_dem_nodata_never_averaged() -> None:
    grid = rio.GridSpec(500_000.0, 3_000_000.0, 4, 3, 500.0, 500.0, 32644)
    ny_px, nx_px = 15, 20  # 100 m pixels, 5x5 per cell
    dem = np.full((ny_px, nx_px), 10.0)
    dem[:5, :5] = 20.0  # north-west block = cell (j=2, i=0)
    dem[0, 0] = -9999.0  # nodata value
    dem[1, 1] = np.nan
    dem[0:5, 5:10] = -9999.0  # whole cell (j=2, i=1) is nodata
    tr = from_origin(500_000.0, 3_001_500.0, 100.0, 100.0)
    vals, vf = rio.aggregate_dem(dem, tr, "EPSG:32644", grid, nodata=-9999.0)
    assert vals[2, 0] == pytest.approx(20.0) and vf[2, 0] == pytest.approx(23 / 25)
    assert np.isnan(vals[2, 1]) and vf[2, 1] == 0.0
    assert vals[0, 0] == pytest.approx(10.0) and vf[0, 0] == 1.0
    vmin, _ = rio.aggregate_dem(np.where(dem == -9999.0, np.nan, dem), tr, "EPSG:32644", grid, how="min")
    vmax, _ = rio.aggregate_dem(np.ma.masked_equal(dem, -9999.0), tr, "EPSG:32644", grid, how="max")
    assert vmin[2, 0] == 20.0 and vmax[0, 3] == 10.0
    with pytest.raises(ValueError):
        rio.aggregate_dem(dem, tr, "EPSG:32644", grid, how="mode")


def test_aggregate_dem_from_lonlat_raster() -> None:
    lon0, lat0 = 80.30, 28.70
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, 32644, always_xy=True).transform(lon0, lat0)
    grid = rio.make_grid((x - 2000, y - 2000, x + 2000, y + 2000), 500, epsg=32644)
    dem = np.full((200, 200), 7.5)
    vals, vf = rio.aggregate_dem(dem, from_origin(lon0 - 0.05, lat0 + 0.05, 0.0005, 0.0005), "EPSG:4326", grid)
    assert np.allclose(vals, 7.5) and np.allclose(vf, 1.0)


def test_slope_and_hillshade_on_plane() -> None:
    ny, nx, dx = 20, 30, 100.0
    x = (np.arange(nx) + 0.5) * dx
    east_rising = np.tile(0.1 * x, (ny, 1))
    assert np.allclose(rio.slope_deg(east_rising, dx), math.degrees(math.atan(0.1)))
    assert np.allclose(rio.slope_deg(np.full((ny, nx), 3.0), dx), 0.0)
    flat = rio.hillshade(np.zeros((ny, nx)), dx, altitude=45.0)
    assert np.allclose(flat, math.sin(math.radians(45.0)))
    # the surface rising to the east faces west: lit from the west (270), dark from the east (90)
    assert rio.hillshade(east_rising, dx, azimuth=270.0).mean() > rio.hillshade(east_rising, dx, azimuth=90.0).mean()
    north_rising = np.tile((0.1 * (np.arange(ny) + 0.5) * dx)[:, None], (1, nx))  # faces south
    assert rio.hillshade(north_rising, dx, azimuth=180.0).mean() > rio.hillshade(north_rising, dx, azimuth=0.0).mean()
    p = np.arange(12.0).reshape(3, 4)
    assert rio.profile(p, "ns", 1).tolist() == [1.0, 5.0, 9.0] and rio.profile(p, "ew", 2).tolist() == [8.0, 9.0, 10.0, 11.0]


# ---------------------------------------------------------------- sources, sites
def test_normalise_source_integral_one() -> None:
    grid = _grid()
    w = np.random.default_rng(0).random(grid.shape)
    f = rio.normalise_source(w, grid)
    assert (f * grid.cell_area).sum() == pytest.approx(1.0, rel=1e-12)
    for bad in (np.zeros(grid.shape), -w, np.full(grid.shape, np.nan)):
        with pytest.raises(ValueError):
            rio.normalise_source(bad, grid)
    with pytest.raises(ValueError):
        rio.normalise_source(w[:3], grid)


def test_candidate_sites_stable_and_clipped() -> None:
    poly = Polygon([(2100, 2100), (9100, 2100), (9100, 5100), (5100, 8100), (2100, 8100)])
    a = rio.candidate_sites(poly, 1500.0, (0.0, 0.0))
    b = rio.candidate_sites(poly, 1500.0, (0.0, 0.0))
    assert a == b and len(a) > 5
    assert all(poly.contains(Point(s.x, s.y)) for s in a)
    assert len({s.id for s in a}) == len(a)
    s0 = a[0]
    assert s0.id == f"c{s0.row:03d}{s0.col:03d}" and (s0.x, s0.y) == (s0.col * 1500.0, s0.row * 1500.0)
    assert {s.id for s in a} == {s.id for s in rio.candidate_sites(poly.buffer(0), 1500.0, (0.0, 0.0))}
    assert [s.id for s in a] == sorted(s.id for s in a)  # row-major order
    outside_corner = [(x, y) for x in (0, 1500, 3000, 9000) for y in (0, 1500) if not poly.contains(Point(x, y))]
    assert outside_corner and all(f"c{int(y / 1500):03d}{int(x / 1500):03d}" not in {s.id for s in a} for x, y in outside_corner)
    with pytest.raises(ValueError):
        rio.candidate_sites(poly, 1500.0, (5000.0, 0.0))


# ---------------------------------------------------------------- fetch (no network)
def test_copernicus_tile_naming() -> None:
    t = fetch.copernicus_dem_tile_urls((80.26, 28.62, 80.43, 28.78))
    assert t == [("Copernicus_DSM_COG_10_N28_00_E080_00_DEM",
                  "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N28_00_E080_00_DEM/"
                  "Copernicus_DSM_COG_10_N28_00_E080_00_DEM.tif")]
    assert [n for n, _ in fetch.copernicus_dem_tile_urls((-70.5, -0.5, -70.2, -0.2))] == ["Copernicus_DSM_COG_10_S01_00_W071_00_DEM"]
    names = [n for n, _ in fetch.copernicus_dem_tile_urls((-0.5, -0.5, 0.5, 0.5))]  # straddles equator and prime meridian
    assert names == [f"Copernicus_DSM_COG_10_{a}_00_{b}_00_DEM" for a in ("S01", "N00") for b in ("W001", "E000")]
    assert len(fetch.copernicus_dem_tile_urls((80.0, 28.0, 81.0, 29.0))) == 1  # integer edges do not add a tile
    assert fetch.tile_name(-12, 105) == "Copernicus_DSM_COG_10_S12_00_E105_00_DEM"


_GJ = {"features": [{"properties": {"shapeName": "Alpha", "shapeID": "A1"}}, {"properties": {"shapeName": "Beta", "shapeID": "B1"}},
                    {"properties": {"shapeName": "Beta", "shapeID": "B2"}}]}


def test_find_feature_errors() -> None:
    assert fetch.find_feature(_GJ, "Alpha")["properties"]["shapeID"] == "A1"
    assert fetch.find_feature(_GJ, "Beta", "B2")["properties"]["shapeID"] == "B2"
    with pytest.raises(ValueError, match="B1.*B2|B2.*B1"):
        fetch.find_feature(_GJ, "Beta")
    with pytest.raises(ValueError, match="closest names.*Alpha"):
        fetch.find_feature(_GJ, "Alpa")
    with pytest.raises(ValueError):
        fetch.find_feature(_GJ, "alpha")  # exact, case-sensitive
    with pytest.raises(ValueError):
        fetch.find_feature(_GJ, "Beta", "ZZ")


class _Resp:
    def __init__(self, data: bytes, headers: dict[str, str] | None = None) -> None:
        self.data, self.headers = data, headers or {}

    def raise_for_status(self) -> None: ...
    def close(self) -> None: ...
    def iter_content(self, chunk_size: int = 10) -> Any:
        for i in range(0, len(self.data), 7):
            yield self.data[i:i + 7]


class _Session:
    def __init__(self, data: bytes = b"synthetic payload", headers: dict[str, str] | None = None) -> None:
        self.data, self.calls, self.headers = data, [], headers

    def get(self, url: str, **kw: Any) -> _Resp:
        self.calls.append((url, kw))
        return _Resp(self.data, self.headers)


class _NoNetwork:
    def get(self, *a: Any, **k: Any) -> Any:
        raise AssertionError("network used")


def test_http_get_download_then_cache_hit(tmp_path: Path) -> None:
    s = _Session()
    dest = tmp_path / "x" / "f.bin"
    r1 = fetch.http_get("https://example.invalid/f.bin?sig=SECRET&a=1", dest, session=s, min_interval_s=0, license="L", attribution="A")
    assert r1["cache_hit"] is False and r1["bytes"] == 17 and "SECRET" not in json.dumps(r1) and "a=1" in r1["url"]
    assert s.calls[0][1]["headers"]["User-Agent"] == fetch.USER_AGENT and s.calls[0][1]["timeout"] == 30.0
    r2 = fetch.http_get("https://example.invalid/f.bin", dest, session=_NoNetwork(), expect_sha256=r1["sha256"])
    assert r2["cache_hit"] is True and r2["sha256"] == r1["sha256"] and r2["retrieved_utc"] == r1["retrieved_utc"]
    with pytest.raises(ValueError, match="sha256"):
        fetch.http_get("https://example.invalid/f.bin", dest, session=_NoNetwork(), expect_sha256="0" * 64)
    assert not list(dest.parent.glob("*.part"))


def test_http_get_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        fetch.http_get("https://example.invalid/a", tmp_path / "a", session=_Session(b"x" * 100), max_bytes=50, min_interval_s=0)
    with pytest.raises(ValueError, match="Content-Length"):
        fetch.http_get("https://example.invalid/b", tmp_path / "b", session=_Session(b"x", {"Content-Length": "999"}), max_bytes=50,
                       min_interval_s=0)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fetch.http_get("https://example.invalid/c", tmp_path / "c", session=_Session(), expect_sha256="1" * 64, min_interval_s=0)
    assert not any(p.exists() for p in (tmp_path / "a", tmp_path / "b", tmp_path / "c")) and not list(tmp_path.glob("*.part"))


def test_nominatim_null_extratags_and_geoboundaries_metadata(tmp_path: Path) -> None:
    raw = [{"osm_type": "relation", "osm_id": 1, "lat": "28.7", "lon": "80.3", "display_name": "Synthetic Town", "extratags": None}]
    res = fetch.parse_nominatim(raw)
    assert res[0]["extratags"] == {} and res[0]["lat"] == 28.7
    s = _Session(json.dumps(raw).encode())
    rec = fetch.nominatim_lookup("Synthetic Town", tmp_path, session=s)
    assert rec["results"][0]["osm_id"] == 1
    assert s.calls[0][1]["params"]["polygon_geojson"] == 0 and s.calls[0][1]["params"]["extratags"] == 1
    meta = {"boundaryID": "SYN-1", "boundaryType": "ADM3", "boundaryYearRepresented": "2019", "boundarySource": "synthetic",
            "boundaryLicense": "CC BY-SA 4.0", "gjDownloadURL": "https://example.invalid/raw/abc1234/file.geojson"}
    rec2 = fetch.geoboundaries_release("XXX", "ADM3", tmp_path, session=_Session(json.dumps(meta).encode()))
    assert rec2["release"]["boundaryID"] == "SYN-1" and rec2["release"]["gjDownloadURL"].endswith("file.geojson")
    with pytest.raises(ValueError, match="not pinned"):
        fetch.fetch_geoboundaries_geojson("XXX", "ADM3", tmp_path, expect_commit="deadbee", session=_NoNetwork())


# ---------------------------------------------------------------- border
def test_border_info_distance_bearing_sides() -> None:
    home = box(0, 0, 100_000, 100_000)
    neighbour = box(0, -100_000, 100_000, 0)  # shares the southern edge of home
    study = box(40_000, 12_000, 50_000, 22_000)
    info = prepare.international_border_info({"HOM": home, "NEI": neighbour}, study, home="HOM", side_tol_km=2.0)
    assert info["distance_to_home_boundary_km"] == pytest.approx(12.0)
    assert info["bearing_deg"] == pytest.approx(180.0, abs=1e-6) and info["sector"] == "S"
    assert info["facing_sides"] == ["south"]
    assert info["distance_by_side_km"]["south"] == pytest.approx(12.0) and info["distance_by_side_km"]["north"] == pytest.approx(78.0)
    assert info["neighbours"]["NEI"]["distance_km"] == pytest.approx(12.0) and not info["neighbours"]["NEI"]["touches"]
    assert info["study_fraction_inside_home"] == pytest.approx(1.0)
    touching = prepare.international_border_info({"HOM": home, "NEI": neighbour}, box(40_000, 0, 50_000, 10_000), home="HOM")
    assert touching["distance_to_home_boundary_km"] == 0.0 and touching["neighbours"]["NEI"]["touches"]
    assert prepare.sector(0) == "N" and prepare.sector(359) == "N" and prepare.sector(90) == "E" and prepare.sector(225) == "SW"


def test_classify_point_international_only_across_adm0() -> None:
    home, nei = box(0, 0, 100, 100), box(0, -100, 100, 0)
    study = box(40, 40, 60, 60)
    c = {"HOM": home, "NEI": nei}
    assert prepare.classify_point(Point(50, 50), study, c, "HOM")["class"] == "local"
    assert prepare.classify_point(Point(70, 50), study, c, "HOM")["class"] == "external_domestic"  # municipality edge only
    assert prepare.classify_point(Point(50, -20), study, c, "HOM") == {"class": "external_international", "country": "NEI"}
    # inside no ADM0 polygon: near the home boundary -> 'external_outside_home' (no international claim); far -> unclassified
    out = prepare.classify_point(Point(50, 105), study, c, "HOM", near_m=10.0)
    assert out["class"] == "external_outside_home" and out["country"] is None and out["distance_to_home_boundary_m"] == pytest.approx(5.0)
    assert prepare.classify_point(Point(500, 500), study, c, "HOM", near_m=10.0)["class"] == "unclassified"
    assert prepare.classify_point(Point(50, 50), study, {"HOM": home, "NEI": box(0, 0, 100, 100)}, "HOM")["class"] == "local"
    assert prepare.classify_point(Point(70, 50), study, {"HOM": home, "NEI": box(0, 0, 100, 100)}, "HOM")["class"] == "unclassified"


# ---------------------------------------------------------------- prepare_region on a synthetic dataset
def _fc(items: list[tuple[Any, dict[str, Any]]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": shapely.geometry.mapping(g), "properties": p}
                                                      for g, p in items]}


def _write_synthetic_raw(raw: Path) -> dict[str, Any]:
    """Synthetic fixture, not Belauri: a fake square 'Synthetic Town' near lon 80.35 / lat 28.70, a fake district around it,
    fake country 'HOM' (lat >= 28.60) and fake neighbour 'NEI' (south of it), a fake plane DEM and fake OSM objects."""
    raw.mkdir(parents=True)
    town = box(80.30, 28.65, 80.40, 28.75)
    files: dict[str, tuple[dict[str, Any], str]] = {
        "study": (_fc([(town, {"shapeName": "Synthetic Town", "shapeID": "SYN1"}), (box(80.4, 28.65, 80.5, 28.75), {"shapeName": "Other", "shapeID": "SYN2"})]), "adm3.geojson"),
        "adm2": (_fc([(box(80.2, 28.6, 80.5, 28.8), {"shapeName": "Synthetic District", "shapeID": "D1"})]), "adm2.geojson"),
        "adm0_HOM": (_fc([(box(79.0, 28.60, 81.5, 29.5), {"shapeName": "Home"})]), "adm0_HOM.geojson"),
        "adm0_NEI": (_fc([(box(79.0, 27.5, 81.5, 28.60), {"shapeName": "Neighbour"})]), "adm0_NEI.geojson"),
        "nominatim": ([{"osm_type": "relation", "osm_id": 424242, "lat": "28.70", "lon": "80.35", "display_name": "Synthetic Town, Synthetic District",
                        "category": "boundary", "type": "administrative", "extratags": None}], "nominatim.json"),
        "osm_roads": (_fc([(LineString([(80.25, 28.70), (80.45, 28.70)]), {"highway": "track"}),
                           (LineString([(80.35, 28.60), (80.35, 28.80)]), {"highway": "primary"})]), "roads.geojson"),
        "osm_buildings": (_fc([(box(80.31 + 0.01 * k, 28.66 + 0.01 * (k % 3), 80.3105 + 0.01 * k, 28.6605 + 0.01 * (k % 3)), {"building": "yes"})
                               for k in range(8)] + [(box(80.43, 28.70, 80.431, 28.701), {"building": "yes"})]), "buildings.geojson"),
        "osm_landuse": (_fc([(box(80.34, 28.68, 80.36, 28.70), {"landuse": "residential"})]), "landuse.geojson"),
        "osm_amenity": (_fc([(Point(80.33, 28.67), {"amenity": "school"}), (box(80.37, 28.72, 80.3705, 28.7205), {"amenity": "hospital"}),
                             (Point(80.35, 28.70), {"amenity": "restaurant"}), (Point(80.44, 28.70), {"amenity": "school"})]), "amenity.geojson"),
    }
    recs: dict[str, dict[str, Any]] = {}
    for role, (obj, name) in files.items():
        (raw / name).write_text(json.dumps(obj))
        recs[role] = {"url": f"https://example.invalid/{name}", "id": f"synthetic:{role}", "retrieved_utc": "2000-01-01T00:00:00+00:00",
                      "bytes": (raw / name).stat().st_size, "sha256": fetch.sha256_file(raw / name), "license": "synthetic", "attribution": "synthetic fixture, not Belauri",
                      "file": name, "feature_count": len(obj["features"]) if isinstance(obj, dict) else len(obj)}
    nx = ny = 300
    x = np.arange(nx) * 0.001
    z = (100.0 + 50.0 * x[None, :] + 20.0 * np.arange(ny)[:, None] * 0.001).astype("float32")
    z[:20, -20:] = -32768.0  # nodata corner far from the town
    with rasterio.open(raw / "dem.tif", "w", driver="GTiff", height=ny, width=nx, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(80.2, 28.85, 0.001, 0.001), nodata=-32768.0) as dst:
        dst.write(z, 1)
    recs["dem"] = {"url": "https://example.invalid/dem.tif", "id": "synthetic:dem", "retrieved_utc": "2000-01-01T00:00:00+00:00",
                   "bytes": (raw / "dem.tif").stat().st_size, "sha256": fetch.sha256_file(raw / "dem.tif"), "license": "synthetic",
                   "attribution": "synthetic fixture, not Belauri", "file": "dem.tif"}
    fetch.write_records(raw, recs)
    return {
        "study": {"name": "Synthetic Town", "geoboundaries": {"iso": "HOM", "level": "ADM3", "shapeName": "Synthetic Town", "shapeID": "SYN1"},
                  "expected": {"parent_district": "Synthetic District", "province": "none", "bbox_lonlat": [80.30, 28.65, 80.40, 28.75], "area_km2": 106.0},
                  "tolerances": {"area_frac": 0.1, "bbox_deg": 0.01}, "expected_epsg": 32644,
                  "independent_locator": {"nominatim_query": "Synthetic Town", "max_distance_km": 2.0}},
        "adm2": {"iso": "HOM", "level": "ADM2"}, "adm0": [{"iso": "HOM", "level": "ADM0"}, {"iso": "NEI", "level": "ADM0"}],
        "dem": {"min_valid_fraction": 0.95}, "grid": {"buffer_m": 6000, "resolutions_m": [1000, 500], "pad_to_m": 1000},
        "osm": {"tags": {}}, "external_patch": {"bearing_deg": "toward_border", "beyond_polygon_m": 3000, "sigma_m": 500},
    }


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    root = tmp_path_factory.mktemp("synthetic_belauri_like")
    cfg = _write_synthetic_raw(root / "raw")
    man = prepare.prepare_region(cfg, root / "raw", root / "data" / "processed")
    return man, root / "data" / "manifest.json", cfg


def test_prepare_manifest_complete_and_hashes_match(prepared: tuple[dict[str, Any], Path, dict[str, Any]]) -> None:
    man, mpath, _ = prepared
    base = mpath.parent
    assert mpath.exists() and json.loads(mpath.read_text())["name"] == "Synthetic Town"
    assert man["crs"]["epsg"] == 32644 and man["verification"]["epsg_matches_expected"] is True
    assert man["verification"]["roundtrip_error_m"] < 1e-6
    assert man["boundary_status"] == "verified", man["boundary"]["checks"]
    assert len(man["datasets"]) == 10 and len(man["processed"]) > 20
    for e in man["datasets"] + man["processed"]:
        for k in prepare.REQUIRED_ENTRY_KEYS:
            assert k in e, (e.get("role", e.get("name")), k)
        assert fetch.sha256_file(mpath.parent / e["file"] if e["kind"] == "processed" else _raw_path(base, e)) == e["sha256"]
    for key in ("w_road", "w_builtup", "w_facility", "w_external_patch"):
        assert any(e["name"].startswith(key) for e in man["processed"])
    for lv in man["verification"]:
        if lv.startswith("polygon_area_rel_err"):
            assert man["verification"][lv] <= 1e-9
    assert man["counts"]["buildings_domain"] == 9 and man["counts"]["facilities_in_study"] == 2 and man["counts"]["facilities_domain"] == 3
    assert any("completeness" in w for w in man["warnings"]) and man["incomplete"] == []


def _raw_path(manifest_dir: Path, e: dict[str, Any]) -> Path:
    return manifest_dir / e["file"]


def test_prepare_patch_external_and_border(prepared: tuple[dict[str, Any], Path, dict[str, Any]]) -> None:
    man, _, _ = prepared
    p, b = man["external_patch"], man["international_border"]
    assert p["group"] == "external" and p["origin"]["class"] == "external_domestic"
    assert 2.9 < p["distance_to_polygon_km"] < 3.1
    assert b["sector"] == "S" and b["facing_sides"] == ["south"] and 5.3 < b["distance_to_home_boundary_km"] < 5.7
    assert b["neighbours"]["NEI"]["touches"] is False
    for lv in man["levels"]:
        assert lv["external_patch_check_passed"] and lv["origin_mass_inside"]["external_patch"] <= 0.01
        assert lv["origin_mass_inside"]["road"] > 0.8  # boundary cells count by area fraction, so slightly below 1


def test_prepare_boundary_mismatch_is_approximate_not_silent(tmp_path: Path) -> None:
    cfg = _write_synthetic_raw(tmp_path / "raw")
    cfg["study"]["expected"]["area_km2"] = 500.0
    man = prepare.prepare_region(cfg, tmp_path / "raw", tmp_path / "d" / "processed")
    assert man["boundary_status"] == "approximate" and man["incomplete"]
    assert [c for c in man["boundary"]["checks"] if c["name"] == "area"][0]["ok"] is False


def test_prepare_fails_loudly_on_missing_data(tmp_path: Path) -> None:
    cfg = _write_synthetic_raw(tmp_path / "raw")
    recs = fetch.read_records(tmp_path / "raw")
    (tmp_path / "raw" / "fetch_records.json").write_text(json.dumps({"records": {k: v for k, v in recs.items() if k != "dem"}}))
    with pytest.raises(FileNotFoundError, match="dem"):
        prepare.prepare_region(cfg, tmp_path / "raw", tmp_path / "d" / "processed")
    (tmp_path / "raw" / "fetch_records.json").write_text(json.dumps({"records": recs}))
    (tmp_path / "raw" / "adm3.geojson").write_text((tmp_path / "raw" / "adm3.geojson").read_text() + " ")
    with pytest.raises(ValueError, match="sha256"):
        prepare.prepare_region(cfg, tmp_path / "raw", tmp_path / "d" / "processed")


def test_prepare_external_patch_inside_polygon_is_refused(tmp_path: Path) -> None:
    cfg = _write_synthetic_raw(tmp_path / "raw")
    cfg["external_patch"] = {"lonlat": [80.35, 28.70], "sigma_m": 500}
    with pytest.raises(ValueError, match="inside the study polygon"):
        prepare.prepare_region(cfg, tmp_path / "raw", tmp_path / "d" / "processed")


def test_load_region_roundtrip(prepared: tuple[dict[str, Any], Path, dict[str, Any]]) -> None:
    from pravah.config import GaussianSource, RasterSource, Scenario, StudyMask
    from pravah.sources import check_origin_tags
    man, mpath, _ = prepared
    reg = prepare.load_region(mpath)
    assert reg.name == "Synthetic Town" and reg.boundary_status == "verified" and sorted(reg.levels) == [500.0, 1000.0]
    lv = reg.level(500.0)
    assert lv.arrays["study_fraction"].shape == lv.grid.shape == (reg.domain(500.0).ny, reg.domain(500.0).nx)
    assert abs(lv.arrays["study_fraction"].sum() * lv.grid.cell_area / 1e6 - man["study"]["area_km2"]) < 1e-6
    assert reg.level(1000.0).grid.x0 == lv.grid.x0 and reg.level(1000.0).grid.y0 == lv.grid.y0  # nested domains
    mask, road, patch = reg.study_mask(500.0), reg.raster_source(500.0, "road"), reg.raster_source(500.0, "external_patch")
    assert isinstance(mask, StudyMask) and isinstance(road, RasterSource) and road.group == "local" and patch.group == "external"
    check_origin_tags(Scenario(domain=reg.domain(500.0), sources=(patch,), study=mask), tol=0.01)
    g = reg.external_gaussian()
    assert isinstance(g, GaussianSource) and g.group == "external"
    # lon/lat <-> local metres
    lon, lat = reg.georef.local_to_lonlat(g.x, g.y)
    x, y = reg.georef.lonlat_to_local(lon, lat)
    assert (float(x), float(y)) == pytest.approx((g.x, g.y), abs=1e-6)
    assert (float(lon), float(lat)) == pytest.approx(tuple(man["external_patch"]["centre_lonlat"]), abs=1e-9)
    cx, cy = reg.georef.lonlat_to_local(80.35, 28.70)
    j, i = (int(a[0]) for a in lv.grid.cell_of(float(cx) + reg.georef.x0, float(cy) + reg.georef.y0))
    assert lv.arrays["study_fraction"][j, i] == pytest.approx(1.0)
    with pytest.raises(KeyError):
        reg.level(123.0)
    # tamper detection
    f = mpath.parent / man["processed"][0]["file"]
    data = f.read_bytes()
    f.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="sha256"):
        prepare.load_region(mpath)
    f.write_bytes(data)


# ---------------------------------------------------------------- import hygiene
def test_cli_import_and_geo_help_do_not_import_gis() -> None:
    code = ("import sys; import pravah, pravah.cli, pravah.geo, pravah.geo.regionio, pravah.geo.fetch, pravah.geo.prepare;"
            "bad=[m for m in ('geopandas','rasterio','shapely','pyproj','osmnx','requests','fiona','pyogrio') if m in sys.modules];"
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    h = subprocess.run([sys.executable, "-m", "pravah", "geo", "--help"], capture_output=True, text=True)
    assert h.returncode == 0 and "prepare" in h.stdout and "fetch" in h.stdout


# ---------------------------------------------------------------------------------------------
# Boundary verification with a coarse district outline (synthetic geometry, not Belauri)
# ---------------------------------------------------------------------------------------------
def _verify_inputs(rel_shift_deg: float = 0.0):
    import shapely
    from pravah.geo.prepare import _project
    study = shapely.box(80.30, 28.65, 80.40, 28.75)
    cfg = {"geoboundaries": {"iso": "XXX", "level": "ADM3", "shapeName": "Synthetic Town", "shapeID": "X1"},
           "expected": {"parent_district": "Synthetic District", "parent_aliases": ["Synthetic District"],
                        "bbox_lonlat": [80.30, 28.65, 80.40, 28.75], "area_km2": None},
           "tolerances": {"area_frac": 0.1, "bbox_deg": 0.03, "district_cover": 0.9, "osm_iou": 0.9},
           "independent_locator": {"max_distance_km": 5.0, "require_osm_relation": True}}
    study_m = _project(study, 4326, 32644)
    cfg["expected"]["area_km2"] = study_m.area / 1e6
    props = {"shapeName": "Synthetic Town", "shapeID": "X1"}
    nom = [{"osm_type": "relation", "osm_id": 1, "lat": 28.70, "lon": 80.35,
            "display_name": "Synthetic Town, Synthetic District, Nowhere", "extratags": {"admin_level": "7"}}]
    rel = shapely.box(80.30 + rel_shift_deg, 28.65, 80.40 + rel_shift_deg, 28.75)
    return study_m, study, props, cfg, nom, rel


def test_coarse_district_outline_is_rescued_only_by_independent_evidence():
    import shapely
    from pravah.geo.prepare import verify_boundary
    study_m, study, props, cfg, nom, rel = _verify_inputs()
    coarse = shapely.box(80.30, 28.65, 80.40, 28.70)  # covers only half of the municipality
    out = verify_boundary(study_m, study, props, cfg, 32644, coarse, nom, rel, other_districts_wgs84=[])
    by = {c["name"]: c for c in out["checks"]}
    assert by["adm2_geometric_coverage"]["ok"] is False and by["adm2_geometric_coverage"]["informational"]
    assert by["osm_relation_agreement"]["ok"] is True and by["parent_district"]["ok"] is True
    assert out["boundary_status"] == "verified"
    # without the independent relation polygon the same coarse outline is NOT accepted
    out2 = verify_boundary(study_m, study, props, cfg, 32644, coarse, nom, None, other_districts_wgs84=[])
    assert out2["boundary_status"] == "approximate"


def test_disagreeing_relation_or_a_competing_district_blocks_verification():
    import shapely
    from pravah.geo.prepare import verify_boundary
    study_m, study, props, cfg, nom, rel_far = _verify_inputs(rel_shift_deg=0.05)  # half the width away: IoU ~ 0.33
    coarse = shapely.box(80.30, 28.65, 80.40, 28.70)
    out = verify_boundary(study_m, study, props, cfg, 32644, coarse, nom, rel_far, other_districts_wgs84=[])
    assert {c["name"]: c["ok"] for c in out["checks"]}["osm_relation_agreement"] is False
    assert out["boundary_status"] == "approximate"
    study_m, study, props, cfg, nom, rel = _verify_inputs()
    rival = shapely.box(80.30, 28.70, 80.40, 28.75)  # another district claims the other half
    out2 = verify_boundary(study_m, study, props, cfg, 32644, coarse, nom, rel, other_districts_wgs84=[rival])
    assert {c["name"]: c["ok"] for c in out2["checks"]}["parent_district"] is False
    assert out2["boundary_status"] == "approximate"

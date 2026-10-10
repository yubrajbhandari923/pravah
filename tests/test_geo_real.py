"""Integrity checks on the REAL prepared Belauri data (data/belauri). Skipped when the data or the geo libraries are absent.

These are not unit tests of formulas (see test_geo.py): they check that what is on disk is internally consistent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("shapely")
pytest.importorskip("pyproj")
rasterio = pytest.importorskip("rasterio")

from pravah.config import Domain, StudyMask  # noqa: E402
from pravah.geo import prepare  # noqa: E402
from pravah.geo import regionio as rio  # noqa: E402
from pravah.geo.fetch import sha256_file  # noqa: E402

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "belauri" / "manifest.json"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="data/belauri/manifest.json not present (run geo fetch/prepare)")
RES = (500.0, 250.0)
POLICY_SOURCES = ("road", "builtup", "facility")


@pytest.fixture(scope="module")
def man() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def region() -> prepare.Region:
    return prepare.load_region(MANIFEST)  # also verifies sha256 and array hashes


def test_processed_files_match_manifest_sha256(man: dict[str, Any]) -> None:
    assert man["processed"], "manifest lists no processed files"
    for e in man["processed"]:
        assert sha256_file(MANIFEST.parent / e["file"]) == e["sha256"], e["file"]


def test_raw_files_match_manifest_sha256(man: dict[str, Any]) -> None:
    for d in man["datasets"]:
        assert sha256_file(MANIFEST.parent / d["file"]) == d["sha256"], d["file"]


@pytest.mark.parametrize("res", RES)
def test_study_mask_area_equals_polygon_area(man: dict[str, Any], region: prepare.Region, res: float) -> None:
    lv = region.level(res)
    area_m2 = float(lv.arrays["study_fraction"].sum() * lv.grid.cell_area)
    assert area_m2 == pytest.approx(man["study"]["area_km2"] * 1e6, rel=1e-9)


@pytest.mark.parametrize("res", RES)
def test_mask_inside_domain_and_grid_contains_buffered_polygon(man: dict[str, Any], region: prepare.Region, res: float) -> None:
    lv = region.level(res)
    frac, g = lv.arrays["study_fraction"], lv.grid
    assert frac.shape == g.shape and frac.min() >= 0.0 and frac.max() <= 1.0 and (frac > 0).any()
    minx, miny, maxx, maxy = man["study"]["bbox_m"]
    buf = man["grid_config"]["buffer_m"]
    gx0, gy0, gx1, gy1 = g.bounds
    assert gx0 <= minx - buf + 1e-6 and gy0 <= miny - buf + 1e-6 and gx1 >= maxx + buf - 1e-6 and gy1 >= maxy + buf - 1e-6
    assert (g.x0, g.y0) == (region.georef.x0, region.georef.y0)
    j, i = np.nonzero(frac)  # every polygon cell lies inside the grid by construction; the polygon never touches the grid edge
    assert j.min() > 0 and i.min() > 0 and j.max() < g.ny - 1 and i.max() < g.nx - 1


@pytest.mark.parametrize("res", RES)
def test_dem_aggregation_masks_nodata_instead_of_averaging(man: dict[str, Any], region: prepare.Region, res: float) -> None:
    z, valid = region.level(res).arrays["dem_mean"], region.level(res).arrays["dem_valid_fraction"]
    assert not np.isnan(z[valid > 0.95]).any()
    assert np.isnan(z[valid == 0]).all()  # a cell with no valid pixel has no value (never an average of nodata)
    dem = [d for d in man["datasets"] if d["id"].startswith("Copernicus")][0]
    with rasterio.open(MANIFEST.parent / dem["file"]) as ds:
        raw = ds.read(1, masked=True)
    assert np.nanmin(z) >= float(raw.min()) - 1e-6 and np.nanmax(z) <= float(raw.max()) + 1e-6  # means stay within the raw range


@pytest.mark.parametrize("res", RES)
def test_source_weights_valid_and_local_sources_zero_outside_polygon(region: prepare.Region, res: float) -> None:
    lv = region.level(res)
    frac = lv.arrays["study_fraction"]
    for name in (*POLICY_SOURCES, "external_patch"):
        w = lv.arrays[f"w_{name}"]
        assert np.isfinite(w).all() and (w >= 0).all() and w.sum() > 0, name
        assert w.sum() * lv.grid.cell_area == pytest.approx(1.0, rel=1e-9), name
    for name in POLICY_SOURCES:
        assert (lv.arrays[f"w_{name}"][frac == 0] == 0).all(), name


def test_external_patch_outside_polygon_inside_domain(man: dict[str, Any], region: prepare.Region) -> None:
    p = man["external_patch"]
    g = region.level(500.0).grid
    cx, cy = p["centre_local_m"]
    assert 0 < cx < g.lx and 0 < cy < g.ly
    assert 3 * p["sigma_m"] <= cx <= g.lx - 3 * p["sigma_m"] and 3 * p["sigma_m"] <= cy <= g.ly - 3 * p["sigma_m"]
    assert p["distance_to_polygon_km"] > 0
    assert p["origin"]["class"] != "local"
    cell_j, cell_i = (int(a[0]) for a in g.cell_of(cx + g.x0, cy + g.y0))
    assert region.level(500.0).arrays["study_fraction"][cell_j, cell_i] == 0


def test_polygon_centroid_roundtrip_is_exact_to_1e6_m(man: dict[str, Any]) -> None:
    lon, lat = man["study"]["centroid_lonlat"]
    assert rio.roundtrip_error_m(lon, lat, man["crs"]["epsg"]) < 1e-6
    assert man["crs"]["roundtrip_error_m"] < 1e-6


@pytest.mark.parametrize("res", RES)
def test_region_yields_valid_domain_and_mask(region: prepare.Region, res: float) -> None:
    dom, mask, g = region.domain(res), region.study_mask(res), region.level(res).grid
    assert isinstance(dom, Domain) and isinstance(mask, StudyMask)
    assert (dom.nx, dom.ny) == (g.nx, g.ny) and dom.lx == g.lx and dom.ly == g.ly
    assert np.asarray(mask.weights).shape == (dom.ny, dom.nx)
    assert region.boundary_status == "verified"

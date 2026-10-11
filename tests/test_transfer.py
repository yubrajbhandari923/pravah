"""Transfer check: the geodesign pipeline runs on a second, UNLIKE synthetic region by changing only config + assets.
ALL data here is a SYNTHETIC fixture (obviously fake: L-shaped 'Lshape Town' at ~10 E / 33 S, home 'ZZA', neighbour 'ZZB', a
dense road grid in the east arm only, no facilities at all).  Real FiPy solves on the tiny smoke settings; no mocks.
Skipped without the optional geo extra (shapely / rasterio / pyproj)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shapely")
pytest.importorskip("pyproj")
pytest.importorskip("rasterio")

import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import LineString, Point, Polygon, box, mapping  # noqa: E402

from pravah import geodesign as G  # noqa: E402
from pravah.config import load_config  # noqa: E402
from pravah.geo import fetch, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BELAURI = ROOT / "data" / "belauri" / "manifest.json"
SMOKE = ROOT / "configs" / "geodesign_smoke.json"
LON0, LAT0, DLON, DLAT = 10.0, -33.20, 0.043, 0.036  # SW corner and a 4 km arm width; arms are 3x as long


def _fc(items: list[tuple[Any, dict[str, Any]]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": mapping(g), "properties": p} for g, p in items]}


def write_lshape_raw(raw: Path) -> dict[str, Any]:
    """Raw fetch cache + prepare config for the unlike region; returns the prepare config."""
    raw.mkdir(parents=True)
    town = Polygon([(LON0, LAT0), (LON0 + 3 * DLON, LAT0), (LON0 + 3 * DLON, LAT0 + DLAT), (LON0 + DLON, LAT0 + DLAT),
                    (LON0 + DLON, LAT0 + 3 * DLAT), (LON0, LAT0 + 3 * DLAT)])
    east = [LineString([(LON0 + DLON + 0.002 * k, LAT0), (LON0 + DLON + 0.002 * k, LAT0 + DLAT)]) for k in range(1, 40)] \
        + [LineString([(LON0, LAT0 + 0.003 * k), (LON0 + 3 * DLON, LAT0 + 0.003 * k)]) for k in range(1, 12)]
    north = [LineString([(LON0 + 0.02, LAT0), (LON0 + 0.02, LAT0 + 3 * DLAT)])]  # one road in the north arm
    files: dict[str, tuple[Any, str]] = {
        "study": (_fc([(town, {"shapeName": "Lshape Town", "shapeID": "LS1"}), (box(10.2, -33.2, 10.3, -33.1), {"shapeName": "Other", "shapeID": "LS2"})]), "adm.geojson"),
        "adm2": (_fc([(box(9.9, -33.3, 10.25, -33.05), {"shapeName": "Lshape District", "shapeID": "D1"})]), "adm2.geojson"),
        "adm0_ZZA": (_fc([(box(9.0, -33.235, 11.0, -32.5), {"shapeName": "Home"})]), "adm0_ZZA.geojson"),
        "adm0_ZZB": (_fc([(box(9.0, -34.0, 11.0, -33.235), {"shapeName": "Neighbour"})]), "adm0_ZZB.geojson"),
        "osm_roads": (_fc([(g, {"highway": "residential"}) for g in east + north]), "roads.geojson"),
        "osm_buildings": (_fc([(box(LON0 + DLON + 0.002 * k + 0.0005, LAT0 + 0.004 + 0.002 * (k % 4), LON0 + DLON + 0.002 * k + 0.0012, LAT0 + 0.0047 + 0.002 * (k % 4)),
                                 {"building": "yes"}) for k in range(1, 40)]), "buildings.geojson"),
        "osm_amenity": (_fc([(Point(LON0 + 0.1, LAT0 + 0.01), {"amenity": "restaurant"})]), "amenity.geojson"),  # no facility at all
    }
    recs: dict[str, dict[str, Any]] = {}
    for role, (obj, name) in files.items():
        (raw / name).write_text(json.dumps(obj))
        recs[role] = {"url": f"https://example.invalid/{name}", "id": f"synthetic:{role}", "retrieved_utc": "2000-01-01T00:00:00+00:00",
                      "bytes": (raw / name).stat().st_size, "sha256": fetch.sha256_file(raw / name), "license": "synthetic",
                      "attribution": "synthetic fixture, not Belauri", "file": name, "feature_count": len(obj["features"])}
    ny, nx = 400, 500
    z = (50.0 + 400.0 * np.arange(nx)[None, :] / nx + 30.0 * np.sin(np.arange(ny)[:, None] / 25.0)).astype("float32")
    with rasterio.open(raw / "dem.tif", "w", driver="GTiff", height=ny, width=nx, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(9.8, -33.0, 0.001, 0.001), nodata=-32768.0) as dst:
        dst.write(z, 1)
    recs["dem"] = {"url": "https://example.invalid/dem.tif", "id": "synthetic:dem", "retrieved_utc": "2000-01-01T00:00:00+00:00",
                   "bytes": (raw / "dem.tif").stat().st_size, "sha256": fetch.sha256_file(raw / "dem.tif"), "license": "synthetic",
                   "attribution": "synthetic fixture, not Belauri", "file": "dem.tif"}
    fetch.write_records(raw, recs)
    from pyproj import Transformer
    t = Transformer.from_crs(4326, 32732, always_xy=True)
    xs, ys = zip(*(t.transform(x, y) for x, y in town.exterior.coords))
    area = Polygon(zip(xs, ys)).area / 1e6
    return {"study": {"name": "Lshape Town", "geoboundaries": {"iso": "ZZA", "level": "ADM3", "shapeName": "Lshape Town", "shapeID": "LS1"},
                      "expected": {"parent_district": "Lshape District", "province": "none", "bbox_lonlat": list(town.bounds), "area_km2": area},
                      "tolerances": {"area_frac": 0.05, "bbox_deg": 0.01}, "expected_epsg": 32732},
            "adm2": {"iso": "ZZA", "level": "ADM2"}, "adm0": [{"iso": "ZZA", "level": "ADM0"}, {"iso": "ZZB", "level": "ADM0"}],
            "dem": {"min_valid_fraction": 0.95}, "grid": {"buffer_m": 5000, "resolutions_m": [500], "pad_to_m": 500},
            "osm": {"tags": {}}, "external_patch": {"bearing_deg": "toward_border", "beyond_polygon_m": 1500, "sigma_m": 400}}


def transfer_cfg(manifest: Path, **region: Any) -> dict[str, Any]:
    """The Belauri smoke config; only the region block differs."""
    cfg = load_config(SMOKE)
    cfg["region"]["manifest"] = str(manifest)
    cfg["region"].update(region)
    return cfg


@pytest.fixture(scope="module")
def lshape(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("lshape")
    prepare.prepare_region(write_lshape_raw(root / "raw"), root / "raw", root / "data" / "processed")
    return root / "data" / "manifest.json"


LSHAPE_REGION = {"crop_buffer_m": 3000.0, "study": {"shapeName": "Lshape Town", "shapeID": "LS1"},
                 "roles": {"home_adm0": "adm0_ZZA", "neighbour_adm0": "adm0_ZZB"}}


def _outputs(out: Path) -> dict[str, list[str]]:
    """Relative file -> csv columns (None-free: non-csv files map to [])."""
    return {str(p.relative_to(out)): (list(pd.read_csv(p, nrows=0).columns) if p.suffix == ".csv" else [])
            for p in sorted(out.rglob("*")) if p.is_file()}


def test_roles_and_no_neighbour_geography(lshape: Path) -> None:
    cfg = transfer_cfg(lshape, **copy.deepcopy(LSHAPE_REGION))
    geo = G.load_geography(cfg)
    assert set(geo.raw_sha256) == {"study", "adm0_ZZA", "adm0_ZZB", "osm_roads", "osm_amenity"}
    assert geo.ind is not None and len(geo.facilities) == 0 and geo.epsg == 32732
    assert not geo.study.convex_hull.equals_exact(geo.study, 1e-6)  # non-convex
    none = G.load_geography({**cfg, "region": {**cfg["region"], "roles": {**LSHAPE_REGION["roles"], "neighbour_adm0": None}}})
    assert none.ind is None and "adm0_ZZB" not in none.raw_sha256
    t = G.build_candidate_table(none, cfg)
    assert not t["inside_coarse_ind_outline"].any() and np.isfinite(t["near_border_km"]).all()
    assert np.isinf(t["dist_facility_m"]).all()  # no facility anywhere: reported as inf, not an error
    with pytest.raises(ValueError, match="no raw dataset with role 'adm0_NPL'"):
        G.load_geography(transfer_cfg(lshape, crop_buffer_m=3000.0, study=LSHAPE_REGION["study"]))  # Belauri default roles do not exist here


def test_pipeline_runs_on_unlike_region_same_outputs_as_belauri(lshape: Path, tmp_path: Path) -> None:
    out_l = tmp_path / "lshape"
    G.run_geodesign(transfer_cfg(lshape, **copy.deepcopy(LSHAPE_REGION)), out_l)
    got = _outputs(out_l)
    assert {"candidates.csv", "cases.csv", "layouts.csv", "summary.csv", "alpha_selected.json", "test_raw.csv"} <= set(got)
    cand = pd.read_csv(out_l / "candidates.csv")
    assert (cand["category"] == G.INTERIOR).sum() >= 3 and not cand["in_polygon"].eq(False)[cand["category"] == G.INTERIOR].any()
    lay = pd.read_csv(out_l / "layouts.csv")
    assert set(lay["method"]) >= {"greedy_total", "greedy_attribution", "random", "spread"}
    assert np.isfinite(pd.read_csv(out_l / "summary.csv").select_dtypes("number").to_numpy()[:, :3]).all()
    if not BELAURI.exists():
        pytest.skip("data/belauri/manifest.json is missing: Belauri side of the comparison not run")
    out_b = tmp_path / "belauri"
    G.run_geodesign(transfer_cfg(BELAURI), out_b)
    assert got == _outputs(out_b)  # same file set, same column names

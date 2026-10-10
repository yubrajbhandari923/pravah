"""Offline preparation: cached raw files (from ``geo fetch``) -> processed arrays + ``manifest.json`` -> ``Region``.

Nothing is invented: a missing or unusable input raises. The boundary is marked ``boundary_status: 'verified'`` only if
every verification check passes including the independent (OSM/Nominatim) locator; otherwise 'approximate' and the
reasons are listed. Emission "source hypotheses" are *synthetic spatial priors* built from real geometry
(roads / built-up / facilities inside the municipality, plus an external patch outside it).
"""
from __future__ import annotations

from collections.abc import Sequence

import datetime as dt
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Domain, GaussianSource, RasterSource, Scenario, StudyMask, array_sha256
from ..io import write_json
from ..sources import check_origin_tags, origin_mass_fractions
from . import regionio as rio
from .fetch import find_feature, parse_nominatim, read_records, sha256_file

SCHEMA = 1
REQUIRED_ENTRY_KEYS = ("url", "id", "retrieved_utc", "license", "attribution", "sha256", "crs", "bbox_lonlat", "units",
                       "resampling", "native_resolution", "source_to_grid", "assumptions")
SECTORS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
SYNTHETIC_PRIOR = "synthetic spatial prior (real geometry, assumed emission pattern; not measured emissions)"


# ---------------------------------------------------------------- small geometry helpers
def bearing_deg(dx: float, dy: float) -> float:
    """Compass bearing (degrees clockwise from north) of the vector (dx east, dy north)."""
    return math.degrees(math.atan2(dx, dy)) % 360.0


def sector(bearing: float) -> str:
    return SECTORS[int((bearing + 11.25) % 360.0 // 22.5)]


def _read_geojson(path: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    import shapely
    gj = json.loads(path.read_text())
    feats = gj["features"]
    geoms = np.array([shapely.geometry.shape(f["geometry"]) if f.get("geometry") else shapely.Point()
                      for f in feats] + [None], dtype=object)[:-1]
    return geoms, [f.get("properties") or {} for f in feats]


def _project(geoms: Any, src: int | str, dst: int | str) -> Any:
    import shapely
    from pyproj import Transformer
    t = Transformer.from_crs(rio.crs_str(src), rio.crs_str(dst), always_xy=True)

    def f(c: np.ndarray) -> np.ndarray:
        x, y = t.transform(c[:, 0], c[:, 1])
        return np.column_stack([x, y])
    return shapely.transform(shapely.segmentize(geoms, 0.02) if src == 4326 else geoms, f)  # densify so long edges follow the projection


def _polygons(geoms: Any) -> np.ndarray:
    """Valid polygonal parts only (OSM polygons are sometimes invalid; make_valid, then drop non-areal remnants)."""
    import shapely
    g = rio.explode_parts(shapely.make_valid(rio.as_geom_array(geoms)))
    return g[shapely.get_type_id(g) == 3] if len(g) else g


def _lines(geoms: Any) -> np.ndarray:
    import shapely
    g = rio.explode_parts(geoms)
    return g[shapely.get_type_id(g) == 1] if len(g) else g


# ---------------------------------------------------------------- international border
def international_border_info(country_polygons: dict[str, Any], study_polygon: Any, home: str = "NPL",
                              side_tol_km: float = 5.0) -> dict[str, Any]:
    """Distance and direction from the study polygon to the national (ADM0) boundary. ``country_polygons`` maps ISO code
    -> ADM0 (Multi)Polygon and ``study_polygon`` is the municipality; BOTH in one projected CRS (metres).

    * ``distance_to_home_boundary_km``: study polygon -> boundary of ``home`` (0 if they touch/cross);
      ``bearing_deg``/``sector``: from the study centroid to the nearest boundary point.
    * per neighbour: distance to that country's polygon, whether it touches, bearing/sector of the nearest point.
    * ``distance_by_side_km``: for each side of the study bounding box, the distance from that side to the part of the
      home boundary lying in the half-strip beyond it (inf if none); ``facing_sides``: sides within ``side_tol_km`` of the
      smallest such distance (compass letters of ``sector`` if the border is diagonal to all four strips).
    A flow is INTERNATIONAL only if it crosses this ADM0 boundary; the municipality edge never counts (see
    ``classify_point``)."""
    import shapely
    from shapely.ops import nearest_points
    if home not in country_polygons:
        raise ValueError(f"home country {home!r} not in country_polygons {sorted(country_polygons)}")
    home_g = country_polygons[home]
    bnd = home_g.boundary
    cen = study_polygon.centroid

    def toward(target: Any) -> tuple[float, str]:
        p = nearest_points(cen, target)[1]
        b = bearing_deg(p.x - cen.x, p.y - cen.y)
        return b, sector(b)

    b, sec = toward(bnd)
    minx, miny, maxx, maxy = study_polygon.bounds
    big = 1e7
    sides = {  # (side segment, half-strip lying beyond that side)
        "west": (shapely.LineString([(minx, miny), (minx, maxy)]), shapely.box(minx - big, miny, minx, maxy)),
        "east": (shapely.LineString([(maxx, miny), (maxx, maxy)]), shapely.box(maxx, miny, maxx + big, maxy)),
        "south": (shapely.LineString([(minx, miny), (maxx, miny)]), shapely.box(minx, miny - big, maxx, miny)),
        "north": (shapely.LineString([(minx, maxy), (maxx, maxy)]), shapely.box(minx, maxy, maxx, maxy + big))}
    by_side: dict[str, float] = {}
    for k, (seg, strip) in sides.items():
        part = bnd.intersection(strip)
        by_side[k] = float("inf") if part.is_empty else float(seg.distance(part)) / 1000.0
    dmin = min(by_side.values())
    if math.isinf(dmin):  # border only diagonal to the bounding box: fall back to the compass sector of the nearest point
        facing = [n for n, ch in (("north", "N"), ("south", "S"), ("east", "E"), ("west", "W")) if ch in sec]
    else:
        facing = [k for k, d in by_side.items() if d <= dmin + side_tol_km]
    neighbours = {}
    for iso, g in country_polygons.items():
        if iso == home:
            continue
        nb, ns = toward(g)
        neighbours[iso] = {"distance_km": float(study_polygon.distance(g)) / 1000.0,
                           "touches": bool(study_polygon.intersects(g)), "bearing_deg": nb, "sector": ns}
    return {
        "home": home,
        "study_fraction_inside_home": float(study_polygon.intersection(home_g).area / study_polygon.area),
        "distance_to_home_boundary_km": float(study_polygon.distance(bnd)) / 1000.0,
        "bearing_deg": b, "sector": sec,
        "distance_by_side_km": by_side,
        "facing_sides": facing,
        "neighbours": neighbours,
        "note": "International means crossing the verified ADM0 boundary of the home country; the municipality edge is "
                "only a local/external (domestic) boundary.",
    }


def classify_point(point: Any, study_polygon: Any, country_polygons: dict[str, Any], home: str = "NPL") -> dict[str, Any]:
    """Origin class of a projected point: 'local' (inside the study polygon), 'external_domestic' (inside the home ADM0 but
    outside the study), 'external_international' (inside another country's ADM0), or 'unclassified' (in no ADM0 polygon
    or in several)."""
    if study_polygon.contains(point):
        return {"class": "local", "country": home}
    inside = [iso for iso, g in country_polygons.items() if g.contains(point)]
    if inside == [home]:
        return {"class": "external_domestic", "country": home}
    if len(inside) == 1:
        return {"class": "external_international", "country": inside[0]}
    return {"class": "unclassified", "country": None, "countries_containing": inside}


# ---------------------------------------------------------------- boundary verification
def verify_boundary(study_m: Any, study_wgs84: Any, props: dict[str, Any], cfg_study: dict[str, Any], epsg: int,
                    district_wgs84: Any | None, nominatim_results: list[dict[str, Any]] | None,
                    osm_relation_wgs84: Any | None = None, other_districts_wgs84: Sequence[Any] = ()) -> dict[str, Any]:
    """Run the boundary checks; each check is {name, ok (True/False/None=not performed), detail[, informational]}. Status
    is 'verified' only if every NON-informational check is True.

    Parent district: the ADM2 layer of a geoBoundaries release can be a coarse outline (e.g. a 2006 district polygon with a
    few dozen vertices), so geometric containment may fail although the municipality is correctly in that district. The
    check passes if EITHER the ADM2 polygon covers >= district_cover of the study polygon OR independent evidence agrees:
    the OSM administrative relation (matching polygon, see osm_relation_agreement) names the district in its hierarchy
    and no OTHER district overlaps the study polygon by more than 1%. The raw ADM2 coverage is always recorded as the
    informational check adm2_geometric_coverage (never hidden)."""
    exp = cfg_study["expected"]
    gb = cfg_study["geoboundaries"]
    tol = cfg_study.get("tolerances", {})
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool | None, detail: str, informational: bool = False) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail, **({"informational": True} if informational else {})})

    add("name", props.get("shapeName") == gb["shapeName"], f"shapeName={props.get('shapeName')!r}")
    add("shape_id", props.get("shapeID") == gb.get("shapeID") if gb.get("shapeID") else None, f"shapeID={props.get('shapeID')!r}")
    add("geometry_valid", bool(study_m.is_valid and not study_m.is_empty), "shapely is_valid on the projected polygon")
    area = study_m.area / 1e6
    ea, at = float(exp["area_km2"]), float(tol.get("area_frac", 0.1))
    add("area", abs(area - ea) <= at * ea, f"area={area:.2f} km2 (UTM {epsg}), expected {ea} +/- {at:.0%}")
    bt = float(tol.get("bbox_deg", 0.03))
    w, s, e, n = study_wgs84.bounds
    ew, es, ee, en = exp["bbox_lonlat"]
    dev = max(abs(w - ew), abs(s - es), abs(e - ee), abs(n - en))
    add("bbox", dev <= bt, f"bbox=({w:.3f},{s:.3f},{e:.3f},{n:.3f}) max deviation {dev:.3f} deg (tol {bt})")
    cov: float | None = None
    if district_wgs84 is not None:
        cov = float(study_wgs84.intersection(district_wgs84).area / study_wgs84.area)
        add("adm2_geometric_coverage", cov >= float(tol.get("district_cover", 0.9)),
            f"{cov:.1%} of the polygon lies inside the ADM2 feature {exp['parent_district']!r}", informational=True)
    rel_ok: bool | None = None
    need_rel = bool(cfg_study.get("independent_locator", {}).get("require_osm_relation", False))
    if osm_relation_wgs84 is None:
        add("osm_relation_agreement", None, "no OSM administrative relation polygon available (not performed)",
            informational=not need_rel)
    else:
        a, b = study_m, _project(osm_relation_wgs84, 4326, epsg).buffer(0)
        iou = float(a.intersection(b).area / a.union(b).area)
        da = abs(a.area - b.area) / a.area
        rel_ok = iou >= float(tol.get("osm_iou", 0.9)) and da <= float(tol.get("area_frac", 0.1))
        add("osm_relation_agreement", rel_ok,
            f"IoU {iou:.4f} with the OSM administrative relation polygon; area differs by {da:.2%}; "
            f"Hausdorff distance {float(a.hausdorff_distance(b)):.0f} m", informational=False)
    names = [str(r.get("display_name", "")) for r in (nominatim_results or [])
             if gb["shapeName"].lower() in str(r.get("display_name", "")).lower()]
    aliases = [str(x).lower() for x in exp.get("parent_aliases", [exp["parent_district"]])]
    hierarchy_names = any(al in n.lower() for n in names for al in aliases)
    other = max([float(study_wgs84.intersection(g).area / study_wgs84.area) for g in other_districts_wgs84], default=0.0)
    if district_wgs84 is None and not names:
        add("parent_district", None, "ADM2 layer and independent hierarchy not available")
    elif cov is not None and cov >= float(tol.get("district_cover", 0.9)):
        add("parent_district", True, f"{cov:.1%} of the polygon lies inside the ADM2 feature {exp['parent_district']!r}")
    else:
        ok = bool(rel_ok and hierarchy_names and other <= 0.01)
        add("parent_district", ok,
            f"ADM2 geometric coverage {cov if cov is None else format(cov, '.1%')} is below the threshold; independent evidence: "
            f"OSM relation polygon agrees={rel_ok}, hierarchy names the district={hierarchy_names}, largest overlap with any "
            f"OTHER district {other:.2%} (must be <= 1%)")
    if nominatim_results is None:
        add("independent_locator", None, "no Nominatim record available (not performed)")
    else:
        import shapely
        maxd = float(cfg_study.get("independent_locator", {}).get("max_distance_km", 5.0))
        best: dict[str, Any] | None = None
        for r in nominatim_results:
            if gb["shapeName"].lower() not in str(r.get("display_name", "")).lower():
                continue
            p = _project(shapely.Point(r["lon"], r["lat"]), 4326, epsg)
            d = float(study_m.distance(p)) / 1000.0
            if best is None or d < best["distance_km"]:
                best = {"osm_type": r["osm_type"], "osm_id": r["osm_id"], "display_name": r["display_name"],
                        "distance_km": d, "extratags": r.get("extratags", {})}
        add("independent_locator", None if not nominatim_results else (best is not None and best["distance_km"] <= maxd),
            f"nearest name-matching Nominatim result: {best}" if best else "no Nominatim result matched the name")
    status = "verified" if all(c["ok"] is True for c in checks if not c.get("informational")) else "approximate"
    return {"boundary_status": status, "checks": checks,
            "level_meaning": f"geoBoundaries {gb['level']} for {gb['iso']} = local-level unit (municipality / rural municipality)"}


# ---------------------------------------------------------------- external patch
def external_patch_centre(spec: dict[str, Any], study_m: Any, epsg: int, toward_border_bearing: float) -> tuple[float, float]:
    """Projected (x, y) of the external patch centre. ``xy_m`` or ``lonlat`` are used as given; otherwise the centre is the
    point ``beyond_polygon_m`` outside the polygon edge along the bearing ``bearing_deg`` (a number, or 'toward_border')
    from the centroid."""
    import shapely
    if "xy_m" in spec:
        return float(spec["xy_m"][0]), float(spec["xy_m"][1])
    if "lonlat" in spec:
        p = _project(shapely.Point(*spec["lonlat"]), 4326, epsg)
        return float(p.x), float(p.y)
    b = spec.get("bearing_deg", "toward_border")
    bear = toward_border_bearing if b == "toward_border" else float(b)
    ux, uy = math.sin(math.radians(bear)), math.cos(math.radians(bear))
    c = study_m.centroid
    ray = shapely.LineString([(c.x, c.y), (c.x + 1e6 * ux, c.y + 1e6 * uy)])
    hit = ray.intersection(study_m.boundary)
    pts = shapely.get_coordinates(hit)
    if len(pts) == 0:
        raise ValueError("ray from the centroid does not leave the polygon")
    exit_d = float(np.max(np.hypot(pts[:, 0] - c.x, pts[:, 1] - c.y)))
    d = exit_d + float(spec["beyond_polygon_m"])
    return float(c.x + d * ux), float(c.y + d * uy)


# ---------------------------------------------------------------- prepare
def _rel(path: Path, base: Path) -> str:
    return os.path.relpath(path, base)


def _raw_entry(role: str, rec: dict[str, Any], raw_dir: Path, base: Path, **meta: Any) -> dict[str, Any]:
    f = raw_dir / rec["file"]
    if not f.exists():
        raise FileNotFoundError(f"{f} (record {role}) is missing")
    if sha256_file(f) != rec["sha256"]:
        raise ValueError(f"{f}: sha256 differs from the fetch record; the cached file was modified")
    e = {k: rec.get(k) for k in ("url", "id", "retrieved_utc", "bytes", "sha256", "license", "attribution")}
    e.update({"role": role, "kind": "raw", "file": _rel(f, base)})
    for k in ("release", "feature_count", "tile_urls", "query", "expected_commit"):
        if k in rec:
            e[k] = rec[k]
    e.update(meta)
    return e


def _write_array(out_dir: Path, base: Path, name: str, arr: np.ndarray, entries: list[dict[str, Any]], common: dict[str, Any],
                 **meta: Any) -> str:
    a = np.asarray(arr, dtype=float)
    p = out_dir / f"{name}.npy"
    np.save(p, a)
    entries.append({**common, "kind": "processed", "name": name, "id": p.name, "file": _rel(p, base), "sha256": sha256_file(p),
                    "array_sha256": array_sha256(a), "shape": list(a.shape), **meta})
    return _rel(p, base)


def prepare_region(cfg: dict[str, Any], raw_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Build the processed arrays in ``out_dir`` and ``manifest.json`` (in out_dir's parent) from cached raw files in
    ``raw_dir`` (written by ``geo fetch``). Offline. Returns the manifest."""
    import rasterio
    import shapely
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = out.parent
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    recs = read_records(raw)
    need = ["study", "adm2", "dem", "osm_roads", "osm_buildings", "osm_amenity", *[f"adm0_{c['iso']}" for c in cfg["adm0"]]]
    missing = [r for r in need if r not in recs]
    if missing:
        raise FileNotFoundError(f"missing fetch records {missing} in {raw}; nothing is fabricated -- run `geo fetch`")
    study_cfg = cfg["study"]
    gb = study_cfg["geoboundaries"]
    warnings: list[str] = []
    incomplete: list[str] = []
    datasets: list[dict[str, Any]] = []

    # --- boundary and CRS
    geoms, props = _read_geojson(raw / recs["study"]["file"])
    gj = {"features": [{"properties": p} for p in props]}
    find_feature(gj, gb["shapeName"], gb.get("shapeID"))  # raises with candidates if absent/ambiguous
    k = [i for i, p in enumerate(props) if p.get("shapeName") == gb["shapeName"] and (not gb.get("shapeID") or p.get("shapeID") == gb["shapeID"])][0]
    poly_ll = geoms[k]
    if not poly_ll.is_valid:
        warnings.append("study polygon was invalid; shapely.make_valid applied")
        poly_ll = shapely.make_valid(poly_ll)
    cen = poly_ll.centroid
    epsg = rio.utm_epsg(cen.x, cen.y)
    study_m = _project(poly_ll, 4326, epsg)
    rt = rio.roundtrip_error_m(cen.x, cen.y, epsg)
    exp_epsg = study_cfg.get("expected_epsg")
    verification: dict[str, Any] = {"epsg_matches_expected": (epsg == exp_epsg) if exp_epsg else None,
                                    "roundtrip_error_m": rt, "features_in_layer": len(props)}
    districts, dprops = _read_geojson(raw / recs["adm2"]["file"])
    dname = study_cfg["expected"]["parent_district"]
    dsel = [i for i, p in enumerate(dprops) if p.get("shapeName") == dname]
    if len(dsel) != 1:
        raise ValueError(f"ADM2 layer: {len(dsel)} features named {dname!r}; candidates {sorted({str(p.get('shapeName')) for p in dprops})[:20]}")
    district_ll = districts[dsel[0]]
    nom_res = parse_nominatim(json.loads((raw / recs["nominatim"]["file"]).read_text())) if "nominatim" in recs else None
    osm_rel_ll = None
    if "osm_relation" in recs:
        _fc = json.loads((raw / recs["osm_relation"]["file"]).read_text())
        _geoms = [shapely.geometry.shape(f["geometry"]) for f in _fc.get("features", []) if f.get("geometry")]
        osm_rel_ll = _geoms[0] if _geoms else None
    others = [districts[i] for i in range(len(districts)) if i != dsel[0]]
    boundary = verify_boundary(study_m, poly_ll, props[k], study_cfg, epsg, district_ll, nom_res, osm_rel_ll, others)
    if boundary["boundary_status"] != "verified":
        incomplete.append("boundary requirement: status is 'approximate' (see boundary.checks)")

    # --- countries
    clip = shapely.box(cen.x - 3, cen.y - 3, cen.x + 3, cen.y + 3)  # +-3 deg: far enough that clip edges are not near the study area
    countries: dict[str, Any] = {}
    for c in cfg["adm0"]:
        g, _ = _read_geojson(raw / recs[f"adm0_{c['iso']}"]["file"])
        countries[c["iso"]] = _project(shapely.union_all(shapely.intersection(shapely.make_valid(g), clip)), 4326, epsg)
    home = gb["iso"]
    border = international_border_info(countries, study_m, home=home)

    # --- grids
    gcfg = cfg["grid"]
    resolutions = [float(r) for r in gcfg["resolutions_m"]]
    pad = float(gcfg.get("pad_to_m", max(resolutions)))
    bounds = rio.domain_bounds_m(study_m, float(gcfg["buffer_m"]))
    grids = {r: rio.make_grid(bounds, r, pad_to=pad, epsg=epsg) for r in resolutions}
    g0 = grids[resolutions[0]]
    if any((g.x0, g.y0, g.nx * g.dx, g.ny * g.dy) != (g0.x0, g0.y0, g0.nx * g0.dx, g0.ny * g0.dy) for g in grids.values()):
        raise ValueError("grids do not share one domain")

    # --- DEM
    drec = recs["dem"]
    with rasterio.open(raw / drec["file"]) as ds:
        dem = ds.read(1, masked=True)
        dtr, dcrs, dnodata = ds.transform, ds.crs.to_string(), ds.nodata
        dbounds = tuple(ds.bounds)
    native_m = (abs(dtr.a) * 111_320.0 * math.cos(math.radians(cen.y)), abs(dtr.e) * 110_574.0)
    dem_native = {"pixel_deg": [abs(dtr.a), abs(dtr.e)], "pixel_m_approx": list(native_m), "crs": dcrs}

    # --- OSM layers
    def osm(role: str) -> Any:
        if role not in recs:
            return None
        g, p = _read_geojson(raw / recs[role]["file"])
        return _project(g, 4326, epsg), p
    roads_m = _lines(osm("osm_roads")[0])
    bld_m = _polygons(osm("osm_buildings")[0])
    lu = osm("osm_landuse")
    lu_m = _polygons(lu[0]) if lu is not None else np.array([], dtype=object)
    am_g, am_p = osm("osm_amenity")
    fac_types = tuple(cfg["osm"].get("facility_amenities", ("school", "college", "kindergarten", "university", "hospital", "clinic", "doctors", "health_post")))
    keep = np.array([p.get("amenity") in fac_types for p in am_p], dtype=bool)
    fac_pts = shapely.get_coordinates(shapely.point_on_surface(shapely.make_valid(rio.as_geom_array(am_g[keep])))) if keep.any() else np.zeros((0, 2))
    fac_in = np.array([study_m.contains(shapely.Point(x, y)) for x, y in fac_pts], dtype=bool)
    roads_in = _lines(shapely.intersection(roads_m, study_m)) if len(roads_m) else roads_m
    builtup_all = np.concatenate([bld_m, lu_m]) if len(lu_m) else bld_m
    builtup_in = _polygons(shapely.intersection(shapely.union_all(builtup_all), study_m)) if len(builtup_all) else builtup_all
    counts = {"roads_segments_domain": int(len(roads_m)), "road_km_domain": float(shapely.length(roads_m).sum() / 1000.0) if len(roads_m) else 0.0,
              "road_km_in_study": float(shapely.length(roads_in).sum() / 1000.0) if len(roads_in) else 0.0,
              "buildings_domain": int(len(bld_m)), "residential_landuse_polygons": int(len(lu_m)),
              "builtup_km2_in_study": float(shapely.area(builtup_in).sum() / 1e6) if len(builtup_in) else 0.0,
              "facilities_domain": int(len(fac_pts)), "facilities_in_study": int(fac_in.sum())}
    warnings.append("OSM completeness is unknown: absent roads/buildings in the data are mapping gaps, not zero sources.")
    if lu is None:
        warnings.append("no residential landuse layer: built-up = building footprints only")
    if counts["buildings_domain"] < 10:
        warnings.append(f"very few OSM buildings ({counts['buildings_domain']}); built-up hypothesis is weak")

    # --- patch
    pspec = cfg["external_patch"]
    sigma = float(pspec["sigma_m"])
    px, py = external_patch_centre(pspec, study_m, epsg, border["bearing_deg"])
    pt = shapely.Point(px, py)
    if study_m.contains(pt):
        raise ValueError("external patch centre lies inside the study polygon")
    in_grid = g0.x0 + 3 * sigma <= px <= g0.x0 + g0.lx - 3 * sigma and g0.y0 + 3 * sigma <= py <= g0.y0 + g0.ly - 3 * sigma
    if not in_grid:
        raise ValueError("external patch (centre +- 3 sigma) does not fit inside the simulation domain; reduce sigma_m/beyond or raise buffer_m")
    patch_class = classify_point(pt, study_m, countries, home)
    plon, plat = rio.utm_to_lonlat(px, py, epsg)
    patch = {"hypothesis": SYNTHETIC_PRIOR, "group": "external", "centre_xy_m": [px, py], "centre_local_m": [px - g0.x0, py - g0.y0],
             "centre_lonlat": [plon, plat], "sigma_m": sigma, "distance_to_polygon_km": float(study_m.distance(pt)) / 1000.0,
             "origin": patch_class}
    if patch_class["class"] == "unclassified":
        warnings.append("external patch centre is in no (or several) ADM0 polygons: 'international' not asserted")

    # --- per-resolution arrays
    wgs_bbox = list(poly_ll.bounds)
    levels: list[dict[str, Any]] = []
    processed: list[dict[str, Any]] = []
    lic = "; ".join(sorted({str(e) for e in (recs['study']['license'], recs['dem']['license'], recs['osm_roads']['license'])}))
    attr = " | ".join(sorted({recs['study']['attribution'], recs['dem']['attribution'], recs['osm_roads']['attribution']}))
    urls = [recs[r]["url"] for r in ("study", "dem", "osm_roads")]
    for res, grid in grids.items():
        tag = f"{int(res) if res == int(res) else res}m"
        common = {"url": urls, "retrieved_utc": now, "license": lic, "attribution": attr, "crs": f"EPSG:{epsg}",
                  "bbox_lonlat": list(rio.lonlat_box(grid.bounds, epsg)), "resolution_m": res,
                  "assumptions": ["synthetic spatial priors: weights follow real geometry, emissions are assumed",
                                  "arrays are [y, x] = [j, i], j increasing northward; local x,y = UTM - (x0, y0)"],
                  "native_resolution": {"dem": dem_native, "grid_cell_m": [grid.dx, grid.dy]}}
        frac = rio.polygon_cell_fractions(study_m, grid)
        area_rel = abs(frac.sum() * grid.cell_area - study_m.area) / study_m.area
        verification[f"polygon_area_rel_err_{tag}"] = float(area_rel)
        if area_rel > 1e-9:
            warnings.append(f"{tag}: polygon not fully inside the grid or area mismatch (rel err {area_rel:.2e})")
        z, zvalid = rio.aggregate_dem(dem, dtr, dcrs, grid, "mean", dnodata)
        inside = frac > 0
        if float(zvalid[inside].mean()) < float(cfg["dem"].get("min_valid_fraction", 0.95)):
            raise ValueError(f"{tag}: DEM valid fraction over the study cells is {zvalid[inside].mean():.3f}; DEM does not cover the area")
        slope, hs = rio.slope_deg(z, grid.dx, grid.dy), rio.hillshade(z, grid.dx, dy=grid.dy)
        cj, ci = (int(a[0]) for a in grid.cell_of(study_m.centroid.x, study_m.centroid.y))
        road = rio.rasterize_line_length(roads_m, grid) if len(roads_m) else np.zeros(grid.shape)
        built = rio.rasterize_polygon_area(builtup_all, grid) if len(builtup_all) else np.zeros(grid.shape)
        fac = rio.rasterize_points(fac_pts, grid)
        w = {"road": rio.rasterize_line_length(roads_in, grid) if len(roads_in) else np.zeros(grid.shape),
             "builtup": rio.rasterize_polygon_area(builtup_in, grid) if len(builtup_in) else np.zeros(grid.shape),
             "facility": rio.rasterize_points(fac_pts[fac_in], grid)}
        xc, yc = np.meshgrid(grid.x_centres(), grid.y_centres())
        gx, gy = px - g0.x0, py - g0.y0
        w["external_patch"] = np.exp(-((xc - gx) ** 2 + (yc - gy) ** 2) / (2 * sigma**2))
        files: dict[str, str] = {}
        kw = dict(resolution_m=res)

        def put(name: str, arr: np.ndarray, units: str, resampling: str, mapping: str, **extra: Any) -> None:
            files[name] = _write_array(out, base, f"{name}_{tag}", arr, processed, common, units=units, resampling=resampling,
                                       source_to_grid=mapping, **kw, **extra)
        put("study_fraction", frac, "1 (area fraction)", "exact shapely polygon-cell intersection", "ADM3 polygon -> cell area fraction")
        put("dem_mean", z, "m (DSM, EGM2008 heights)", "mean of valid source pixel centres in cell", "pixel centre -> containing cell")
        put("dem_valid_fraction", zvalid, "1", "valid pixels / pixels in cell", "pixel centre -> containing cell")
        put("slope_deg", slope, "degrees", "central differences of the aggregated DEM (understates native slopes)", "from dem_mean")
        put("hillshade", hs, "1", f"azimuth 315, altitude 45 on dem_mean", "from dem_mean")
        put("road_length_m", road, "m of road per cell", "exact line-cell intersection (whole domain)", "OSM edges -> cells")
        put("builtup_fraction", built, "1 (area fraction)", "union of footprints/residential landuse, exact intersection", "OSM polygons -> cells")
        put("facility_count", fac, "count per cell", "point in cell (polygons via point_on_surface)", "OSM amenity -> cells")
        put("profile_ns_z", rio.profile(z, "ns", ci), "m", "column through the centroid cell", f"col {ci}")
        put("profile_ew_z", rio.profile(z, "ew", cj), "m", "row through the centroid cell", f"row {cj}")
        raster_sources: list[RasterSource] = []
        for sname, wt in w.items():
            if not wt.sum() > 0:
                incomplete.append(f"{tag}: source hypothesis '{sname}' has no weight (no data inside the polygon): not created")
                continue
            f = rio.normalise_source(wt, grid)
            put(f"w_{sname}", f, "1/m^2 (sum f*cell_area = 1)", "normalised raster weights",
                "local: clipped to the polygon before rasterising; external: Gaussian at cell centres", group="external" if sname == "external_patch" else "local",
                hypothesis=SYNTHETIC_PRIOR)
            raster_sources.append(RasterSource.from_array(sname, "external" if sname == "external_patch" else "local", f))
        dom = Domain(grid.lx, grid.ly, grid.nx, grid.ny)
        mask = StudyMask.from_array(frac)
        ext = [s for s in raster_sources if s.group == "external"]
        ext_ok = bool(ext)
        if ext:
            check_origin_tags(Scenario(domain=dom, sources=tuple(ext), study=mask), tol=0.01)  # raises if the patch leaks into the polygon
        mass = origin_mass_fractions(Scenario(domain=dom, sources=tuple(raster_sources), study=mask)) if raster_sources else []
        zin = z[inside & np.isfinite(z)]
        kmax = int(np.nanargmax(np.where(inside, z, -np.inf)))
        jm, im = divmod(kmax, grid.nx)
        dxm, dym = grid.x_centres()[im] - (study_m.centroid.x - g0.x0), grid.y_centres()[jm] - (study_m.centroid.y - g0.y0)
        levels.append({"resolution_m": res, "tag": tag, "grid": grid.to_dict(), "files": files,
                       "centroid_cell": {"j": cj, "i": ci},
                       "polygon_cells": int(inside.sum()), "origin_mass_inside": dict(zip((s.name for s in raster_sources), map(float, mass))),
                       "external_patch_check_passed": ext_ok,
                       "relief": {"min_m": float(zin.min()), "max_m": float(zin.max()), "range_m": float(zin.max() - zin.min()),
                                  "max_at_distance_km": math.hypot(dxm, dym) / 1000.0, "max_at_bearing_deg": bearing_deg(dxm, dym),
                                  "ns_profile_range_m": float(np.nanmax(rio.profile(z, 'ns', ci)) - np.nanmin(rio.profile(z, 'ns', ci))),
                                  "ew_profile_range_m": float(np.nanmax(rio.profile(z, 'ew', cj)) - np.nanmin(rio.profile(z, 'ew', cj)))},
                       "mean_slope_deg_in_study": float(np.nansum(slope * frac) / frac[np.isfinite(slope)].sum())})

    # --- raw dataset entries (after the processing so metadata is known)
    common_raw = {"resampling": "none (raw)", "source_to_grid": "see processed entries"}
    datasets.append(_raw_entry("study", recs["study"], raw, base, crs="EPSG:4326", bbox_lonlat=wgs_bbox, units="degrees (lon, lat)",
                               native_resolution="vector", counts={"features_in_layer": len(props)}, assumptions=[
                                   "polygon taken as authoritative only if boundary_status == 'verified'"], boundary=boundary, **common_raw))
    datasets.append(_raw_entry("adm2", recs["adm2"], raw, base, crs="EPSG:4326", bbox_lonlat=list(district_ll.bounds), units="degrees",
                               native_resolution="vector", counts={"features_in_layer": len(dprops)}, assumptions=["parent-district check"], **common_raw))
    for iso, g in countries.items():
        datasets.append(_raw_entry(f"adm0_{iso}", recs[f"adm0_{iso}"], raw, base, crs="EPSG:4326", bbox_lonlat=list(
            rio.lonlat_box(g.bounds, epsg)), units="degrees", native_resolution="vector", counts={},
            assumptions=["clipped to +-3 deg around the study centroid for distance computations only"], **common_raw))
    if "nominatim" in recs:
        datasets.append(_raw_entry("nominatim", recs["nominatim"], raw, base, crs="EPSG:4326", bbox_lonlat=wgs_bbox, units="degrees",
                                   native_resolution="point", counts={"results": len(nom_res or [])},
                                   assumptions=["independent locator only; its geometry is not used"], **common_raw))
    for role in ("osm_roads", "osm_buildings", "osm_landuse", "osm_amenity"):
        if role in recs:
            datasets.append(_raw_entry(role, recs[role], raw, base, crs="EPSG:4326", bbox_lonlat=list(rio.lonlat_box(bounds, epsg)),
                                       units="degrees", native_resolution="vector", counts={"features": recs[role].get("feature_count")},
                                       assumptions=["OSM completeness unknown"], **common_raw))
    datasets.append(_raw_entry("dem", drec, raw, base, crs=dcrs, bbox_lonlat=list(dbounds), units="m (DSM)", native_resolution=dem_native,
                               counts={"shape": list(dem.shape), "nodata": dnodata, "nodata_pixels": int(np.ma.count_masked(dem))},
                               assumptions=["Copernicus GLO-30 is a surface model (DSM), not bare earth"],
                               **{**common_raw, "resampling": "mean of valid pixel centres per cell (aggregate_dem)",
                                  "source_to_grid": "native DEM pixels -> grid cells by pixel centre"}))

    manifest = {
        "schema": SCHEMA, "name": study_cfg["name"], "created_utc": now,
        "boundary_status": boundary["boundary_status"], "boundary": boundary, "incomplete": incomplete, "warnings": warnings,
        "crs": {"epsg": epsg, "zone": epsg % 100, "hemisphere": "N" if epsg < 32700 else "S", "axis_order": "geographic (lon, lat); "
                "projected (x=easting, y=northing); pyproj always_xy", "roundtrip_error_m": rt},
        "georef": {"epsg": epsg, "x0": g0.x0, "y0": g0.y0}, "study": {
            "name": study_cfg["name"], "area_km2": study_m.area / 1e6, "centroid_lonlat": [cen.x, cen.y], "bbox_lonlat": wgs_bbox,
            "bbox_m": list(study_m.bounds)},
        "grid_config": {"buffer_m": float(gcfg["buffer_m"]), "resolutions_m": resolutions, "pad_to_m": pad},
        "levels": levels, "datasets": datasets, "processed": processed, "counts": counts,
        "external_patch": patch, "international_border": border, "verification": verification,
        "assumptions": [SYNTHETIC_PRIOR, "DEM is a surface model used as context only unless a layer alters source geometry",
                        "local sources are clipped to the polygon; boundary cells count by area fraction"],
    }
    write_json(base / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------- Region
@dataclass(frozen=True)
class GeoRef:
    """Local simulation coordinates = projected (UTM ``epsg``) metres minus (x0, y0); x east, y north."""
    epsg: int
    x0: float
    y0: float

    def utm_to_local(self, x: Any, y: Any) -> tuple[Any, Any]:
        return np.asarray(x, float) - self.x0, np.asarray(y, float) - self.y0

    def local_to_utm(self, x: Any, y: Any) -> tuple[Any, Any]:
        return np.asarray(x, float) + self.x0, np.asarray(y, float) + self.y0

    def lonlat_to_local(self, lon: Any, lat: Any) -> tuple[Any, Any]:
        from pyproj import Transformer
        x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True).transform(lon, lat)
        return self.utm_to_local(x, y)

    def local_to_lonlat(self, x: Any, y: Any) -> tuple[Any, Any]:
        from pyproj import Transformer
        ux, uy = self.local_to_utm(x, y)
        lon, lat = Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True).transform(ux, uy)
        return lon, lat


@dataclass(frozen=True, eq=False)
class GridLevel:
    resolution_m: float
    grid: rio.GridSpec
    arrays: dict[str, np.ndarray] = field(repr=False)
    info: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass(frozen=True, eq=False)
class Region:
    """Prepared geography for scenario builders: grids (one per resolution, same domain), arrays, georeference."""
    name: str
    georef: GeoRef
    levels: dict[float, GridLevel]
    manifest: dict[str, Any] = field(repr=False)

    @property
    def boundary_status(self) -> str:
        return str(self.manifest["boundary_status"])

    def level(self, resolution_m: float) -> GridLevel:
        if resolution_m not in self.levels:
            raise KeyError(f"no grid at {resolution_m} m; have {sorted(self.levels)}")
        return self.levels[resolution_m]

    def domain(self, resolution_m: float) -> Domain:
        g = self.level(resolution_m).grid
        return Domain(g.lx, g.ly, g.nx, g.ny)

    def study_mask(self, resolution_m: float) -> StudyMask:
        return StudyMask.from_array(self.level(resolution_m).arrays["study_fraction"])

    def raster_source(self, resolution_m: float, name: str, profile: tuple[tuple[float, float], ...] = ((0.0, 1.0),)) -> RasterSource:
        """Source hypothesis 'road' | 'builtup' | 'facility' (group local) or 'external_patch' (group external)."""
        lv = self.level(resolution_m)
        key = f"w_{name}"
        if key not in lv.arrays:
            raise KeyError(f"source hypothesis {name!r} is not available at {resolution_m} m (see manifest 'incomplete')")
        return RasterSource.from_array(name, "external" if name == "external_patch" else "local", lv.arrays[key], profile)

    def external_gaussian(self, profile: tuple[tuple[float, float], ...] = ((0.0, 1.0),)) -> GaussianSource:
        p = self.manifest["external_patch"]
        return GaussianSource("external_patch", p["centre_local_m"][0], p["centre_local_m"][1], p["sigma_m"], profile, "external")


def load_region(manifest_path: str | Path) -> Region:
    """Load a prepared region, verifying every processed file's sha256 against the manifest."""
    mp = Path(manifest_path)
    m = json.loads(mp.read_text())
    base = mp.parent
    georef = GeoRef(int(m["georef"]["epsg"]), float(m["georef"]["x0"]), float(m["georef"]["y0"]))
    by_file = {e["file"]: e for e in m["processed"]}
    levels: dict[float, GridLevel] = {}
    for lv in m["levels"]:
        arrays: dict[str, np.ndarray] = {}
        for name, rel in lv["files"].items():
            e = by_file[rel]
            if sha256_file(base / rel) != e["sha256"]:
                raise ValueError(f"{rel}: sha256 does not match the manifest")
            a = np.load(base / rel)
            if array_sha256(a) != e["array_sha256"]:
                raise ValueError(f"{rel}: array hash does not match the manifest")
            arrays[name] = a
        levels[float(lv["resolution_m"])] = GridLevel(float(lv["resolution_m"]), rio.GridSpec.from_dict(lv["grid"]), arrays,
                                                       {k: v for k, v in lv.items() if k not in ("files", "grid")})
    return Region(m["name"], georef, levels, m)

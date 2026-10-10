"""Bounded, cached, polite network access. Nothing here runs at import time; ``requests``/``osmnx``/``rasterio`` are
imported inside the functions that use them. No credentials are used; query strings that look like secrets are redacted
from recorded URLs.

Every fetch returns/records a dict with at least: url, id, retrieved_utc, bytes, sha256, license, attribution, file
(path relative to the cache dir). ``fetch_records.json`` in the cache dir maps a *role* ('study', 'adm2', 'adm0_NPL',
'nominatim', 'dem', 'osm_roads', ...) to its record; ``geo prepare`` reads only that file and the files it names.
"""
from __future__ import annotations

import datetime as dt
import difflib
import hashlib
import json
import math
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

USER_AGENT = "pravah-research/0.1 (yubraj.bhandari@duke.edu)"
MIN_INTERVAL_S = 1.0  # minimum delay between requests to one host (Nominatim/Overpass policies)
GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso}/{level}/"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"  # osmnx default endpoint
COPERNICUS_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
                  "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")
OSM_LICENSE = "ODbL 1.0 (https://opendatacommons.org/licenses/odbl/1-0/)"
OSM_ATTRIBUTION = "(c) OpenStreetMap contributors"
COPERNICUS_LICENSE = "Copernicus DEM GLO-30 (ESA/EU Copernicus programme licence: free use, attribution required)"
COPERNICUS_ATTRIBUTION = ("Contains modified Copernicus DEM GLO-30 data, (c) DLR e.V. 2010-2014 and (c) Airbus Defence and "
                          "Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA; all rights reserved")
_SECRET = re.compile(r"(sig|token|key|secret|credential|x-amz|auth)", re.I)
_last_request: dict[str, float] = {}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def redact_url(url: str) -> str:
    """URL safe to log/record: query values whose name looks like a signature/token/key are replaced by 'REDACTED'."""
    s = urlsplit(url)
    q = [(k, "REDACTED" if _SECRET.search(k) else v) for k, v in parse_qsl(s.query, keep_blank_values=True)]
    return urlunsplit((s.scheme, s.netloc.split("@")[-1], s.path, urlencode(q), ""))


def _throttle(host: str, min_interval_s: float) -> None:
    wait = _last_request.get(host, -math.inf) + min_interval_s - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request[host] = time.monotonic()


def _record(url: str, dest: Path, base: Path, *, rec_id: str, retrieved_utc: str, license: str, attribution: str,
            extra: dict[str, Any] | None = None) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "url": redact_url(url), "id": rec_id, "retrieved_utc": retrieved_utc, "bytes": dest.stat().st_size,
        "sha256": sha256_file(dest), "license": license, "attribution": attribution,
        "file": str(dest.relative_to(base)) if dest.is_relative_to(base) else str(dest)}
    rec.update(extra or {})
    return rec


def http_get(url: str, dest: str | Path, *, timeout: float = 30.0, max_bytes: int = 50_000_000,
             expect_sha256: str | None = None, session: Any = None, params: dict[str, Any] | None = None,
             min_interval_s: float = MIN_INTERVAL_S, license: str = "unknown", attribution: str = "unknown",
             rec_id: str | None = None, cache_root: str | Path | None = None) -> dict[str, Any]:
    """GET ``url`` into ``dest`` (streamed, at most ``max_bytes``, ``timeout`` s per socket operation) and return a
    record. Cache: if ``dest`` exists it is NOT re-downloaded; its sha256 is checked against ``expect_sha256`` (mismatch
    -> ValueError, never silently re-fetched) and the original retrieval time is read from ``dest.record.json``.
    A new download is written to a temporary file and moved into place only after the size/hash checks pass."""
    dest = Path(dest)
    base = Path(cache_root) if cache_root is not None else dest.parent
    side = dest.with_name(dest.name + ".record.json")
    if dest.exists():
        sha = sha256_file(dest)
        if expect_sha256 is not None and sha != expect_sha256:
            raise ValueError(f"cached {dest} has sha256 {sha}, expected {expect_sha256}; delete it deliberately to refetch")
        old = json.loads(side.read_text()) if side.exists() else {}
        when = old.get("retrieved_utc") or dt.datetime.fromtimestamp(dest.stat().st_mtime, dt.timezone.utc).isoformat(
            timespec="seconds")
        return _record(url, dest, base, rec_id=rec_id or dest.name, retrieved_utc=when, license=license,
                       attribution=attribution, extra={"cache_hit": True})
    if session is None:
        import requests
        session = requests.Session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    _throttle(urlsplit(url).netloc, min_interval_s)
    tmp = dest.with_name(dest.name + ".part")
    when = _now()
    resp = session.get(url, params=params, headers={"User-Agent": USER_AGENT}, stream=True, timeout=timeout)
    try:
        resp.raise_for_status()
        declared = resp.headers.get("Content-Length")
        if declared is not None and int(declared) > max_bytes:
            raise ValueError(f"{redact_url(url)}: Content-Length {declared} exceeds max_bytes={max_bytes}")
        n = 0
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                n += len(chunk)
                if n > max_bytes:
                    raise ValueError(f"{redact_url(url)}: download exceeds max_bytes={max_bytes}")
                f.write(chunk)
        if expect_sha256 is not None and sha256_file(tmp) != expect_sha256:
            raise ValueError(f"{redact_url(url)}: sha256 mismatch (expected {expect_sha256})")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
        close = getattr(resp, "close", None)
        if close is not None:
            close()
    rec = _record(url, dest, base, rec_id=rec_id or dest.name, retrieved_utc=when, license=license,
                  attribution=attribution, extra={"cache_hit": False})
    side.write_text(json.dumps({"retrieved_utc": when, "url": rec["url"]}) + "\n")
    return rec


# ---------------------------------------------------------------- records file
def write_records(cache_dir: str | Path, records: dict[str, dict[str, Any]]) -> Path:
    p = Path(cache_dir) / "fetch_records.json"
    old = read_records(cache_dir) if p.exists() else {}
    old.update(records)
    p.write_text(json.dumps({"records": old}, indent=2, sort_keys=True) + "\n")
    return p


def read_records(cache_dir: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(cache_dir) / "fetch_records.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run `python -m pravah geo fetch` first")
    return dict(json.loads(p.read_text())["records"])


# ---------------------------------------------------------------- geoBoundaries
def geoboundaries_release(iso: str, level: str, cache_dir: str | Path, *, session: Any = None) -> dict[str, Any]:
    """Release metadata for a country/level from the geoBoundaries gbOpen API. Returns a record with the API fields
    boundaryID, boundaryType, year, source, license and the (commit-pinned) gjDownloadURL under key ``release``."""
    cache = Path(cache_dir)
    url = GEOBOUNDARIES_API.format(iso=quote(iso), level=quote(level))
    rec = http_get(url, cache / f"geoboundaries_{iso}_{level}.meta.json", timeout=30, max_bytes=2_000_000, session=session,
                   license="CC BY 4.0 (API metadata)", attribution="geoBoundaries (William & Mary geoLab)",
                   rec_id=f"geoBoundaries-api:{iso}-{level}", cache_root=cache)
    meta = json.loads((cache / rec["file"]).read_text())
    if isinstance(meta, list):
        if len(meta) != 1:
            raise ValueError(f"geoBoundaries API returned {len(meta)} entries for {iso}/{level}")
        meta = meta[0]
    if "gjDownloadURL" not in meta:
        raise ValueError(f"geoBoundaries metadata for {iso}/{level} has no gjDownloadURL: keys {sorted(meta)}")
    rec["release"] = {
        "boundaryID": meta.get("boundaryID"), "boundaryType": meta.get("boundaryType"),
        "year": meta.get("boundaryYearRepresented"), "source": meta.get("boundarySource"),
        "license": meta.get("boundaryLicense"), "licenseDetail": meta.get("licenseDetail"),
        "buildDate": meta.get("buildDate"), "gjDownloadURL": meta["gjDownloadURL"]}
    return rec


def fetch_geoboundaries_geojson(iso: str, level: str, cache_dir: str | Path, *, expect_commit: str | None = None,
                                max_bytes: int = 200_000_000, session: Any = None) -> dict[str, Any]:
    """Download the GeoJSON named by the release metadata. If ``expect_commit`` is given, the download URL must contain it
    (a moving 'current' release is refused: the data would not be reproducible)."""
    cache = Path(cache_dir)
    meta = geoboundaries_release(iso, level, cache, session=session)
    rel = meta["release"]
    url = rel["gjDownloadURL"]
    if expect_commit is not None and expect_commit not in url:
        raise ValueError(f"geoBoundaries {iso}/{level}: download URL {redact_url(url)} is not pinned to commit "
                         f"{expect_commit}; the current release has changed")
    lic = f"{rel.get('license') or 'see release metadata'}"
    rec = http_get(url, cache / f"geoBoundaries-{iso}-{level}.geojson", timeout=60, max_bytes=max_bytes, session=session,
                   license=lic, attribution=f"geoBoundaries gbOpen {iso} {level}; source: {rel.get('source')}",
                   rec_id=f"geoBoundaries:{iso}-{level}:{rel.get('boundaryID')}", cache_root=cache)
    rec["release"] = rel
    rec["expected_commit"] = expect_commit
    return rec


def find_feature(geojson: dict[str, Any], name: str, shape_id: str | None = None) -> dict[str, Any]:
    """The single feature whose properties.shapeName equals ``name`` exactly (case-sensitive); if ``shape_id`` is given it
    must also match properties.shapeID. Zero or several matches raise ValueError listing the candidates."""
    feats = geojson.get("features", [])
    hits = [f for f in feats if f.get("properties", {}).get("shapeName") == name]
    if shape_id is not None:
        by_id = [f for f in hits if f["properties"].get("shapeID") == shape_id]
        if len(by_id) != 1:
            ids = [(f["properties"].get("shapeName"), f["properties"].get("shapeID")) for f in hits]
            raise ValueError(f"no unique feature {name!r} with shapeID {shape_id!r}; same-name features: {ids}")
        return by_id[0]
    if len(hits) == 1:
        return hits[0]
    if hits:
        cands = [(f["properties"].get("shapeName"), f["properties"].get("shapeID")) for f in hits]
        raise ValueError(f"{len(hits)} features named {name!r} (give shape_id): {cands}")
    names = sorted({str(f.get("properties", {}).get("shapeName")) for f in feats})
    near = difflib.get_close_matches(name, names, n=8, cutoff=0.5)
    raise ValueError(f"no feature named {name!r} among {len(feats)} features; closest names: {near}")


def load_geojson(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------- Nominatim (independent locator only)
def nominatim_lookup(query: str, cache_dir: str | Path, *, session: Any = None, limit: int = 5) -> dict[str, Any]:
    """One Nominatim search (cached; >= 1 s between requests). Used ONLY as an independent locator for the boundary check,
    never as the boundary itself (polygon_geojson=0). Returns the record plus ``results``: osm_type, osm_id, lat, lon,
    display_name, class, type, boundingbox, extratags ({} when Nominatim returns null)."""
    cache = Path(cache_dir)
    tag = hashlib.sha256(query.encode()).hexdigest()[:12]
    params = {"q": query, "format": "jsonv2", "limit": limit, "extratags": 1, "polygon_geojson": 0, "addressdetails": 0}
    rec = http_get(NOMINATIM_URL, cache / f"nominatim_{tag}.json", timeout=30, max_bytes=1_000_000, session=session,
                   params=params, license=OSM_LICENSE, attribution=OSM_ATTRIBUTION + " via Nominatim",
                   rec_id=f"nominatim:{query}", cache_root=cache)
    rec["url"] = f"{NOMINATIM_URL}?{urlencode({k: v for k, v in params.items()})}"
    rec["results"] = parse_nominatim(json.loads((cache / rec["file"]).read_text()))
    return rec


def parse_nominatim(raw: Any) -> list[dict[str, Any]]:
    out = []
    for r in raw:
        out.append({"osm_type": r.get("osm_type"), "osm_id": r.get("osm_id"), "lat": float(r["lat"]), "lon": float(r["lon"]),
                    "display_name": r.get("display_name"), "class": r.get("category", r.get("class")),
                    "type": r.get("type"), "boundingbox": r.get("boundingbox"), "extratags": r.get("extratags") or {}})
    return out


# ---------------------------------------------------------------- OSM via osmnx
def _retry(fn: Any, what: str, attempts: int = 5, base_sleep: float = 20.0) -> Any:
    """Call ``fn()``; on a connection error or timeout wait base_sleep*attempt seconds and retry (bounded). Overpass is
    served by several backends and one can be briefly unreachable; a hard failure after ``attempts`` is raised."""
    import requests
    last: Exception | None = None
    for k in range(1, attempts + 1):
        try:
            return fn()
        except (requests.ConnectionError, requests.Timeout) as e:
            last = e
            print(f"  {what}: attempt {k}/{attempts} failed ({type(e).__name__}); waiting {base_sleep * k:.0f} s", flush=True)
            time.sleep(base_sleep * k)
    raise RuntimeError(f"{what}: gave up after {attempts} attempts") from last


def osm_features(polygon_wgs84: Any, tags: dict[str, Any], cache_dir: str | Path, *, max_features: int = 500_000,
                 timeout: int = 120) -> dict[str, dict[str, Any]]:
    """Fetch OSM layers inside ``polygon_wgs84`` through osmnx (Overpass), caching each as GeoJSON in ``cache_dir`` and
    returning {role: record} with feature counts. ``tags`` = {'buildings': {'building': True}, 'landuse':
    {'landuse': 'residential'}, 'amenity': {'amenity': [...]}}; roads are always fetched as the 'all' street network.
    osmnx rate-limits Overpass itself. A layer with more than ``max_features`` features aborts (bounded)."""
    import osmnx as ox
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    s = ox.settings
    s.use_cache, s.cache_folder, s.requests_timeout, s.log_console = True, str(cache / "osmnx_cache"), timeout, False
    setattr(s, "http_user_agent", USER_AGENT)
    setattr(s, "overpass_rate_limit", True)
    now = _now()
    out: dict[str, dict[str, Any]] = {}

    def save(role: str, gdf: Any, keep: Sequence[str], query: str) -> None:
        if len(gdf) > max_features:
            raise ValueError(f"OSM layer {role}: {len(gdf)} features exceeds max_features={max_features}")
        cols = [c for c in keep if c in gdf.columns]
        g = gdf.reset_index(drop=True)[cols + ["geometry"]].copy()
        for c in cols:
            g[c] = g[c].map(lambda v: v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
        path = cache / f"{role}.geojson"
        g.to_file(path, driver="GeoJSON")
        out[role] = _record(OVERPASS_URL, path, cache, rec_id=f"overpass:{role}", retrieved_utc=now, license=OSM_LICENSE,
                            attribution=OSM_ATTRIBUTION,
                            extra={"query": query, "feature_count": int(len(g)),
                                   "geometry_types": sorted(map(str, g.geom_type.unique()))})

    print("  OSM roads (Overpass via osmnx) ...", flush=True)
    graph = _retry(lambda: ox.graph.graph_from_polygon(polygon_wgs84, network_type="all", simplify=True), "osm_roads")
    save("osm_roads", ox.convert.graph_to_gdfs(graph, nodes=False), ("highway", "name", "length"), "network_type=all")
    for role, key in (("osm_buildings", "buildings"), ("osm_landuse", "landuse"), ("osm_amenity", "amenity")):
        if key not in tags:
            continue
        print(f"  OSM {role} (Overpass via osmnx) ...", flush=True)
        try:
            gdf = _retry(lambda k=key: ox.features.features_from_polygon(polygon_wgs84, tags[k]), role)
        except Exception as e:  # osmnx raises InsufficientResponseError when nothing matches
            if type(e).__name__ != "InsufficientResponseError":
                raise
            import geopandas as gpd
            gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        save(role, gdf, ("building", "landuse", "amenity", "name"), json.dumps(tags[key], sort_keys=True))
    return out


# ---------------------------------------------------------------- Copernicus DEM
def tile_name(lat_floor: int, lon_floor: int) -> str:
    """GLO-30 tile id for the 1x1 degree cell whose SW corner is (lat_floor, lon_floor)."""
    ns, ew = ("N" if lat_floor >= 0 else "S"), ("E" if lon_floor >= 0 else "W")
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"


def copernicus_dem_tile_urls(bounds: Sequence[float]) -> list[tuple[str, str]]:
    """[(tile_name, url)] of the GLO-30 COG tiles covering lon/lat ``bounds`` = (west, south, east, north)."""
    west, south, east, north = bounds
    if not (west < east and south < north):
        raise ValueError("bounds must be (west, south, east, north) with positive extent")
    out = []
    for lat in range(math.floor(south), math.ceil(north)):
        for lon in range(math.floor(west), math.ceil(east)):
            n = tile_name(lat, lon)
            out.append((n, COPERNICUS_URL.format(ns="N" if lat >= 0 else "S", lat=abs(lat),
                                                 ew="E" if lon >= 0 else "W", lon=abs(lon))))
    return out


def read_dem_window(urls: Sequence[str], bounds: Sequence[float], *, max_pixels: int = 25_000_000,
                    timeout_s: int = 30) -> tuple[Any, Any, str, float | None]:
    """Windowed read of COG tiles (HTTP range requests via GDAL), merged to ``bounds`` (west, south, east, north).
    Returns (array float32 [rows north->south], affine transform, crs string, nodata). Nodata is preserved as the
    source value (not converted); the window must not exceed ``max_pixels`` (checked before reading)."""
    import rasterio
    from rasterio.merge import merge
    with rasterio.Env(GDAL_HTTP_TIMEOUT=str(timeout_s), GDAL_HTTP_MAX_RETRY="2", GDAL_HTTP_USERAGENT=USER_AGENT,
                      GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        srcs = [rasterio.open(u) for u in urls]
        try:
            res = srcs[0].res
            npx = ((bounds[2] - bounds[0]) / res[0]) * ((bounds[3] - bounds[1]) / res[1])
            if npx > max_pixels:
                raise ValueError(f"DEM window would be {npx:.3g} pixels > max_pixels={max_pixels}")
            crs, nodata = srcs[0].crs.to_string(), srcs[0].nodata
            arr, transform = merge(srcs, bounds=tuple(bounds), nodata=nodata)
        finally:
            for s in srcs:
                s.close()
    return arr[0].astype("float32"), transform, crs, nodata


def write_dem_window(path: str | Path, array: Any, transform: Any, crs: str, nodata: float | None) -> None:
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1, dtype="float32",
                       crs=crs, transform=transform, nodata=nodata, compress="deflate") as dst:
        dst.write(array.astype("float32"), 1)


def fetch_dem(bounds: Sequence[float], cache_dir: str | Path, *, buffer_deg: float = 0.0,
              max_pixels: int = 25_000_000) -> dict[str, Any]:
    """Read the GLO-30 window for lon/lat ``bounds`` (+ ``buffer_deg``) and cache it as ``dem_window.tif`` (float32 GeoTIFF)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    b = (bounds[0] - buffer_deg, bounds[1] - buffer_deg, bounds[2] + buffer_deg, bounds[3] + buffer_deg)
    tiles = copernicus_dem_tile_urls(b)
    when = _now()
    arr, tr, crs, nodata = read_dem_window([u for _, u in tiles], b, max_pixels=max_pixels)
    path = cache / "dem_window.tif"
    write_dem_window(path, arr, tr, crs, nodata)
    return _record(tiles[0][1], path, cache, rec_id="Copernicus GLO-30 DSM: " + ", ".join(n for n, _ in tiles),
                   retrieved_utc=when, license=COPERNICUS_LICENSE, attribution=COPERNICUS_ATTRIBUTION,
                   extra={"tile_urls": [u for _, u in tiles], "window_bounds_lonlat": list(b), "nodata": nodata,
                          "crs": crs, "shape": list(arr.shape), "product": "Copernicus GLO-30 (DSM, ~30 m)"})


# ---------------------------------------------------------------- orchestration
def fetch_all(cfg: dict[str, Any], cache_dir: str | Path, *, session: Any = None) -> dict[str, dict[str, Any]]:
    """Fetch everything named in the geo config into ``cache_dir`` and write ``fetch_records.json``. Fails loudly on any
    missing piece. Network use: geoBoundaries API + GitHub-hosted GeoJSON, one Nominatim query, Overpass (osmnx), the
    Copernicus DEM S3 bucket."""
    from .regionio import domain_bounds_m, lonlat_box, project_geom, utm_epsg
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    study = cfg["study"]
    gb = study["geoboundaries"]
    recs: dict[str, dict[str, Any]] = {}

    def save(role: str, rec: dict[str, Any]) -> None:
        recs[role] = rec
        write_records(cache, {role: rec})
        print(f"fetched {role}: {rec['url']} ({rec['bytes']} bytes, sha256 {rec['sha256'][:12]})", flush=True)

    save("study", fetch_geoboundaries_geojson(gb["iso"], gb["level"], cache, expect_commit=gb["release"].get("commit"),
                                              session=session))
    for role, spec in (("adm2", cfg["adm2"]), *((f"adm0_{c['iso']}", c) for c in cfg["adm0"])):
        save(role, fetch_geoboundaries_geojson(spec["iso"], spec["level"], cache, session=session))
    loc = study.get("independent_locator", {}).get("nominatim_query")
    if loc:
        save("nominatim", nominatim_lookup(loc, cache, session=session))
    import shapely
    feat = find_feature(load_geojson(cache / recs["study"]["file"]), gb["shapeName"], gb.get("shapeID"))
    poly = shapely.geometry.shape(feat["geometry"])
    epsg = utm_epsg(poly.centroid.x, poly.centroid.y)
    dom_m = domain_bounds_m(project_geom(poly, 4326, epsg), float(cfg["grid"]["buffer_m"]))
    west, south, east, north = lonlat_box(dom_m, epsg)
    for role, rec in osm_features(shapely.box(west, south, east, north), cfg["osm"]["tags"], cache).items():
        save(role, rec)
    dem = cfg["dem"]
    save("dem", fetch_dem((west, south, east, north), cache, buffer_deg=float(dem.get("buffer_deg", 0.02)),
                          max_pixels=int(dem.get("max_window_pixels", 25_000_000))))
    return recs

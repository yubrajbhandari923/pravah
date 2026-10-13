"""Stage F2a: geographic sensor design on the REAL Belauri region.  Real geography; synthetic emissions and meteorology.

Everything numerical is synthetic: emission rates, inflow, winds, noise.  Only the geometry is real (municipality polygon,
OSM roads / facilities, Nepal ADM0 outline).  Candidate sites, layouts and all errors are a PROVISIONAL SIMULATION DESIGN,
not field-validated: mapped accessibility is not permission, power, security or connectivity.

Pipeline (``run_geodesign``), in this order (the order is a guarantee, see tests):
  1. geography + candidate table (interior lattice, reference pools) and held-out evaluation points;
  2. DESIGN and VALIDATION ensembles are built (cached unit responses, direct truth solve, truth components);
  3. alpha (ridge, scaled units) is chosen per noise fraction on the VALIDATION ensemble with alpha-independent layouts
     (random + spread) and frozen in ``alpha_selected.json``;
  4. layouts: random, spread, greedy_total, greedy_attribution per setting / regime / budget, selected on DESIGN cases only;
  5. only now the independent TEST ensemble (own seed, fresh noise) is built and every layout is evaluated on it.
Test seeds never influence steps 1-4 (``design_stage`` returns before the test ensemble exists).

Parameters theta = [road (local, ug/s), builtup (local, ug/s), external_patch (external, ug/s), bg (inflow, ug/m^3)];
emission and inflow errors are never mixed.  Noise sigma = fraction x signal scale, where the scale is the median over the
VALIDATION cases of the peak noiseless reading at the candidate sites, for ALL ensembles: the measurement-noise level the
estimator is told is fixed and independent of the test truths (each ensemble's own scale is only recorded).  Greedy layouts are heuristics, never called optimal.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ...runner import Progress, save_fig as _save_fig, say as _say
from ... import design as D
from ...attribution import base_parts, contributions, external_part, identifiability_report
from .belauri import draw_map, file_sha256, resolve_manifest
from ...config import SIDES, FieldResult, InflowBasis, SolverOptions, WindSegment
from ...design import Candidates, Constraints, DesignCase, DesignOptions, Layout
from ..toy.experiments import _peak_mb
from ...forward import FIPY
from ...geo import regionio as rio
from ...geo.prepare import Region, load_region
from ...geo.scenarios import LABEL as REGION_LABEL
from ...geo.scenarios import BelauriCase, belauri_scenario, crop_region, study_weights_of, toward_vector
from ...inverse import FitResult, ResponseCache, run_forward
from ...io import archive_git_state, data_provenance, make_run_dir, provenance, write_csv, write_json


LABEL = "Real geography; synthetic emissions and meteorology"
PROVISIONAL = "provisional simulation design, not field-validated"
ASSUMPTIONS = ("mapped accessibility is not permission/power/security/connectivity",
               "OSM gaps are not absence of feasible sites",
               "sites outside the Nepal ADM0 polygon are hypothetical (international access is not assumed)")
INTERIOR, REF_DOMESTIC, REF_HYPOTHETICAL = "interior", "reference_domestic", "reference_hypothetical"
SETTING_A, SETTING_B, SETTING_C = "interior_only", "with_reference", "hypothetical_sensitivity"
STANDARD, BLOCK = "standard", "block_holdout"
METHODS = ("random", "spread", "greedy_total", "greedy_attribution")
SELECTED, ALPHA_ZERO, STRONG = "selected", "alpha_zero", "strong_prior"
SD_NOTE = "sd across layouts x cases x noise reps (descriptive, never a confidence interval; cases/reps are not independent)"
PARAMS = ("road", "builtup", "external_patch", "bg")
DEFAULT_FACILITIES = ("school", "college", "kindergarten", "university", "hospital", "clinic", "doctors", "health_post")


def config_hash(cfg: dict[str, Any], drop: tuple[str, ...] = ()) -> str:
    """sha256 of the config JSON; ``drop`` removes dotted keys (e.g. 'ensembles.test') first."""
    c = json.loads(json.dumps(cfg, default=str))
    for key in drop:
        node = c
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.get(p, {})
        node.pop(parts[-1], None)
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------- geography
@dataclass(frozen=True, eq=False)
class Geography:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    full: Region
    region: Region  # the (possibly cropped) simulation region
    epsg: int
    study: Any  # shapely polygon, UTM metres
    nepal: Any  # Nepal ADM0 (clipped near the study), UTM metres
    roads: Any  # array of shapely line parts, UTM metres
    facilities: np.ndarray  # (n, 2) UTM metres
    raw_sha256: dict[str, str]
    ind: Any = None  # coarse geoBoundaries IND ADM0 outline (clipped near the study), UTM metres; for the candidate flags only


_RAW_CACHE: dict[str, dict[str, Any]] = {}


def _raw_file(manifest: dict[str, Any], base: Path, role: str) -> tuple[Path, str]:
    """Path of a raw dataset named in the manifest, with its sha256 VERIFIED against the manifest (no network)."""
    rec = next((d for d in manifest["datasets"] if d["role"] == role and d["kind"] == "raw"), None)
    if rec is None:
        raise ValueError(f"manifest has no raw dataset with role {role!r}")
    p = base / rec["file"]
    sha = file_sha256(p)
    if sha != rec["sha256"]:
        raise ValueError(f"{p}: sha256 {sha[:12]} does not match the manifest {rec['sha256'][:12]}")
    return p, sha


def _to_utm(geom: Any, epsg: int) -> Any:
    import shapely
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    def f(c: np.ndarray) -> np.ndarray:
        x, y = t.transform(c[:, 0], c[:, 1])
        return np.column_stack([x, y])
    return shapely.transform(shapely.segmentize(geom, 0.02), f)


def _load_raw(manifest_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Study polygon, home-country ADM0, roads and facilities (UTM) from the sha256-verified raw files; cached per process.

    Manifest roles come from optional ``cfg["region"]["roles"]`` (defaults = Belauri): ``home_adm0`` "adm0_NPL",
    ``neighbour_adm0`` "adm0_IND" (null, or a role absent from the manifest, means no neighbour: ``ind`` is None), ``roads``
    "osm_roads", ``amenity`` "osm_amenity".  The ``sha`` keys are the resolved role names.  Output names keep their Belauri
    spelling: ``nepal`` / ``in_nepal_adm0`` mean "home-country ADM0", ``inside_coarse_ind_outline`` means "inside the
    neighbour outline" (always False without one), and ``near_border_km`` is the distance to the HOME ADM0 boundary line
    whether or not a neighbour exists (for a region without a neighbour it is the distance to that outline, not to a border)."""
    roles = {"home_adm0": "adm0_NPL", "neighbour_adm0": "adm0_IND", "roads": "osm_roads", "amenity": "osm_amenity",
             **cfg["region"].get("roles", {})}
    key = f"{manifest_path.resolve()}|{json.dumps(cfg['region'].get('study', {}), sort_keys=True)}|{json.dumps(roles, sort_keys=True)}"
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]
    import shapely
    man = json.loads(manifest_path.read_text())
    base, epsg = manifest_path.parent, int(man["georef"]["epsg"])
    sha: dict[str, str] = {}
    study_spec = cfg["region"]["study"]
    p, sha["study"] = _raw_file(man, base, "study")
    feats = [f for f in json.loads(p.read_text())["features"]
             if f["properties"].get("shapeName") == study_spec["shapeName"]
             and f["properties"].get("shapeID") == study_spec["shapeID"]]
    if len(feats) != 1:
        raise ValueError(f"expected exactly one study feature, found {len(feats)}")
    poly_ll = shapely.geometry.shape(feats[0]["geometry"])
    study = shapely.make_valid(_to_utm(poly_ll, epsg))
    area_km2 = study.area / 1e6
    if abs(area_km2 - man["study"]["area_km2"]) > 1e-6 * man["study"]["area_km2"]:
        raise ValueError(f"study polygon area {area_km2:.4f} km2 differs from the manifest {man['study']['area_km2']:.4f}")
    p, sha[roles["home_adm0"]] = _raw_file(man, base, roles["home_adm0"])
    parts = [shapely.make_valid(shapely.geometry.shape(f["geometry"])) for f in json.loads(p.read_text())["features"]]
    cen = poly_ll.centroid
    clip = shapely.box(cen.x - 3, cen.y - 3, cen.x + 3, cen.y + 3)  # +-3 deg, as in geo.prepare
    nepal = shapely.make_valid(_to_utm(shapely.intersection(shapely.union_all(parts), clip), epsg))
    ind = None
    nb = roles["neighbour_adm0"]
    if nb is not None and any(d["role"] == nb and d["kind"] == "raw" for d in man["datasets"]):
        p, sha[nb] = _raw_file(man, base, nb)
        iparts = [shapely.make_valid(shapely.geometry.shape(f["geometry"])) for f in json.loads(p.read_text())["features"]]
        ind = shapely.make_valid(_to_utm(shapely.intersection(shapely.union_all(iparts), clip), epsg))
    p, sha[roles["roads"]] = _raw_file(man, base, roles["roads"])
    g = np.array([shapely.geometry.shape(f["geometry"]) for f in json.loads(p.read_text())["features"] if f.get("geometry")]
                 + [None], dtype=object)[:-1]
    roads = rio.explode_parts(_to_utm(g, epsg))
    roads = roads[shapely.get_type_id(roads) == 1]
    p, sha[roles["amenity"]] = _raw_file(man, base, roles["amenity"])
    fc = json.loads(p.read_text())["features"]
    types = tuple(cfg["candidates"].get("facility_amenities", DEFAULT_FACILITIES))
    keep = [f for f in fc if f.get("geometry") and (f.get("properties") or {}).get("amenity") in types]
    fac = np.zeros((0, 2))
    if keep:
        ag = _to_utm(np.array([shapely.geometry.shape(f["geometry"]) for f in keep], dtype=object), epsg)
        fac = shapely.get_coordinates(shapely.point_on_surface(shapely.make_valid(rio.as_geom_array(ag))))
    out = {"study": study, "nepal": nepal, "ind": ind, "roads": roads, "facilities": fac, "sha": sha}
    _RAW_CACHE[key] = out
    return out


def load_geography(cfg: dict[str, Any]) -> Geography:
    mp = resolve_manifest(cfg["region"]["manifest"])
    full = load_region(mp)
    buf = cfg["region"].get("crop_buffer_m")
    region = full if buf is None else crop_region(full, float(buf))
    raw = _load_raw(mp, cfg)
    return Geography(mp, file_sha256(mp), json.loads(mp.read_text()), full, region, full.georef.epsg, raw["study"],
                     raw["nepal"], raw["roads"], raw["facilities"], raw["sha"], raw["ind"])


# ---------------------------------------------------------------------------------------------- candidates
def _hull_box(geo: Geography, res: float, inset_m: float) -> Any:
    import shapely
    g = geo.region.level(res).grid
    return shapely.box(g.x0 + g.dx / 2 + inset_m, g.y0 + g.dy / 2 + inset_m, g.x0 + g.lx - g.dx / 2 - inset_m,
                       g.y0 + g.ly - g.dy / 2 - inset_m)


def _lattice(area: Any, spacing: float, origin: tuple[float, float], prefix: str) -> list[rio.CandidateSite]:
    if area.is_empty or area.area <= 0:
        return []
    return [dataclasses.replace(s, id=f"{prefix}_{s.id}") for s in rio.candidate_sites(area, spacing, origin)]


def build_candidate_table(geo: Geography, cfg: dict[str, Any]) -> pd.DataFrame:
    """All candidate sites with stable IDs and flags.  'interior': lattice (spacing candidates.interior_spacing_m, origin =
    the prepared grid's south-west corner) inside the municipality polygon.  'reference_domestic': lattice
    (candidates.reference_spacing_m) outside the polygon, inside the simulation domain (inset candidates.reference_inset_m
    from the sensor hull) and inside Nepal ADM0.  'reference_hypothetical': the same lattice inside the domain but OUTSIDE
    Nepal ADM0 (across the national border): hypothetical, not available by default.  Every site also carries
    ``inside_coarse_ind_outline`` (inside the coarse geoBoundaries IND ADM0 outline) and ``near_border_km`` (distance to
    the Nepal ADM0 boundary line): flags for a reader, nothing is excluded by them."""
    import shapely
    from pyproj import Transformer
    cc = cfg["candidates"]
    res = float(cfg["mesh"]["resolution_m"])
    origin = (geo.full.georef.x0, geo.full.georef.y0)
    hull = _hull_box(geo, res, float(cc["reference_inset_m"]))
    s_int, s_ref = float(cc["interior_spacing_m"]), float(cc["reference_spacing_m"])
    interior = _lattice(geo.study, s_int, origin, "int")
    domestic = _lattice(shapely.difference(shapely.intersection(hull, geo.nepal), geo.study), s_ref, origin, "refd")
    hypothetical = _lattice(shapely.difference(hull, geo.nepal), s_ref, origin, "refh")
    sites = [(s, INTERIOR) for s in interior] + [(s, REF_DOMESTIC) for s in domestic] + [(s, REF_HYPOTHETICAL) for s in hypothetical]
    if not interior:
        raise ValueError("no interior candidate sites (spacing too large for the polygon)")
    x = np.array([s.x for s, _ in sites])
    y = np.array([s.y for s, _ in sites])
    pts = shapely.points(x, y)
    bad = ~shapely.contains(_hull_box(geo, res, 0.0), pts)
    if bad.any():
        raise ValueError(f"candidate sites outside the simulation sensor hull: {[sites[i][0].id for i in np.flatnonzero(bad)]}")

    def nearest(targets: Any) -> np.ndarray:
        if len(targets) == 0:
            return np.full(len(x), np.inf)
        tree = shapely.STRtree(targets)
        return np.asarray(tree.query_nearest(pts, return_distance=True, all_matches=False)[1], dtype=float)
    d_road = nearest(geo.roads)
    d_fac = nearest(shapely.points(geo.facilities[:, 0], geo.facilities[:, 1]) if len(geo.facilities) else [])
    lon, lat = Transformer.from_crs(f"EPSG:{geo.epsg}", "EPSG:4326", always_xy=True).transform(x, y)
    lx, ly = geo.region.georef.utm_to_local(x, y)
    near = float(cc["road_near_m"])
    cat = [c for _, c in sites]
    # flags against the two outlines (the border is NOT independently verified, docs/geography.md): inside the coarse
    # geoBoundaries IND ADM0 outline, and the distance to the Nepal ADM0 boundary line
    in_ind = shapely.contains(geo.ind, pts) if geo.ind is not None else np.zeros(len(x), dtype=bool)
    near_border = shapely.distance(shapely.boundary(geo.nepal), pts) / 1000.0
    return pd.DataFrame({
        "id": [s.id for s, _ in sites], "category": cat, "lon": np.asarray(lon), "lat": np.asarray(lat),
        "utm_x": x, "utm_y": y, "local_x": np.asarray(lx), "local_y": np.asarray(ly),
        "in_polygon": shapely.contains(geo.study, pts), "in_nepal_adm0": shapely.contains(geo.nepal, pts),
        "inside_coarse_ind_outline": in_ind, "near_border_km": near_border,
        "dist_road_m": d_road, "dist_facility_m": d_fac, "on_or_near_road": d_road <= near,
        "hypothetical": [c == REF_HYPOTHETICAL for c in cat], "available_default": [c != REF_HYPOTHETICAL for c in cat],
        "assumption_flags": "; ".join(ASSUMPTIONS)})


def table_to_candidates(table: pd.DataFrame, include_hypothetical: bool = False) -> Candidates:
    """The ``design.Candidates`` used by the experiments: interior + reference_domestic, plus reference_hypothetical only
    when ``include_hypothetical`` (the clearly labelled sensitivity setting)."""
    t: Any = table if include_hypothetical else table[table["available_default"]]
    return Candidates(tuple(t["id"]), t["local_x"].to_numpy(float), t["local_y"].to_numpy(float), tuple(t["category"]),
                      {"lon": t["lon"].tolist(), "lat": t["lat"].tolist(), "utm_x": t["utm_x"].tolist(),
                       "utm_y": t["utm_y"].tolist()})


def candidates_geojson(table: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    feats = []
    for rec in table.to_dict("records"):
        props = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in rec.items() if k not in ("lon", "lat")}
        props = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in props.items()}
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(rec["lon"]), float(rec["lat"])]},
                      "properties": props})
    return {"type": "FeatureCollection", "metadata": meta, "features": feats}


def heldout_points(geo: Geography, cfg: dict[str, Any], table: pd.DataFrame) -> dict[str, Candidates]:
    """Held-out evaluation groups (local metres): 'random' = n deterministic random points inside the polygon, each at
    least heldout.min_distance_m (default 500 m) from EVERY candidate site (all categories), so they are genuinely
    held out (ValueError if no such point can be found); 'block' = a fixed lattice inside the polygon, restricted to the chosen spatial block (heldout_blocks)."""
    import shapely
    hc = cfg["heldout"]
    rng = np.random.default_rng(int(hc["random_seed"]))
    minx, miny, maxx, maxy = geo.study.bounds
    n, pts = int(hc["n_random_points"]), []
    cx, cy = table["utm_x"].to_numpy(float), table["utm_y"].to_numpy(float)
    dmin, tries = float(hc.get("min_distance_m", 500.0)), 0
    while len(pts) < n:
        tries += 1
        if tries > 200000:
            raise ValueError(f"cannot place {n} random held-out points >= {dmin:g} m from every candidate site inside the polygon")
        p = (float(rng.uniform(minx, maxx)), float(rng.uniform(miny, maxy)))
        if shapely.contains_xy(geo.study, *p) and np.hypot(cx - p[0], cy - p[1]).min() >= dmin:
            pts.append(p)
    ux, uy = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    lx, ly = geo.region.georef.utm_to_local(ux, uy)
    rand = Candidates(tuple(f"hr{i:02d}" for i in range(n)), np.asarray(lx), np.asarray(ly))
    b = hc["block"]
    sp, off = float(b["spacing_m"]), b["lattice_offset_m"]
    org = (geo.full.georef.x0 - off[0], geo.full.georef.y0 - off[1])
    lat = rio.candidate_sites(geo.study, sp, org)
    bx, by = geo.region.georef.utm_to_local(np.array([s.x for s in lat]), np.array([s.y for s in lat]))
    pool = Candidates(tuple(f"hb{i:03d}" for i in range(len(lat))), np.asarray(bx), np.asarray(by))
    inside, _ = D.heldout_blocks(pool, int(b["n_blocks_x"]), int(b["n_blocks_y"]), b["which"])
    return {"random": rand, "block": Candidates(tuple(pool.ids[i] for i in inside), pool.x[inside], pool.y[inside])}


# ---------------------------------------------------------------------------------------------- ensembles
@dataclass(frozen=True)
class CaseDraw:
    """One synthetic truth: amplitudes theta = [road, builtup, external_patch (ug/s), bg (ug/m^3)] and wind segments
    (t_start_s, from_deg, speed_m_s)."""
    name: str
    theta: tuple[float, float, float, float]
    winds: tuple[tuple[float, float, float], ...]


def draw_cases(cfg: dict[str, Any], n: int, seed: int, prefix: str) -> list[CaseDraw]:
    """n cases, deterministic in ``seed``.  road, builtup x U(amplitude_factor_range); patch x U(patch_factor_range) (so some
    cases are nearly local-only, some external-dominated); bg U(bg_range); wind direction = a random one of n_sectors compass
    sectors +- sector_half_width_deg, speed U(speed_range); every ``two_segment_every``-th case (index % k == 1) has a second
    segment (independent direction/speed draw: a turn, reversal or none by chance) at a random multiple of switch_step_s in
    [T/3, 2T/3]."""
    e, m = cfg["ensemble"], cfg["model"]
    T, step = float(cfg["physics"]["T"]), float(e["switch_step_s"])
    k_lo, k_hi = math.ceil(T / 3 / step - 1e-9), math.floor(2 * T / 3 / step + 1e-9)
    if k_hi < k_lo:
        raise ValueError("no multiple of switch_step_s lies within [T/3, 2T/3]")
    base = m["base_amplitudes_ug_s"]
    rng = np.random.default_rng(seed)

    def wind() -> tuple[float, float]:
        sector = int(rng.integers(0, int(e["n_sectors"])))
        jit = rng.uniform(-float(e["sector_half_width_deg"]), float(e["sector_half_width_deg"]))
        return float((sector * 360.0 / int(e["n_sectors"]) + jit) % 360.0), float(rng.uniform(*e["speed_range"]))

    out = []
    for i in range(n):
        road = float(base["road"]) * rng.uniform(*e["amplitude_factor_range"])
        built = float(base["builtup"]) * rng.uniform(*e["amplitude_factor_range"])
        patch = float(base["external"]) * rng.uniform(*e["patch_factor_range"])
        bg = float(rng.uniform(*e["bg_range"]))
        winds = [(0.0, *wind())]
        if i % int(e["two_segment_every"]) == 1:
            winds.append((float(rng.integers(k_lo, k_hi + 1) * step), *wind()))
        out.append(CaseDraw(f"{prefix}{i}", (road, built, patch, bg), tuple(winds)))
    return out


def _draw_key(d: CaseDraw) -> str:
    return json.dumps([list(d.theta), [list(w) for w in d.winds]])


def check_ensembles(ens: dict[str, tuple[list[CaseDraw], int, int]]) -> None:
    """ens[name] = (draws, scenario_seed, first_noise_seed); case i of an ensemble uses noise seed first + i.  Raise
    ValueError unless, pairwise, scenario seeds differ, noise-seed ranges are disjoint, no (theta, wind) draw is shared and
    no case name is shared (cf. experiments.check_ensembles)."""
    names = list(ens)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (da, sa, na), (db, sb, nb) = ens[a], ens[b]
            if sa == sb:
                raise ValueError(f"{a} and {b} scenario seeds must differ")
            if set(range(na, na + len(da))) & set(range(nb, nb + len(db))):
                raise ValueError(f"{a} and {b} noise-seed ranges overlap")
            if {_draw_key(x) for x in da} & {_draw_key(x) for x in db}:
                raise ValueError(f"{a} and {b} ensembles share a (theta, wind) draw")
            if {x.name for x in da} & {x.name for x in db}:
                raise ValueError(f"{a} and {b} case names overlap")


@dataclass(frozen=True, eq=False)
class Ensemble:
    role: str
    cases: list[DesignCase]
    info: list[dict[str, Any]]  # per-case record (truth amplitudes, wind, checks): evaluation/reporting only
    scale: float  # this ensemble's OWN median over cases of the peak noiseless reading at the candidates [ug/m^3] (recorded only)
    sigma: dict[float, float]  # noise fraction -> sigma [ug/m^3] = fraction x the VALIDATION scale (same for all ensembles)


class Setup:
    """Everything shared by the stages: geography, candidates, held-out points, base options, response cache."""

    def __init__(self, cfg: dict[str, Any], cache: ResponseCache | None = None) -> None:
        self.cfg, self.cache = cfg, cache or ResponseCache()
        self.geo = load_geography(cfg)
        self.table = build_candidate_table(self.geo, cfg)
        self.hyp = bool(cfg["settings"].get("hypothetical", False))
        self.cands = table_to_candidates(self.table, self.hyp)
        self.heldout = heldout_points(self.geo, cfg, self.table)
        self.solver = SolverOptions(**cfg.get("solver", {}))
        self.res, self.dt = float(cfg["mesh"]["resolution_m"]), float(cfg["mesh"]["dt_s"])
        ph = cfg["physics"]
        self.T, self.t_exclude, self.out_dt = float(ph["T"]), float(ph["t_exclude"]), float(ph["output_interval"])
        step = float(cfg["noise"]["obs_interval_s"])
        self.obs_times = tuple(float(t) for t in np.arange(step, self.T + 1e-9, step))
        self.weight = np.asarray(self.geo.region.level(self.res).arrays["study_fraction"], dtype=float)
        self.fractions = [float(f) for f in cfg["noise"]["fractions"]]
        self.reps = int(cfg["noise"]["reps"])
        self.budgets = [int(b) for b in cfg["methods"]["budgets"]]
        self.settings = [SETTING_A, SETTING_B] + ([SETTING_C] if self.hyp else [])
        self.alpha_grid = sorted(float(a) for a in cfg["alpha"]["grid"])

    def opts(self, sigma: float, alpha: float = 0.0, scales: tuple[float, float] | None = None) -> DesignOptions:
        return DesignOptions(noise_std=sigma, obs_times=self.obs_times, alpha=alpha, t_exclude=self.t_exclude,
                             weight=self.weight, eval_sites=self.heldout, solver=self.solver, attribution_scales=scales)

    def scenario(self, d: CaseDraw) -> BelauriCase:
        m = self.cfg["model"]
        winds = tuple(WindSegment(t, *toward_vector(f, s)) for t, f, s in d.winds)
        bg = m["bg"]
        basis = InflowBasis("bg", tuple(bg.get("sides", SIDES)), tuple((float(t), float(v)) for t, v in bg["profile"]))
        ph = self.cfg["physics"]
        amp: dict[str, float] = {"road": d.theta[0], "builtup": d.theta[1], "external": d.theta[2], "bg": d.theta[3]}
        return belauri_scenario(
            self.geo.region, self.res, winds, groups=tuple(m["groups"]), duration_s=self.T, dt_s=self.dt,
            output_interval_s=self.out_dt, K=float(ph["K"]), lam=float(ph["lam"]), H=float(ph["H"]), amplitudes=amp,
            bases=(basis,), external_profile=tuple((float(t), float(v)) for t, v in m["external_profile"]))

    def build(self, d: CaseDraw, role: str, noise_seed: int) -> tuple[DesignCase, dict[str, Any]]:
        """Matched case: the inference model is the truth scenario (declared wind), truth = a DIRECT forward solve; truth
        components come from the tags of the same cached unit responses (used only for design objectives / evaluation)."""
        bc = self.scenario(d)
        sc, theta = bc.scenario, bc.theta
        miss = self.cache.misses
        resp = self.cache.get(sc, self.solver)
        truth: FieldResult = run_forward(FIPY, sc, theta, self.solver)
        tags = contributions(resp, theta, base_parts(sc, self.solver, FIPY, self.cache))
        lin = np.asarray(resp.field(theta).values)
        direct = np.asarray(truth.concentration.transpose("time", "y", "x").values)
        rel = float(np.abs(lin - direct).max() / max(float(np.abs(direct).max()), 1e-300))
        if rel > 1e-6:
            raise ValueError(f"case {d.name}: direct truth and the linear response sum differ by {rel:.2e} (relative)")
        # opts() sigma is irrelevant at build time (applied at fit time); only baked options matter
        case = D.build_case(d.name, role, sc, theta, sc, self.cands, self.opts(0.0), noise_seed, self.reps, self.cache,
                            truth_field=truth.concentration,
                            truth_components={"local": tags["local"], "external": external_part(tags)})
        speed = max(s for _, _, s in d.winds)
        info = {"case": d.name, "role": role, "noise_seed": noise_seed, "road_ug_s": theta[0], "builtup_ug_s": theta[1],
                "external_patch_ug_s": theta[2], "bg_ug_m3": theta[3],
                "wind": ";".join(f"t={t:g}s from {f:.1f} deg at {s:.2f} m/s" for t, f, s in d.winds),
                "n_wind_segments": len(d.winds), "max_courant": speed * self.dt / self.res,
                "linear_vs_direct_max_rel_diff": rel, "external_fraction_true": case.external_fraction_true,
                "new_response_sets": self.cache.misses - miss}
        return case, info

    def ensemble(self, role: str, progress: Progress = None, noise_scale: float | None = None) -> Ensemble:
        """Build the ensemble of ``role``.  sigma = fraction x ``noise_scale``, the signal scale of the VALIDATION
        ensemble: a fixed, test-independent measurement-noise level (the estimator is told sigma, so it must not be
        derived from the test truths; noise realisations stay fresh per ensemble).  For role 'validation' ``noise_scale``
        defaults to its own scale; for the others it is required."""
        e = self.cfg["ensembles"][role]
        prefix = role[0]
        draws = draw_cases(self.cfg, int(e["n_cases"]), int(e["seed"]), prefix)
        cases, infos = [], []
        for i, d in enumerate(draws):
            t0 = time.perf_counter()
            c, info = self.build(d, role, int(e["noise_seed"]) + i)
            info["build_s"] = time.perf_counter() - t0
            cases.append(c)
            infos.append(info)
            _say(progress, f"{role} case {d.name} built in {info['build_s']:.1f} s")
        scale = float(np.median([c.truth_readings.max() for c in cases]))
        if noise_scale is None:
            if role != "validation":
                raise ValueError(f"ensemble {role!r}: pass noise_scale (the VALIDATION ensemble's signal scale)")
            noise_scale = scale
        return Ensemble(role, cases, infos, scale, {f: f * float(noise_scale) for f in self.fractions})

    def category_mask(self, cats: tuple[str, ...]) -> np.ndarray:
        return np.isin(np.asarray(self.cands.categories), cats)

    def constraints(self, setting: str, regime: str) -> Constraints:
        cap = int(self.cfg["settings"]["max_reference"])
        if setting == SETTING_A:
            allowed, caps = self.category_mask((INTERIOR,)), None
        elif setting == SETTING_B:
            allowed, caps = self.category_mask((INTERIOR, REF_DOMESTIC)), {REF_DOMESTIC: cap}
        elif setting == SETTING_C:
            allowed, caps = self.category_mask((INTERIOR, REF_DOMESTIC, REF_HYPOTHETICAL)), {REF_DOMESTIC: cap, REF_HYPOTHETICAL: cap}
        else:
            raise ValueError(f"unknown setting {setting!r}")
        if regime == STANDARD:
            return Constraints(allowed, 0.0, caps)
        b = self.heldout["block"]
        return Constraints(allowed, 0.0, caps, float(self.cfg["heldout"]["block"]["exclusion_radius_m"]),
                           tuple((float(x), float(y)) for x, y in zip(b.x, b.y)))


# ---------------------------------------------------------------------------------------------- layouts
@dataclass(frozen=True)
class LayoutRec:
    setting: str
    regime: str
    method: str
    budget: int
    k: int
    layout: Layout
    noise_fraction: float | None = None  # None: alpha-independent layout (random, spread), shared by all fractions
    objective: str | None = None
    alpha: float | None = None
    design_sigma: float | None = None
    seed: int | None = None
    order: tuple[int, ...] = ()
    trace_score: float | None = None

    @property
    def layout_id(self) -> str:
        f = "" if self.noise_fraction is None else f"-f{self.noise_fraction:g}"
        return f"{self.setting}|{self.regime}|{self.method}{f}|b{self.budget}|{self.k}"


def alpha_independent_layouts(st: Setup, setting: str, regime: str) -> tuple[list[LayoutRec], dict[int, str]]:
    """random (n_random seeded feasible layouts per budget) and spread layouts; they do not depend on any case, noise or
    alpha.  Returns (layouts, infeasible budgets with reasons)."""
    cons = st.constraints(setting, regime)
    infeas = D.infeasible_budgets(st.cands, cons, st.budgets)
    mc = st.cfg["methods"]
    si, ri = st.settings.index(setting), (STANDARD, BLOCK).index(regime)
    out: list[LayoutRec] = []
    for b in (b for b in st.budgets if b not in infeas):
        seed = int(mc["layout_seed"]) + 10000 * si + 1000 * ri + b
        for k, lay in enumerate(D.random_layouts(st.cands, b, int(mc["n_random"]), cons, seed)):
            out.append(LayoutRec(setting, regime, "random", b, k, lay, seed=seed))
        out.append(LayoutRec(setting, regime, "spread", b, 0, D.spread_layout(st.cands, b, cons)))
    return out, infeas


def select_alpha(st: Setup, validation: Ensemble) -> dict[str, Any]:
    """Per noise fraction: alpha from the pre-declared grid with the lowest mean total regional RMSE (study-weighted, over
    alpha-independent layouts x VALIDATION cases x noise reps); exact tie -> smaller alpha.  Returns the
    alpha_selected.json payload (contains nothing from the test ensemble) plus the raw rows under key '_raw'."""
    lays: list[LayoutRec] = []
    for setting in st.settings:
        lays += alpha_independent_layouts(st, setting, STANDARD)[0]
    flat: dict[str, list[Layout]] = {}
    for r in lays:
        flat.setdefault(f"{r.setting}|{r.method}", []).append(r.layout)
    raw_parts, tables, chosen = [], {}, {}
    for f in st.fractions:
        sigma = validation.sigma[f]
        parts = []
        for a in st.alpha_grid:
            df = D.evaluate_validation(flat, validation.cases, st.opts(sigma, a))
            df.insert(0, "alpha", a)
            parts.append(df)
        raw = pd.concat(parts, ignore_index=True)
        raw.insert(0, "noise_fraction", f)
        raw.insert(1, "sigma_ug_m3", sigma)
        tab = cast(Any, raw).groupby("alpha")["rmse"].agg(["mean", "median", "count"]).reset_index()
        tab = tab.rename(columns={"mean": "criterion_mean_rmse", "median": "median_rmse", "count": "n_rows"})
        best = min((math.inf if not math.isfinite(c) else float(c), float(a))
                   for a, c in zip(tab["alpha"], tab["criterion_mean_rmse"]))
        chosen[f"{f:g}"], tables[f"{f:g}"] = best[1], tab.to_dict("records")
        raw_parts.append(raw)
    e = st.cfg["ensembles"]["validation"]
    return {
        "alpha_by_noise_fraction": chosen, "alpha_grid": st.alpha_grid, "tables": tables,
        "criterion": "mean over alpha-independent layouts (random + spread, every setting, standard regime), budgets, "
                     "VALIDATION cases and noise reps of the total regional RMSE (study-area x time weighted, ug/m^3); "
                     "lowest wins, exact ties go to the smaller alpha",
        "units": "alpha acts in scaled units (q / q_scale, emission 1e5 ug/s, inflow 0.05 ug/m^3), prior mean 0",
        "validation": {"seed": e["seed"], "noise_seed": e["noise_seed"], "n_cases": e["n_cases"], "signal_scale_ug_m3": validation.scale,
                       "sigma_ug_m3": {f"{f:g}": validation.sigma[f] for f in st.fractions}},
        "noise_reps": st.reps, "n_random": st.cfg["methods"]["n_random"], "layout_seed": st.cfg["methods"]["layout_seed"],
        "budgets": st.budgets, "settings": st.settings, "label": LABEL,
        "config_hash_without_test_ensemble": config_hash(st.cfg, ("ensembles.test",)),
        "uses_test_ensemble": False, "_raw": pd.concat(raw_parts, ignore_index=True)}


def make_layouts(st: Setup, design: Ensemble, alphas: dict[float, float]) -> tuple[list[LayoutRec], dict[str, dict[int, str]]]:
    """All layouts.  Greedy layouts are selected on DESIGN cases only, per noise fraction (sigma = fraction x the validation
    ensemble's scale, alpha = the frozen validation choice); one greedy trace to the largest feasible budget is truncated
    to every budget (a greedy layout is the prefix of the trace).  Returns (layouts, infeasible budgets per
    '<setting>|<regime>')."""
    scales = D.attribution_scales(design.cases)
    out: list[LayoutRec] = []
    infeasible: dict[str, dict[int, str]] = {}
    for setting in st.settings:
        for regime in (STANDARD, BLOCK):
            lays, infeas = alpha_independent_layouts(st, setting, regime)
            infeasible[f"{setting}|{regime}"] = infeas
            out += lays
            feas = [b for b in st.budgets if b not in infeas]
            if not feas:
                continue
            cons = st.constraints(setting, regime)
            for f in st.fractions:
                opts = st.opts(design.sigma[f], alphas[f], scales)
                for method, obj in (("greedy_total", "total"), ("greedy_attribution", "attribution")):
                    tr = D.greedy_trace(st.cands, max(feas), design.cases, cons, opts, obj)
                    for b in feas:
                        out.append(LayoutRec(setting, regime, method, b, 0, tuple(sorted(tr.order[:b])), f, obj,
                                             alphas[f], design.sigma[f], None, tr.order[:b], tr.scores[b - 1]))
    return out, infeasible


@dataclass
class DesignStage:
    setup: Setup
    design: Ensemble
    validation: Ensemble
    alpha: dict[str, Any]  # payload (without '_raw')
    alpha_by_f: dict[float, float]
    validation_raw: pd.DataFrame
    layouts: list[LayoutRec]
    infeasible: dict[str, dict[int, str]]
    timings: dict[str, float] = field(default_factory=dict)


def design_stage(cfg: dict[str, Any], cache: ResponseCache | None = None, progress: Progress = None,
                 setup: Setup | None = None) -> DesignStage:
    """Steps 1-4 of the pipeline; the TEST ensemble is not built here and no test seed is read."""
    tim: dict[str, float] = {}
    t0 = time.perf_counter()
    st = setup or Setup(cfg, cache)
    tim["geography_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    _say(progress, "building validation ensemble")  # first: its signal scale defines sigma for every ensemble
    validation = st.ensemble("validation", progress)
    _say(progress, "building design ensemble")
    design = st.ensemble("design", progress, validation.scale)
    tim["design_validation_cases_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    payload = select_alpha(st, validation)
    raw = payload.pop("_raw")
    alphas = {f: float(payload["alpha_by_noise_fraction"][f"{f:g}"]) for f in st.fractions}
    _say(progress, f"alpha selected on validation: {payload['alpha_by_noise_fraction']}")
    tim["alpha_selection_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    layouts, infeas = make_layouts(st, design, alphas)
    tim["layouts_s"] = time.perf_counter() - t0
    return DesignStage(st, design, validation, payload, alphas, raw, layouts, infeas, tim)


# ---------------------------------------------------------------------------------------------- evaluation
def _fit_result(sol: Any, names: tuple[str, ...], kinds: tuple[str, ...], units: tuple[str, ...], n_obs: int,
                alpha: float) -> FitResult:
    return FitResult(
        q_hat=sol.q_hat, success=sol.success, message=sol.message, weighted_residual=sol.weighted_residual, n_obs=n_obs,
        singular_values=sol.singular_values, rank=sol.rank, condition=sol.condition, effective_rank=sol.effective_rank,
        q_std=sol.q_std, reg_singular_values=sol.reg_singular_values, reg_condition=sol.reg_condition, method="linear",
        q_scale=sol.q_scale, alpha=alpha, prior=sol.prior, heldout_ids=(), runtime_s=0.0, param_names=names,
        param_kinds=kinds, param_units=units, column_norms=sol.column_norms, column_cosines=sol.column_cosines)


def fit_row(case: DesignCase, rec: LayoutRec, rep: int, opts: DesignOptions, cats: tuple[str, ...], flags: bool,
            units: tuple[str, ...]) -> dict[str, Any]:
    """One evaluation row for (layout, case, rep): metrics from ``design.fit_layout`` (the fit sees only the noisy readings
    at the layout), ``chi2_per_dof`` (below) and identifiability flags (``attribution.identifiability_report``, only
    when ``flags``).

    chi2_per_dof = ||(b + A theta_hat - y) / sigma||^2 / (n_obs - n_params): the weighted data residual of the fitted
    estimate (``solve_linear``'s data rows only; the ridge prior rows are not counted) over the fitted sensors x times
    minus the parameters.  Unlike the flags it depends on the DATA, so it responds to model error; it is about 1 when the
    model and sigma are right (up to the ridge/bound effects on the degrees of freedom).  NaN when n_obs <= n_params."""
    f = D.fit_layout(case, rec.layout, rep, opts)
    nan = float("nan")
    err = f["q_hat"] - case.q_true[case.truth_index]
    cnt = Counter(cats[i] for i in rec.layout)
    row: dict[str, Any] = {
        "case": case.name, "rep": rep, "sensors": ",".join(case.candidate_ids[i] for i in rec.layout),
        "rmse": f["rmse"], "rmse_local": math.sqrt(f["mse_local"]), "rmse_external": math.sqrt(f["mse_external"]),
        "mse_attribution": f["mse_attribution"], "attr_err": math.sqrt(f["mse_attribution"]),
        "external_fraction_true": _nn(f.get("external_fraction_true")), "external_fraction_est": _nn(f.get("external_fraction_est")),
        "external_fraction_error": _nn(f.get("external_fraction_error")),
        "source_rel_err": f["source_rel_err"], "inflow_abs_err_bg": float(f["inflow_abs_err"].get("bg", nan)),
        **{f"err_{n}": float(e) for n, e in zip(case.param_names, err)},
        "heldout_rmse": f["heldout_rmse"], "heldout_random_rmse": f["heldout_by_group"].get("random", nan),
        "heldout_block_rmse": f["heldout_by_group"].get("block", nan),
        "rank": f["rank"], "effective_rank": f["effective_rank"], "cond_number": f["condition"], "success": f["success"],
        "n_reference_sites": sum(v for c, v in cnt.items() if c != INTERIOR),
        "category_counts": ";".join(f"{c}={cnt[c]}" for c in sorted(cnt)), "id_flags": None}
    sol = D._solve(case, rec.layout, rep, opts)  # same solve as fit_layout's: gives the weighted residual
    n_obs = len(rec.layout) * len(case.obs_times)
    dof = n_obs - case.n_params
    row["chi2_per_dof"] = float(sol.weighted_residual ** 2 / dof) if dof > 0 else nan
    if flags:
        names, kinds = case.param_names, case.param_kinds
        fit = _fit_result(sol, names, kinds, units, n_obs, opts.alpha)
        row["id_flags"] = ";".join(identifiability_report(fit)["flags"])
    return row


def _nn(v: float | None) -> float:
    return float("nan") if v is None else float(v)


def evaluate_test(st: Setup, layouts: list[LayoutRec], test: Ensemble, alphas: dict[float, float],
                  scales: tuple[float, float], progress: Progress = None) -> pd.DataFrame:
    """Rows (layout x test case x rep) per noise fraction and alpha label ('selected', 'alpha_zero', 'strong_prior').
    Test cases bring fresh noise; sigma = fraction x the VALIDATION signal scale (test.sigma).  Identifiability flags are computed
    for the selected-alpha rows only (rank / effective rank / condition are in every row)."""
    D._require_role(test.cases, "test", "evaluate_test")
    units = tuple(p.units for p in st.scenario(draw_cases(st.cfg, 1, 0, "u")[0]).scenario.param_table())
    labels = {SELECTED: None, ALPHA_ZERO: 0.0, STRONG: float(st.cfg["alpha"]["strong_prior"])}
    cats = test.cases[0].category
    rows: list[dict[str, Any]] = []
    for f in st.fractions:
        for label, fixed in labels.items():
            a = alphas[f] if fixed is None else fixed
            opts = st.opts(test.sigma[f], a, scales)
            for rec in layouts:
                if rec.noise_fraction not in (None, f):
                    continue
                for c in test.cases:
                    for rep in range(c.n_noise):
                        r = fit_row(c, rec, rep, opts, cats, label == SELECTED, units)
                        rows.append({"setting": rec.setting, "regime": rec.regime, "noise_fraction": f,
                                     "sigma_ug_m3": test.sigma[f], "alpha_label": label, "alpha": a, "method": rec.method,
                                     "layout_id": rec.layout_id, "budget": rec.budget, **r})
        _say(progress, f"test evaluation done for noise fraction {f:g} ({len(rows)} rows)")
    return pd.DataFrame(rows)


METRICS = ("rmse", "rmse_local", "rmse_external", "attr_err", "ext_frac_abs_err", "source_rel_err", "inflow_abs_err_bg_abs",
           "heldout_random_rmse", "heldout_block_rmse", "effective_rank", "n_reference_sites", "chi2_per_dof")
GROUP = ["setting", "regime", "noise_fraction", "alpha_label", "method", "budget"]


def _prep(raw: pd.DataFrame) -> Any:
    d: Any = raw.copy()
    d["ext_frac_abs_err"] = d["external_fraction_error"].abs()
    d["inflow_abs_err_bg_abs"] = d["inflow_abs_err_bg"].abs()
    return d


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd (ddof=1) of every metric per (setting, regime, noise fraction, alpha label, method, budget), pooled over
    layouts x cases x reps; ext_frac_abs_err is over rows where the external fraction is defined (n_ext_defined)."""
    d = _prep(raw)
    g = d.groupby(GROUP)
    out = pd.DataFrame({"n_rows": g.size(), "n_layouts": g["layout_id"].nunique(), "n_cases": g["case"].nunique(),
                        "n_ext_defined": g["ext_frac_abs_err"].count()})
    for m in METRICS:
        out[f"mean_{m}"], out[f"sd_{m}"] = g[m].mean(), g[m].std(ddof=1)
    out["mean_ext_frac_bias"] = g["external_fraction_error"].mean()
    out["median_chi2_per_dof"] = g["chi2_per_dof"].median()
    out["frac_rows_id_flagged"] = g["id_flags"].apply(lambda s: float(np.mean([bool(v) for v in s if isinstance(v, str)]))
                                                         if s.map(lambda v: isinstance(v, str)).any() else float("nan"))
    out["sd_note"] = SD_NOTE
    return out.reset_index()


def _layout_mean(d: Any, metrics: list[str]) -> Any:
    """Average over layouts of a method (random has several) per (setting, regime, fraction, budget, case, rep)."""
    keys = ["setting", "regime", "noise_fraction", "method", "budget", "case", "rep"]
    return d[d["alpha_label"] == SELECTED].groupby(keys)[metrics].mean().reset_index()


def _nested(raw: pd.DataFrame) -> dict[tuple[Any, ...], bool]:
    """(setting, regime, noise_fraction, method, previous budget, budget) -> True when every layout of the previous budget
    is contained in some layout of the larger budget (read from the 'sensors' column) and the method is not 'random'.
    Random layouts are independent draws per budget, so their budget-to-budget difference compares non-nested layouts."""
    sets: dict[tuple[Any, ...], set[frozenset[str]]] = {}
    cols = ["setting", "regime", "noise_fraction", "method", "budget", "sensors"]
    for st_, rg, f, m, b, sens in raw[cols].drop_duplicates().itertuples(index=False):
        sets.setdefault((st_, rg, f, m, b), set()).add(frozenset(str(sens).split(",")))
    out: dict[tuple[Any, ...], bool] = {}
    for (s_, r_, f_, m_, b1), big in sets.items():
        for (s0, r0, f0, m0, b0), small in sets.items():
            if (s0, r0, f0, m0) == (s_, r_, f_, m_) and b0 < b1:
                out[(s_, r_, f_, m_, b0, b1)] = m_ != "random" and all(any(a <= b for b in big) for a in small)
    return out


def paired_tables(raw: pd.DataFrame) -> pd.DataFrame:
    """Paired differences (a - b, negative = a better) per (case, rep), after averaging over layouts of a method:
    greedy_total vs spread and vs random (mean); greedy_attribution vs greedy_total and vs spread, on total RMSE and on the
    attribution error; with_reference vs interior_only at equal budget for every method; consecutive-budget improvement
    (b - previous budget, 'per_added_sensor' = mean difference / added sensors; column ``nested`` is False when the two budgets'
    layouts are not nested, as for 'random', whose per-added-sensor value then compares different layouts, not added sensors).  Rows give mean, sd (labelled sd, not a CI),
    n_pairs, how many of the cases a wins (case means over reps) and the fraction of pairs a wins."""
    d = _prep(raw)
    mets = ["rmse", "attr_err"]
    lm = _layout_mean(d, mets)
    rows: list[dict[str, Any]] = []
    nest = _nested(raw)

    def add(comp: str, a: Any, b: Any, keys: list[str], meta: dict[str, Any], metric: str, per: float = 1.0, nested: bool | None = None) -> None:
        m = a.merge(b, on=keys, suffixes=("_a", "_b"))
        if m.empty:
            return
        diff = m[f"{metric}_a"] - m[f"{metric}_b"]
        case_diff = diff.groupby(m["case"]).mean()
        rows.append({"comparison": comp, **meta, "metric": metric, "n_pairs": int(len(m)), "mean_diff": float(diff.mean()),
                     "sd_diff": float(diff.std(ddof=1)) if len(m) > 1 else float("nan"),
                     "per_added_sensor": float(diff.mean() / per), "a_wins_cases": int((case_diff < 0).sum()),
                     "n_cases": int(len(case_diff)), "frac_pairs_a_better": float((diff < 0).mean()), "sd_note": SD_NOTE,
                     "nested": nested})

    base = ["setting", "regime", "noise_fraction", "budget", "case", "rep"]
    for a_m, b_m in (("greedy_total", "spread"), ("greedy_total", "random"), ("greedy_attribution", "greedy_total"),
                     ("greedy_attribution", "spread")):
        for metric in mets:
            a, b = lm[lm["method"] == a_m].drop(columns="method"), lm[lm["method"] == b_m].drop(columns="method")
            for (s, rg, f), _ in a.groupby(["setting", "regime", "noise_fraction"]):
                aa = a[(a["setting"] == s) & (a["regime"] == rg) & (a["noise_fraction"] == f)]
                bb = b[(b["setting"] == s) & (b["regime"] == rg) & (b["noise_fraction"] == f)]
                for bud in sorted(aa["budget"].unique()):
                    add(f"{a_m} vs {b_m if b_m != 'random' else 'random_mean'}", aa[aa["budget"] == bud], bb[bb["budget"] == bud],
                        ["case", "rep"], {"setting": s, "regime": rg, "noise_fraction": f, "budget": int(bud), "method_a": a_m,
                                          "method_b": b_m}, metric)
    ai = lm[lm["setting"] == SETTING_B].drop(columns="setting")
    bi = lm[lm["setting"] == SETTING_A].drop(columns="setting")
    for metric in mets:
        for m_ in sorted(lm["method"].unique()):
            for (rg, f, bud), _ in ai[ai["method"] == m_].groupby(["regime", "noise_fraction", "budget"]):
                x = ai[(ai["method"] == m_) & (ai["regime"] == rg) & (ai["noise_fraction"] == f) & (ai["budget"] == bud)]
                y = bi[(bi["method"] == m_) & (bi["regime"] == rg) & (bi["noise_fraction"] == f) & (bi["budget"] == bud)]
                add("with_reference vs interior_only", x, y, ["case", "rep"],
                    {"setting": SETTING_B, "regime": rg, "noise_fraction": f, "budget": int(bud), "method_a": m_,
                     "method_b": m_}, metric)
    for metric in mets:
        for (s, rg, f, m_), grp in lm.groupby(["setting", "regime", "noise_fraction", "method"]):
            buds = sorted(grp["budget"].unique())
            for b0, b1 in zip(buds, buds[1:]):
                add("consecutive_budgets", grp[grp["budget"] == b1], grp[grp["budget"] == b0], ["case", "rep"],
                    {"setting": s, "regime": rg, "noise_fraction": f, "budget": int(b1), "method_a": m_, "method_b": m_,
                     "budget_previous": int(b0)}, metric, float(b1 - b0), nest.get((s, rg, f, m_, b0, b1), False))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- layout exports
def layouts_table(recs: list[LayoutRec], st: Setup, chash: str) -> pd.DataFrame:
    cats = st.cands.categories
    rows = []
    for r in recs:
        c = Counter(cats[i] for i in r.layout)
        rows.append({
            "layout_id": r.layout_id, "setting": r.setting, "regime": r.regime, "method": r.method, "budget": r.budget, "k": r.k,
            "noise_fraction": r.noise_fraction, "objective": r.objective or "none (alpha-independent)", "alpha": r.alpha,
            "design_sigma_ug_m3": r.design_sigma, "layout_seed": r.seed,
            "design_seed": st.cfg["ensembles"]["design"]["seed"], "validation_seed": st.cfg["ensembles"]["validation"]["seed"],
            "sites": ",".join(st.cands.ids[i] for i in (r.order or r.layout)), "n_sites": len(r.layout),
            "category_counts": ";".join(f"{k_}={c[k_]}" for k_ in sorted(c)), "config_hash": chash,
            "trace_score": r.trace_score, "label": LABEL, "provenance": PROVISIONAL})
    return pd.DataFrame(rows)


def layouts_geojson(recs: list[LayoutRec], st: Setup, chash: str) -> dict[str, Any]:
    meta = st.cands.meta or {}
    feats = []
    for r in recs:
        for pos, i in enumerate(r.order or r.layout):
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [meta["lon"][i], meta["lat"][i]]},
                          "properties": {"layout_id": r.layout_id, "setting": r.setting, "regime": r.regime, "method": r.method,
                                         "budget": r.budget, "k": r.k, "site_id": st.cands.ids[i], "category": st.cands.categories[i],
                                         "order": pos + 1 if r.order else None, "objective": r.objective, "alpha": r.alpha,
                                         "noise_fraction": r.noise_fraction, "layout_seed": r.seed, "config_hash": chash,
                                         "label": LABEL, "provenance": PROVISIONAL}})
    return {"type": "FeatureCollection", "metadata": {"label": LABEL, "provenance": PROVISIONAL, "config_hash": chash},
            "features": feats}


def design_scores(st: Setup, layouts: list[LayoutRec], design: Ensemble, alphas: dict[float, float]) -> pd.DataFrame:
    """Design-ensemble scores of every layout at every design noise fraction (diagnostic; selection used the design
    cases only): total-objective RMSE and the attribution objective."""
    scales = D.attribution_scales(design.cases)
    rows = []
    for f in st.fractions:
        opts = st.opts(design.sigma[f], alphas[f], scales)
        for r in layouts:
            if r.noise_fraction not in (None, f):
                continue
            tot = D.score_layout(r.layout, design.cases, opts, "total")
            att = D.score_layout(r.layout, design.cases, opts, "attribution")
            rows.append({"layout_id": r.layout_id, "setting": r.setting, "regime": r.regime, "method": r.method,
                         "budget": r.budget, "k": r.k, "noise_fraction": f, "design_sigma_ug_m3": design.sigma[f],
                         "alpha": alphas[f], "design_rmse": tot["rmse"], "design_attribution": att["value"],
                         "selection_trace_score": r.trace_score, "selection_objective": r.objective})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- figures
def _stamp(fig: Figure, extra: str = "") -> Figure:
    eng = fig.get_layout_engine()
    if eng is not None:
        cast(Any, eng).set(rect=(0.0, 0.03, 1.0, 0.97))
    fig.text(0.01, 0.005, f"{LABEL}. {PROVISIONAL}. {extra}".strip(), fontsize=7, style="italic", color="0.25")
    return fig


def _map(ax: Any, st: Setup) -> None:
    draw_map(ax, st.geo.region, st.res)


def _scatter_sites(ax: Any, st: Setup, idx: list[int] | tuple[int, ...], **kw: Any) -> None:
    ax.scatter(st.cands.x[list(idx)] / 1000.0, st.cands.y[list(idx)] / 1000.0, **kw)


def fig_layout_maps(st: Setup, recs: list[LayoutRec], budget: int, f: float) -> Figure:
    fig, axes = plt.subplots(len(st.settings), len(METHODS), figsize=(4.0 * len(METHODS), 3.8 * len(st.settings)),
                             squeeze=False, layout="constrained")
    cats = np.asarray(st.cands.categories)
    for i, s in enumerate(st.settings):
        for j, m in enumerate(METHODS):
            ax = axes[i, j]
            _map(ax, st)
            _scatter_sites(ax, st, list(range(len(st.cands))), s=6, c="0.85", edgecolors="0.4", linewidths=0.3, zorder=3)
            hit = [r for r in recs if (r.setting, r.regime, r.method, r.budget) == (s, STANDARD, m, budget)
                   and r.noise_fraction in (None, f) and r.k == 0]
            if hit:
                lay = list(hit[0].layout)
                ref = [k for k in lay if cats[k] != INTERIOR]
                _scatter_sites(ax, st, [k for k in lay if cats[k] == INTERIOR], s=70, c="tab:blue", marker="o", edgecolors="k", zorder=5)
                if ref:
                    _scatter_sites(ax, st, ref, s=90, c="tab:orange", marker="s", edgecolors="k", zorder=5)
            else:
                ax.text(0.5, 0.5, f"budget {budget} infeasible", transform=ax.transAxes, ha="center")
            ax.set_title(f"{s}\n{m}" + (" (layout 0 of random)" if m == "random" else ""), fontsize=8)
    fig.suptitle(f"Selected sites, budget {budget}, noise fraction {f:g} (blue interior, orange reference; greedy = heuristic)", fontsize=9)
    return _stamp(fig)


def fig_error_curves(summ: pd.DataFrame, setting: str, regime: str) -> Figure:
    s: Any = summ[(summ["setting"] == setting) & (summ["regime"] == regime) & (summ["alpha_label"] == SELECTED)]
    fr = sorted(s["noise_fraction"].unique())
    fig, axes = plt.subplots(1, max(len(fr), 1), figsize=(5.2 * max(len(fr), 1), 4.0), squeeze=False, layout="constrained")
    for ax, f in zip(axes[0], fr):
        for m in METHODS:
            d = s[(s["noise_fraction"] == f) & (s["method"] == m)].sort_values("budget")
            if d.empty:
                continue
            ax.plot(d["budget"], d["mean_rmse"], marker="o", label=m)
            ax.fill_between(d["budget"], d["mean_rmse"] - d["sd_rmse"].fillna(0), d["mean_rmse"] + d["sd_rmse"].fillna(0), alpha=0.15)
        ax.set(xlabel="number of sensors", ylabel="total regional RMSE (ug/m^3, study-weighted)", title=f"noise fraction {f:g}")
        ax.legend(fontsize=7)
    fig.suptitle(f"{setting}, {regime}: mean +/- 1 s.d. (shaded) across layouts x cases x reps; s.d. is descriptive, not a CI",
                 fontsize=8)
    return _stamp(fig)


def fig_attribution_vs_total(summ: pd.DataFrame, setting: str, regime: str) -> Figure:
    s: Any = summ[(summ["setting"] == setting) & (summ["regime"] == regime) & (summ["alpha_label"] == SELECTED)]
    fr = sorted(s["noise_fraction"].unique())
    fig, axes = plt.subplots(len(fr) or 1, 2, figsize=(9.5, 3.6 * max(len(fr), 1)), squeeze=False, layout="constrained")
    for i, f in enumerate(fr):
        for j, (m, lab) in enumerate((("rmse", "total regional RMSE (ug/m^3)"), ("attr_err", "attribution error sqrt(objective)"))):
            ax = axes[i, j]
            for meth in ("greedy_total", "greedy_attribution"):
                d = s[(s["noise_fraction"] == f) & (s["method"] == meth)].sort_values("budget")
                if d.empty:
                    continue
                ax.errorbar(d["budget"], d[f"mean_{m}"], yerr=d[f"sd_{m}"].fillna(0), marker="o", capsize=3, label=meth)
            ax.set(xlabel="number of sensors", ylabel=lab, title=f"noise fraction {f:g}")
            ax.legend(fontsize=7)
    fig.suptitle(f"{setting}, {regime}: greedy_attribution vs greedy_total on both metrics (error bars: +/- 1 s.d., descriptive)", fontsize=8)
    return _stamp(fig)


def fig_block_map(st: Setup, recs: list[LayoutRec], f: float) -> Figure:
    fig, axes = plt.subplots(1, len(st.settings), figsize=(5.2 * len(st.settings), 4.6), squeeze=False, layout="constrained")
    b = st.heldout["block"]
    rad = float(st.cfg["heldout"]["block"]["exclusion_radius_m"])
    for ax, s in zip(axes[0], st.settings):
        _map(ax, st)
        ax.scatter(b.x / 1000.0, b.y / 1000.0, s=5, c="tab:red", alpha=0.6, label="block evaluation points", zorder=3)
        ax.scatter(st.heldout["random"].x / 1000.0, st.heldout["random"].y / 1000.0, s=14, c="tab:green", marker="x",
                   label="random held-out points", zorder=3)
        ok = D.feasible_sites(st.cands, st.constraints(s, BLOCK))
        _scatter_sites(ax, st, list(range(len(st.cands))), s=7, c="0.8", edgecolors="0.4", linewidths=0.3, zorder=4, label="candidates")
        _scatter_sites(ax, st, list(ok), s=12, c="0.2", zorder=4, label=f"feasible (>= {rad:g} m from block points)")
        top = [r for r in recs if r.setting == s and r.regime == BLOCK and r.method == "greedy_total" and r.noise_fraction == f]
        if top:
            r = max(top, key=lambda q: q.budget)
            _scatter_sites(ax, st, list(r.layout), s=80, c="tab:blue", edgecolors="k", zorder=5, label=f"greedy_total, budget {r.budget}")
        ax.set_title(s, fontsize=9)
        ax.legend(fontsize=6, loc="lower left")
    fig.suptitle("Spatial-block hold-out: layouts built with an exclusion radius around the block's evaluation points", fontsize=8)
    return _stamp(fig)


# ---------------------------------------------------------------------------------------------- the runner
@dataclass
class GeodesignResult:
    out_dir: Path
    alpha: dict[str, Any]
    files: list[str]
    timings: dict[str, float]
    figures: list[str] = field(default_factory=list)


def noise_levels(st: Setup, ens: dict[str, Ensemble]) -> pd.DataFrame:
    return pd.DataFrame([{"ensemble": r, "noise_fraction": f, "signal_scale_ug_m3": e.scale, "sigma_ug_m3": e.sigma[f],
                          "sigma_pct_of_signal_scale": 100.0 * e.sigma[f] / e.scale}
                         for r, e in ens.items() for f in st.fractions])


def run_geodesign(cfg: dict[str, Any], out_dir: Path | str, overwrite: bool = False, progress: Progress = None,
                  cache: ResponseCache | None = None) -> GeodesignResult:
    """Run the whole study into ``out_dir`` (refused when non-empty unless ``overwrite``).  Files: config.json,
    provenance.json, timings.json, candidates.{csv,geojson}, noise_levels.csv, cases.csv, alpha_selected.json,
    validation_raw.csv, layouts.{csv,geojson}, design_scores.csv, infeasible_budgets.json, test_raw.csv, summary.csv,
    paired.csv, figures/*.png."""
    t_all = time.perf_counter()
    out = make_run_dir(out_dir, overwrite)
    write_json(out / "config.json", cfg)
    files = ["config.json"]
    st = Setup(cfg, cache)
    chash = config_hash(cfg)
    ens_cfg = cfg["ensembles"]
    draws = {r: (draw_cases(cfg, int(e["n_cases"]), int(e["seed"]), r[0]), int(e["seed"]), int(e["noise_seed"]))
             for r, e in ens_cfg.items()}
    check_ensembles(draws)
    meta = {"label": LABEL, "provenance": PROVISIONAL, "manifest_sha256": st.geo.manifest_sha256, "raw_sha256": st.geo.raw_sha256,
            "config_hash": chash, "assumptions": list(ASSUMPTIONS), "candidate_config": cfg["candidates"],
            "hypothetical_in_experiments": st.hyp}
    write_csv(out / "candidates.csv", st.table.assign(manifest_sha256=st.geo.manifest_sha256, config_hash=chash))
    write_json(out / "candidates.geojson", candidates_geojson(st.table, meta))
    files += ["candidates.csv", "candidates.geojson"]

    stage = design_stage(cfg, progress=progress, setup=st)
    tim = dict(stage.timings)
    write_json(out / "alpha_selected.json", stage.alpha)  # frozen BEFORE the test ensemble exists
    alpha_sha = hashlib.sha256((out / "alpha_selected.json").read_bytes()).hexdigest()
    write_csv(out / "validation_raw.csv", stage.validation_raw)
    write_json(out / "infeasible_budgets.json", {"settings": st.settings, "budgets": st.budgets,
                                                 "infeasible": {k: {str(b): r for b, r in v.items()} for k, v in stage.infeasible.items()},
                                                 "note": "a budget is infeasible when the allowed sites (after the exclusion radius and category caps) cannot hold it; layouts are not produced for it"})
    write_csv(out / "layouts.csv", layouts_table(stage.layouts, st, chash))
    write_json(out / "layouts.geojson", layouts_geojson(stage.layouts, st, chash))
    t0 = time.perf_counter()
    scores = design_scores(st, stage.layouts, stage.design, stage.alpha_by_f)
    write_csv(out / "design_scores.csv", scores)
    tim["design_scores_s"] = time.perf_counter() - t0
    files += ["alpha_selected.json", "validation_raw.csv", "infeasible_budgets.json", "layouts.csv", "layouts.geojson",
              "design_scores.csv"]

    _say(progress, "alpha and layouts frozen; building the independent TEST ensemble")
    t0 = time.perf_counter()
    test = st.ensemble("test", progress, stage.validation.scale)
    D.assert_disjoint(stage.design.cases, test.cases, stage.validation.cases)
    tim["test_cases_s"] = time.perf_counter() - t0
    scales = D.attribution_scales(stage.design.cases)
    t0 = time.perf_counter()
    raw = evaluate_test(st, stage.layouts, test, stage.alpha_by_f, scales, progress)
    tim["test_evaluation_s"] = time.perf_counter() - t0
    write_csv(out / "test_raw.csv", raw)
    summ = summarise(raw)
    paired = paired_tables(raw)
    write_csv(out / "summary.csv", summ)
    write_csv(out / "paired.csv", paired)
    ens = {"design": stage.design, "validation": stage.validation, "test": test}
    write_csv(out / "noise_levels.csv", noise_levels(st, ens))
    write_csv(out / "cases.csv", pd.DataFrame([i for e in ens.values() for i in e.info]))
    files += ["test_raw.csv", "summary.csv", "paired.csv", "noise_levels.csv", "cases.csv"]

    figs: list[str] = []
    if cfg.get("figures", {}).get("enabled", True):
        t0 = time.perf_counter()
        feas_b = [b for b in st.budgets if b not in stage.infeasible[f"{SETTING_A}|{STANDARD}"]]
        fb = max(feas_b) if feas_b else st.budgets[0]
        f0 = st.fractions[0]
        _save_fig(out, fig_layout_maps(st, stage.layouts, fb, f0), "selected_sites.png", figs)
        for s in st.settings:
            _save_fig(out, fig_error_curves(summ, s, STANDARD), f"error_vs_count_{s}.png", figs)
        _save_fig(out, fig_attribution_vs_total(summ, SETTING_A, STANDARD), "attribution_vs_total.png", figs)
        _save_fig(out, fig_block_map(st, stage.layouts, f0), "heldout_block_map.png", figs)
        tim["figures_s"] = time.perf_counter() - t0
    tim["total_s"] = time.perf_counter() - t_all
    write_json(out / "timings.json", {"stages_s": tim, "total_s": tim["total_s"], "peak_memory_mb": _peak_mb(),
                                      "n_response_sets": st.cache.misses, "n_cases": {r: len(e.cases) for r, e in ens.items()}})
    seeds = {r: {"scenario_seed": e["seed"], "first_noise_seed": e["noise_seed"], "n_cases": e["n_cases"]} for r, e in ens_cfg.items()}
    seeds.update(layout_seed=cfg["methods"]["layout_seed"], heldout_random_seed=cfg["heldout"]["random_seed"])
    git_archive = archive_git_state(out)  # dirty tree: git_diff.patch + git_status.txt (sha256 below)
    write_json(out / "provenance.json", provenance(cfg, seeds, {
        "experiment": "geodesign", "label": LABEL, "status": PROVISIONAL, "manifest": str(st.geo.manifest_path),
        "manifest_sha256": st.geo.manifest_sha256, "raw_sha256": st.geo.raw_sha256, "alpha_selected_sha256": alpha_sha,
        "config_hash": chash, "alpha_by_noise_fraction": stage.alpha["alpha_by_noise_fraction"],
        "noise_sigma_ug_m3": {r: {f"{f:g}": e.sigma[f] for f in st.fractions} for r, e in ens.items()},
        "infeasible_budgets": {k: sorted(v) for k, v in stage.infeasible.items()},
        "git_archive": git_archive,
        "note": "emissions, inflow, winds and noise are SYNTHETIC; layouts are greedy heuristics, never called optimal"}, data=data_provenance("geography_only", [{"role": "geography_manifest", "path": str(st.geo.manifest_path), "sha256": st.geo.manifest_sha256}])))
    files += ["timings.json", "provenance.json"] + (["git_diff.patch", "git_status.txt"] if git_archive.get("dirty") else [])
    return GeodesignResult(out, stage.alpha, files, tim, figs)

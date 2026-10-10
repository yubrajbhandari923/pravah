"""Geography report and figures for a prepared region (offline, deterministic).

Usage: python scripts/geo_report.py --manifest data/belauri/manifest.json --out results/geography

Writes five PNG figures and geography_report.json into --out. Arrays come from the PROCESSED data (loaded with
``load_region``, which verifies every sha256); vector layers for the maps/border numbers (study polygon, ADM0 outlines,
OSM roads and facilities) and the native 30 m DEM statistics are read from the raw files named in the manifest
(sha256-checked against it). Nothing is downloaded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from pravah.geo import prepare, regionio as rio  # noqa: E402
from pravah.geo.fetch import sha256_file  # noqa: E402
from pravah.io import write_json  # noqa: E402

TITLE_SYNTH = "Real geography; synthetic emissions and meteorology."
PNG_META: dict[str, str | None] = {"Software": None}
BORDER_TOL_M = 50.0
SCHOOL = ("school", "college", "kindergarten", "university")


# ---------------------------------------------------------------- helpers
def quadrant(bearing: float) -> str:
    return ("north", "east", "south", "west")[int(((bearing + 45.0) % 360.0) // 90.0)]


def vector_info(dx_m: float, dy_m: float) -> dict[str, Any]:
    b = prepare.bearing_deg(dx_m, dy_m)
    return {"distance_km": math.hypot(dx_m, dy_m) / 1000.0, "bearing_deg": b, "sector": prepare.sector(b)}


def raw_file(man: dict[str, Any], base: Path, ds_id_prefix: str) -> Path:
    ds = [d for d in man["datasets"] if d["id"].startswith(ds_id_prefix)]
    if len(ds) != 1:
        raise ValueError(f"expected exactly one dataset {ds_id_prefix!r} in the manifest, found {len(ds)}")
    p = base / ds[0]["file"]
    if sha256_file(p) != ds[0]["sha256"]:
        raise ValueError(f"{p}: sha256 does not match the manifest")
    return p


def load_vectors(man: dict[str, Any], base: Path, epsg: int, x0: float, y0: float) -> dict[str, Any]:
    """Raw vector layers projected to UTM and shifted to local km (x0, y0 = grid SW corner)."""
    import shapely
    check = [c for c in man["boundary"]["checks"] if c["name"] == "shape_id"][0]
    shape_id = check["detail"].split("'")[1]  # detail is "shapeID='...'"
    geoms, props = prepare._read_geojson(raw_file(man, base, "geoBoundaries:NPL-ADM3"))
    k = [i for i, p in enumerate(props) if p.get("shapeID") == shape_id]
    if len(k) != 1:
        raise ValueError(f"study polygon {shape_id} not found uniquely")
    study = prepare._project(geoms[k[0]], 4326, epsg)
    win = shapely.box(study.centroid.x - 60_000, study.centroid.y - 60_000, study.centroid.x + 60_000, study.centroid.y + 60_000)
    out: dict[str, Any] = {"study": study, "window": win}
    for iso in ("NPL", "IND"):
        g, _ = prepare._read_geojson(raw_file(man, base, f"geoBoundaries:{iso}-ADM0"))
        full = shapely.union_all(shapely.make_valid(g))
        out[f"{iso}_vertices_full"] = int(shapely.get_num_coordinates(full))
        out[iso] = prepare._project(shapely.intersection(full, shapely.box(80.0, 28.3, 80.7, 29.1)), 4326, epsg)
    rg, _ = prepare._read_geojson(raw_file(man, base, "overpass:osm_roads"))
    out["roads"] = prepare._lines(prepare._project(rg, 4326, epsg))
    ag, ap = prepare._read_geojson(raw_file(man, base, "overpass:osm_amenity"))
    pts = prepare._project(shapely.point_on_surface(shapely.make_valid(ag)), 4326, epsg)
    xy = shapely.get_coordinates(pts)
    out["facilities"] = [(float(x), float(y), str(p.get("amenity"))) for (x, y), p in zip(xy, ap)]
    return out


def lines_of(g: Any) -> list[np.ndarray]:
    import shapely
    parts = shapely.get_parts(prepare._lines(g)) if not g.is_empty else []
    return [np.asarray(p.coords) for p in parts]


def to_local_km(coords: np.ndarray, x0: float, y0: float) -> np.ndarray:
    return (coords - np.array([x0, y0])) / 1000.0


# ---------------------------------------------------------------- numbers
def elevation_numbers(z: np.ndarray, frac: np.ndarray, grid: rio.GridSpec, cen_local: tuple[float, float]) -> dict[str, Any]:
    inside = (frac > 0) & np.isfinite(z)
    zin, win = z[inside], frac[inside]
    zd = z[np.isfinite(z)]
    xc, yc = np.meshgrid(grid.x_centres(), grid.y_centres())
    p90 = float(np.nanpercentile(z, 90))
    top = np.isfinite(z) & (z >= p90)
    tx, ty = float(xc[top].mean()), float(yc[top].mean())
    kmax = int(np.nanargmax(z))
    jm, im = divmod(kmax, grid.nx)
    edge = np.zeros(grid.shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    # plane fit z = a + b*E + c*N (E, N in km from the domain SW corner) over the whole domain
    ok = np.isfinite(z)
    A = np.column_stack([np.ones(ok.sum()), xc[ok] / 1000.0, yc[ok] / 1000.0])
    (a0, bE, cN), *_ = np.linalg.lstsq(A, z[ok], rcond=None)
    rows = np.array_split(np.arange(grid.ny), 3)
    thirds = [float(np.nanmean(z[r, :])) for r in rows]
    jin = np.where(inside.any(axis=1))[0]
    prow = np.array_split(jin, 3)
    pthirds = [float(np.sum((z * frac)[r, :][inside[r, :]]) / np.sum(frac[r, :][inside[r, :]])) for r in prow]
    ascent = prepare.bearing_deg(float(bE), float(cN))
    return {
        "inside_polygon": {"min_m": float(zin.min()), "mean_m": float(np.sum(zin * win) / np.sum(win)), "max_m": float(zin.max()),
                           "range_m": float(zin.max() - zin.min()), "note": "cell means of the 500 m aggregation; mean weighted by polygon area fraction"},
        "over_domain": {"min_m": float(zd.min()), "mean_m": float(zd.mean()), "max_m": float(zd.max()), "range_m": float(zd.max() - zd.min())},
        "highest_terrain": {
            "p90_elevation_m": p90, "cells_above_p90": int(top.sum()),
            "centroid_of_cells_above_p90_vs_polygon_centroid": vector_info(tx - cen_local[0], ty - cen_local[1]),
            "domain_max_cell": {"elevation_m": float(z[jm, im]), "on_domain_edge": bool(edge[jm, im]),
                                **vector_info(float(xc[jm, im]) - cen_local[0], float(yc[jm, im]) - cen_local[1])},
            "fraction_of_top_decile_cells_inside_polygon": float((top & (frac > 0)).sum() / top.sum()),
            "fraction_of_top_decile_cells_on_domain_edge": float((top & edge).sum() / top.sum()),
        },
        "gradient": {
            "domain_plane_fit_dz_per_km_east_m": float(bE), "domain_plane_fit_dz_per_km_north_m": float(cN),
            "steepest_ascent_bearing_deg": ascent, "steepest_ascent_sector": prepare.sector(ascent),
            "plane_fit_total_rise_across_domain_m": float(math.hypot(bE * grid.lx / 1000.0, cN * grid.ly / 1000.0)),
            "domain_mean_elevation_by_thirds_south_mid_north_m": thirds,
            "domain_north_minus_south_third_m": thirds[2] - thirds[0],
            "polygon_mean_elevation_by_thirds_south_mid_north_m": pthirds,
            "polygon_north_minus_south_third_m": pthirds[2] - pthirds[0],
        },
    }


def native_dem_numbers(raw_dem: Path, polygon_ll_mask: Any) -> dict[str, Any]:
    import rasterio
    from rasterio.mask import mask as rmask
    with rasterio.open(raw_dem) as ds:
        arr, _ = rmask(ds, [polygon_ll_mask], crop=True, filled=False)
        z = arr[0]
        return {"native_pixels_in_polygon": int(z.count()), "min_m": float(z.min()), "mean_m": float(z.mean()), "max_m": float(z.max()),
                "pixel_size_deg": [abs(ds.transform.a), abs(ds.transform.e)]}


def border_numbers(v: dict[str, Any], man: dict[str, Any]) -> dict[str, Any]:
    import shapely
    from shapely.ops import nearest_points
    study, npl, ind = v["study"], v["NPL"], v["IND"]
    bnd = npl.boundary
    cen = study.centroid
    perim = study.boundary
    n = max(int(perim.length // 10.0), 1)
    samples = shapely.line_interpolate_point(perim, np.linspace(0.0, perim.length, n, endpoint=False))
    d = shapely.distance(samples, bnd)
    on = d <= BORDER_TOL_M
    sides: dict[str, float] = {k: 0.0 for k in ("north", "east", "south", "west")}
    for p in np.asarray(samples)[on]:
        sides[quadrant(prepare.bearing_deg(p.x - cen.x, p.y - cen.y))] += 1.0 / n
    nearest = nearest_points(cen, bnd)[1]
    # coarseness of the Indian outline: distance of the (detailed) Nepal border points near the domain to the Indian outline
    c0 = study.centroid
    box = shapely.box(c0.x - 30_000, c0.y - 30_000, c0.x + 30_000, c0.y + 30_000)
    nb = shapely.intersection(bnd, box)
    pts = shapely.line_interpolate_point(nb, np.linspace(0.0, nb.length, 200)) if nb.length > 0 else np.array([])
    dev = shapely.distance(pts, ind.boundary) if len(pts) else np.array([np.nan])
    return {
        "border_source": "geoBoundaries NPL ADM0 boundary (2019 release data)",
        "centroid_to_border_km": float(cen.distance(bnd)) / 1000.0,
        "centroid_to_nearest_border_point": vector_info(nearest.x - cen.x, nearest.y - cen.y),
        "polygon_to_border_km": float(study.distance(bnd)) / 1000.0,
        "polygon_perimeter_km": float(perim.length) / 1000.0,
        "perimeter_share_within_tolerance_of_border": float(on.mean()),
        "tolerance_m": BORDER_TOL_M,
        "border_share_by_polygon_side_of_perimeter": sides,
        "sides_touching_border": [k for k, s in sides.items() if s > 0],
        "bounding_box_side_distances_km": man["international_border"]["distance_by_side_km"],
        "ind_outline_touches_polygon": bool(man["international_border"]["neighbours"]["IND"]["touches"]),
        "polygon_fraction_inside_ind_outline": float(study.intersection(ind).area / study.area),
        "polygon_fraction_inside_npl_adm0": float(study.intersection(npl).area / study.area),
        "ind_outline_coarseness": {
            "vertices_ind_adm0_total": v["IND_vertices_full"], "vertices_npl_adm0_total": v["NPL_vertices_full"],
            "npl_border_to_ind_outline_distance_m_near_domain": {"mean": float(np.nanmean(dev)), "max": float(np.nanmax(dev)), "n_points": int(len(pts))},
            "note": "if the two outlines described the same line these distances would be near 0; they measure the coarseness/offset of the IND outline"},
    }


def count_numbers(man: dict[str, Any]) -> dict[str, Any]:
    c = man["counts"]
    return {**c, "source": "manifest counts (OSM via Overpass mirror; completeness unknown, gaps are not zero sources)"}


def provenance(man: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for d in man["datasets"]:
        rel = d.get("release") or {}
        rows.append({"id": d["id"], "url": d["url"], "release": rel.get("buildDate") or "n/a (live query; see retrieved_utc)",
                     "data_year": rel.get("year"), "retrieved_utc": d["retrieved_utc"], "licence": d["license"],
                     "sha256_prefix": d["sha256"][:16]})
    return rows


# ---------------------------------------------------------------- figures
def lonlat_ticks(ax: Any, grid: rio.GridSpec, georef_x0: float, georef_y0: float) -> None:
    """Secondary lon/lat annotation: lon along the top edge (true values at the domain's northern edge), lat along the
    right edge (true values at the eastern edge). Parallels/meridians are tilted by < 0.5 deg, so the labels are good to
    well under one cell anywhere in the domain."""
    xs = np.linspace(0.0, grid.lx, 65)
    ys = np.linspace(0.0, grid.ly, 65)
    lon_top = np.array([rio.utm_to_lonlat(georef_x0 + x, georef_y0 + grid.ly, grid.epsg)[0] for x in xs])
    lat_right = np.array([rio.utm_to_lonlat(georef_x0 + grid.lx, georef_y0 + y, grid.epsg)[1] for y in ys])
    sx = ax.secondary_xaxis("top", functions=(lambda x: np.interp(x, xs / 1000.0, lon_top), lambda lo: np.interp(lo, lon_top, xs / 1000.0)))
    sy = ax.secondary_yaxis("right", functions=(lambda y: np.interp(y, ys / 1000.0, lat_right), lambda la: np.interp(la, lat_right, ys / 1000.0)))
    sx.set_xticks([lo for lo in np.arange(math.ceil(lon_top.min() * 10) / 10, lon_top.max(), 0.1)])
    sy.set_yticks([la for la in np.arange(math.ceil(lat_right.min() * 10) / 10, lat_right.max(), 0.1)])
    sx.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}\u00b0E")
    sy.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}\u00b0N")
    sx.tick_params(labelsize=8)
    sy.tick_params(labelsize=8)


def style_km(ax: Any, grid: rio.GridSpec) -> None:
    ax.set_xlim(0, grid.lx / 1000.0)
    ax.set_ylim(0, grid.ly / 1000.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x east of domain SW corner (km, UTM 44N)")
    ax.set_ylabel("y north of domain SW corner (km, UTM 44N)")


def outline(ax: Any, geom: Any, x0: float, y0: float, **kw: Any) -> None:
    import shapely
    for part in shapely.get_parts(geom):
        ax.plot(*to_local_km(np.asarray(part.exterior.coords), x0, y0).T, **kw)


def fig_map(v: dict[str, Any], man: dict[str, Any], grid: rio.GridSpec, out: Path) -> None:
    x0, y0 = grid.x0, grid.y0
    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    style_km(ax, grid)
    ax.add_collection(LineCollection([to_local_km(c, x0, y0) for c in sum((lines_of(g) for g in v["roads"]), [])],
                                     colors="#9a9a9a", linewidths=0.35, label="OSM roads"))
    outline(ax, v["study"], x0, y0, color="#1b6ca8", lw=2.0, label="Belauri municipality (geoBoundaries ADM3)")
    borders: list[tuple[Any, dict[str, Any]]] = [
        (v["NPL"].boundary, {"color": "#c0392b", "lw": 1.6, "label": "Nepal ADM0 border (detailed)"}),
        (v["IND"].boundary, {"color": "#e08a00", "lw": 1.6, "ls": "--", "label": "India ADM0 outline (coarse, 2014 data)"})]
    for g, kw in borders:
        for c in lines_of(g):
            ax.plot(*to_local_km(c, x0, y0).T, **kw)
            kw.pop("label", None)
    sch = np.array([(x, y) for x, y, a in v["facilities"] if a in SCHOOL])
    hea = np.array([(x, y) for x, y, a in v["facilities"] if a not in SCHOOL])
    if len(sch):
        ax.scatter(*to_local_km(sch, x0, y0).T, s=22, marker="s", c="#2a9d8f", edgecolors="k", linewidths=0.4, label="schools (OSM)", zorder=5)
    if len(hea):
        ax.scatter(*to_local_km(hea, x0, y0).T, s=40, marker="P", c="#d81b60", edgecolors="k", linewidths=0.4, label="health facilities (OSM)", zorder=5)
    p = man["external_patch"]
    cx, cy = p["centre_local_m"][0] / 1000.0, p["centre_local_m"][1] / 1000.0
    for r, ls in ((1.0, "-"), (2.0, ":")):
        ax.add_patch(Circle((cx, cy), r * p["sigma_m"] / 1000.0, fill=False, ec="#6a3d9a", lw=1.4, ls=ls))
    ax.plot([cx], [cy], marker="*", ms=14, c="#6a3d9a", mec="k", label=f"external patch centre (synthetic; circles 1 and 2 sigma, sigma = {p['sigma_m']:.0f} m)", zorder=6)
    ax.set_title(TITLE_SYNTH + "\nSimulation domain (frame) with the real layers; the patch is a synthetic hypothesis", fontsize=11, pad=34)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, ncol=2, frameon=False)
    lonlat_ticks(ax, grid, x0, y0)
    fig.savefig(out / "fig1_domain_map.png", dpi=150, bbox_inches="tight", metadata=PNG_META)
    plt.close(fig)


def fig_dem(z: np.ndarray, hs: np.ndarray, v: dict[str, Any], grid: rio.GridSpec, out: Path, tag: str) -> None:
    x0, y0 = grid.x0, grid.y0
    ext = (0, grid.lx / 1000.0, 0, grid.ly / 1000.0)
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    ax.imshow(hs, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
    im = ax.imshow(np.ma.masked_invalid(z), origin="lower", extent=ext, cmap="viridis", alpha=0.6)
    outline(ax, v["study"], x0, y0, color="k", lw=1.8)
    style_km(ax, grid)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.12, label="elevation, m a.s.l. (Copernicus GLO-30 DSM, cell mean)")
    ax.set_title(f"DSM elevation + hillshade, {tag} cells; black = municipality\n"
                 f"range over domain {np.nanmin(z):.0f} to {np.nanmax(z):.0f} m (context only, not used by the PDE)", fontsize=10, pad=34)
    lonlat_ticks(ax, grid, x0, y0)
    fig.savefig(out / "fig2_dem_hillshade.png", dpi=150, bbox_inches="tight", metadata=PNG_META)
    plt.close(fig)


def fig_profiles(reg: Any, out: Path) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    info: dict[str, Any] = {}
    for res, col in ((500.0, "#1b6ca8"), (250.0, "#e08a00")):
        lv = reg.level(res)
        g, frac = lv.grid, lv.arrays["study_fraction"]
        cj, ci = lv.info["centroid_cell"]["j"], lv.info["centroid_cell"]["i"]
        for ax, key, xs, line, axis in ((axes[0], "profile_ns_z", g.y_centres(), frac[:, ci], "N-S"),
                                        (axes[1], "profile_ew_z", g.x_centres(), frac[cj, :], "E-W")):
            prof = lv.arrays[key]
            ax.plot(xs / 1000.0, prof, color=col, label=f"{res:.0f} m cells")
            idx = np.where(line > 0)[0]
            lo, hi = xs[idx[0]] / 1000.0 - g.dx / 2000.0, xs[idx[-1]] / 1000.0 + g.dx / 2000.0
            if res == 500.0:
                ax.axvspan(lo, hi, color="#1b6ca8", alpha=0.12, label="municipality extent on this line")
            info[f"{axis}_{res:.0f}m"] = {"extent_km": [float(lo), float(hi)], "min_m": float(np.nanmin(prof)), "max_m": float(np.nanmax(prof)),
                                          "range_m": float(np.nanmax(prof) - np.nanmin(prof))}
    axes[0].set_xlabel("y north of domain SW corner (km)")
    axes[1].set_xlabel("x east of domain SW corner (km)")
    axes[0].set_title("N-S profile through the polygon centroid column", fontsize=10)
    axes[1].set_title("E-W profile through the polygon centroid row", fontsize=10)
    for ax in axes:
        ax.set_ylabel("elevation, m a.s.l. (DSM)")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
    fig.suptitle("Elevation profiles (vertical range is only a few tens of metres)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig3_profiles.png", dpi=150, metadata=PNG_META)
    plt.close(fig)
    return info


def fig_slope(slope: np.ndarray, valid: np.ndarray, v: dict[str, Any], grid: rio.GridSpec, out: Path) -> None:
    x0, y0 = grid.x0, grid.y0
    ext = (0, grid.lx / 1000.0, 0, grid.ly / 1000.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, a, cm, lab, vmin, vmax in ((axes[0], slope, "magma", "slope of the 500 m mean DSM (degrees)", None, None),
                                       (axes[1], valid, "Greens", "DEM valid fraction (1 = every 30 m pixel valid)", 0.0, 1.0)):
        im = ax.imshow(np.ma.masked_invalid(a), origin="lower", extent=ext, cmap=cm, vmin=vmin, vmax=vmax)
        outline(ax, v["study"], x0, y0, color="k", lw=1.5)
        style_km(ax, grid)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.12, label=lab)
    axes[0].set_title(f"Slope (max {np.nanmax(slope):.2f} deg; coarse cells understate native slopes)", fontsize=9)
    axes[1].set_title(f"DEM valid fraction (min over domain {valid.min():.3f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig4_slope_valid.png", dpi=150, metadata=PNG_META)
    plt.close(fig)


def fig_sources(reg: Any, v: dict[str, Any], out: Path) -> dict[str, float]:
    lv = reg.level(500.0)
    g = lv.grid
    x0, y0 = g.x0, g.y0
    ext = (0, g.lx / 1000.0, 0, g.ly / 1000.0)
    sums: dict[str, float] = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, name, label in zip(axes.ravel(), ("road", "builtup", "facility", "external_patch"),
                               ("road length", "built-up area", "facility count", "external patch (Gaussian)")):
        f = lv.arrays[f"w_{name}"]
        s = float(f.sum() * g.cell_area)
        sums[name] = s
        im = ax.imshow(np.ma.masked_less_equal(f * 1e6, 0), origin="lower", extent=ext, cmap="YlOrRd")
        outline(ax, v["study"], x0, y0, color="k", lw=1.2)
        style_km(ax, g)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.12, label="f (1/km$^2$)")
        ax.set_title(f"{label}: $\\sum f\\,A$ = {s:.6f}", fontsize=10)
    fig.suptitle("Synthetic source hypotheses (spatial priors on real geometry; not an emissions inventory)\n"
                 "normalised shapes f, 500 m cells; " + TITLE_SYNTH, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig5_source_hypotheses.png", dpi=150, metadata=PNG_META)
    plt.close(fig)
    return sums


# ---------------------------------------------------------------- main
def build(manifest_path: Path, out: Path) -> dict[str, Any]:
    import shapely
    base = manifest_path.parent
    man = json.loads(manifest_path.read_text())
    reg = prepare.load_region(manifest_path)  # verifies every processed sha256 / array hash
    out.mkdir(parents=True, exist_ok=True)
    epsg, x0, y0 = reg.georef.epsg, reg.georef.x0, reg.georef.y0
    v = load_vectors(man, base, epsg, x0, y0)
    study = v["study"]
    cen_local = (study.centroid.x - x0, study.centroid.y - y0)
    lv5 = reg.level(500.0)
    g5, z5, f5 = lv5.grid, lv5.arrays["dem_mean"], lv5.arrays["study_fraction"]

    fig_map(v, man, g5, out)
    fig_dem(z5, lv5.arrays["hillshade"], v, g5, out, "500 m")
    prof = fig_profiles(reg, out)
    fig_slope(lv5.arrays["slope_deg"], lv5.arrays["dem_valid_fraction"], v, g5, out)
    sums = fig_sources(reg, v, out)

    dem_ds = [d for d in man["datasets"] if d["id"].startswith("Copernicus")][0]
    poly_ll = prepare._project(study, epsg, 4326)
    levels: dict[str, Any] = {}
    for res in sorted(reg.levels):
        lv = reg.level(res)
        g, frac, valid, z = lv.grid, lv.arrays["study_fraction"], lv.arrays["dem_valid_fraction"], lv.arrays["dem_mean"]
        inside = frac > 0
        levels[f"{res:.0f}m"] = {
            "grid": g.to_dict(), "cells": int(g.nx * g.ny), "polygon_cells": int(inside.sum()),
            "elevation": elevation_numbers(z, frac, g, cen_local),
            "mean_slope_deg_in_polygon": float(np.nansum(lv.arrays["slope_deg"] * frac) / frac[np.isfinite(lv.arrays["slope_deg"])].sum()),
            "max_slope_deg_domain": float(np.nanmax(lv.arrays["slope_deg"])),
            "dem_valid_fraction": {"min_domain": float(valid.min()), "mean_domain": float(valid.mean()), "min_in_polygon_cells": float(valid[inside].min()),
                                   "cells_below_0.95": int((valid < 0.95).sum()), "cells_nan_dem": int((~np.isfinite(z)).sum())},
            "polygon_area_from_mask_km2": float(frac.sum() * g.cell_area / 1e6),
            "polygon_area_km2": float(study.area / 1e6),
            "polygon_area_relative_error": float(abs(frac.sum() * g.cell_area - study.area) / study.area),
            "source_sums_f_times_A": {k: float(lv.arrays[f"w_{k}"].sum() * g.cell_area) for k in ("road", "builtup", "facility", "external_patch")},
        }
    clon, clat = rio.utm_to_lonlat(study.centroid.x, study.centroid.y, epsg)
    rep: dict[str, Any] = {
        "region": man["name"], "manifest": str(manifest_path), "manifest_created_utc": man["created_utc"],
        "boundary_status": man["boundary_status"],
        "elevation_semantics": "m a.s.l. from the Copernicus GLO-30 DSM (surface model: includes canopy and buildings), EGM2008 heights; "
                               "cell values are means of valid 30 m pixel centres in each cell",
        "relief_statement": None,
        "levels": levels,
        "native_dem_inside_polygon_30m": native_dem_numbers(raw_file(man, base, "Copernicus"), poly_ll),
        "profiles_through_polygon_centroid": prof,
        "border": border_numbers(v, man),
        "counts": count_numbers(man),
        "dem_native_resolution": {"pixel_m_approx_ew_ns": dem_ds["native_resolution"]["pixel_m_approx"] if isinstance(dem_ds["native_resolution"], dict) else None,
                                  "nominal_m": 30, "grid_cell_m": [500, 250],
                                  "dem_raw_nodata_pixels": dem_ds["counts"]["nodata_pixels"], "dem_raw_shape": dem_ds["counts"]["shape"]},
        "external_patch": {**{k: man["external_patch"][k] for k in ("centre_lonlat", "centre_local_m", "sigma_m", "distance_to_polygon_km")},
                           "origin": man["external_patch"]["origin"], "label": "synthetic hypothesis"},
        "crs": {"epsg": epsg, "axis_conventions": man["crs"]["axis_order"],
                "roundtrip_error_m_manifest": man["crs"]["roundtrip_error_m"],
                "roundtrip_error_m_recomputed": rio.roundtrip_error_m(clon, clat, epsg),
                "polygon_centroid_lonlat_from_utm_centroid": [clon, clat],
                "local_coordinates": "x, y = UTM easting/northing minus the domain SW corner (x0, y0); arrays are [y, x] with y increasing northward"},
        "source_hypothesis_sums_figure5": sums,
        "provenance": provenance(man),
        "manifest_warnings": man["warnings"],
        "boundary_checks": man["boundary"]["checks"],
        "assumptions": [
            "ASSUMPTION: emissions, source strengths and the external patch (centre, sigma) are synthetic spatial priors, not measurements or an inventory",
            "ASSUMPTION: meteorology is synthetic; the DEM is context only and is not used by the PDE",
            "ASSUMPTION: the municipality polygon (geoBoundaries ADM3, Open Data Nepal / MoFAGA 2019) is treated as the study region; no survey-department polygon was obtained",
            "ASSUMPTION: the Nepal ADM0 line and the municipality polygon share a lineage, so their coincidence says little about the true border position",
            "ASSUMPTION: OSM completeness is unknown; empty cells are mapping gaps, not zero sources",
            "ASSUMPTION: a 500 m (or 250 m) cell mean of a 30 m DSM removes local relief; slopes from coarse cells understate native slopes",
        ],
    }
    e5 = levels["500m"]["elevation"]
    top = e5["highest_terrain"]
    hv = top["centroid_of_cells_above_p90_vs_polygon_centroid"]
    rep["relief_statement"] = (
        f"Within the domain the DSM spans {e5['over_domain']['min_m']:.1f} to {e5['over_domain']['max_m']:.1f} m "
        f"(range {e5['over_domain']['range_m']:.1f} m at 500 m cells); inside the polygon {e5['inside_polygon']['min_m']:.1f} to "
        f"{e5['inside_polygon']['max_m']:.1f} m. The domain maximum ({top['domain_max_cell']['elevation_m']:.1f} m) lies "
        f"{top['domain_max_cell']['distance_km']:.1f} km {top['domain_max_cell']['sector']} of the centroid"
        f"{', on the domain edge, so terrain may keep rising outside the domain' if top['domain_max_cell']['on_domain_edge'] else ''}. Cells above the 90th percentile centre {hv['distance_km']:.1f} km towards "
        f"{hv['sector']} ({hv['bearing_deg']:.0f} deg) of the polygon centroid. Whether higher terrain (e.g. Siwalik hills) lies beyond the "
        f"domain cannot be read from this DEM window, which extends only a small margin beyond the domain; the data show a plane-fit "
        f"rise of {e5['gradient']['plane_fit_total_rise_across_domain_m']:.1f} m across the domain, toward {e5['gradient']['steepest_ascent_sector']}.")
    write_json(out / "geography_report.json", rep)
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/belauri/manifest.json")
    ap.add_argument("--out", default="results/geography")
    a = ap.parse_args(argv)
    rep = build(Path(a.manifest), Path(a.out))
    print(rep["relief_statement"])
    print(f"wrote {a.out}/geography_report.json and 5 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

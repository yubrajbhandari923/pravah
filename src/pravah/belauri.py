"""Stage E: forward simulations on the REAL Belauri region.  Real geography; synthetic emissions and meteorology.

``Study`` builds the scenarios (``geo.scenarios``) and solves them; ``run_belauri_forward`` runs the configured studies
(winds, outer domain, mesh, what-changes table, figures) into a run directory with ``pravah.io``.  Notebook 05 uses the
same ``Study`` object with a reduced set.

Honesty rules.  Emission rates, the external onset and all winds are SYNTHETIC; K, lam, H are nominal.  The municipality
polygon (as cell-area fractions), the road / built-up / facility geometry and the external patch position are real
(clipped to the polygon for local sources).  DEM, hillshade and slope are plotting-only layers: no solver input reads them
(``plotting_only_check`` demonstrates a change of exactly zero).  The numerical diffusion of the baseline scheme is
reported SEPARATELY from the physical K (``geo.scenarios.numerical_diffusion_estimate``): at 500 m cells and u = 1.5 m/s it
is several times K, so K = 100 m^2/s is a nominal input, not the effective spreading.

Tags.  local = road + built-up (+ facility) emissions, external = the external patch; both come from the linear response
set (``attribution.contributions`` on a ``ResponseCache``), and an independent direct solve of the combined problem is
compared with their sum.  "Study" metrics weight cells by the polygon area fraction; peaks are over cells with area
fraction >= 0.5; time windows start at ``t_exclude``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from .attribution import contributions, external_fraction
from .config import FieldResult, InflowBasis, SolverOptions, to_dict
from .forward import FIPY
from .geo.prepare import GridLevel, Region, load_region
from .geo.scenarios import (LABEL, BelauriCase, WindFactory, belauri_scenario, build_wind, crop_region,
                            numerical_diffusion_estimate, study_weights_of)
from .inverse import ResponseCache, Responses, run_forward
from .io import make_run_dir, provenance, save_field, write_csv, write_json
from .metrics import quadrature_weights
from .sources import cell_centres
from .transboundary import arrival_time, fraction_timeseries

Progress = Callable[[str], None] | None
PLOTTING_ONLY_LAYERS = ("dem_mean", "dem_valid_fraction", "slope_deg", "hillshade", "profile_ns_z", "profile_ew_z")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _say(progress: Progress, msg: str) -> None:
    if progress is not None:
        progress(msg)


def resolve_manifest(path: str | Path) -> Path:
    """The manifest path as given if it exists (cwd-relative), else relative to the repository root."""
    p = Path(path)
    if p.exists():
        return p
    q = REPO_ROOT / p
    if q.exists():
        return q
    raise FileNotFoundError(f"manifest {path} not found (tried the working directory and {REPO_ROOT})")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------- data carriers
@dataclass(frozen=True, eq=False)
class Fields:
    """Concentration fields (nt, ny, nx) microgram/m^3 of the three views of one problem."""
    times: np.ndarray
    local: np.ndarray
    external: np.ndarray
    total: np.ndarray

    def da(self, which: str, case: BelauriCase) -> xr.DataArray:
        x, y = cell_centres(case.scenario)
        return xr.DataArray(getattr(self, which), dims=("time", "y", "x"), coords={"time": self.times, "x": x, "y": y},
                            name=which, attrs={"units": "microgram/m^3"})


@dataclass
class WindResult:
    name: str
    case: BelauriCase
    fields: Fields
    resp: Responses
    summary: dict[str, Any]
    series: pd.DataFrame
    arrival: list[dict[str, Any]]
    consistency: dict[str, Any]


def block_average(a: np.ndarray, factor: int) -> np.ndarray:
    """Mean over factor x factor blocks of the last two axes (a fine field -> the coarse grid)."""
    if factor == 1:
        return a
    *lead, ny, nx = a.shape
    return np.asarray(a.reshape(*lead, ny // factor, factor, nx // factor, factor).mean(axis=(-3, -1)))


def weighted_rmse(a: np.ndarray, b: np.ndarray, w: np.ndarray, times: np.ndarray, t_exclude: float) -> float:
    """Study-weighted, trapezoid-in-time RMSE (same quadrature as ``metrics.field_mse``) of two (nt, ny, nx) arrays."""
    sel, dt, ww, den = quadrature_weights(times, a.shape[1], a.shape[2], w, t_exclude)
    return float(np.sqrt(np.einsum("t,tyx,yx->", dt, ((a - b) ** 2)[sel], ww) / den))


def _contrib(f: Fields, case: BelauriCase) -> dict[str, xr.DataArray]:
    z = xr.zeros_like(f.da("total", case))
    return {"local": f.da("local", case), "external_sources": f.da("external", case), "boundary": z,
            "background": z, "initial": z}


def regional_metrics(f: Fields, case: BelauriCase, t_exclude: float) -> tuple[dict[str, Any], pd.DataFrame]:
    """Study-weighted summary of ``f``: per tag the time-mean regional mean, the peak over study cells (area fraction
    >= 0.5) and times >= t_exclude and its time, plus the concentration_external_fraction (None when undefined), and the
    series (mean, peak, fraction) at every output time."""
    w = study_weights_of(case)
    times = f.times
    sel, dt, ww, den = quadrature_weights(times, w.shape[0], w.shape[1], w, t_exclude)
    mask = w >= 0.5
    summ: dict[str, Any] = {}
    ser: dict[str, Any] = {"time_s": times}
    for tag, a in (("local", f.local), ("external", f.external), ("total", f.total)):
        mean_t = np.einsum("tyx,yx->t", a, w) / w.sum()
        peak_t = a[:, mask].max(axis=1)
        ts = times[sel]
        k = int(np.argmax(peak_t[sel]))
        summ[tag] = {"time_mean_regional_mean": float(np.einsum("t,tyx,yx->", dt, a[sel], ww) / den),
                     "peak": float(peak_t[sel][k]), "peak_time_s": float(ts[k])}
        ser[f"mean_{tag}"], ser[f"peak_{tag}"] = mean_t, peak_t
    c = _contrib(f, case)
    summ["external_fraction"] = external_fraction(c, case.scenario, t_exclude)
    _, frac = fraction_timeseries(c, case.scenario)
    ser["external_fraction"] = np.array([np.nan if v is None else v for v in frac])
    return summ, pd.DataFrame(ser)


def sample_cell(a: np.ndarray, case: BelauriCase, x: float, y: float) -> np.ndarray:
    """Trace (nt,) of the cell containing the point (x, y) given in the STANDARD domain's local metres."""
    off = case.georef["offset_in_standard_domain_m"]
    d = case.scenario.domain
    i, j = int(math.floor((x - off[0]) / d.dx)), int(math.floor((y - off[1]) / d.dy))
    if not (0 <= i < d.nx and 0 <= j < d.ny):
        raise ValueError(f"point ({x}, {y}) lies outside the domain")
    return np.asarray(a[:, j, i])


def arrival_rows(ext: np.ndarray, times: np.ndarray, case: BelauriCase, points: list[dict[str, Any]], level: float,
                 onset_s: float) -> list[dict[str, Any]]:
    """First time (linear interpolation of output times) at which the EXTERNAL-tag concentration at each point exceeds the
    absolute ``level`` (microgram/m^3); None if it never does.  Also the delay after the external onset."""
    rows = []
    for p in points:
        tr = sample_cell(ext, case, p["x"], p["y"])
        t = arrival_time(tr, times, level / max(float(tr.max()), 1e-300), level) if tr.max() > level else None
        rows.append({"point": p["label"], "x_m": p["x"], "y_m": p["y"], "arrival_s": t,
                     "arrival_after_onset_s": None if t is None else t - onset_s, "max_external": float(tr.max())})
    return rows


# ---------------------------------------------------------------------------------------------- the study object
class Study:
    """Everything shared by the runner and the notebook for one config: region, winds, points, solver, caches."""

    def __init__(self, cfg: dict[str, Any], progress: Progress = None, cache: ResponseCache | None = None) -> None:
        self.cfg, self.progress = cfg, progress
        self.cache = cache or ResponseCache()
        self.solver = SolverOptions(**cfg.get("solver", {}))
        self.manifest_path = resolve_manifest(cfg["region"]["manifest"])
        self.manifest_sha256 = file_sha256(self.manifest_path)
        self.full = load_region(self.manifest_path)
        sb = cfg["region"].get("standard_buffer_m")
        self.region = self.full if sb is None else crop_region(self.full, float(sb))
        ph, ms, so = cfg["physics"], cfg["mesh"], cfg["sources"]
        self.K, self.lam, self.H = float(ph["K"]), float(ph["lam"]), float(ph["H"])
        self.T, self.out_dt, self.t_exclude = float(ph["T"]), float(ph["output_interval"]), float(ph["t_exclude"])
        self.res, self.dt = float(ms["resolution_m"]), float(ms["dt_s"])
        self.groups = tuple(so["groups"])
        self.amplitudes = {k: float(v) for k, v in so["amplitudes_ug_s"].items()}
        self.ext_profile = tuple((float(t), float(v)) for t, v in so["external_profile"])
        self.onset_s = next((t for t, v in self.ext_profile if v > 0), 0.0)
        self.speed = float(cfg["wind"]["speed"])
        self.arrival_level = float(cfg["arrival"]["level_ug_m3"])
        lv = self.region.level(self.res)
        w = lv.arrays["study_fraction"]
        jj, ii = np.mgrid[0:w.shape[0], 0:w.shape[1]]
        self.centroid = (float(((ii + 0.5) * lv.grid.dx * w).sum() / w.sum()), float(((jj + 0.5) * lv.grid.dy * w).sum() / w.sum()))
        self.patch = (float(self.region.manifest["external_patch"]["centre_local_m"][0]),
                      float(self.region.manifest["external_patch"]["centre_local_m"][1]))
        self.winds: dict[str, WindFactory] = {n: build_wind(n, s, self.speed, self.T, self.centroid)
                                              for n, s in cfg["wind"]["scenarios"].items()}
        self.points = select_points(self.region.level(self.res), self.patch)
        self.records: list[dict[str, Any]] = []
        self._results: dict[str, WindResult] = {}

    # ---- scenario construction
    def case(self, wind: str, *, resolution_m: float | None = None, dt_s: float | None = None,
             crop_buffer_m: float | None = None, K: float | None = None, lam: float | None = None,
             amplitudes: dict[str, float] | None = None, region: Region | None = None,
             bases: Sequence[InflowBasis] = (), duration_s: float | None = None) -> BelauriCase:
        return belauri_scenario(
            region or self.region, resolution_m or self.res, self.winds[wind], groups=self.groups,
            crop_buffer_m=crop_buffer_m, duration_s=duration_s or self.T, dt_s=dt_s or self.dt,
            output_interval_s=self.out_dt, K=K or self.K, lam=self.lam if lam is None else lam, H=self.H,
            amplitudes=amplitudes or self.amplitudes, bases=bases, external_profile=self.ext_profile)

    # ---- solving
    def record(self, label: str, kind: str, n_solves: int, wall_s: float, res: FieldResult | None, g: np.ndarray | None,
               case: BelauriCase) -> dict[str, Any]:
        d = res.diagnostics if res is not None else {}
        sc = case.scenario
        rec = {"label": label, "kind": kind, "resolution_m": case.georef["resolution_m"], "dt_s": sc.times.dt,
               "n_cells": sc.domain.nx * sc.domain.ny, "n_solves": n_solves, "wall_s": wall_s,
               "wall_s_per_solve": wall_s / n_solves,
               "n_negative": int(d["n_negative"]) if res is not None else int((g < 0).sum()) if g is not None else None,
               "min_concentration": float(d["min_concentration"]) if res is not None else
               float(g.min()) if g is not None else None,
               "max_rel_budget_residual": d.get("max_rel_budget_residual"),
               "n_steps": d.get("n_steps"), "max_courant": self.max_courant(case)}
        self.records.append(rec)
        return rec

    def max_courant(self, case: BelauriCase) -> float:
        """Speed Courant number max |u| dt / dx over the winds (for a stream function: hypot of the largest face speeds,
        an upper bound); the guideline is <= 0.3."""
        from .forward import psi_face_fluxes
        sc = case.scenario
        d = sc.domain
        best = 0.0
        for wseg in sc.transport.wind:
            if wseg.psi is None:
                sp = math.hypot(wseg.u, wseg.v)
            else:
                uf, vf = psi_face_fluxes(sc, wseg.psi)
                sp = math.hypot(float(np.abs(uf).max()), float(np.abs(vf).max()))
            best = max(best, sp * sc.times.dt / min(d.dx, d.dy))
        return best

    def direct(self, case: BelauriCase, label: str, theta: np.ndarray | None = None) -> tuple[np.ndarray, FieldResult]:
        t0 = time.perf_counter()
        res = run_forward(FIPY, case.scenario, case.theta if theta is None else theta, self.solver)
        self.record(label, "direct", 1, time.perf_counter() - t0, res, None, case)
        return np.asarray(res.concentration.transpose("time", "y", "x").values, dtype=float), res

    def pair(self, case: BelauriCase, label: str) -> Fields:
        """Direct solves of the combined problem and of the external-only problem (theta of the local sources zeroed);
        local = combined - external (exact by linearity; verified in ``wind_result``)."""
        total, res = self.direct(case, f"{label}:combined")
        th_ext = np.where(np.array([lab["group"] == "external" for lab in case.labels]), case.theta, 0.0)
        ext, _ = self.direct(case, f"{label}:external_only", th_ext)
        return Fields(np.asarray(res.concentration["time"].values, dtype=float), total - ext, ext, total)

    def wind_result(self, wind: str) -> WindResult:
        if wind in self._results:
            return self._results[wind]
        case = self.case(wind)
        t0 = time.perf_counter()
        n0 = self.cache.misses
        resp = self.cache.get(case.scenario, self.solver)
        wall = time.perf_counter() - t0
        if self.cache.misses > n0:
            self.record(f"{wind}:response_set", "response_set", case.scenario.n_params + 1, wall, None, resp.g, case)
        zeros = {"initial": np.zeros_like(resp.base), "background": np.zeros_like(resp.base)}
        c = contributions(resp, case.theta, zeros)
        local, ext = np.asarray(c["local"].values), np.asarray(c["external_sources"].values)
        total = np.asarray(resp.field(case.theta).values)
        f = Fields(resp.times, local, ext, total)
        direct, res = self.direct(case, f"{wind}:combined_direct")
        scale = float(np.abs(direct).max())
        cons = {"max_abs_err_direct_vs_tag_sum": float(np.abs(direct - (local + ext)).max()),
                "max_rel_err_direct_vs_tag_sum": float(np.abs(direct - (local + ext)).max() / scale) if scale > 0 else 0.0,
                "max_abs_err_sum_vs_field": float(np.abs((local + ext) - total).max()),
                "max_rel_budget_residual": res.diagnostics["max_rel_budget_residual"],
                "n_negative_direct": res.diagnostics["n_negative"], "n_negative_unit_responses": int((resp.g < 0).sum()),
                "max_courant": self.max_courant(case)}
        summ, series = regional_metrics(f, case, self.t_exclude)
        arr = arrival_rows(ext, f.times, case, self.points, self.arrival_level, self.onset_s)
        out = WindResult(wind, case, f, resp, summ, series, arr, cons)
        self._results[wind] = out
        _say(self.progress, f"wind {wind}: total peak {summ['total']['peak']:.3f}, external fraction "
                            f"{summ['external_fraction']}, direct-vs-sum {cons['max_rel_err_direct_vs_tag_sum']:.1e}")
        return out


def select_points(level: GridLevel, patch: tuple[float, float]) -> list[dict[str, Any]]:
    """Five report points in local metres (cell centres): the polygon cell nearest to / farthest from the external patch,
    the facility cell (facility layer) nearest to / farthest from it, and the area-weighted polygon centroid cell.  Without
    facility cells the two facility points are replaced by polygon cells at the 25 % / 75 % distance quantiles."""
    g = level.grid
    w = level.arrays["study_fraction"]
    xs, ys = g.x_centres()[None, :] + 0 * w, g.y_centres()[:, None] + 0 * w
    dist = np.hypot(xs - patch[0], ys - patch[1])
    inside = w >= 0.5

    def pick(mask: np.ndarray, far: bool) -> tuple[float, float]:
        d = np.where(mask, dist, -np.inf if far else np.inf)
        j, i = np.unravel_index(int(np.argmax(d) if far else np.argmin(d)), d.shape)
        return float(xs[j, i]), float(ys[j, i])

    fac = inside & (level.arrays["facility_count"] > 0) if "facility_count" in level.arrays else np.zeros_like(inside)
    cj, ci = np.unravel_index(int(np.argmin(np.hypot(xs - (xs * w).sum() / w.sum(), ys - (ys * w).sum() / w.sum()) + np.where(inside, 0, np.inf))), w.shape)
    pts = [("polygon_nearest_to_patch", *pick(inside, False)), ("polygon_farthest_from_patch", *pick(inside, True))]
    if fac.any():
        pts += [("facility_nearest_to_patch", *pick(fac, False)), ("facility_farthest_from_patch", *pick(fac, True))]
    else:
        q = np.quantile(dist[inside], [0.25, 0.75])
        for lab, qq in (("polygon_q25_distance", q[0]), ("polygon_q75_distance", q[1])):
            j, i = np.unravel_index(int(np.argmin(np.where(inside, np.abs(dist - qq), np.inf))), w.shape)
            pts.append((lab, float(xs[j, i]), float(ys[j, i])))
    pts.append(("polygon_centroid", float(xs[cj, ci]), float(ys[cj, ci])))
    return [{"label": lab, "x": x, "y": y, "distance_to_patch_m": float(math.hypot(x - patch[0], y - patch[1]))}
            for lab, x, y in pts]


# ---------------------------------------------------------------------------------------------- plotting-only layers
def plotting_only_check(region: Region, build: Callable[[Region], BelauriCase]) -> dict[str, Any]:
    """Rebuild the case after REPLACING every plotting-only layer (DEM, valid fraction, slope, hillshade, profiles) by
    garbage: the Scenario (sources, mask, wind, parameters) must be identical, hence every number is identical."""
    def scrambled(lv: GridLevel) -> GridLevel:
        arrays = {k: (np.full_like(a, 1234.5) if k in PLOTTING_ONLY_LAYERS else a) for k, a in lv.arrays.items()}
        return dataclasses.replace(lv, arrays=arrays)
    other = Region(region.name, region.georef, {r: scrambled(lv) for r, lv in region.levels.items()}, region.manifest)
    a, b = build(region), build(other)
    same = to_dict(a.scenario) == to_dict(b.scenario) and bool(np.array_equal(a.theta, b.theta))
    return {"layers": list(PLOTTING_ONLY_LAYERS), "scenario_identical": same, "case_a": a, "case_b": b}


# ---------------------------------------------------------------------------------------------- figures
def _finish(fig: Figure) -> Figure:
    """Reserve a strip at the bottom and print the mandatory label on the figure."""
    eng = fig.get_layout_engine()
    if eng is not None:
        cast(Any, eng).set(rect=(0.0, 0.03, 1.0, 0.97))
    fig.text(0.01, 0.006, LABEL, fontsize=8, style="italic", color="0.25")
    return fig


def _km(level: GridLevel) -> tuple[float, float, float, float]:
    g = level.grid
    return 0.0, g.lx / 1000.0, 0.0, g.ly / 1000.0


def draw_map(ax: Any, region: Region, res: float, *, hillshade: bool = True, outline: bool = True) -> None:
    """Hillshade (context only) with the municipality outline (0.5 contour of the 250 m area fraction, or the finest
    level available)."""
    lv = region.level(res)
    ext = _km(lv)
    if hillshade:
        ax.imshow(lv.arrays["hillshade"], origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
    if outline:
        fine = region.level(min(region.levels))
        g = fine.grid
        ax.contour(g.x_centres() / 1000.0, g.y_centres() / 1000.0, fine.arrays["study_fraction"], levels=[0.5],
                   colors="tab:red", linewidths=1.6)
    ax.set(xlabel="x (km, local)", ylabel="y (km, local)", aspect="equal")


def fig_overview(study: Study, wind: str | None = None) -> Figure:
    """Municipality over hillshade, the prepared (standard) domain, the small outer domain, source layers, the external
    patch (2 sigma), the five report points; and the N-S elevation profile (context only)."""
    reg = study.region
    lv = reg.level(study.res)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2.2, 1]}, constrained_layout=True)
    draw_map(ax, reg, study.res)
    ext = _km(lv)
    g = lv.grid
    for key, col in (("w_road", "tab:orange"), ("w_builtup", "tab:cyan")):
        a = lv.arrays[key]
        ax.contourf(g.x_centres() / 1000.0, g.y_centres() / 1000.0, np.where(a > 0, 1.0, np.nan), levels=[0.5, 1.5], colors=[col], alpha=0.35)
    if "w_facility" in lv.arrays:
        fj, fi = np.nonzero(lv.arrays["w_facility"] > 0)
        ax.plot(g.x_centres()[fi] / 1000.0, g.y_centres()[fj] / 1000.0, "m^", ms=4, label="facility cells")
    px, py, sg = study.patch[0] / 1000.0, study.patch[1] / 1000.0, float(reg.manifest["external_patch"]["sigma_m"]) / 1000.0
    for k in (1, 2):
        ax.add_patch(Circle((px, py), k * sg, fill=False, ec="tab:brown", ls="--" if k == 2 else "-"))
    ax.annotate("external patch\n(1, 2 sigma)", (px, py), textcoords="offset points", xytext=(-70, -35), color="tab:brown")
    small = study.cfg["region"].get("small_buffer_m")
    if small is not None:
        sm = crop_region(reg, float(small))
        o = (sm.georef.x0 - reg.georef.x0, sm.georef.y0 - reg.georef.y0)
        sg2 = sm.level(study.res).grid
        ax.add_patch(Rectangle((o[0] / 1000.0, o[1] / 1000.0), sg2.lx / 1000.0, sg2.ly / 1000.0, fill=False, ec="tab:blue", lw=1.5,
                                   label=f"small outer domain ({float(small) / 1000:g} km buffer)"))
    for p in study.points:
        ax.plot(p["x"] / 1000.0, p["y"] / 1000.0, "o", mfc="yellow", mec="k", ms=6)
        ax.annotate(p["label"].replace("_", " "), (p["x"] / 1000.0, p["y"] / 1000.0), textcoords="offset points", xytext=(4, 3), fontsize=6)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_title("Belauri municipality (red outline), road/built-up cells (orange/cyan), hillshade = context only")
    ax.legend(loc="lower right", fontsize=7)
    z = lv.arrays["profile_ns_z"]
    ax2.plot(z, g.y_centres() / 1000.0)
    ax2.set(xlabel="Copernicus DEM mean elevation (m), context only", ylabel="y (km, local)",
            title="N-S profile through the centroid column")
    return _finish(fig)


def _mapsnap(ax: Any, study: Study, a: np.ndarray, case: BelauriCase, vmax: float, title: str) -> Any:
    reg = study.region
    draw_map(ax, reg, case.georef["resolution_m"])
    d = case.scenario.domain
    off = case.georef["offset_in_standard_domain_m"]
    im = ax.imshow(np.ma.masked_less_equal(a, 1e-12), origin="lower", extent=(off[0] / 1000, (off[0] + d.lx) / 1000, off[1] / 1000, (off[1] + d.ly) / 1000),
                   cmap="magma_r", vmin=0, vmax=vmax, alpha=0.8)
    lv = reg.level(study.res)
    ax.set_xlim(*_km(lv)[:2])
    ax.set_ylim(*_km(lv)[2:])
    ax.set_title(title, fontsize=9)
    return im


def fig_contributions(study: Study, wr: WindResult, times: Sequence[float]) -> Figure:
    """Rows local / external / combined at the given times, ONE common colour scale for all panels."""
    vmax = float(wr.fields.total.max()) or 1.0
    fig, axes = plt.subplots(3, len(times), figsize=(4.2 * len(times), 11), squeeze=False, constrained_layout=True)
    im: Any = None
    for c, t in enumerate(times):
        k = int(np.argmin(np.abs(wr.fields.times - t)))
        for r, (lab, a) in enumerate((("local", wr.fields.local), ("external", wr.fields.external), ("combined", wr.fields.total))):
            im = _mapsnap(axes[r, c], study, a[k], wr.case, vmax, f"{lab}, t = {wr.fields.times[k] / 3600:.1f} h")
    fig.colorbar(im, ax=axes, label="concentration (microgram/m$^3$), common scale", shrink=0.5)
    fig.suptitle(f"wind: {wr.name} (synthetic)")
    return _finish(fig)


def fig_arrival(study: Study, results: Sequence[WindResult]) -> Figure:
    """External-tag concentration at the five report points for each wind; vertical line = onset, dotted = level."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), squeeze=False, sharey=True, constrained_layout=True)
    for ax, wr in zip(axes[0], results):
        for p in study.points:
            ax.plot(wr.fields.times / 3600, sample_cell(wr.fields.external, wr.case, p["x"], p["y"]), label=p["label"].replace("_", " "))
        ax.axhline(study.arrival_level, c="k", ls=":", lw=0.8)
        ax.axvline(study.onset_s / 3600, c="0.5", ls="--", lw=0.8)
        ax.set(title=wr.name, xlabel="t (h)")
    axes[0, 0].set_ylabel("external-tag concentration (microgram/m$^3$)")
    axes[0, 0].legend(fontsize=6)
    return _finish(fig)


def fig_regional(results: Sequence[WindResult]) -> Figure:
    """Study-weighted regional mean and study peak (total, local, external) and the external fraction versus time."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for wr in results:
        s = wr.series
        axes[0].plot(s.time_s / 3600, s.mean_total, label=wr.name)
        axes[1].plot(s.time_s / 3600, s.peak_total, label=wr.name)
        axes[2].plot(s.time_s / 3600, s.external_fraction, label=wr.name)
    axes[0].set(title="regional mean, total", ylabel="microgram/m$^3$", xlabel="t (h)")
    axes[1].set(title="study peak, total", xlabel="t (h)")
    axes[2].set(title="external fraction (blank = undefined)", xlabel="t (h)", ylim=(-0.02, 1.02))
    axes[0].legend(fontsize=7)
    return _finish(fig)


def fig_bars(df: pd.DataFrame, label: str, ys: Sequence[str], title: str) -> Figure:
    """Horizontal bars of the columns ``ys`` (one panel each), one bar per row labelled by ``label`` (shortened)."""
    labs = [str(v)[:58] for v in df[label]]
    fig, axes = plt.subplots(1, len(ys), figsize=(7 * len(ys), 0.35 * len(df) + 2), squeeze=False, sharey=True, constrained_layout=True)
    for ax, y in zip(axes[0], ys):
        ax.barh(labs, np.nan_to_num(np.asarray(pd.to_numeric(df[y]), dtype=float)))
        ax.invert_yaxis()
        ax.set_title(y, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(title)
    return _finish(fig)


def animate_on_map(study: Study, wr: WindResult, which: str = "total", max_frames: int = 49) -> FuncAnimation:
    """Animation of one field over the hillshade/outline (at most ``max_frames`` frames)."""
    a = getattr(wr.fields, which)
    idx = np.unique(np.linspace(0, a.shape[0] - 1, min(max_frames, a.shape[0])).astype(int))
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    vmax = float(a.max()) or 1.0
    im = _mapsnap(ax, study, a[0], wr.case, vmax, "")
    fig.colorbar(im, label="microgram/m$^3$")
    fig.text(0.01, 0.005, LABEL, fontsize=7, style="italic")

    def upd(n: int) -> tuple[Any, ...]:
        im.set_data(np.ma.masked_less_equal(a[idx[n]], 1e-12))
        ax.set_title(f"{wr.name}, {which}, t = {wr.fields.times[idx[n]] / 3600:.2f} h", fontsize=9)
        return (im,)
    return FuncAnimation(fig, upd, frames=len(idx), interval=150, blit=False)


# ---------------------------------------------------------------------------------------------- comparisons
def outer_domain_rows(study: Study, wr: WindResult, buffer_m: float) -> tuple[dict[str, Any], list[dict[str, Any]], Fields, BelauriCase]:
    """Same physical problem on the small (cropped) outer domain vs the standard one."""
    case = study.case(wr.name, crop_buffer_m=buffer_m)
    f = study.pair(case, f"{wr.name}:small_domain")
    off, d = case.georef["offset_in_standard_domain_m"], case.scenario.domain
    i0, j0 = int(round(off[0] / d.dx)), int(round(off[1] / d.dy))
    sl = (slice(None), slice(j0, j0 + d.ny), slice(i0, i0 + d.nx))
    ref = Fields(wr.fields.times, wr.fields.local[sl], wr.fields.external[sl], wr.fields.total[sl])
    w = study_weights_of(case)
    peak = wr.summary["total"]["peak"]
    sm, _ = regional_metrics(f, case, study.t_exclude)
    row: dict[str, Any] = {"wind": wr.name, "buffer_m": buffer_m, "n_cells_small": d.nx * d.ny,
                           "n_cells_standard": wr.case.scenario.domain.nx * wr.case.scenario.domain.ny,
                           "study_peak_standard": peak, "study_peak_small": sm["total"]["peak"],
                           "peak_change_pct": 100 * (sm["total"]["peak"] / peak - 1) if peak > 0 else None,
                           "mean_standard": wr.summary["total"]["time_mean_regional_mean"],
                           "mean_small": sm["total"]["time_mean_regional_mean"],
                           "max_abs_diff_whole_small_domain": float(np.abs(f.total - ref.total).max())}
    row["mean_change_pct"] = 100 * (row["mean_small"] / row["mean_standard"] - 1) if row["mean_standard"] > 0 else None
    for tag in ("total", "local", "external"):
        r = weighted_rmse(getattr(f, tag), getattr(ref, tag), w, f.times, study.t_exclude)
        row[f"rmse_{tag}"], row[f"rmse_{tag}_rel_peak"] = r, (r / peak if peak > 0 else None)
    arr_small = arrival_rows(f.external, f.times, case, study.points, study.arrival_level, study.onset_s)
    arr_rows = []
    for a, b in zip(wr.arrival, arr_small):
        arr_rows.append({"wind": wr.name, "buffer_m": buffer_m, "point": a["point"], "arrival_standard_s": a["arrival_s"],
                         "arrival_small_s": b["arrival_s"],
                         "difference_s": None if a["arrival_s"] is None or b["arrival_s"] is None else b["arrival_s"] - a["arrival_s"]})
    return row, arr_rows, f, case


def mesh_rows(study: Study, wind: str, runs: list[dict[str, Any]], reference: str, base: WindResult | None = None
              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mesh / time-step sensitivity at one wind.  Every run is block-averaged to the baseline mesh (``study.res``) for the
    comparison; ``reference`` (a run label) is the finest.  Numerical-diffusion excess is reported separately from K."""
    out: dict[str, tuple[Fields, BelauriCase, float]] = {}
    for r in runs:
        res, dt = float(r["resolution_m"]), float(r["dt_s"])
        if base is not None and res == study.res and dt == study.dt:
            out[r["label"]] = (base.fields, base.case, base.case.scenario.times.dt)
            continue
        case = study.case(wind, resolution_m=res, dt_s=dt)
        f = study.pair(case, f"mesh:{r['label']}")
        fac = int(round(study.res / res))
        out[r["label"]] = (Fields(f.times, block_average(f.local, fac), block_average(f.external, fac), block_average(f.total, fac)), case, dt)
    ref_f = out[reference][0]
    case0 = study.case(wind)  # baseline-mesh case: grid, mask and point lookup for the common comparison level
    w = study_weights_of(case0)
    sref, _ = regional_metrics(ref_f, case0, study.t_exclude)
    arr_ref = arrival_rows(ref_f.external, ref_f.times, case0, study.points, study.arrival_level, study.onset_s)
    u = study.speed
    rows, arr_rows = [], []
    for r in runs:
        f, case, dt = out[r["label"]]
        s, _ = regional_metrics(f, case0, study.t_exclude)
        res = float(r["resolution_m"])
        ex = numerical_diffusion_estimate(u, res, dt, study.K)
        rec = next((x for x in reversed(study.records) if x["label"] in (f"mesh:{r['label']}:combined", f"{wind}:combined_direct")
                    and x["resolution_m"] == res and x["dt_s"] == dt), None)
        rows.append({"run": r["label"], "resolution_m": res, "dt_s": dt, "courant_u_dt_over_dx": u * dt / res,
                     "mean_total": s["total"]["time_mean_regional_mean"], "peak_total": s["total"]["peak"],
                     "mean_change_vs_ref_pct": 100 * (s["total"]["time_mean_regional_mean"] / sref["total"]["time_mean_regional_mean"] - 1),
                     "peak_change_vs_ref_pct": 100 * (s["total"]["peak"] / sref["total"]["peak"] - 1),
                     "external_fraction": s["external_fraction"],
                     "rmse_total_rel_ref_peak": weighted_rmse(f.total, ref_f.total, w, f.times, study.t_exclude) / sref["total"]["peak"],
                     "solve_wall_s": None if rec is None else rec["wall_s_per_solve"],
                     "numerical_excess_Keff_over_K_minus_1": ex, "K_physical_m2_s": study.K,
                     "K_effective_m2_s_estimate": study.K * (1 + ex),
                     "spreading_length_physical_m_at_3h": math.sqrt(2 * study.K * 3 * 3600),
                     "spreading_length_effective_m_at_3h": math.sqrt(2 * study.K * (1 + ex) * 3 * 3600),
                     "note": "estimate is exact for wind along a grid axis; scalar-speed approximation for oblique wind"})
        for a, b in zip(arr_ref, arrival_rows(f.external, f.times, case0, study.points, study.arrival_level, study.onset_s)):
            arr_rows.append({"run": r["label"], "point": a["point"], "arrival_ref_s": a["arrival_s"], "arrival_s": b["arrival_s"],
                             "difference_s": None if a["arrival_s"] is None or b["arrival_s"] is None else b["arrival_s"] - a["arrival_s"]})
    return rows, arr_rows


def _metric_triplet(f: Fields, case: BelauriCase, t_exclude: float) -> dict[str, Any]:
    s, _ = regional_metrics(f, case, t_exclude)
    return {"mean": s["total"]["time_mean_regional_mean"], "peak": s["total"]["peak"], "ext_fraction": s["external_fraction"]}


def _pct(a: float, b: float) -> float | None:
    return 100.0 * (a / b - 1.0) if b > 0 else None


def what_changes(study: Study, base: WindResult, results: dict[str, WindResult], outer: list[dict[str, Any]],
                 mesh: list[dict[str, Any]], plot_check: dict[str, Any] | None, scfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Measured change of three regional metrics (time-mean regional mean, study peak, external fraction) when ONE input
    is changed from the baseline (reference wind, standard domain, baseline mesh)."""
    b = _metric_triplet(base.fields, base.case, study.t_exclude)
    rows: list[dict[str, Any]] = []

    def add(inp: str, var: str, m: dict[str, Any], solves: str, note: str = "") -> None:
        rows.append({"input": inp, "variation": var, "mean": m["mean"], "peak": m["peak"], "external_fraction": m["ext_fraction"],
                     "mean_change_pct": _pct(m["mean"], b["mean"]), "peak_change_pct": _pct(m["peak"], b["peak"]),
                     "external_fraction_change": None if m["ext_fraction"] is None or b["ext_fraction"] is None else m["ext_fraction"] - b["ext_fraction"],
                     "solves": solves, "note": note})

    add("baseline", f"{base.name}, K={study.K:g}, lam=1/{1 / study.lam:g} s", b, "0")
    for n, wr in results.items():
        if n != base.name:
            add("wind", n, _metric_triplet(wr.fields, wr.case, study.t_exclude), "from wind runs", "synthetic wind direction/segments")
    for fct in scfg.get("K_factors", []):
        f = study.pair(study.case(base.name, K=study.K * fct), f"sens:K x{fct}")
        add("K (physical diffusivity)", f"x{fct:g}", _metric_triplet(f, base.case, study.t_exclude), "2")
    for fct in scfg.get("lam_factors", []):
        f = study.pair(study.case(base.name, lam=study.lam * fct), f"sens:lam x{fct}")
        add("lam (removal rate)", f"x{fct:g}", _metric_triplet(f, base.case, study.t_exclude), "2")
    th, g = base.case.theta, base.resp.g
    names = [lab["name"] for lab in base.case.labels]

    def from_theta(t: np.ndarray) -> Fields:
        loc = sum((t[j] * g[j] for j, lab in enumerate(base.case.labels) if lab["group"] == "local"), np.zeros_like(base.resp.base))
        ext = sum((t[j] * g[j] for j, lab in enumerate(base.case.labels) if lab["group"] == "external"), np.zeros_like(base.resp.base))
        return Fields(base.resp.times, np.asarray(loc), np.asarray(ext), np.asarray(loc + ext))
    fa = float(scfg.get("amplitude_factor", 2.0))
    for grp, lab in (("external", "external patch amplitude"), ("road", "road amplitude")):
        j = names.index("external_patch" if grp == "external" else grp)
        t2 = th.copy()
        t2[j] *= fa
        add("source amplitude", f"{lab} x{fa:g}", _metric_triplet(from_theta(t2), base.case, study.t_exclude), "0 (linear superposition)")
    loc_total = sum(th[j] for j, lab in enumerate(base.case.labels) if lab["group"] == "local")
    for layer in [n for n in names if n in ("road", "builtup")]:
        t2 = th.copy()
        for j, lab in enumerate(base.case.labels):
            if lab["group"] == "local":
                t2[j] = loc_total if names[j] == layer else 0.0
        add("source geometry", f"all local emission ({loc_total:.3g} ug/s) on the {layer} layer only",
            _metric_triplet(from_theta(t2), base.case, study.t_exclude), "0 (linear superposition)", "same total local emission, different real geometry")
    for r in outer:
        if r["wind"] == base.name:
            add("outer domain", f"{r['buffer_m'] / 1000:g} km buffer instead of the standard one",
                {"mean": r["mean_small"], "peak": r["study_peak_small"], "ext_fraction": r.get("external_fraction_small")}, "2 (outer-domain study)")
    for r in mesh:
        if r["run"] != f"{study.res:g}m_dt{study.dt:g}":
            add("mesh / time step", r["run"], {"mean": r["mean_total"], "peak": r["peak_total"], "ext_fraction": r["external_fraction"]}, "2 (mesh study)",
                "block-averaged to the baseline mesh")
    if plot_check is not None:
        add("DEM / hillshade / slope (plotting-only)", "layers replaced by constants",
            {"mean": plot_check["mean"], "peak": plot_check["peak"], "ext_fraction": plot_check["ext_fraction"]}, "1",
            f"scenario identical: {plot_check['scenario_identical']}; max |change| = {plot_check['max_abs_change']:.3g}")
    return rows


# ---------------------------------------------------------------------------------------------- the runner
@dataclass
class BelauriResult:
    out_dir: Path
    winds: list[str]
    figures: list[str]
    timings: dict[str, float]
    files: list[str] = field(default_factory=list)


def _save_fig(out: Path, fig: Figure, name: str, figures: list[str]) -> None:
    d = out / "figures"
    d.mkdir(exist_ok=True)
    fig.savefig(d / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    figures.append(name)


def run_belauri_forward(cfg: dict[str, Any], out_dir: Path | str, overwrite: bool = False, progress: Progress = None
                        ) -> BelauriResult:
    """Run the configured studies into ``out_dir`` (refused when non-empty unless ``overwrite``).  Files: config.json,
    provenance.json (incl. the manifest sha256, resolution, no seeds), timings.json, runs.csv (per solve: runtime,
    budget residual, negative count), wind_summary.csv, consistency.json, arrival_times.csv, series/<wind>.csv,
    fields/<wind>.nc, outer_domain*.csv, mesh_sensitivity*.csv, what_changes.{csv,json}, figures/*.png."""
    t_all = time.perf_counter()
    out = make_run_dir(out_dir, overwrite)
    write_json(out / "config.json", cfg)
    study = Study(cfg, progress)
    tim: dict[str, float] = {}
    figs: list[str] = []
    wcfg = cfg["wind"]
    ref = wcfg["reference"]
    run_names = list(wcfg["run"])
    if ref not in run_names:
        run_names.insert(0, ref)
    (out / "series").mkdir(exist_ok=True)
    (out / "fields").mkdir(exist_ok=True)

    t0 = time.perf_counter()
    results: dict[str, WindResult] = {}
    for n in run_names:
        results[n] = wr = study.wind_result(n)
        wr.series.to_csv(out / "series" / f"{n}.csv", index=False)
        ds = xr.Dataset({"local": wr.fields.da("local", wr.case), "external": wr.fields.da("external", wr.case),
                         "combined": wr.fields.da("total", wr.case)})
        ds.attrs["label"] = LABEL
        save_field(out / "fields" / f"{n}.nc", ds)
    tim["winds_s"] = time.perf_counter() - t0
    srows = []
    for n, wr in results.items():
        s = wr.summary
        row: dict[str, Any] = {"wind": n}
        for tag in ("local", "external", "total"):
            row.update({f"{tag}_mean": s[tag]["time_mean_regional_mean"], f"{tag}_peak": s[tag]["peak"],
                        f"{tag}_peak_time_s": s[tag]["peak_time_s"]})
        row["external_fraction"] = s["external_fraction"]
        srows.append(row)
    write_csv(out / "wind_summary.csv", pd.DataFrame(srows))
    write_csv(out / "arrival_times.csv", pd.DataFrame([{"wind": n, **a} for n, wr in results.items() for a in wr.arrival]))
    write_json(out / "consistency.json", {n: wr.consistency for n, wr in results.items()})
    write_json(out / "points.json", study.points)

    outer_rows: list[dict[str, Any]] = []
    outer_arr: list[dict[str, Any]] = []
    ocfg = cfg.get("outer_domain", {})
    if ocfg.get("enabled", True):
        t0 = time.perf_counter()
        buf = float(cfg["region"]["small_buffer_m"])
        for n in ocfg.get("winds", run_names):
            row, arr, f, case = outer_domain_rows(study, results[n], buf)
            s, _ = regional_metrics(f, case, study.t_exclude)
            row["external_fraction_small"], row["external_fraction_standard"] = s["external_fraction"], results[n].summary["external_fraction"]
            outer_rows.append(row)
            outer_arr += arr
        write_csv(out / "outer_domain.csv", pd.DataFrame(outer_rows))
        write_csv(out / "outer_domain_arrival.csv", pd.DataFrame(outer_arr))
        tim["outer_domain_s"] = time.perf_counter() - t0

    mesh: list[dict[str, Any]] = []
    mcfg = cfg.get("mesh_sensitivity", {})
    if mcfg.get("enabled", True):
        t0 = time.perf_counter()
        mesh, marr = mesh_rows(study, mcfg.get("wind", ref), list(mcfg["runs"]), mcfg["reference"], results[mcfg.get("wind", ref)])
        write_csv(out / "mesh_sensitivity.csv", pd.DataFrame(mesh))
        write_csv(out / "mesh_sensitivity_arrival.csv", pd.DataFrame(marr))
        tim["mesh_s"] = time.perf_counter() - t0

    plot_check: dict[str, Any] | None = None
    if cfg.get("plotting_only", {}).get("enabled", True):
        chk = plotting_only_check(study.region, lambda r: study.case(ref, region=r))
        tot, _ = study.direct(chk["case_b"], "plotting_only:scrambled_layers")
        base_tot, _ = study.direct(chk["case_a"], "plotting_only:original_layers")
        m = _metric_triplet(Fields(results[ref].fields.times, tot - results[ref].fields.external, results[ref].fields.external, tot),
                            results[ref].case, study.t_exclude)
        plot_check = {**m, "scenario_identical": chk["scenario_identical"], "max_abs_change": float(np.abs(tot - base_tot).max()),
                      "layers": chk["layers"]}

    wc: list[dict[str, Any]] = []
    if cfg.get("sensitivity", {}).get("enabled", True):
        t0 = time.perf_counter()
        wc = what_changes(study, results[ref], results, outer_rows, mesh, plot_check, cfg["sensitivity"])
        write_csv(out / "what_changes.csv", pd.DataFrame(wc))
        write_json(out / "what_changes.json", {"rows": wc, "plotting_only_layers": {
            "layers": list(PLOTTING_ONLY_LAYERS), "changes_any_number": False, "check": plot_check,
            "meaning": "DEM, hillshade, slope and the elevation profiles are drawn on maps only; no solver input reads them"}})
        tim["what_changes_s"] = time.perf_counter() - t0
    write_csv(out / "runs.csv", pd.DataFrame(study.records))

    if cfg.get("figures", {}).get("enabled", True):
        t0 = time.perf_counter()
        _save_fig(out, fig_overview(study), "overview_map.png", figs)
        snaps = [float(t) for t in cfg["figures"]["snapshot_times_s"]]
        for n, wr in results.items():
            _save_fig(out, fig_contributions(study, wr, snaps), f"contributions_{n}.png", figs)
        _save_fig(out, fig_arrival(study, list(results.values())), "arrival_traces.png", figs)
        _save_fig(out, fig_regional(list(results.values())), "regional_summaries.png", figs)
        if wc:
            df = pd.DataFrame(wc)
            df["label"] = df["input"] + ": " + df["variation"]
            _save_fig(out, fig_bars(df, "label", ["mean_change_pct", "peak_change_pct"], "what changes (percent vs baseline)"), "what_changes.png", figs)
        if mesh:
            _save_fig(out, fig_bars(pd.DataFrame(mesh), "run", ["mean_change_vs_ref_pct", "peak_change_vs_ref_pct"], "mesh / time step vs reference"), "mesh_sensitivity.png", figs)
        tim["figures_s"] = time.perf_counter() - t0

    tim["total_s"] = time.perf_counter() - t_all
    write_json(out / "timings.json", tim)
    write_json(out / "provenance.json", provenance(cfg, {}, {
        "experiment": "belauri_forward", "label": LABEL, "manifest": str(study.manifest_path), "manifest_sha256": study.manifest_sha256,
        "boundary_status": study.full.boundary_status, "resolution_m": study.res, "dt_s": study.dt, "seeds": "none (deterministic)",
        "amplitudes_ug_s": study.amplitudes, "points": study.points,
        "note": "emissions and meteorology are SYNTHETIC; DEM/hillshade/slope are plotting-only context"}))
    return BelauriResult(out, run_names, figs, tim)

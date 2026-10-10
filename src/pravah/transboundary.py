"""SYNTHETIC study of EXTERNAL vs LOCAL pollution (Stage B3a).

Terms (used literally): local = emitted inside the study region; external = outside it (an explicit source or an
outer-inflow-boundary input).  The study region is only a reporting mask / weight, never a PDE wall.  There is no
national border here, so the phrase 'international transboundary' is not used.  All numbers in ``configs/
transboundary*.json`` are DEMONSTRATION values, not estimates for any place.

Coordinates in the config are STUDY-RELATIVE metres (origin = south-west corner of the study region).  The simulation
domain adds a buffer on every side (domain A: 5 km, domain B: 10 km by default) so that sources and study region keep
the same positions relative to each other.  A and B hold the same physical sources and study region, but the open
boundary (and so any boundary basis) sits 5 km vs 10 km from the study region: for scenarios without boundary bases the
A-B difference is pure domain truncation; for scenarios with bases ('mixed') it mixes truncation with a MOVED forcing.

Contents
* ``build_scenario(cfg, name, domain, wind)`` -> (Scenario, theta_true, labels) for the named scenarios, with
  ``check_no_double_count`` (no plume is represented twice) and ``sources.check_origin_tags`` (tags agree with the study);
* ``draw_cases`` / ``check_ensembles``: validation / test ensembles with disjoint scenario seeds and noise-seed ranges;
* ``run_transboundary(cfg, out_dir, ...)``: four switchable studies (fields, recovery from total observations,
  special cases, domain-size / boundary-basis equivalence); everything is written with ``pravah.io``.

KNOWN inputs of the inversion (not estimated): wind (speed, direction, switch times), K, lam, H, source positions and
widths, boundary-basis shapes AND the TEMPORAL PROFILES: E1 switches on 0 -> 1 at its true onset (3600 s in the default
config) and every basis uses a step profile 0 -> 1 at ``basis_onset_s`` (7200 s in the scenarios; the equivalence study
runs the basis at E1's true onset and, as a sensitivity check, at a different onset).  Only the amplitudes are
estimated, so the recovery numbers are conditional on these profiles being right.

Scales.  ``inverse.SCALE_BY_KIND`` (emission 1e5 ug/s, inflow 0.05 ug/m^3) is a numerical scaling at alpha = 0, but a
PRIOR WIDTH when alpha > 0 (penalty alpha (a/0.05)^2 for an inflow amplitude, alpha (q/1e5)^2 for an emission), and the
emission scale 1e5 equals the true L1 amplitude.  ``recovery/scale_sensitivity.csv`` reruns the fits with the scales
multiplied by 0.1 and 10.

Honesty rules.  The inversion sees only ``Observations`` (readings + noise std + layout) and declared model inputs
(wind, K, lam, H, source geometry, basis shapes); origin tags are used only afterwards, to evaluate.  The ridge strength
alpha (scaled units) is chosen on VALIDATION cases only, written to ``recovery/alpha_selected.json`` BEFORE the test
ensemble is built, and the test ensemble is never used for any choice.  Noise is set relative to a documented signal
scale (median over cases of the peak noiseless reading of the ``scale_layout``), reported in ug/m^3 and in % of the scale.
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import xarray as xr

from .attribution import (component_errors, contributions, emission_error, external_part, identifiability_report,
                          origin_fractions, total_field)
from .config import (Boundary, Domain, GaussianSource, InflowBasis, Scenario, SolverOptions, StudyRegion, Times,
                     Transport, WindSegment)
from .experiments import perturb_wind
from .forward import FIPY
from .inverse import SCALE_BY_KIND, InferenceOptions, ResponseCache, Responses, fit_sources, run_forward
from .io import make_run_dir, provenance, write_csv, write_json
from .metrics import field_rmse, quadrature_weights
from .observations import ObservationSpec, Observations, SensorLayout, observe, sample
from .sources import check_origin_tags, study_weights

Progress = Callable[[str], None] | None
EPS_REL = 1e-3  # fraction definitions: None when the (time-mean or instantaneous) total is below EPS_REL x the peak
Rect = tuple[float, float, float, float]


def _say(progress: Progress, msg: str) -> None:
    if progress is not None:
        progress(msg)


# ------------------------------------------------------------------ scenarios
def _r12(x: float) -> float:
    """Round a direction-derived wind component to 1e-12 m/s (kills sin(pi) = 1.2e-16 style round-off)."""
    return round(float(x), 12) + 0.0  # + 0.0 turns -0.0 into 0.0


def uniform_wind(speed: float, toward_deg: float) -> tuple[WindSegment, ...]:
    """One constant wind segment: ``speed`` m/s toward ``toward_deg`` (counter-clockwise from +x = east); components are
    rounded to 1e-12 m/s so that a side that is exactly tangential is tangential in the float representation too."""
    th = math.radians(toward_deg)
    return (WindSegment(0.0, _r12(speed * math.cos(th)), _r12(speed * math.sin(th))),)


def _nominal_angle(cfg: dict[str, Any], spec: dict[str, Any]) -> float:
    return float(spec.get("wind_toward_deg", cfg["wind"]["toward_deg"]))


def check_no_double_count(sources: list[tuple[str, float, float]], bases: list[tuple[str, Rect | None]],
                          domain_rect: Rect | None = None) -> None:
    """Raise ValueError if an explicit source lies in a region that a basis is flagged to represent (that plume would be
    counted twice: once explicitly, once through the boundary), or lies outside the simulation domain.
    ``sources``: (name, x, y); ``bases``: (name, represents) with ``represents`` a rectangle (x0, y0, x1, y1) in the same
    coordinates, or None (= unspecified sources beyond the domain).  Minimal on purpose."""
    for name, x, y in sources:
        if domain_rect is not None and not (domain_rect[0] <= x <= domain_rect[2] and domain_rect[1] <= y <= domain_rect[3]):
            raise ValueError(f"explicit source {name} lies outside the simulation domain: represent it by a boundary basis")
        for bname, rect in bases:
            if rect is not None and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                raise ValueError(f"double count: explicit source {name} lies inside the region that basis {bname} represents")


def _profile(p: Any) -> tuple[tuple[float, float], ...]:
    return tuple((float(t), float(v)) for t, v in p)


def _resolve_source(cfg: dict[str, Any], name: str, angle_deg: float) -> dict[str, Any]:
    s = dict(cfg["sources"][name])
    if "upwind_of" in s:
        ref = cfg["sources"][s["upwind_of"]]
        th = math.radians(angle_deg)
        s["x"] = ref["x"] - s["distance"] * math.cos(th)
        s["y"] = ref["y"] - s["distance"] * math.sin(th)
    return s


def build_model(cfg: dict[str, Any], spec: dict[str, Any], domain: str = "A",
                wind: tuple[WindSegment, ...] | None = None) -> tuple[Scenario, np.ndarray, list[dict[str, Any]]]:
    """Scenario from a spec ``{sources: [names], bases: [names], amplitudes: {basis: value}, represents: rect|None,
    wind_toward_deg: float|None}`` on domain 'A' or 'B'.  Returns (Scenario, theta_true, labels): theta = [source
    amplitudes (ug/s)..., basis amplitudes (ug/m^3)...]; ``labels`` is the parameter table (name, kind, units, group,
    true value, study-relative position)."""
    dom = cfg["geometry"]["domains"][domain]
    size, buf, dx = float(cfg["geometry"]["study_size"]), float(dom["buffer"]), float(dom["dx"])
    length = size + 2 * buf
    n = int(round(length / dx))
    if abs(n * dx - length) > 1e-6:
        raise ValueError(f"domain {domain}: length {length} is not a multiple of dx = {dx}")
    ph = cfg["physics"]
    angle = _nominal_angle(cfg, spec)
    srcs = [_resolve_source(cfg, nm, angle) for nm in spec["sources"]]
    bnames: list[str] = list(spec.get("bases", []))
    rep = spec.get("represents")
    bases_cfg = [cfg["bases"][b] for b in bnames]
    rect: Rect | None = None if rep is None else (float(rep[0]), float(rep[1]), float(rep[2]), float(rep[3]))
    represents: list[tuple[str, Rect | None]] = [(b, rect) for b in bnames]
    check_no_double_count([(nm, s["x"], s["y"]) for nm, s in zip(spec["sources"], srcs)], represents,
                          (-buf, -buf, size + buf, size + buf))
    sources = tuple(GaussianSource(nm, s["x"] + buf, s["y"] + buf, s["sigma"],
                                   _profile(s["onset"]) if "onset" in s else ((0.0, 1.0),), s["group"])
                    for nm, s in zip(spec["sources"], srcs))
    bases = tuple(InflowBasis(b, tuple(c["sides"]), _profile(c["profile"])) for b, c in zip(bnames, bases_cfg))
    if wind is None:
        wind = uniform_wind(float(spec.get("wind_speed", cfg["wind"]["speed"])), angle)
    scn = Scenario(
        domain=Domain(length, length, n, n), sources=sources,
        transport=Transport(wind=wind, K=float(ph["K"]), lam=float(ph["lam"]), H=float(ph["H"])),
        boundary=Boundary("open", 0.0, bases),
        times=Times(float(ph["T"]), float(ph["dt"]), float(ph["output_interval"])),
        study=StudyRegion(buf, buf, buf + size, buf + size))
    check_origin_tags(scn)
    amps = dict(spec.get("amplitudes", {}))
    theta = np.array([float(s["q"]) for s in srcs] + [float(amps.get(b, c["amplitude"])) for b, c in zip(bnames, bases_cfg)])
    labels = []
    for j, p in enumerate(scn.param_table()):
        row: dict[str, Any] = {"name": p.name, "kind": p.kind, "units": p.units, "group": p.group, "true_value": float(theta[j])}
        if j < len(srcs):
            row.update({"x_study": srcs[j]["x"], "y_study": srcs[j]["y"], "sigma_m": srcs[j]["sigma"]})
        else:
            row.update({"sides": ",".join(bases[j - len(srcs)].sides)})
        labels.append(row)
    return scn, theta, labels


def build_scenario(cfg: dict[str, Any], name: str, domain: str = "A", wind: tuple[WindSegment, ...] | None = None
                   ) -> tuple[Scenario, np.ndarray, list[dict[str, Any]]]:
    """A NAMED scenario of ``cfg['scenarios']`` (local_only, external_sources, boundary_only, mixed, sector, confounded,
    no_arrival) on domain 'A' or 'B', see ``build_model``.  ``wind`` overrides the nominal wind."""
    if name not in cfg["scenarios"]:
        raise ValueError(f"unknown scenario {name!r}; known: {sorted(cfg['scenarios'])}")
    return build_model(cfg, cfg["scenarios"][name], domain, wind)


def param_table(cfg: dict[str, Any], names: list[str], domain: str = "A") -> pd.DataFrame:
    rows = []
    for nm in names:
        _, _, labels = build_scenario(cfg, nm, domain)
        rows += [{"scenario": nm, "domain": domain, **lab} for lab in labels]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ ensembles and seeds
@dataclass(frozen=True, eq=False)
class Case:
    """One ensemble member: the declared scenario (its wind is the truth wind) and the true parameter vector theta."""
    name: str
    scenario: Scenario
    theta: np.ndarray


def draw_cases(cfg: dict[str, Any], base: Scenario, theta0: np.ndarray, n: int, seed: int, prefix: str) -> list[Case]:
    """n cases: theta = theta0 * U(q_factor_range) per parameter; wind speed U(speed_range) and direction = nominal
    +- direction_spread_deg; every second case (1, 3, ...) has a two-segment wind with a switch time that is a multiple
    of output_interval within [T/3, 2T/3].  Deterministic in ``seed``."""
    e = cfg["ensemble"]
    rng = np.random.default_rng(seed)
    w0 = base.transport.wind[0]
    th0 = math.atan2(w0.v, w0.u)
    oi, T = base.times.output_interval, base.times.duration
    k_lo, k_hi = math.ceil(T / 3 / oi - 1e-9), math.floor(2 * T / 3 / oi + 1e-9)
    if k_hi < k_lo:
        raise ValueError("no output-interval multiple lies within [T/3, 2T/3]")

    def draw_wind() -> tuple[float, float]:
        sp = rng.uniform(*e["speed_range"])
        th = th0 + math.radians(rng.uniform(-e["direction_spread_deg"], e["direction_spread_deg"]))
        return _r12(sp * math.cos(th)), _r12(sp * math.sin(th))

    out = []
    for i in range(n):
        theta = theta0 * rng.uniform(*e["q_factor_range"], size=theta0.size)
        u, v = draw_wind()
        segs = [WindSegment(0.0, u, v)]
        if i % 2 == 1:
            t_sw = float(rng.integers(k_lo, k_hi + 1) * oi)
            u2, v2 = draw_wind()
            segs.append(WindSegment(t_sw, u2, v2))
        scn = dataclasses.replace(base, transport=dataclasses.replace(base.transport, wind=tuple(segs)))
        out.append(Case(f"{prefix}{i}", scn, theta))
    return out


def _draw_key(c: Case) -> str:
    return json.dumps([[float(x) for x in c.theta], [dataclasses.astuple(w)[:3] for w in c.scenario.transport.wind]])


def check_ensembles(validation: list[Case], test: list[Case], validation_seed: int, test_seed: int,
                    validation_noise_seed: int, test_noise_seed: int, noise_reps: int) -> None:
    """Raise ValueError unless the validation and test ensembles have different scenario seeds, disjoint noise-seed
    ranges (case i, repetition r uses seed noise_seed + i * noise_reps + r), no shared (theta, wind) draw and no shared
    case name (cf. experiments.check_ensembles)."""
    if validation_seed == test_seed:
        raise ValueError("validation_seed and test_seed must differ")
    check_noise_ranges({"validation": (validation_noise_seed, len(validation) * noise_reps),
                        "test": (test_noise_seed, len(test) * noise_reps)})
    if {_draw_key(c) for c in validation} & {_draw_key(c) for c in test}:
        raise ValueError("validation and test ensembles share a (theta, wind) draw")
    if {c.name for c in validation} & {c.name for c in test}:
        raise ValueError("validation and test case names overlap")


def check_noise_ranges(ranges: dict[str, tuple[int, int]]) -> None:
    """Raise ValueError if any two (first_seed, count) noise-seed ranges overlap."""
    names = list(ranges)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = ranges[a], ranges[b]
            if set(range(ra[0], ra[0] + ra[1])) & set(range(rb[0], rb[0] + rb[1])):
                raise ValueError(f"{a} and {b} noise seeds overlap")


def _noise_ranges(cfg: dict[str, Any]) -> dict[str, tuple[int, int]]:
    r, s, e = cfg["recovery"], cfg["special"], cfg["domain_size"]["equivalence"]
    out = {"validation": (int(r["validation_noise_seed"]), int(r["n_validation"]) * int(r["noise_reps"])),
           "test": (int(r["test_noise_seed"]), int(r["n_test"]) * int(r["noise_reps"])),
           "special": (int(s["noise_seed"]), 16 * int(s["noise_reps"])),  # up to 16 special runs
           "equivalence": (int(e["noise_seed"]), int(e["noise_reps"]))}
    return out


# ------------------------------------------------------------------ small numerical helpers
def _arr(da: xr.DataArray) -> np.ndarray:
    return np.asarray(da.transpose("time", "y", "x").values, dtype=float)


def _zero_parts(resp: Responses) -> dict[str, np.ndarray]:
    """base_parts for scenarios with zero initial field and zero static background (all scenarios here).  Raises if the
    response has a nonzero base, because then the tags would no longer sum to the total."""
    if np.any(resp.base):
        raise ValueError("scenario has a nonzero initial field or background: _zero_parts would lose it from the tags")
    z = np.zeros_like(resp.base)
    return {"initial": z, "background": z}


def _get_resp(cache: ResponseCache, scn: Scenario, solver: SolverOptions) -> Responses:
    return cache.get(scn, solver)


def fraction_timeseries(contrib: dict[str, xr.DataArray], scenario: Scenario, eps_rel: float = EPS_REL
                        ) -> tuple[np.ndarray, list[float | None]]:
    """Instantaneous concentration_external_fraction F(t) = sum_w c_external / sum_w c_total over the study cells,
    None where the study-mean total is below eps_rel x its maximum over time (undefined) or not positive."""
    w = study_weights(scenario)
    if w is None:
        raise ValueError("scenario has no study region")
    tot = np.einsum("tyx,yx->t", _arr(total_field(contrib)), w) / w.sum()
    ext = np.einsum("tyx,yx->t", _arr(external_part(contrib)), w) / w.sum()
    scale = float(tot.max())
    out: list[float | None] = [float(e / t) if (scale > 0 and t > 0 and t >= eps_rel * scale) else None
                               for e, t in zip(ext, tot)]
    return np.asarray(contrib["local"]["time"].values, dtype=float), out


def arrival_time(trace: np.ndarray, times: np.ndarray, threshold: float, abs_floor: float) -> float | None:
    """First time (linear interpolation between output times) at which ``trace`` exceeds ``threshold`` x its maximum;
    None when the maximum does not exceed ``abs_floor`` (the trace is numerically zero: no arrival)."""
    peak = float(trace.max())
    if not peak > abs_floor:
        return None
    level = threshold * peak
    idx = np.nonzero(trace > level)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(times[0])
    a, b = trace[i - 1], trace[i]
    return float(times[i - 1] + (level - a) / (b - a) * (times[i] - times[i - 1]))


def _study_block(da: xr.DataArray, scn: Scenario) -> np.ndarray:
    """(nt, n, n) cells of the study region; needs the region to coincide with cell edges."""
    st = scn.study
    assert isinstance(st, StudyRegion)
    d = scn.domain
    i0, i1, j0, j1 = st.x0 / d.dx, st.x1 / d.dx, st.y0 / d.dy, st.y1 / d.dy
    if any(abs(v - round(v)) > 1e-9 for v in (i0, i1, j0, j1)):
        raise ValueError("study region must coincide with cell edges for the block comparison")
    return _arr(da)[:, int(round(j0)):int(round(j1)), int(round(i0)):int(round(i1))]


def _block_rmse(a: np.ndarray, b: np.ndarray, times: np.ndarray, t_exclude: float) -> float:
    sel, dt, w, den = quadrature_weights(times, a.shape[1], a.shape[2], None, t_exclude)
    return float(np.sqrt(np.einsum("t,tyx,yx->", dt, ((a - b) ** 2)[sel], w) / den))


# ------------------------------------------------------------------ run context
@dataclass
class _Run:
    cfg: dict[str, Any]
    out: Path
    cache: ResponseCache
    solver: SolverOptions
    progress: Progress
    tim: dict[str, float] = field(default_factory=dict)
    alpha_by_noise: dict[str, float] = field(default_factory=dict)  # frozen validation choice per noise fraction
    figures: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def t_exclude(self) -> float:
        return float(self.cfg["physics"]["t_exclude"])

    def sub(self, name: str) -> Path:
        p = self.out / name
        p.mkdir(exist_ok=True)
        return p

    def save_fig(self, fig: Any, name: str) -> None:
        d = self.sub("figures")
        fig.savefig(d / name, dpi=110, bbox_inches="tight")
        plt.close(fig)
        self.figures.append(name)

    def alpha_for(self, f: float) -> float | None:
        return self.alpha_by_noise.get(f"{f:g}")


def _layouts(cfg: dict[str, Any], scn: Scenario) -> dict[str, SensorLayout]:
    """Sensor layouts, study-relative in the config, shifted into simulation coordinates."""
    st = scn.study
    assert isinstance(st, StudyRegion)
    out = {}
    for name, pts in cfg["recovery"]["layouts"].items():
        out[name] = SensorLayout(tuple(f"S{i + 1}" for i in range(len(pts))), tuple(float(p[0]) + st.x0 for p in pts),
                                 tuple(float(p[1]) + st.y0 for p in pts))
    return out


def _origin(cfg: dict[str, Any]) -> float:
    """Domain-A buffer: study-relative -> simulation coordinates."""
    return float(cfg["geometry"]["domains"]["A"]["buffer"])


def _obs_times(cfg: dict[str, Any]) -> tuple[float, ...]:
    dt, T = float(cfg["recovery"]["obs_interval"]), float(cfg["physics"]["T"])
    return tuple(dt * (i + 1) for i in range(int(round(T / dt))))


# ------------------------------------------------------------------ fit + score
def _aligned_truth(model: Scenario, truth: Scenario, theta_true: np.ndarray) -> np.ndarray:
    """True value for each MODEL parameter, matched by name (0 for a parameter the truth does not have)."""
    tv = {p.name: float(theta_true[j]) for j, p in enumerate(truth.param_table())}
    return np.array([tv.get(p.name, 0.0) for p in model.param_table()])


def _study_peak(contrib: dict[str, xr.DataArray], scn: Scenario, t_exclude: float) -> float:
    """Peak true total concentration over the study cells and the evaluation window (scale for relative errors)."""
    w = study_weights(scn)
    assert w is not None
    times = np.asarray(contrib["local"]["time"].values, dtype=float)
    return float(_arr(total_field(contrib))[times >= t_exclude][:, w > 0].max())


def _scaled(model: Scenario, emission_mult: float, inflow_mult: float) -> tuple[float, ...] | None:
    """q_scale per parameter = default SCALE_BY_KIND x multiplier of its kind; None when both multipliers are 1."""
    if emission_mult == 1.0 and inflow_mult == 1.0:
        return None
    mult = {"emission": emission_mult, "inflow": inflow_mult}
    return tuple(SCALE_BY_KIND[p.kind] * mult[p.kind] for p in model.param_table())


def _fit_and_score(run: _Run, model: Scenario, truth: Scenario, theta_true: np.ndarray,
                   truth_contrib: dict[str, xr.DataArray], obs: Observations, alpha: float,
                   emission_mult: float = 1.0, inflow_mult: float = 1.0) -> dict[str, Any]:
    """Fit on Observations only, then evaluate against the truth tags (the inversion never sees them).  The multipliers
    scale the default per-kind q_scale (prior width when alpha > 0; see the module docstring)."""
    fit = fit_sources(model, obs, InferenceOptions(alpha=alpha, solver=run.solver,
                                                   q_scale=_scaled(model, emission_mult, inflow_mult)), run.cache)
    resp = _get_resp(run.cache, model, run.solver)
    est = contributions(resp, fit.q_hat, _zero_parts(resp))
    ce = component_errors(est, truth_contrib, model, run.t_exclude)
    em = emission_error(fit.q_hat, _aligned_truth(model, truth, theta_true), model)
    ident = identifiability_report(fit)
    peak = _study_peak(truth_contrib, model, run.t_exclude)
    row: dict[str, Any] = {
        "alpha": alpha, "true_study_peak_ug_m3": peak, "rmse_total_rel_to_peak": cast(float, ce["rmse_total"]) / peak,
        "rmse_local_rel_to_peak": cast(float, ce["rmse_local"]) / peak, "rmse_external_rel_to_peak": cast(float, ce["rmse_external"]) / peak, "rmse_total": ce["rmse_total"], "rmse_local": ce["rmse_local"], "rmse_external": ce["rmse_external"],
        "f_ext_true": ce["external_fraction_true"], "f_ext_est": ce["external_fraction_est"],
        "f_ext_error": ce["external_fraction_error"],
        "emission_rel_err_local": em.get("local"), "emission_rel_err_external": em.get("external"),
        "emission_rel_err_all": em["all_emission"],
        "rank": ident["rank"], "effective_rank": ident["effective_rank"], "condition": ident["condition"],
        "flags": ";".join(ident["flags"]), "fit_runtime_s": fit.runtime_s}
    for name, v in em["inflow"]["abs_error_ug_m3"].items():
        row[f"inflow_abs_err_ug_m3_{name}"] = abs(v)
    for j, p in enumerate(model.param_table()):
        row[f"theta_hat_{p.name}"] = float(fit.q_hat[j])
    for j, p in enumerate(truth.param_table()):
        row[f"theta_true_{p.name}"] = float(theta_true[j])
    return row


def _case_rows(run: _Run, label: str, truth: Scenario, theta_true: np.ndarray, model: Scenario, layout: SensorLayout,
               noise_fraction: float, alphas: dict[str, float], noise_seed0: int, reps: int,
               extra: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Single deterministic case: truth observed at ``layout`` with noise = noise_fraction x (this case's peak noiseless
    reading); fitted by ``model`` at each alpha for ``reps`` noise draws.  Returns (rows, details of the first fit)."""
    times = _obs_times(run.cfg)
    truth_resp = _get_resp(run.cache, truth, run.solver)
    truth_da = run_forward(FIPY, truth, theta_true, run.solver).concentration
    truth_contrib = contributions(truth_resp, theta_true, _zero_parts(truth_resp))
    scale = float(np.max(sample(truth_da, layout, times).values))
    sigma = noise_fraction * scale
    rows: list[dict[str, Any]] = []
    for r in range(reps):
        obs = observe(truth_da, layout, ObservationSpec(times, sigma, noise_seed0 + r))
        for role, a in alphas.items():
            row = _fit_and_score(run, model, truth, theta_true, truth_contrib, obs, a)
            row.update({"case": label, "alpha_role": role, "rep": r, "noise_seed": noise_seed0 + r,
                        "noise_fraction": noise_fraction, "signal_scale_ug_m3": scale, "sigma_ug_m3": sigma,
                        "sigma_pct_of_scale": 100.0 * sigma / scale if scale > 0 else None, **(extra or {})})
            rows.append(row)
    return rows, {"signal_scale_ug_m3": scale, "sigma_ug_m3": sigma}


def _special_alphas(run: _Run, f: float) -> dict[str, float]:
    sp = run.cfg["special"]
    out = {"none": 0.0}
    a_sel = run.alpha_for(f)
    if a_sel is not None:
        out["selected"] = a_sel
    out["strong"] = float(sp["alpha_strong"])
    return out


# ------------------------------------------------------------------ study 1: fields
def _study_fields(run: _Run) -> None:
    cfg, fl = run.cfg, run.cfg["fields"]
    names = list(fl["scenarios"])
    d1 = run.sub("fields")
    write_csv(run.sub("scenarios") / "param_table.csv", param_table(cfg, names))
    contribs: dict[str, dict[str, xr.DataArray]] = {}
    scns: dict[str, Scenario] = {}
    summary, ts_rows, arr_rows = [], [], []
    points = fl["arrival_points"]
    for nm in names:
        _say(run.progress, f"fields: scenario {nm}")
        scn, theta, _ = build_scenario(cfg, nm, "A")
        resp = _get_resp(run.cache, scn, run.solver)
        c = contributions(resp, theta, _zero_parts(resp))
        contribs[nm], scns[nm] = c, scn
        st = scn.study
        assert isinstance(st, StudyRegion)
        of = origin_fractions(c, scn, run.t_exclude, EPS_REL)
        w = study_weights(scn)
        assert w is not None
        times, fts = fraction_timeseries(c, scn)
        for t, f in zip(times, fts):
            ts_rows.append({"scenario": nm, "time_s": float(t), "concentration_external_fraction": f})
        row: dict[str, Any] = {"scenario": nm, "n_params": scn.n_params, "origin_fraction_local": of["local"],
                               "origin_fraction_external": of["external"],
                               "concentration_external_fraction": of["external"],
                               "peak_total_study_ug_m3": float(_arr(total_field(c)).max())}
        for tag, da in c.items():
            row[f"study_mean_{tag}_ug_m3"] = float(np.einsum("tyx,yx->", _arr(da), w) / w.sum() / len(times))
        summary.append(row)
        ext = external_part(c)
        floor = 1e-3 * float(_arr(total_field(c)).max())
        for k, (px, py) in enumerate(points):
            lay = SensorLayout((f"P{k}",), (px + st.x0,), (py + st.y0,))
            tr = sample(ext, lay, times).values[:, 0]
            arr_rows.append({"scenario": nm, "point": f"P{k}", "x_study": px, "y_study": py,
                             "arrival_time_s": arrival_time(np.asarray(tr, dtype=float), times, float(fl["arrival_threshold"]), floor),
                             "max_external_ug_m3": float(tr.max())})
    write_csv(d1 / "scenario_summary.csv", pd.DataFrame(summary))
    write_csv(d1 / "fraction_timeseries.csv", pd.DataFrame(ts_rows))
    write_csv(d1 / "arrival_times.csv", pd.DataFrame(arr_rows))
    # wind-direction sweep (temporary caches: only the study-weighted fractions are kept)
    n_dir = int(fl["sweep_n_directions"])
    sweep = []
    for nm in fl["sweep_scenarios"]:
        for k in range(n_dir):
            deg = 360.0 * k / n_dir
            _say(run.progress, f"fields: wind sweep {nm} toward {deg:.0f} deg")
            scn, theta, _ = build_scenario(cfg, nm, "A", uniform_wind(float(cfg["wind"]["speed"]), deg))
            tmp = ResponseCache()
            resp = _get_resp(tmp, scn, run.solver)
            c = contributions(resp, theta, _zero_parts(resp))
            of = origin_fractions(c, scn, run.t_exclude, EPS_REL)
            sweep.append({"scenario": nm, "toward_deg": deg, "concentration_external_fraction": of["external"],
                          "local_fraction": of["local"]})
    write_csv(d1 / "wind_direction_sweep.csv", pd.DataFrame(sweep))
    _fields_figures(run, contribs, scns, pd.DataFrame(ts_rows), pd.DataFrame(arr_rows), pd.DataFrame(sweep))


def _study_rect(ax: Any, scn: Scenario) -> None:
    st = scn.study
    assert isinstance(st, StudyRegion)
    ax.add_patch(Rectangle((st.x0, st.y0), st.x1 - st.x0, st.y1 - st.y0, fill=False, ec="k", lw=1.2))


def _fields_figures(run: _Run, contribs: dict[str, dict[str, xr.DataArray]], scns: dict[str, Scenario],
                    ts: Any, arr: Any, sweep: Any) -> None:
    fl = run.cfg["fields"]
    nm = fl["snapshot_scenario"]
    if nm in contribs:
        c, scn = contribs[nm], scns[nm]
        tags = [t for t in ("local", "external_sources", "boundary") if float(np.abs(_arr(c[t])).max()) > 0] + ["total"]
        times = [float(t) for t in fl["snapshot_times_s"]]
        fig, axes = plt.subplots(len(tags), len(times), figsize=(3.4 * len(times), 3.0 * len(tags)), squeeze=False)
        vmax = float(_arr(total_field(c)).max())
        d = scn.domain
        im = None
        for i, tag in enumerate(tags):
            da = total_field(c) if tag == "total" else c[tag]
            for j, t in enumerate(times):
                ax = axes[i][j]
                im = ax.imshow(da.sel(time=t, method="nearest").values, origin="lower", extent=(0, d.lx, 0, d.ly),
                               vmin=0, vmax=vmax, cmap="viridis")
                _study_rect(ax, scn)
                ax.set_title(f"{tag}, t = {t:.0f} s", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), label="ug/m3", shrink=0.8)
        fig.suptitle(f"Contribution fields by origin tag, scenario '{nm}' (black box = study region)")
        run.save_fig(fig, "fields_snapshots.png")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for s, g in ts.groupby("scenario"):
        v = pd.to_numeric(g["concentration_external_fraction"], errors="coerce")
        ax.plot(g["time_s"], v, label=str(s))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("external fraction (concentration)")
    ax.set_title("Study-weighted external fraction vs time (gaps = undefined)")
    ax.legend(fontsize=7)
    run.save_fig(fig, "external_fraction_timeseries.png")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for (s, p), g in arr.groupby(["scenario", "point"]):
        if s in ("external_sources", "mixed"):
            ax.plot([f"{s}\n{p}"], [g["arrival_time_s"].iloc[0] if g["arrival_time_s"].notna().any() else np.nan], "o")
    ax.set_ylabel("arrival time of external tag (s)")
    ax.set_title("First time external concentration exceeds 5% of its maximum")
    ax.tick_params(axis="x", labelsize=6, rotation=45)
    run.save_fig(fig, "arrival_times.png")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for s, g in sweep.groupby("scenario"):
        ax.plot(g["toward_deg"], pd.to_numeric(g["concentration_external_fraction"], errors="coerce"), "o-", label=str(s))
    ax.set_xlabel("wind blows toward (deg, CCW from east)")
    ax.set_ylabel("external fraction (study-weighted, time-integrated)")
    ax.legend()
    run.save_fig(fig, "wind_direction_sweep.png")


# ------------------------------------------------------------------ study 2: recovery
@dataclass(frozen=True, eq=False)
class _Built:
    case: Case
    resp: Responses
    truth_da: xr.DataArray  # direct forward run (what the sensors see)
    truth_total: xr.DataArray  # theta applied to the cached responses (what the metrics compare against)
    contrib: dict[str, xr.DataArray]
    clean: dict[str, np.ndarray]  # layout -> noiseless readings (nt, ns)
    build_s: float
    super_rel_diff: float


def _build_ensemble(run: _Run, cases: list[Case], layouts: dict[str, SensorLayout]) -> list[_Built]:
    times = _obs_times(run.cfg)
    out = []
    for c in cases:
        t0 = time.perf_counter()
        resp = _get_resp(run.cache, c.scenario, run.solver)
        truth_da = run_forward(FIPY, c.scenario, c.theta, run.solver).concentration
        total = resp.field(c.theta)
        diff = float(np.max(np.abs(_arr(truth_da) - _arr(total))) / max(float(np.max(_arr(truth_da))), 1e-300))
        contrib = contributions(resp, c.theta, _zero_parts(resp))
        clean = {n: np.asarray(sample(truth_da, lay, times).values, dtype=float) for n, lay in layouts.items()}
        out.append(_Built(c, resp, truth_da, total, contrib, clean, time.perf_counter() - t0, diff))
    return out


def _signal_scale(built: list[_Built], scale_layout: str) -> float:
    return float(np.median([b.clean[scale_layout].max() for b in built]))


def _noise_table(role: str, scale: float, scale_layout: str, fractions: list[float]) -> list[dict[str, Any]]:
    return [{"ensemble": role, "noise_fraction": f, "signal_scale_ug_m3": scale, "scale_layout": scale_layout,
             "sigma_ug_m3": f * scale, "sigma_pct_of_scale": 100.0 * (f * scale) / scale} for f in fractions]


def _observation(b: _Built, layout: SensorLayout, sigma: float, seed: int, run: _Run) -> Observations:
    return observe(b.truth_da, layout, ObservationSpec(_obs_times(run.cfg), sigma, seed))


def _select_alpha(raw: Any) -> tuple[float, pd.DataFrame]:
    """Mean total-field regional RMSE per alpha; lowest wins, an exact tie goes to the smaller alpha."""
    tab = raw.groupby("alpha")["rmse_total"].agg(["mean", "median", "count"]).reset_index()
    tab = tab.rename(columns={"mean": "criterion_mean_rmse_total", "median": "median_rmse_total", "count": "n_rows"})
    best = min((math.inf if not math.isfinite(c) else float(c), float(a))
               for a, c in zip(tab["alpha"], tab["criterion_mean_rmse_total"]))
    return best[1], tab


def _study_recovery(run: _Run) -> None:
    cfg, rec = run.cfg, run.cfg["recovery"]
    d2 = run.sub("recovery")
    base, theta0, labels = build_scenario(cfg, rec["scenario"], "A")
    layouts = _layouts(cfg, base)
    fractions = [float(f) for f in rec["noise_fractions"]]
    reps, scale_layout = int(rec["noise_reps"]), rec["scale_layout"]
    grid = sorted(float(a) for a in rec["alpha_grid"])
    seeds = {"validation_seed": int(rec["validation_seed"]), "test_seed": int(rec["test_seed"]),
             "validation_noise_seed": int(rec["validation_noise_seed"]), "test_noise_seed": int(rec["test_noise_seed"])}
    val_cases = draw_cases(cfg, base, theta0, int(rec["n_validation"]), seeds["validation_seed"], "val")
    test_cases = draw_cases(cfg, base, theta0, int(rec["n_test"]), seeds["test_seed"], "test")
    check_ensembles(val_cases, test_cases, seeds["validation_seed"], seeds["test_seed"], seeds["validation_noise_seed"],
                    seeds["test_noise_seed"], reps)
    st = base.study
    assert isinstance(st, StudyRegion)
    write_csv(d2 / "layouts.csv", pd.DataFrame([{"layout": n, "sensor": i, "x_study": x - st.x0, "y_study": y - st.y0}
                                                 for n, lay in layouts.items() for i, x, y in zip(lay.ids, lay.x, lay.y)]))
    # ---- validation: the ONLY data used to choose alpha
    _say(run.progress, f"recovery: building {len(val_cases)} validation cases")
    t0 = time.perf_counter()
    vb = _build_ensemble(run, val_cases, layouts)
    run.tim["recovery_validation_build_s"] = time.perf_counter() - t0
    run.tim["seconds_per_ensemble_case"] = run.tim["recovery_validation_build_s"] / max(len(vb), 1)
    vscale = _signal_scale(vb, scale_layout)
    w = study_weights(base)
    t0 = time.perf_counter()
    vrows = []
    for f in fractions:
        sigma = f * vscale
        for i, b in enumerate(vb):
            for ln, lay in layouts.items():
                for r in range(reps):
                    obs = _observation(b, lay, sigma, seeds["validation_noise_seed"] + i * reps + r, run)
                    for a in grid:
                        fit = fit_sources(b.case.scenario, obs, InferenceOptions(alpha=a, solver=run.solver), run.cache)
                        vrows.append({"ensemble": "validation", "case": b.case.name, "layout": ln, "noise_fraction": f,
                                      "rep": r, "alpha": a,
                                      "rmse_total": field_rmse(b.resp.field(fit.q_hat), b.truth_total, w, run.t_exclude)})
        _say(run.progress, f"recovery: validation noise fraction {f:g} done")
    vraw = pd.DataFrame(vrows)
    run.tim["recovery_validation_fits_s"] = time.perf_counter() - t0
    selected: dict[str, Any] = {}
    tables = {}
    for f in fractions:
        a, tab = _select_alpha(vraw[vraw["noise_fraction"] == f])
        selected[f"{f:g}"] = a
        tables[f"{f:g}"] = tab.to_dict("records")
    run.alpha_by_noise = {k: float(v) for k, v in selected.items()}
    write_csv(d2 / "validation_raw.csv", vraw)
    write_csv(d2 / "noise_levels_validation.csv", pd.DataFrame(_noise_table("validation", vscale, scale_layout, fractions)))
    # freeze BEFORE the test ensemble is built or evaluated
    write_json(d2 / "alpha_selected.json", {
        "alpha_by_noise_fraction": selected,
        "criterion": "mean over validation cases, layouts and noise reps of the total-field regional RMSE "
                     "(study-region area weights, time window [t_exclude, T]); lowest wins, exact tie -> smaller alpha",
        "alpha_grid": grid, "alpha_units": "ridge strength in scaled units (alpha * ||theta/q_scale||^2 on whitened residuals)",
        "prior": "zeros (prior mean 0, no truth)", "tables": tables, "validation_cases": [c.name for c in val_cases],
        "validation_signal_scale_ug_m3": vscale, "seeds": seeds, "noise_reps": reps, "scenario": rec["scenario"],
        "layouts": list(layouts), "note": "chosen on validation cases only; the test ensemble was not built yet"})
    frozen = json.loads((d2 / "alpha_selected.json").read_text())["alpha_by_noise_fraction"]
    # ---- test: fresh seed, never used for any choice
    _say(run.progress, f"recovery: building {len(test_cases)} test cases")
    t0 = time.perf_counter()
    tb = _build_ensemble(run, test_cases, layouts)
    run.tim["recovery_test_build_s"] = time.perf_counter() - t0
    tscale = _signal_scale(tb, scale_layout)
    write_csv(d2 / "noise_levels_test.csv", pd.DataFrame(_noise_table("test", tscale, scale_layout, fractions)))
    write_csv(d2 / "cases.csv", pd.DataFrame([
        {"ensemble": role, "case": b.case.name, "wind": json.dumps([[s.t_start, s.u, s.v] for s in b.case.scenario.transport.wind]),
         **{f"theta_true_{p['name']}": float(b.case.theta[j]) for j, p in enumerate(labels)},
         "build_s": b.build_s, "direct_vs_superposition_rel_diff": b.super_rel_diff}
        for role, bl in (("validation", vb), ("test", tb)) for b in bl]))
    t0 = time.perf_counter()
    trows = []
    for f in fractions:
        sigma = f * tscale
        alphas = {"selected": float(frozen[f"{f:g}"]), "none": 0.0, "strong": float(rec["alpha_strong"])}
        for i, b in enumerate(tb):
            for ln, lay in layouts.items():
                for r in range(reps):
                    seed = seeds["test_noise_seed"] + i * reps + r
                    obs = _observation(b, lay, sigma, seed, run)
                    for role, a in alphas.items():
                        row = _fit_and_score(run, b.case.scenario, b.case.scenario, b.case.theta, b.contrib, obs, a)
                        row.update({"ensemble": "test", "case": b.case.name, "layout": ln, "n_sensors": len(lay),
                                    "noise_fraction": f, "signal_scale_ug_m3": tscale, "sigma_ug_m3": sigma,
                                    "sigma_pct_of_scale": 100.0 * sigma / tscale, "rep": r, "noise_seed": seed,
                                    "alpha_role": role})
                        trows.append(row)
        _say(run.progress, f"recovery: test noise fraction {f:g} done")
    traw = pd.DataFrame(trows)
    run.tim["recovery_test_fits_s"] = time.perf_counter() - t0
    write_csv(d2 / "test_raw.csv", traw)
    ss = rec.get("scale_sensitivity", {})
    if ss.get("enabled", True):
        _scale_sensitivity(run, tb, layouts[ss.get("layout", "interior8")], ss.get("layout", "interior8"), tscale, frozen,
                           [float(x) for x in ss.get("factors", [0.1, 10.0])], fractions, reps, seeds["test_noise_seed"], d2)
    summ, paired = summarise_recovery(traw, [tuple(p) for p in rec["paired"]])
    write_csv(d2 / "summary.csv", summ)
    write_csv(d2 / "paired.csv", paired)
    _recovery_figures(run, pd.DataFrame([{"noise_fraction": float(k), **r} for k, t in tables.items() for r in t]), summ, selected)


def _scale_sensitivity(run: _Run, tb: list[_Built], layout: SensorLayout, layout_name: str, tscale: float,
                       frozen: dict[str, Any], factors: list[float], fractions: list[float], reps: int, noise_seed0: int,
                       d2: Path) -> None:
    """TEST ensemble only: refit ``layout_name`` at each noise level with the emission and then the inflow q_scale
    multiplied by each factor (alpha as selected at the baseline scales, NOT reselected).  Same noise draws as the main
    test fits, so the baseline rows reproduce the 'selected' rows of ``test_raw.csv``."""
    _say(run.progress, "recovery: scale sensitivity")
    variants = [("baseline", 1.0, 1.0)] + [(f"{kind}_x{fac:g}", fac if kind == "emission" else 1.0,
                                           fac if kind == "inflow" else 1.0)
                                          for kind in ("inflow", "emission") for fac in factors]
    rows = []
    for f in fractions:
        sigma, alpha = f * tscale, float(frozen[f"{f:g}"])
        for i, b in enumerate(tb):
            for r in range(reps):
                obs = _observation(b, layout, sigma, noise_seed0 + i * reps + r, run)
                for name, em, inf in variants:
                    row = _fit_and_score(run, b.case.scenario, b.case.scenario, b.case.theta, b.contrib, obs, alpha, em, inf)
                    row.update({"variant": name, "emission_scale_mult": em, "inflow_scale_mult": inf, "layout": layout_name,
                                "case": b.case.name, "rep": r, "noise_fraction": f, "alpha_role": "selected"})
                    rows.append(row)
    write_csv(d2 / "scale_sensitivity.csv", pd.DataFrame(rows))


RECOVERY_METRICS = ("rmse_total", "rmse_local", "rmse_external", "f_ext_error", "emission_rel_err_local",
                    "emission_rel_err_external")
SD_LABEL = "sd (standard deviation across rows, not a confidence interval)"


def _num(x: Any) -> Any:
    return pd.to_numeric(x, errors="coerce")


def summarise_recovery(raw: pd.DataFrame, pairs: list[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (layout, noise_fraction, alpha_role): n, mean, median, sd of each metric (|f_ext_error| for the fraction);
    paired differences (second layout minus first) on identical (case, rep, noise_fraction, alpha_role) rows."""
    df: Any = raw.copy()
    metrics = list(RECOVERY_METRICS) + sorted(c for c in raw.columns if c.startswith("inflow_abs_err_ug_m3_"))
    df["f_ext_error"] = _num(df["f_ext_error"]).abs()
    rows = []
    for key, g in df.groupby(["layout", "noise_fraction", "alpha_role"]):  # key: tuple
        row: dict[str, Any] = dict(zip(["layout", "noise_fraction", "alpha_role"], key))
        row["n_rows"] = len(g)
        for m in metrics:
            v = _num(g[m])
            row.update({f"{m}_mean": v.mean(), f"{m}_median": v.median(), f"{m}_sd": v.std(ddof=1) if v.notna().sum() > 1 else np.nan,
                        f"{m}_n_defined": int(v.notna().sum())})
        rows.append(row)
    prow = []
    keys = ["case", "rep", "noise_fraction", "alpha_role"]
    for a, b in pairs:
        ga, gb = df[df["layout"] == a].set_index(keys), df[df["layout"] == b].set_index(keys)
        j = ga.join(gb, lsuffix="_a", rsuffix="_b", how="inner")
        for (f, role), g in j.groupby(["noise_fraction", "alpha_role"]):
            row = {"layout_a": a, "layout_b": b, "noise_fraction": f, "alpha_role": role, "n_pairs": len(g),
                   "n_sensors_a": int(g["n_sensors_a"].iloc[0]), "n_sensors_b": int(g["n_sensors_b"].iloc[0]),
                   "equal_budget": bool(g["n_sensors_a"].iloc[0] == g["n_sensors_b"].iloc[0]), "spread_label": SD_LABEL}
            for m in metrics:
                d = _num(g[f"{m}_b"]) - _num(g[f"{m}_a"])
                row.update({f"{m}_diff_mean": d.mean(), f"{m}_diff_sd": d.std(ddof=1) if d.notna().sum() > 1 else np.nan,
                            f"{m}_n_b_lower": int((d < 0).sum())})
            prow.append(row)
    return pd.DataFrame(rows), pd.DataFrame(prow)


def _recovery_figures(run: _Run, alpha_tab: Any, summ: Any, selected: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for f, g in alpha_tab.groupby("noise_fraction"):
        g = g.sort_values("alpha")
        x = np.where(g["alpha"] > 0, g["alpha"], 1e-5)
        ax.plot(x, g["criterion_mean_rmse_total"], "o-", label=f"noise {f:g} (selected {selected[f'{f:g}']:g})")
    ax.set_xscale("log")
    ax.set_xlabel("alpha (0 plotted at 1e-5)")
    ax.set_ylabel("validation criterion: mean total-field RMSE (ug/m3)")
    ax.legend(fontsize=8)
    run.save_fig(fig, "recovery_alpha_selection.png")
    s = summ[summ["alpha_role"] == "selected"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    for ax, f in zip(axes, sorted(s["noise_fraction"].unique())):
        g = s[s["noise_fraction"] == f]
        xs = np.arange(len(g))
        for k, m in enumerate(("rmse_total", "rmse_local", "rmse_external")):
            ax.bar(xs + 0.27 * (k - 1), g[f"{m}_mean"], 0.27, yerr=g[f"{m}_sd"], label=m, capsize=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(g["layout"], fontsize=7)
        ax.set_title(f"noise = {f:g} of signal scale (test, selected alpha)", fontsize=8)
        ax.set_ylabel("RMSE (ug/m3), bars = mean, whiskers = sd")
    axes[0].legend(fontsize=7)
    run.save_fig(fig, "recovery_rmse_by_layout.png")


# ------------------------------------------------------------------ study 3: special cases
def verify_no_arrival(contrib: dict[str, xr.DataArray], scn: Scenario, max_ratio: float) -> float:
    """Ratio (study-mean, time-mean external concentration) / (same for local) from the MODEL; ValueError unless it is
    below ``max_ratio`` (the no-arrival property is verified, never assumed)."""
    w = study_weights(scn)
    assert w is not None
    ext = float(np.einsum("tyx,yx->", _arr(external_part(contrib)), w) / w.sum())
    loc = float(np.einsum("tyx,yx->", _arr(contrib["local"]), w) / w.sum())
    ratio = ext / loc
    if not ratio < max_ratio:
        raise ValueError(f"no_arrival scenario: the external source DOES reach the study region (ratio {ratio:.3g})")
    return ratio



def _study_special(run: _Run) -> None:
    cfg, sp = run.cfg, run.cfg["special"]
    d3 = run.sub("special")
    layouts = _layouts(cfg, build_scenario(cfg, "mixed", "A")[0])
    layout = layouts[sp["layout"]]
    cl = sp["confounded_layout"]
    cfracs = [float(x) for x in sp["confounded_noise_fractions"]]
    c_layout = SensorLayout(tuple(f"C{i + 1}" for i in range(len(cl))), tuple(float(p[0]) + _origin(cfg) for p in cl),
                            tuple(float(p[1]) + _origin(cfg) for p in cl))
    f, reps, seed0 = float(sp["noise_fraction"]), int(sp["noise_reps"]), int(sp["noise_seed"])
    alphas = _special_alphas(run, f)
    details: dict[str, Any] = {"alphas": alphas, "noise_fraction": f, "layout": sp["layout"],
                               "alpha_default": sp["alpha_default"]}
    nxt = [0]

    def next_seed() -> int:
        s = seed0 + nxt[0] * reps
        nxt[0] += 1
        return s

    # --- confounded
    _say(run.progress, "special: confounded")
    scn, theta, _ = build_scenario(cfg, "confounded", "A")
    rows = []
    by_noise: dict[str, Any] = {}
    truth_da = run_forward(FIPY, scn, theta, run.solver).concentration
    for cf in cfracs:
        cseed = next_seed()
        r_, det = _case_rows(run, "confounded", scn, theta, scn, c_layout, cf, _special_alphas(run, cf), cseed, reps)
        rows += r_
        # diagnostics of the FIRST noise draw of THIS noise level (alpha = 0); one entry per noise level
        fit0 = fit_sources(scn, observe(truth_da, c_layout, ObservationSpec(_obs_times(cfg), det["sigma_ug_m3"], cseed)),
                           InferenceOptions(alpha=0.0, solver=run.solver), run.cache)
        by_noise[f"{cf:g}"] = {**det, "noise_seed": cseed, "identifiability_alpha0": identifiability_report(fit0),
                               "column_cosines": fit0.column_cosines.tolist(), "param_names": list(fit0.param_names)}
    conf: Any = pd.DataFrame(rows)
    write_csv(d3 / "confounded.csv", conf)
    details["confounded"] = {"by_noise_fraction": by_noise}
    # --- no arrival (verified with the model)
    _say(run.progress, "special: no_arrival")
    scn, theta, labels = build_scenario(cfg, "no_arrival", "A")
    resp = _get_resp(run.cache, scn, run.solver)
    c = contributions(resp, theta, _zero_parts(resp))
    ratio = verify_no_arrival(c, scn, float(sp["no_arrival_max_ratio"]))
    rows, det = _case_rows(run, "no_arrival", scn, theta, scn, layout, f, alphas, next_seed(), reps)
    write_csv(d3 / "no_arrival.csv", pd.DataFrame(rows))
    details["no_arrival"] = {**det, "external_to_local_study_mean_ratio": ratio, "max_allowed_ratio": sp["no_arrival_max_ratio"],
                             "verified_by": "forward model (unit-source responses), not by assertion"}
    # --- wind mismatch: truth wind rotated / scaled, inference declares the nominal wind
    _say(run.progress, "special: wind mismatch")
    wm = sp["wind_mismatch"]
    model, theta, _ = build_scenario(cfg, wm["scenario"], "A")
    rows = []
    for tag, rot, spd in (("matched", 0.0, 0.0), ("mismatched", float(wm["rotation_deg"]), float(wm["speed_error"]))):
        truth = perturb_wind(model, spd, rot)
        r, det = _case_rows(run, f"wind_{tag}", truth, theta, model, layout, f, alphas, next_seed(), reps,
                            {"wind_rotation_deg": rot, "wind_speed_error": spd})
        rows += r
    write_csv(d3 / "wind_mismatch.csv", pd.DataFrame(rows))
    # --- prior sensitivity
    _say(run.progress, "special: prior sensitivity")
    rows = []
    pal = {f"{a:g}": float(a) for a in sp["prior_alphas"]}
    for nm in sp["prior_scenarios"]:
        scn, theta, _ = build_scenario(cfg, nm, "A")
        r, _ = _case_rows(run, f"prior_{nm}", scn, theta, scn, c_layout if nm == "confounded" else layout, f, pal, next_seed(), reps, {"scenario": nm})
        rows += r
    prior: Any = pd.DataFrame(rows)
    write_csv(d3 / "prior_sensitivity.csv", prior)
    # --- omitted / unmodelled external source
    _say(run.progress, "special: omitted source")
    rows = []
    for om in sp["omitted"]:
        truth, theta, _ = build_scenario(cfg, om["truth"], "A")
        model, _, _ = build_scenario(cfg, om["model"], "A")
        r, _ = _case_rows(run, f"omitted_{om['truth']}_by_{om['model']}", truth, theta, model, layout, f, alphas,
                          next_seed(), reps, {"truth_scenario": om["truth"], "model_scenario": om["model"]})
        rows += r
    write_csv(d3 / "omitted_source.csv", pd.DataFrame(rows))
    write_json(d3 / "diagnostics.json", details)
    # figures
    sel = conf[(conf["alpha_role"] == "none") & (conf["noise_fraction"] == max(cfracs))]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    cols = ["rmse_total", "rmse_local", "rmse_external"]
    ax.bar(cols, [sel[c].mean() for c in cols], yerr=[sel[c].std(ddof=1) if len(sel) > 1 else 0 for c in cols], capsize=3)
    ax.set_ylabel("RMSE over study region (ug/m3)")
    ax.set_title(f"Confounded case, alpha = 0, noise {max(cfracs):g}: total vs component RMSE\n(bars = mean, whiskers = sd)", fontsize=9)
    run.save_fig(fig, "special_confounded.png")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for nm, g in prior.groupby("scenario"):
        m = g.groupby("alpha")[["rmse_total", "rmse_local", "rmse_external"]].mean().reset_index()
        x = np.where(m["alpha"] > 0, m["alpha"], 1e-3)
        ax.plot(x, m["rmse_total"], "o-", label=f"{nm}: total")
        ax.plot(x, m["rmse_external"], "s--", label=f"{nm}: external")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("alpha (0 plotted at 1e-3)")
    ax.set_ylabel("RMSE (ug/m3)")
    ax.legend(fontsize=7)
    run.save_fig(fig, "special_prior_sensitivity.png")


# ------------------------------------------------------------------ study 4: domain size + boundary-basis equivalence
def _study_domain(run: _Run) -> None:
    cfg, ds = run.cfg, run.cfg["domain_size"]
    d4 = run.sub("domain_size")
    rows = []
    fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for nm in ds["scenarios"]:
        _say(run.progress, f"domain_size: {nm} on A and B")
        blocks, ext_blocks, fe = {}, {}, {}
        for dom in ("A", "B"):
            scn, theta, _ = build_scenario(cfg, nm, dom)
            scn_has_bases = bool(scn.boundary.bases)
            resp = _get_resp(run.cache, scn, run.solver)
            c = contributions(resp, theta, _zero_parts(resp))
            blocks[dom] = _study_block(total_field(c), scn)
            ext_blocks[dom] = _study_block(external_part(c), scn)
            fe[dom] = origin_fractions(c, scn, run.t_exclude, EPS_REL)["external"]
            times = np.asarray(c["local"]["time"].values, dtype=float)
        peak = float(blocks["B"].max())
        rm = _block_rmse(blocks["A"], blocks["B"], times, run.t_exclude)
        re = _block_rmse(ext_blocks["A"], ext_blocks["B"], times, run.t_exclude)
        rows.append({"scenario": nm, "rmse_total_A_vs_B": rm, "peak_B_study_ug_m3": peak, "rmse_total_rel_to_peak": rm / peak,
                     "rmse_external_A_vs_B": re, "external_fraction_A": fe["A"], "external_fraction_B": fe["B"],
                     "note": ("same sources and study region; the open boundary is 5 km (A) vs 10 km (B) from it: pure domain "
                              "truncation" if not scn_has_bases else
                              "same sources and study region, but the boundary basis sits at the domain edge (5 km vs 10 km "
                              "away): truncation mixed with a moved forcing")})
        fields[nm] = (blocks["A"], blocks["B"])
    write_csv(d4 / "field_difference.csv", pd.DataFrame(rows))
    fig, axes = plt.subplots(len(fields), 3, figsize=(10, 3.2 * len(fields)), squeeze=False)
    for i, (nm, (a, b)) in enumerate(fields.items()):
        vmax = float(b[-1].max())
        for j, (arr_, ttl, vm) in enumerate(((a[-1], "domain A", vmax), (b[-1], "domain B", vmax), (a[-1] - b[-1], "A - B", None))):
            ax = axes[i][j]
            im = ax.imshow(arr_, origin="lower", cmap="viridis" if vm else "RdBu_r",
                           vmin=0 if vm else -float(np.abs(a[-1] - b[-1]).max() or 1.0), vmax=vm or float(np.abs(a[-1] - b[-1]).max() or 1.0))
            ax.set_title(f"{nm}: {ttl}, study region, t = T", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.7)
    run.save_fig(fig, "domain_size_fields.png")
    eq = ds["equivalence"]
    if eq.get("enabled", True):
        _equivalence(run, eq)


def _e1_true_onset(cfg: dict[str, Any]) -> float:
    """First time at which E1's configured emission profile is positive (its true onset)."""
    prof = _profile(cfg["sources"]["E1"].get("onset", ((0.0, 1.0),)))
    return next(t for t, v in prof if v > 0)


def _relocate(obs: Observations, layout: SensorLayout) -> Observations:
    """The same readings, declared at the sites ``layout`` (the model's coordinates of the same physical sites)."""
    vals = obs.values.assign_coords(x=("sensor", np.asarray(layout.x, dtype=float)),
                                    y=("sensor", np.asarray(layout.y, dtype=float)))
    return Observations(vals, obs.noise_std, layout, obs.times)


def _equivalence(run: _Run, eq: dict[str, Any]) -> None:
    """The SAME physical external source E1 is (i) explicit in domain B (the TRUTH) and (ii) represented by whole-side
    boundary-basis inflow in domain A, fitted to synthetic study observations of (i).  An APPROXIMATION study: E1 is
    never present in the fitted model, and each basis carries ``represents`` = E1's footprint, so no plume is counted
    twice.  The sensors sit at the same STUDY-RELATIVE sites in both domains: the observations are sampled from the
    truth with the layout shifted by B's study origin, and declared to the fit at A's coordinates of the same sites.

    The basis temporal profile is a KNOWN input: it is run at each onset of ``eq['basis_onsets_s']``; the one equal to
    E1's true onset (3600 s) gives the basis the true timing, any other onset measures the sensitivity to that
    assumption."""
    cfg = run.cfg
    d4 = run.sub("domain_size")
    _say(run.progress, "domain_size: boundary-basis equivalence")
    truth, theta, _ = build_scenario(cfg, eq["truth_scenario"], "B")
    tresp = _get_resp(run.cache, truth, run.solver)
    tc = contributions(tresp, theta, _zero_parts(tresp))
    tda = run_forward(FIPY, truth, theta, run.solver).concentration
    ref_a, ref_theta, _ = build_scenario(cfg, eq["truth_scenario"], "A")
    layout_truth = _layouts(cfg, truth)[eq["layout"]]  # study-relative sites in B coordinates (where the truth is)
    layout_fit = _layouts(cfg, ref_a)[eq["layout"]]  # the same sites in A coordinates (what the fit declares)
    times = _obs_times(cfg)
    scale = float(np.max(sample(tda, layout_truth, times).values))
    sigma = float(eq["noise_fraction"]) * scale
    a_sel = run.alpha_for(float(eq["noise_fraction"]))
    alpha, a_src = (a_sel, "frozen validation choice (recovery study)") if a_sel is not None else (
        float(eq["fallback_alpha"]), "config fallback_alpha (recovery study not run)")
    tblock = _study_block(total_field(tc), truth)
    tblock_ext = _study_block(external_part(tc), truth)
    ttimes = np.asarray(tc["local"]["time"].values, dtype=float)
    peak = float(tblock.max())
    rows = []
    fit_block = tblock[-1]
    # reference: E1 explicit in A with the true parameters (domain effect only, no fitting)
    ra = _get_resp(run.cache, ref_a, run.solver)
    rc = contributions(ra, ref_theta, _zero_parts(ra))
    rm = _block_rmse(_study_block(total_field(rc), ref_a), tblock, ttimes, run.t_exclude)
    rows.append({"model": "A_E1_explicit_true_theta", "rep": 0, "rmse_total": rm, "rmse_total_rel_to_peak": rm / peak,
                 "rmse_external": _block_rmse(_study_block(external_part(rc), ref_a), tblock_ext, ttimes, run.t_exclude),
                 "alpha": None, "note": "reference: domain effect only (E1 explicit in A, true amplitudes, no fit)"})
    e1_onset = _e1_true_onset(cfg)
    onsets = [float(x) for x in eq["basis_onsets_s"]]
    if e1_onset not in onsets:
        raise ValueError(f"basis_onsets_s must contain E1's true onset {e1_onset}")
    for bo in onsets:
        for mname, spec in eq["models"].items():
            spec = dict(spec)
            spec["represents"] = eq["represents"]
            model, _, _ = build_model(cfg, spec, "A")
            model = dataclasses.replace(model, boundary=dataclasses.replace(
                model.boundary, bases=tuple(dataclasses.replace(b, profile=((0.0, 0.0), (bo, 1.0))) for b in model.boundary.bases)))
            if any(s.name == "E1" for s in model.sources):
                raise ValueError("double count: E1 must not be an explicit source of a boundary-basis model")
            has_basis = bool(model.boundary.bases)
            if not has_basis and bo != e1_onset:
                continue  # no basis, so no onset: this model is fitted once
            resp = _get_resp(run.cache, model, run.solver)
            for r in range(int(eq["noise_reps"])):
                obs = _relocate(observe(tda, layout_truth, ObservationSpec(times, sigma, int(eq["noise_seed"]) + r)), layout_fit)
                fit = fit_sources(model, obs, InferenceOptions(alpha=float(alpha), solver=run.solver), run.cache)
                est = contributions(resp, fit.q_hat, _zero_parts(resp))
                mrm = _block_rmse(_study_block(total_field(est), model), tblock, ttimes, run.t_exclude)
                ident = identifiability_report(fit)
                row = {"model": mname, "rep": r, "basis_onset_s": bo if has_basis else None, "e1_true_onset_s": e1_onset,
                       "onset_matches_e1": (bo == e1_onset) if has_basis else None, "rmse_total": mrm, "rmse_total_rel_to_peak": mrm / peak,
                       "rmse_external": _block_rmse(_study_block(external_part(est), model), tblock_ext, ttimes, run.t_exclude),
                       "alpha": float(alpha), "alpha_source": a_src, "noise_fraction": float(eq["noise_fraction"]),
                       "sigma_ug_m3": sigma, "signal_scale_ug_m3": scale, "flags": ";".join(ident["flags"]),
                       "effective_rank": ident["effective_rank"], "condition": ident["condition"]}
                for j, p in enumerate(model.param_table()):
                    row[f"theta_hat_{p.name}"] = float(fit.q_hat[j])
                rows.append(row)
                if r == 0 and mname == "A_sector_bases" and bo == e1_onset:
                    fit_block = _study_block(total_field(est), model)[-1]
    write_csv(d4 / "boundary_basis_equivalence.csv", pd.DataFrame(rows))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, arr_, ttl in zip(axes, (tblock[-1], fit_block, fit_block - tblock[-1]), ("truth: E1 explicit in B", "A, sector bases (fit)", "fit - truth")):
        im = ax.imshow(arr_, origin="lower", cmap="RdBu_r" if "-" in ttl else "viridis")
        ax.set_title(ttl + ", t = T", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.7)
    run.save_fig(fig, "boundary_basis_equivalence.png")


# ------------------------------------------------------------------ driver
@dataclass
class TransboundaryResult:
    out_dir: Path
    alpha_by_noise: dict[str, float]
    timings: dict[str, float]
    figures: list[str]
    studies: list[str]


def run_transboundary(cfg: dict[str, Any], out_dir: Path | str, overwrite: bool = False, cache: ResponseCache | None = None,
                      progress: Progress = None) -> TransboundaryResult:
    """Run the enabled sub-studies of ``cfg`` (fields, recovery, special, domain_size) into ``out_dir`` (refused when
    non-empty unless ``overwrite``).  Writes CSV/JSON tables, figures/, config.json, provenance.json, timings.json."""
    t_all = time.perf_counter()
    check_noise_ranges(_noise_ranges(cfg))
    out = make_run_dir(out_dir, overwrite)
    run = _Run(cfg, out, cache or ResponseCache(), SolverOptions(**cfg.get("solver", {})), progress)
    write_json(out / "config.json", cfg)
    done = []
    for key, fn in (("fields", _study_fields), ("recovery", _study_recovery), ("special", _study_special),
                    ("domain_size", _study_domain)):
        if cfg[key].get("enabled", True):
            t0 = time.perf_counter()
            fn(run)
            run.tim[f"{key}_s"] = time.perf_counter() - t0
            done.append(key)
    run.tim["total_s"] = time.perf_counter() - t_all
    run.tim["response_cache_hits"] = float(run.cache.hits)
    run.tim["response_cache_misses"] = float(run.cache.misses)
    r = cfg["recovery"]
    seeds = {"design": cfg["seeds"]["design"], "validation_seed": r["validation_seed"], "test_seed": r["test_seed"],
             "validation_noise_seed": r["validation_noise_seed"], "test_noise_seed": r["test_noise_seed"],
             "special_noise_seed": cfg["special"]["noise_seed"], "equivalence_noise_seed": cfg["domain_size"]["equivalence"]["noise_seed"]}
    write_json(out / "provenance.json", provenance(cfg, seeds, {
        "experiment": "transboundary", "studies": done, "alpha_by_noise_fraction": run.alpha_by_noise,
        "note": "synthetic demonstration, not an estimate for any place; tags never reach fit_sources"}))
    write_json(out / "timings.json", run.tim)
    return TransboundaryResult(out, run.alpha_by_noise, run.tim, run.figures, done)

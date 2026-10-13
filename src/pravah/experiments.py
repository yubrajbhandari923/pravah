"""Sensor-placement experiments: scenario ensembles, alpha selection, the design benchmark and the mismatch study.

Config sections used (see configs/benchmark.json, configs/mismatch.json):
* ``scenario``, ``q`` (nominal amplitudes), ``solver``, ``inverse.noise_std`` / ``inverse.sensor_interval``;
* ``design``: candidates_n, budgets, n_design, n_test, n_validation, noise_reps, n_random, weighted, min_separation,
  heldout_n, alpha, t_exclude, the explicit seeds design_seed, test_seed, validation_seed, design_noise_seed,
  test_noise_seed, validation_noise_seed, layout_seed, and alpha_grid / alpha_selection_file (see below);
* ``mismatch`` (run_mismatch only): refine, truth_dt_factor, wind_speed_error, wind_rotation_deg, extra_source,
  poor_layout_sites, layouts_from.

Honesty rules.  Three disjoint scenario ensembles exist: 'design' (greedy layout selection), 'validation' (choice of
the Tikhonov strength alpha, ``run_alpha_selection``) and 'test' (every reported error).  Layouts are SELECTED only
from role='design' cases (design.py enforces it); every reported error comes from independent role='test' cases
with fresh noise.  alpha is chosen once on the validation ensemble, frozen in ``alpha_selected.json``, and then read
by ``run_benchmark`` / ``run_mismatch`` through ``design.alpha_selection_file`` (design.alpha null = take it from the
file; if both are given they must agree).  A config with alpha set and no selection file (e.g. the alpha = 0
sensitivity run, or the smoke config) is labelled 'not selected: sensitivity' in provenance.  Random layouts are never selected; their design-stage
score is computed only as a diagnostic.  All numbers are synthetic and idealised.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

from .runner import Progress, say as _say
from .config import (GaussianSource, Scenario, SolverOptions, Times, Transport, WindSegment, scenario_from_dict,
                     to_dict)
from .design import (Candidates, Constraints, DesignCase, DesignOptions, Layout, assert_disjoint, build_case,
                     candidate_grid, evaluate_layouts, evaluate_validation, fit_layout, greedy_trace, paired_comparison, random_layouts,
                     score_layout, spread_layout)
from .forward import FIPY, ForwardSolver, forward_info, get_forward
from .inverse import ResponseCache, run_forward
from .io import make_run_dir, provenance, write_csv, write_json
from .metrics import population_weight


WIND_SPEED_RANGE = (0.6, 1.6)  # m/s
WIND_DIRECTION_SPREAD_DEG = 40.0  # +/- around the nominal direction
Q_FACTOR_RANGE = (0.5, 1.5)
SD_LABEL = "sd of per-row RMSE across layouts x cases x noise reps"
MISMATCH_KINDS = ("refined_truth", "wind_perturbed", "extra_source")
DESIGN_DEFAULTS: dict[str, Any] = {
    "candidates_n": 10, "budgets": [1, 2, 3, 5, 8], "n_design": 6, "n_test": 6, "noise_reps": 3, "n_random": 20,
    "weighted": False, "min_separation": 0.0, "heldout_n": 7, "alpha": 0.0, "t_exclude": 0.0,
    "design_seed": 11, "test_seed": 33, "design_noise_seed": 1000, "test_noise_seed": 7000, "layout_seed": 7,
    "n_validation": 6, "validation_seed": 44, "validation_noise_seed": 9000,
    "alpha_grid": [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0], "alpha_selection_file": None}
SEED_KEYS = ("design_seed", "test_seed", "validation_seed", "design_noise_seed", "test_noise_seed",
             "validation_noise_seed", "layout_seed")
DEFAULT_Q_SCALE = 1e5  # ug/s; the default of inverse.InferenceOptions.q_scale, recorded in alpha_selected.json
NO_EMISSION_FACTOR = 10.0  # blow-up count: rows with RMSE > 10 x the RMSE of the zero-emission estimate


# ------------------------------------------------------------------ scenario ensembles
@dataclass(frozen=True)
class CaseSpec:
    """One ensemble member: the declared scenario (its wind is the truth wind) and its source amplitudes [ug/s]."""
    name: str
    scenario: Scenario
    q: tuple[float, ...]


def _draw_wind(rng: np.random.Generator, theta0: float) -> tuple[float, float]:
    speed = rng.uniform(*WIND_SPEED_RANGE)
    theta = theta0 + math.radians(rng.uniform(-WIND_DIRECTION_SPREAD_DEG, WIND_DIRECTION_SPREAD_DEG))
    return float(speed * math.cos(theta)), float(speed * math.sin(theta))


def make_ensemble(base: Scenario, n: int, seed: int, q_base: tuple[float, ...] | list[float] | np.ndarray,
                  prefix: str = "case") -> list[CaseSpec]:
    """n cases: q = q_base * U(0.5, 1.5) per source; wind speed U(0.6, 1.6) m/s and direction = nominal direction
    (of the first base wind segment) + U(-40, +40) degrees.  Every second case (index 1, 3, ...) has a 2-segment wind:
    the switch time is a random multiple of output_interval within [T/3, 2T/3], the second wind is drawn the same
    way.  Deterministic in ``seed``.  The declared wind is also the inference wind in matched experiments."""
    qb = np.asarray(q_base, dtype=float)
    if qb.shape != (len(base.sources),):
        raise ValueError("q_base needs one entry per base source")
    w0 = base.transport.wind[0]
    theta0 = math.atan2(w0.v, w0.u)
    oi, T = base.times.output_interval, base.times.duration
    k_lo, k_hi = math.ceil(T / 3 / oi - 1e-9), math.floor(2 * T / 3 / oi + 1e-9)
    if k_hi < k_lo:
        raise ValueError("no output-interval multiple lies within [T/3, 2T/3] for the two-segment winds")
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        q = qb * rng.uniform(*Q_FACTOR_RANGE, size=qb.size)
        u, v = _draw_wind(rng, theta0)
        wind = [WindSegment(0.0, u, v)]
        if i % 2 == 1:
            t_switch = float(rng.integers(k_lo, k_hi + 1) * oi)
            u2, v2 = _draw_wind(rng, theta0)
            wind.append(WindSegment(t_switch, u2, v2))
        scn = dataclasses.replace(base, transport=dataclasses.replace(base.transport, wind=tuple(wind)))
        out.append(CaseSpec(f"{prefix}{i}", scn, tuple(float(x) for x in q)))
    return out


def _draw_key(spec: CaseSpec) -> str:
    return json.dumps([list(spec.q), [dataclasses.astuple(w) for w in spec.scenario.transport.wind]])


def check_ensembles(design: list[CaseSpec], test: list[CaseSpec], design_seed: int, test_seed: int,
                    design_noise_seed: int, test_noise_seed: int, validation: list[CaseSpec] | None = None,
                    validation_seed: int | None = None, validation_noise_seed: int | None = None) -> None:
    """Raise ValueError unless, pairwise over design / test (/ validation), the seeds differ, the noise-seed ranges
    are disjoint, no (q, wind) draw is shared and no case name is shared."""
    ens = {"design": (design, design_seed, design_noise_seed), "test": (test, test_seed, test_noise_seed)}
    if validation is not None:
        if validation_seed is None or validation_noise_seed is None:
            raise ValueError("validation ensemble needs validation_seed and validation_noise_seed")
        ens["validation"] = (validation, validation_seed, validation_noise_seed)
    names = list(ens)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (sa, seed_a, nz_a), (sb, seed_b, nz_b) = ens[a], ens[b]
            if seed_a == seed_b:
                raise ValueError(f"{a}_seed and {b}_seed must differ")
            if set(range(nz_a, nz_a + len(sa))) & set(range(nz_b, nz_b + len(sb))):
                raise ValueError(f"{a} and {b} noise seeds overlap")
            if {_draw_key(x) for x in sa} & {_draw_key(x) for x in sb}:
                raise ValueError(f"{a} and {b} ensembles share a (q, wind) draw")
            if {x.name for x in sa} & {x.name for x in sb}:
                raise ValueError(f"{a} and {b} case names overlap")


# ------------------------------------------------------------------ mismatch helpers
def refine_scenario(scenario: Scenario, factor: int, dt_factor: float = 1.0) -> Scenario:
    """Same physics on a grid `factor` times finer in each direction; dt scaled by dt_factor; output grid unchanged."""
    if factor < 1 or int(factor) != factor:
        raise ValueError("refine factor must be a positive integer")
    if not dt_factor > 0:
        raise ValueError("dt_factor must be positive")
    dom = dataclasses.replace(scenario.domain, nx=scenario.domain.nx * factor, ny=scenario.domain.ny * factor)
    times = Times(scenario.times.duration, scenario.times.dt * dt_factor, scenario.times.output_interval)
    return dataclasses.replace(scenario, domain=dom, times=times)


def perturb_wind(scenario: Scenario, speed_error: float, rotation_deg: float) -> Scenario:
    """Every wind segment: speed x (1 + speed_error), rotated counter-clockwise by rotation_deg."""
    th = math.radians(rotation_deg)
    f = 1.0 + speed_error
    seg = tuple(WindSegment(w.t_start, f * (w.u * math.cos(th) - w.v * math.sin(th)),
                            f * (w.u * math.sin(th) + w.v * math.cos(th))) for w in scenario.transport.wind)
    return dataclasses.replace(scenario, transport=dataclasses.replace(scenario.transport, wind=seg))


def block_average(fine: xr.DataArray, coarse: Scenario) -> xr.DataArray:
    """Mean of the fine (time, y, x) field over the fine cells inside each coarse cell, on the coarse coordinates."""
    d = coarse.domain
    v = np.asarray(fine.transpose("time", "y", "x").values, dtype=float)
    fy, fx = v.shape[1] // d.ny, v.shape[2] // d.nx
    if (fy * d.ny, fx * d.nx) != v.shape[1:]:
        raise ValueError("fine grid is not an integer refinement of the coarse grid")
    avg = v.reshape(v.shape[0], d.ny, fy, d.nx, fx).mean(axis=(2, 4))
    return xr.DataArray(avg, dims=("time", "y", "x"), name="concentration",
                        coords={"time": fine["time"].values, "y": (np.arange(d.ny) + 0.5) * d.dy,
                                "x": (np.arange(d.nx) + 0.5) * d.dx})


# ------------------------------------------------------------------ shared set-up
@dataclass
class _Setup:
    cfg: dict[str, Any]
    d: dict[str, Any]
    base: Scenario
    solver: SolverOptions
    noise_std: float
    obs_times: tuple[float, ...]
    cands: Candidates
    heldout: Candidates
    design_specs: list[CaseSpec]
    test_specs: list[CaseSpec]
    weight: np.ndarray
    validation_specs: list[CaseSpec] = field(default_factory=list)
    alpha: float | None = None  # None only for the alpha-selection run (alpha is then set per fit)
    alpha_info: dict[str, Any] | None = None
    truth_forward: ForwardSolver = FIPY  # generates every synthetic truth (incl. refined / wind / extra-source)
    inference_forward: ForwardSolver = FIPY  # response bases, case building, fits, reconstructions

    def backend_dict(self) -> dict[str, Any]:
        """Both BackendInfo records, for provenance / summary.json."""
        return {"truth": forward_info(self.truth_forward).to_dict(),
                "inference": forward_info(self.inference_forward).to_dict()}


def resolve_backends(cfg: dict[str, Any], truth_forward: ForwardSolver | None = None,
                     inference_forward: ForwardSolver | None = None) -> tuple[ForwardSolver, ForwardSolver]:
    """(truth, inference) forward solvers, resolved once.  Explicit objects win; otherwise
    cfg['backends'] = {'truth': {'spec': 'fipy' | 'module:callable', 'kwargs': {...}}, 'inference': {...}}
    (see forward.get_forward); default FiPy for both.  cfg['solver'] stays the SolverOptions kwargs."""
    b = cfg.get("backends", {})
    unknown = set(b) - {"truth", "inference"}
    if unknown:
        raise ValueError(f"unknown cfg['backends'] roles {sorted(unknown)}; use 'truth' and 'inference'")

    def one(role: str, given: ForwardSolver | None) -> ForwardSolver:
        if given is not None:
            return given
        c = b.get(role, {})
        return get_forward(c.get("spec"), c.get("kwargs"))
    return one("truth", truth_forward), one("inference", inference_forward)


def load_alpha_selection(path: Path | str) -> tuple[float, str]:
    """(selected alpha, sha256 of the file) from an ``alpha_selected.json``.  The path is used as given (relative
    paths are relative to the working directory)."""
    raw = Path(path).read_bytes()
    alpha = float(json.loads(raw)["alpha"])
    if not (math.isfinite(alpha) and alpha >= 0):
        raise ValueError(f"{path}: selected alpha must be finite and >= 0, got {alpha}")
    return alpha, hashlib.sha256(raw).hexdigest()


def resolve_alpha(d: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """alpha used by benchmark / mismatch and a provenance record.  With design.alpha_selection_file: alpha is read
    from that file; design.alpha must be null or equal to it (ValueError otherwise).  Without a file design.alpha
    must be a number and the run is labelled 'not selected: sensitivity'."""
    f, a = d.get("alpha_selection_file"), d.get("alpha")
    if f:
        sel, sha = load_alpha_selection(f)
        if a is not None and not math.isclose(float(a), sel, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"design.alpha = {a} disagrees with the alpha {sel} selected in {f}; set alpha to null "
                             "or to the selected value")
        return sel, {"value": sel, "source": "alpha_selection_file", "alpha_selection_file": str(f),
                     "alpha_selection_sha256": sha}
    if a is None:
        raise ValueError("design.alpha is null but design.alpha_selection_file is not set")
    return float(a), {"value": float(a), "source": "config", "alpha_selection_file": None,
                      "alpha_selection_sha256": None, "label": "not selected: sensitivity (no alpha_selection_file)"}


def _setup(cfg: dict[str, Any], need_alpha: bool = True, truth_forward: ForwardSolver | None = None,
           inference_forward: ForwardSolver | None = None) -> _Setup:
    d = {**DESIGN_DEFAULTS, **cfg.get("design", {})}
    base = scenario_from_dict(cfg["scenario"])
    inv = cfg.get("inverse", {})
    step = float(inv.get("sensor_interval", 600.0))
    obs_times = tuple(float(t) for t in np.arange(step, base.times.duration + 1e-9, step))
    n_h = int(d["heldout_n"])
    hx = (np.arange(n_h) + 0.5) * base.domain.lx / n_h
    hy = (np.arange(n_h) + 0.5) * base.domain.ly / n_h
    X, Y = np.meshgrid(hx, hy)
    held = Candidates(tuple(f"h{k:03d}" for k in range(n_h * n_h)), X.reshape(-1), Y.reshape(-1))
    q_base = cfg["q"]
    ds = make_ensemble(base, int(d["n_design"]), int(d["design_seed"]), q_base, "d")
    ts = make_ensemble(base, int(d["n_test"]), int(d["test_seed"]), q_base, "t")
    vs = make_ensemble(base, int(d["n_validation"]), int(d["validation_seed"]), q_base, "v")
    check_ensembles(ds, ts, int(d["design_seed"]), int(d["test_seed"]), int(d["design_noise_seed"]),
                    int(d["test_noise_seed"]), vs, int(d["validation_seed"]), int(d["validation_noise_seed"]))
    alpha, info = resolve_alpha(d) if need_alpha else (None, None)
    tf, inf = resolve_backends(cfg, truth_forward, inference_forward)
    return _Setup(cfg, d, base, SolverOptions(**cfg.get("solver", {})), float(inv.get("noise_std", 0.005)), obs_times,
                  candidate_grid(base.domain, int(d["candidates_n"])), held, ds, ts, population_weight(base.domain),
                  vs, alpha, info, tf, inf)


def _opts(st: _Setup, weighted: bool) -> DesignOptions:
    # st.alpha is None only in run_alpha_selection, which overrides alpha on every fit (dataclasses.replace)
    return DesignOptions(noise_std=st.noise_std, obs_times=st.obs_times, alpha=float(st.alpha or 0.0),
                         t_exclude=float(st.d["t_exclude"]), weight=st.weight if weighted else None,
                         eval_sites=st.heldout, solver=st.solver)


def _constraints(st: _Setup) -> Constraints:
    return Constraints(None, float(st.d["min_separation"]))


def _evaluate(flat: dict[str, list[Layout]], cases: list[DesignCase], opts: DesignOptions,
              validation: bool = False) -> pd.DataFrame:
    """design.evaluate_layouts (or evaluate_validation for the alpha selection); its matrix 'condition' number is
    renamed 'cond_number' (the word 'condition' tags the mismatch kind in this module's tables)."""
    df = evaluate_validation(flat, cases, opts) if validation else evaluate_layouts(flat, cases, opts)
    return df.rename(columns={"condition": "cond_number"})


def _flatten(layouts: dict[str, dict[int, list[Layout]]]) -> dict[str, list[Layout]]:
    return {m: [lay for b in sorted(by_b) for lay in by_b[b]] for m, by_b in layouts.items()}


def _peak_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # bytes on macOS, KiB on Linux


def _jsonable_layouts(layouts: dict[str, dict[int, list[Layout]]]) -> dict[str, dict[str, list[list[int]]]]:
    return {m: {str(b): [list(lay) for lay in lays] for b, lays in sorted(by_b.items())}
            for m, by_b in layouts.items()}


def load_layouts(src: Path | str | dict[str, Any]) -> tuple[int | None, dict[str, dict[int, list[Layout]]]]:
    """(candidates_n, layouts) from a layouts.json dict, file, or run directory."""
    data: dict[str, Any]
    if isinstance(src, (str, Path)):
        p = Path(src)
        data = json.loads((p / "layouts.json" if p.is_dir() else p).read_text())
    else:
        data = src
    n = data.get("candidates_n")
    lays = {m: {int(b): [tuple(int(i) for i in lay) for lay in ls] for b, ls in by_b.items()}
            for m, by_b in data["layouts"].items()}
    return (None if n is None else int(n)), lays


# ------------------------------------------------------------------ alpha selection (validation ensemble)
CRITERION_TEXT = ("mean over budgets, layouts, validation cases and noise reps of the regional field RMSE "
                  "(uniform weights, ug/m^3, vs the validation truth); lowest wins, exact ties go to the smaller alpha")


@dataclass
class AlphaSelectionResult:
    alpha: float
    table: pd.DataFrame  # one row per alpha: criterion and companions
    raw: pd.DataFrame  # one row per (alpha, layout, case, rep)
    selected: dict[str, Any]  # the alpha_selected.json payload
    layouts: dict[str, dict[int, list[Layout]]]
    timings: dict[str, float]
    runtime_s: float
    peak_memory_mb: float
    out_dir: Path | None = None
    figures: list[str] = field(default_factory=list)


def _alpha_grid(grid: Any) -> list[float]:
    try:
        g = sorted({float(a) for a in grid})
    except TypeError as e:
        raise ValueError("design.alpha_grid must be a list of numbers") from e
    if not g or not all(math.isfinite(a) and a >= 0 for a in g):
        raise ValueError("design.alpha_grid must be a non-empty list of finite alphas >= 0")
    return g


def _alpha_table(raw: pd.DataFrame) -> pd.DataFrame:
    g: Any = raw.groupby("alpha")
    t = pd.DataFrame({
        "criterion": g["rmse"].mean(), "median_rmse": g["rmse"].median(), "p90_rmse": g["rmse"].quantile(0.9),
        "n_rmse_gt_10x_no_emission": g.apply(lambda r: int((r["rmse"] > NO_EMISSION_FACTOR * r["rmse_no_emission"]).sum())),
        "mean_source_rel_err": g["source_rel_err"].mean(), "n_rows": g.size()}).reset_index()
    return t


def _pick_alpha(table: pd.DataFrame) -> float:
    """Lowest criterion; exact ties (and a nan criterion, treated as +inf) resolve to the smaller alpha."""
    key = [(math.inf if not math.isfinite(c) else float(c), float(a)) for a, c in zip(table["alpha"], table["criterion"])]
    return min(key)[1]


def run_alpha_selection(cfg: dict[str, Any], out_dir: Path | str | None = None, overwrite: bool = False,
                        cache: ResponseCache | None = None, progress: Progress = None,
                        truth_forward: ForwardSolver | None = None,
                        inference_forward: ForwardSolver | None = None) -> AlphaSelectionResult:
    """Choose the Tikhonov strength alpha on a VALIDATION ensemble, independently of every test scenario.

    Pre-declared procedure (nothing below depends on the test ensemble, nor on design.alpha / alpha_selection_file):
    1. build design.n_validation matched validation cases (validation_seed; fresh noise from validation_noise_seed);
    2. layouts: only alpha-INDEPENDENT ones, per budget the design.n_random seeded random layouts
       (layout_seed + budget) and the spread layout (greedy is excluded: its choice depends on alpha);
    3. for every alpha in design.alpha_grid fit every layout x case x noise rep (prior mean = 0 for all sources;
       alpha is applied at fit time on the cached case arrays, no forward solve is repeated);
    4. criterion = mean over budgets, layouts, validation cases and noise reps of the regional field RMSE (uniform
       weights; CRITERION_TEXT); the lowest criterion wins and an exact tie goes to the smaller alpha.
       Also recorded per alpha: median, 90th percentile, number of rows whose RMSE exceeds 10 x the RMSE of the
       zero-emission estimate, mean source relative error.
    The grid spans several decades of alpha / q_scale^2 (the only combination that matters in the linear route).
    Writes alpha_selected.json, validation_raw.csv, layouts.json, figures/criterion_vs_alpha.png, config.json,
    provenance.json, timings.json when ``out_dir`` is given."""
    t_all = time.perf_counter()
    run_dir = None if out_dir is None else make_run_dir(out_dir, overwrite)
    cache = cache or ResponseCache()
    st = _setup(cfg, False, truth_forward, inference_forward)
    d = st.d
    grid = _alpha_grid(d["alpha_grid"])
    budgets = sorted(int(b) for b in d["budgets"])
    tim: dict[str, float] = {}
    t0 = time.perf_counter()
    _say(progress, f"building {len(st.validation_specs)} validation cases")
    vcs = _build_cases(st, st.validation_specs, "validation", int(d["validation_noise_seed"]), cache,
                       ["uniform"])["uniform"]
    tim["cases_s"] = time.perf_counter() - t0
    k = _constraints(st)
    layouts: dict[str, dict[int, list[Layout]]] = {"random": {}, "spread": {}}
    for b in budgets:
        layouts["random"][b] = random_layouts(st.cands, b, int(d["n_random"]), k, int(d["layout_seed"]) + b)
        layouts["spread"][b] = [spread_layout(st.cands, b, k)]
    flat = _flatten(layouts)
    base_opts = _opts(st, False)
    no_emission = {c.name: math.sqrt(c.c0) for c in vcs}
    t0 = time.perf_counter()
    parts = []
    for a in grid:
        _say(progress, f"alpha = {a:g}")
        df = _evaluate(flat, vcs, dataclasses.replace(base_opts, alpha=a), validation=True)
        df.insert(0, "alpha", a)
        df["rmse_no_emission"] = [no_emission[c] for c in df["case"]]
        parts.append(df)
    raw = pd.concat(parts, ignore_index=True)
    table = _alpha_table(raw)
    alpha = _pick_alpha(table)
    tim["evaluation_s"] = time.perf_counter() - t0
    seeds = {k_: d[k_] for k_ in SEED_KEYS}
    selected = {
        "alpha": alpha, "selected_criterion": float(table.loc[table["alpha"] == alpha, "criterion"].iloc[0]),
        "criterion": CRITERION_TEXT, "tie_rule": "exact tie -> smaller alpha",
        "table": table.to_dict("records"), "alpha_grid": grid,
        "validation_seed": d["validation_seed"], "validation_noise_seed": d["validation_noise_seed"],
        "n_validation": d["n_validation"], "noise_reps": d["noise_reps"], "n_random": d["n_random"],
        "budgets": budgets, "layout_seed": d["layout_seed"], "layout_methods": ["random", "spread"],
        "noise_std": st.noise_std, "q_scale": DEFAULT_Q_SCALE,
        "prior": "zeros (prior mean 0 for every source)", "prior_mean_uses_truth": False,
        "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
        "package_versions": provenance({}, {})["versions"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    res = AlphaSelectionResult(alpha, table, raw, selected, layouts, tim, time.perf_counter() - t_all, _peak_mb(),
                               run_dir)
    if run_dir is not None:
        write_json(run_dir / "config.json", cfg)
        write_json(run_dir / "alpha_selected.json", selected)
        write_json(run_dir / "provenance.json", provenance(cfg, seeds, {
            "experiment": "alpha_selection", "criterion": CRITERION_TEXT, "backends": st.backend_dict(),
            "note": "synthetic idealised experiment; independent of every test scenario"}))
        write_csv(run_dir / "validation_raw.csv", raw)
        write_json(run_dir / "layouts.json", {"candidates_n": int(d["candidates_n"]),
                                              "layouts": _jsonable_layouts(layouts)})
        (run_dir / "figures").mkdir(exist_ok=True)
        fig = plot_alpha_criterion(table, alpha)
        fig.savefig(run_dir / "figures" / "criterion_vs_alpha.png", dpi=120)
        plt.close(fig)
        res.figures.append("criterion_vs_alpha.png")
        res.runtime_s = time.perf_counter() - t_all
        res.peak_memory_mb = _peak_mb()
        write_json(run_dir / "timings.json", {"stages_s": tim, "total_s": res.runtime_s,
                                              "peak_memory_mb": res.peak_memory_mb})
    return res


def plot_alpha_criterion(table: pd.DataFrame, selected: float | None = None) -> Figure:
    """Validation criterion (mean RMSE) and median RMSE vs alpha, log axes; alpha = 0 sits at the left edge,
    one decade below the smallest positive alpha, labelled '0'."""
    t = table.sort_values("alpha")
    pos = [a for a in t["alpha"] if a > 0]
    x0 = (min(pos) / 10.0) if pos else 1.0
    pos_x = lambda a: x0 if a == 0 else a  # noqa: E731
    xs = [pos_x(a) for a in t["alpha"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(xs, t["criterion"], "o-", color="tab:red", label="criterion (mean RMSE)")
    ax.plot(xs, t["median_rmse"], "s--", color="tab:blue", label="median RMSE")
    ax.plot(xs, t["p90_rmse"], "^:", color="tab:gray", label="90th percentile RMSE")
    if selected is not None:
        ax.axvline(pos_x(selected), color="k", lw=0.8, alpha=0.5, label=f"selected alpha = {selected:g}")
    ax.set(xscale="log", yscale="log", xlabel="alpha (Tikhonov strength; 0 = no prior)",
           ylabel="validation RMSE (microgram/m$^3$)", title="Alpha selection on the validation ensemble")
    ax.set_xticks(xs)
    ax.set_xticklabels(["0" if a == 0 else f"{a:g}" for a in t["alpha"]])
    ax.minorticks_off()
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ benchmark
@dataclass
class BenchmarkResult:
    raw: pd.DataFrame  # one row per (weighting, method, layout, case, rep)
    summary: pd.DataFrame
    paired: pd.DataFrame
    design_scores: pd.DataFrame
    layouts: dict[str, dict[int, list[Layout]]]  # design-stage layouts: method -> budget -> layouts
    greedy_orders: dict[str, tuple[int, ...]]
    candidates: Candidates
    base: Scenario
    timings: dict[str, float]
    runtime_s: float
    peak_memory_mb: float
    seeds: dict[str, Any]
    out_dir: Path | None = None
    figures: list[str] = field(default_factory=list)

    def layouts_json(self, candidates_n: int) -> dict[str, Any]:
        """The layouts.json payload (accepted directly by run_mismatch / load_layouts)."""
        return {"candidates_n": candidates_n, "layouts": _jsonable_layouts(self.layouts)}


def _build_cases(st: _Setup, specs: list[CaseSpec], role: str, noise_seed0: int, cache: ResponseCache,
                 weightings: list[str]) -> dict[str, list[DesignCase]]:
    """Cases per weighting; each truth is simulated once and shared by the (cheap) rebuilds."""
    out: dict[str, list[DesignCase]] = {w: [] for w in weightings}
    for i, s in enumerate(specs):
        q = np.array(s.q)
        truth = run_forward(st.truth_forward, s.scenario, q, st.solver).concentration
        for w in weightings:
            out[w].append(build_case(s.name, role, s.scenario, q, s.scenario, st.cands, _opts(st, w == "population"),
                                     noise_seed0 + i, int(st.d["noise_reps"]), cache, truth_field=truth,
                                     forward=st.inference_forward))
    return out


def _random_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Per (weighting, budget, case, rep): mean over the random layouts, as one pseudo-layout 'random_mean'."""
    r: Any = df[df["method"] == "random"]
    g = r.groupby(["weighting", "budget", "case", "rep"], as_index=False)[["mse_w", "rmse"]].mean()
    g["method"], g["layout_id"] = "random_mean", "random_mean-0"
    return g


PAIRS = [("greedy", "spread"), ("greedy", "random_mean"), ("spread", "random_mean"),
         ("greedy_weighted", "greedy")]


def _paired_table(raw: pd.DataFrame) -> pd.DataFrame:
    full = pd.concat([raw, _random_mean(raw)], ignore_index=True)
    rows = []
    for w in sorted(set(raw["weighting"])):
        sub: Any = full[full["weighting"] == w]
        for a, b in PAIRS:
            if a not in set(sub["method"]) or b not in set(sub["method"]):
                continue
            _, s = paired_comparison(sub, a, b, "rmse")
            s.insert(0, "weighting", w)
            s.insert(0, "comparison", f"{a} - {b}")
            rows.append(s)
    cols = ["comparison", "weighting", "budget", "mean_diff", "sd_diff", "n", "frac_a_better"]
    out: Any = pd.concat(rows, ignore_index=True)[cols] if rows else pd.DataFrame(columns=cols)
    return out


def _summary_table(raw: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "weighting", "budget"]
    g: Any = raw.groupby(keys)
    s = pd.DataFrame({
        "n_layouts": g["layout_id"].nunique(), "n_rows": g.size(), "mean_rmse": g["rmse"].mean(),
        "sd_rmse": g["rmse"].std(ddof=1), "median_rmse": g["rmse"].median(), "pooled_rmse": np.sqrt(g["mse_w"].mean()),
        "mean_heldout_rmse": g["heldout_rmse"].mean(), "mean_source_rel_err": g["source_rel_err"].mean(),
        "mean_effective_rank": g["effective_rank"].mean()}).reset_index()
    s["sd_label"] = SD_LABEL
    dg: Any = design.groupby(keys, as_index=False)["design_mse_w"].mean()
    dg["design_rmse"] = np.sqrt(dg.pop("design_mse_w"))
    s = s.merge(dg, on=keys, how="left")
    s["gap_rmse"] = s["pooled_rmse"] - s["design_rmse"]  # test minus design, both pooled-RMSE; > 0: worse on test
    return s


def run_benchmark(cfg: dict[str, Any], out_dir: Path | str | None = None, overwrite: bool = False,
                  cache: ResponseCache | None = None, progress: Progress = None,
                  truth_forward: ForwardSolver | None = None,
                  inference_forward: ForwardSolver | None = None) -> BenchmarkResult:
    """Compare random / spread / greedy (and, if design.weighted, greedy_weighted) layouts over several budgets.
    See the module docstring.  Writes a run directory when ``out_dir`` is given (refuses a non-empty one unless
    ``overwrite``)."""
    t_all = time.perf_counter()
    run_dir = None if out_dir is None else make_run_dir(out_dir, overwrite)
    cache = cache or ResponseCache()
    tim: dict[str, float] = {}
    st = _setup(cfg, True, truth_forward, inference_forward)
    d = st.d
    budgets = sorted(int(b) for b in d["budgets"])
    weighted = bool(d["weighted"])
    wnames = ["uniform"] + (["population"] if weighted else [])
    # ---- cases
    t0 = time.perf_counter()
    _say(progress, f"building {len(st.design_specs)} design + {len(st.test_specs)} test cases")
    dcs = _build_cases(st, st.design_specs, "design", int(d["design_noise_seed"]), cache, wnames)
    tcs = _build_cases(st, st.test_specs, "test", int(d["test_noise_seed"]), cache, wnames)
    for w in wnames:
        assert_disjoint(dcs[w], tcs[w])
    tim["cases_s"] = time.perf_counter() - t0
    # ---- layouts
    t0 = time.perf_counter()
    k = _constraints(st)
    layouts: dict[str, dict[int, list[Layout]]] = {"random": {}, "spread": {}, "greedy": {}}
    orders: dict[str, tuple[int, ...]] = {}
    for b in budgets:
        layouts["random"][b] = random_layouts(st.cands, b, int(d["n_random"]), k, int(d["layout_seed"]) + b)
        layouts["spread"][b] = [spread_layout(st.cands, b, k)]
    _say(progress, "greedy selection (design cases only)")
    selectors = [("greedy", "uniform")] + ([("greedy_weighted", "population")] if weighted else [])
    for method, w in selectors:
        trace = greedy_trace(st.cands, budgets[-1], dcs[w], k, _opts(st, w == "population"))
        orders[method] = trace.order
        layouts[method] = {b: [tuple(sorted(trace.order[:b]))] for b in budgets}  # nested greedy layouts
    tim["selection_s"] = time.perf_counter() - t0
    # ---- evaluation on test cases and design-stage scores
    t0 = time.perf_counter()
    flat = _flatten(layouts)
    raws, dscores = [], []
    for w in wnames:
        opt = _opts(st, w == "population")
        _say(progress, f"evaluating on test cases ({w} metric)")
        df = _evaluate(flat, tcs[w], opt)
        df.insert(0, "weighting", w)
        raws.append(df)
        for m, lays in flat.items():
            for j, lay in enumerate(lays):
                sc = score_layout(lay, dcs[w], opt)
                dscores.append({"method": m, "layout_id": f"{m}-{j}", "budget": len(lay), "weighting": w,
                                "sensors": ",".join(st.cands.ids[i] for i in lay), "design_mse_w": sc["mse_w"],
                                "design_rmse": sc["rmse"]})
    raw = pd.concat(raws, ignore_index=True)
    design = pd.DataFrame(dscores)
    tim["evaluation_s"] = time.perf_counter() - t0
    summary, paired = _summary_table(raw, design), _paired_table(raw)
    seeds = {k_: d[k_] for k_ in SEED_KEYS}
    res = BenchmarkResult(raw, summary, paired, design, layouts, orders, st.cands, st.base, tim,
                          time.perf_counter() - t_all, _peak_mb(), seeds, run_dir)
    if run_dir is not None:
        t0 = time.perf_counter()
        _write_benchmark(res, cfg, run_dir, st)
        tim["output_s"] = time.perf_counter() - t0
        res.runtime_s = time.perf_counter() - t_all
        res.peak_memory_mb = _peak_mb()
        write_json(run_dir / "timings.json", {"stages_s": tim, "total_s": res.runtime_s,
                                              "peak_memory_mb": res.peak_memory_mb})
    return res


def _write_benchmark(res: BenchmarkResult, cfg: dict[str, Any], run_dir: Path, st: _Setup) -> None:
    write_json(run_dir / "config.json", cfg)
    write_json(run_dir / "provenance.json", provenance(cfg, res.seeds, {
        "experiment": "benchmark", "noise_std": st.noise_std, "alpha": st.alpha_info, "backends": st.backend_dict(),
        "note": "synthetic idealised experiment; RMSE in microgram/m^3 on the inference grid"}))
    write_csv(run_dir / "raw_metrics.csv", res.raw)
    write_csv(run_dir / "summary.csv", res.summary)
    write_json(run_dir / "summary.json", {"sd_label": SD_LABEL, "alpha": st.alpha_info, "backend": st.backend_dict(),
                                          "summary": res.summary.to_dict("records"),
                                          "paired": res.paired.to_dict("records")})
    write_csv(run_dir / "paired_comparisons.csv", res.paired)
    write_csv(run_dir / "design_scores.csv", res.design_scores)
    write_json(run_dir / "layouts.json", {
        **res.layouts_json(int(st.d["candidates_n"])), "candidate_ids": list(res.candidates.ids),
        "greedy_orders": {m: list(o) for m, o in res.greedy_orders.items()}})
    figs = run_dir / "figures"
    figs.mkdir(exist_ok=True)
    for w in sorted(set(res.summary["weighting"])):
        _save(plot_error_bands(res.summary, w), figs / f"error_vs_sensors_{w}.png", res)
    _save(plot_selected_sites(res), figs / "selected_sites.png", res)
    _save(plot_heldout_vs_regional(res.summary), figs / "heldout_vs_regional.png", res)


def _save(fig: Figure, path: Path, res: BenchmarkResult | MismatchResult) -> None:
    fig.savefig(path, dpi=120)
    plt.close(fig)
    res.figures.append(path.name)


# ------------------------------------------------------------------ benchmark figures
_STYLE = {"random": ("tab:gray", "o"), "spread": ("tab:blue", "s"), "greedy": ("tab:red", "^"),
          "greedy_weighted": ("tab:purple", "v"), "poor": ("black", "x")}


def plot_error_bands(summary: pd.DataFrame, weighting: str = "uniform", methods: list[str] | None = None,
                     ax: Any = None, condition: str | None = None, log: bool = True) -> Figure:
    """Mean test RMSE vs number of sensors per method; the shaded band is mean +/- 1 s.d. of the per-row RMSE
    (a spread across layouts x cases x noise reps, NOT a confidence interval).  Unregularised fits with poorly
    placed sensors can blow up by orders of magnitude, so the axis is logarithmic by default (lower band edge then
    clipped at 1% of the mean)."""
    s: Any = summary[summary["weighting"] == weighting] if "weighting" in summary else summary
    if condition is not None:
        s = s[s["condition"] == condition]
    fig = ax.figure if ax is not None else plt.subplots(figsize=(6.5, 4.2))[0]
    ax = ax if ax is not None else fig.axes[0]
    for m in (methods or [m for m in _STYLE if m in set(s["method"])]):
        g = s[s["method"] == m].sort_values("budget")
        if g.empty:
            continue
        c, mk = _STYLE.get(m, (None, "o"))
        ax.plot(g["budget"], g["mean_rmse"], marker=mk, color=c, label=f"{m} (mean)")
        sd = g["sd_rmse"].fillna(0.0)
        lo = g["mean_rmse"] - sd
        lo = np.maximum(lo, 0.01 * g["mean_rmse"]) if log else lo
        ax.fill_between(g["budget"], lo, g["mean_rmse"] + sd, color=c, alpha=0.15,
                        label=f"{m} (mean $\\pm$ 1 s.d.)")
    ax.set(xlabel="number of sensors", ylabel="test RMSE (microgram/m$^3$)",
           yscale="log" if log else "linear", title=f"Held-out cases, {weighting} error weights"
                 + (f", {condition}" if condition else "") + "\nline = mean, band = $\\pm$1 s.d. across layouts, cases, reps")
    ax.set_xticks(sorted(set(s["budget"])))
    ax.legend(fontsize=7, ncol=2)
    return fig


def plot_selected_sites(res: BenchmarkResult, budget: int | None = None) -> Figure:
    """Greedy (numbered by selection order) and spread sites at the largest budget, with the nominal sources."""
    b = budget or max(next(iter(res.layouts.values())))
    methods = [m for m in ("greedy", "greedy_weighted", "spread") if m in res.layouts]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4.3), squeeze=False)
    for ax, m in zip(axes[0], methods):
        lay = res.layouts[m][b][0]
        ax.plot(res.candidates.x, res.candidates.y, ".", c="0.75", label="candidate sites")
        ax.plot([s.x for s in res.base.sources if isinstance(s, GaussianSource)], [s.y for s in res.base.sources if isinstance(s, GaussianSource)], "b*", ms=12, label="sources")
        ax.plot(res.candidates.x[list(lay)], res.candidates.y[list(lay)], "ro", label="selected")
        if m in res.greedy_orders:
            for rank, i in enumerate(res.greedy_orders[m][:b], 1):
                ax.annotate(str(rank), (res.candidates.x[i], res.candidates.y[i]), textcoords="offset points",
                            xytext=(5, 5), fontsize=8)
        ax.set(aspect="equal", xlim=(0, res.base.domain.lx), ylim=(0, res.base.domain.ly), xlabel="x (m)",
               ylabel="y (m)", title=f"{m}, {b} sensors" + (" (numbers = pick order)" if m in res.greedy_orders else ""))
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_heldout_vs_regional(summary: pd.DataFrame, weighting: str = "uniform") -> Figure:
    """Regional field RMSE (whole domain) next to held-out sensor RMSE (noiseless readings at 49 unused sites)."""
    s: Any = summary[summary["weighting"] == weighting]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, col, title in zip(axes, ("mean_rmse", "mean_heldout_rmse"),
                              ("regional field error (mean RMSE)", "held-out sensor error (mean RMSE)")):
        for m in [m for m in _STYLE if m in set(s["method"])]:
            g = s[s["method"] == m].sort_values("budget")
            ax.plot(g["budget"], g[col], marker=_STYLE[m][1], color=_STYLE[m][0], label=m)
        ax.set(xlabel="number of sensors", title=title, ylim=(0, None))
    axes[0].set_ylabel("RMSE (microgram/m$^3$)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ mismatch
@dataclass
class MismatchResult:
    raw: pd.DataFrame  # column 'condition' in {matched, refined_truth, wind_perturbed, extra_source}
    summary: pd.DataFrame
    paired_rows: pd.DataFrame  # mismatch - matched per (condition, method, layout, case, rep)
    paired: pd.DataFrame  # aggregated over those rows
    poor_diagnostics: pd.DataFrame
    poor_paired: pd.DataFrame
    layouts: dict[str, dict[int, list[Layout]]]
    timings: dict[str, float]
    runtime_s: float
    peak_memory_mb: float
    out_dir: Path | None = None
    figures: list[str] = field(default_factory=list)


MISMATCH_DEFAULTS: dict[str, Any] = {
    "refine": 2, "truth_dt_factor": 0.5, "wind_speed_error": 0.2, "wind_rotation_deg": 15.0,
    "extra_source": {"x": 3000.0, "y": 4500.0, "sigma": 500.0, "q": 60000.0},
    "poor_layout_sites": [[500.0, 9500.0], [1500.0, 9500.0], [500.0, 8500.0]]}


def _nearest(cands: Candidates, sites: list[list[float]]) -> Layout:
    idx = [int(np.argmin(np.hypot(cands.x - x, cands.y - y))) for x, y in sites]
    if len(set(idx)) != len(idx):
        raise ValueError("poor_layout_sites map to duplicate candidate sites")
    return tuple(sorted(idx))


def _mismatch_case(kind: str, spec: CaseSpec, i: int, st: _Setup, opts: DesignOptions, cache: ResponseCache,
                   m: dict[str, Any], matched_truth: xr.DataArray) -> DesignCase:
    inf, q = spec.scenario, np.array(spec.q)
    seed = int(st.d["test_noise_seed"]) + i
    reps = int(st.d["noise_reps"])
    err: xr.DataArray | None = None
    truth_scn, truth_q = inf, q
    truth = matched_truth
    if kind == "refined_truth":
        truth_scn = refine_scenario(inf, int(m["refine"]), float(m["truth_dt_factor"]))
        truth = run_forward(st.truth_forward, truth_scn, q, st.solver).concentration
        err = block_average(truth, inf)
    elif kind == "wind_perturbed":
        truth_scn = perturb_wind(inf, float(m["wind_speed_error"]), float(m["wind_rotation_deg"]))
        truth = run_forward(st.truth_forward, truth_scn, q, st.solver).concentration
    elif kind == "extra_source":
        e = m["extra_source"]
        truth_scn = dataclasses.replace(inf, sources=inf.sources + (GaussianSource(
            "X", float(e["x"]), float(e["y"]), float(e["sigma"])),))
        truth_q = np.append(q, float(e["q"]))
        truth = run_forward(st.truth_forward, truth_scn, truth_q, st.solver).concentration
    elif kind != "matched":
        raise ValueError(f"unknown mismatch kind {kind!r}")
    return build_case(spec.name, "test", truth_scn, truth_q, inf, st.cands, opts, seed, reps, cache,
                      error_field=err, truth_field=truth, forward=st.inference_forward)


def _poor_diagnostics(cases: list[DesignCase], layouts: dict[str, Layout], opts: DesignOptions,
                      q_names: list[str]) -> pd.DataFrame:
    rows = []
    for method, lay in layouts.items():
        for c in cases:
            for r in range(c.n_noise):
                f = fit_layout(c, lay, r, opts)
                sv = np.asarray(f["singular_values"], dtype=float)
                row = {"method": method, "budget": len(lay), "case": c.name, "rep": r, "rank": f["rank"],
                       "effective_rank": f["effective_rank"], "cond_number": f["condition"],
                       "sv_max": float(sv.max()), "sv_min": float(sv.min()),
                       "singular_values": ";".join(f"{v:.6g}" for v in sv), "rmse": f["rmse"],
                       "source_rel_err": f["source_rel_err"]}
                for j, nm in enumerate(q_names):
                    row[f"q_std_{nm}"] = float(f["q_std"][j])
                    row[f"q_std_rel_{nm}"] = float(f["q_std"][j] / c.q_true[j])
                rows.append(row)
    return pd.DataFrame(rows)


def _mismatch_pairs(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["method", "layout_id", "budget", "case", "rep"]
    base: Any = raw[raw["condition"] == "matched"][keys + ["rmse", "heldout_rmse", "source_rel_err"]]
    other: Any = raw[raw["condition"] != "matched"]
    m = other[["condition"] + keys + ["rmse", "heldout_rmse", "source_rel_err"]].merge(
        base, on=keys, suffixes=("", "_matched"))
    for c in ("rmse", "heldout_rmse", "source_rel_err"):
        m[f"diff_{c}"] = m[c] - m[f"{c}_matched"]
    m["worse"] = m["diff_rmse"] > 0
    g: Any = m.groupby(["condition", "method", "budget"])
    agg = pd.DataFrame({"mean_diff_rmse": g["diff_rmse"].mean(), "sd_diff_rmse": g["diff_rmse"].std(ddof=1),
                        "n": g.size(), "frac_worse": g["worse"].mean(),
                        "mean_diff_heldout_rmse": g["diff_heldout_rmse"].mean(),
                        "mean_diff_source_rel_err": g["diff_source_rel_err"].mean()}).reset_index()
    return m, agg


def run_mismatch(cfg: dict[str, Any], layouts_from: Path | str | dict[str, Any] | None = None,
                 out_dir: Path | str | None = None, overwrite: bool = False, cache: ResponseCache | None = None,
                 progress: Progress = None, truth_forward: ForwardSolver | None = None,
                 inference_forward: ForwardSolver | None = None) -> MismatchResult:
    """Evaluate FIXED design-stage layouts (from a benchmark run's layouts.json) on the test ensemble regenerated from
    the same config seeds, under matched conditions and three separate mismatches (refined-grid truth, hidden wind
    perturbation, unmodelled 4th source), plus a deliberately poor fixed layout.  ``layouts_from`` defaults to
    cfg['mismatch']['layouts_from']."""
    t_all = time.perf_counter()
    run_dir = None if out_dir is None else make_run_dir(out_dir, overwrite)
    cache = cache or ResponseCache()
    st = _setup(cfg, True, truth_forward, inference_forward)
    m = {**MISMATCH_DEFAULTS, **cfg.get("mismatch", {})}
    src = layouts_from if layouts_from is not None else m.get("layouts_from")
    if src is None:
        raise ValueError("no layouts: pass layouts_from or set mismatch.layouts_from")
    n_c, layouts = load_layouts(src)
    if n_c is not None and n_c != int(st.d["candidates_n"]):
        raise ValueError(f"layouts were designed on a {n_c}x{n_c} candidate grid, config has {st.d['candidates_n']}")
    layouts = {k: v for k, v in layouts.items() if k != "poor"}
    poor = _nearest(st.cands, m["poor_layout_sites"])
    layouts["poor"] = {len(poor): [poor]}
    flat = _flatten(layouts)
    opts = _opts(st, False)
    tim: dict[str, float] = {}
    raws, cases_matched = [], []
    for kind in ("matched",) + MISMATCH_KINDS:
        t0 = time.perf_counter()
        _say(progress, f"condition {kind}")
        cases = []
        for i, s in enumerate(st.test_specs):
            matched_truth = (run_forward(st.truth_forward, s.scenario, np.array(s.q), st.solver).concentration
                             if kind == "matched" else None)
            cases.append(_mismatch_case(kind, s, i, st, opts, cache, m, matched_truth))  # type: ignore[arg-type]
        if kind == "matched":
            cases_matched = cases
        df = _evaluate(flat, cases, opts)
        df.insert(0, "condition", kind)
        raws.append(df)
        tim[f"{kind}_s"] = time.perf_counter() - t0
    raw = pd.concat(raws, ignore_index=True)
    g: Any = raw.groupby(["condition", "method", "budget"])
    summary = pd.DataFrame({
        "n_layouts": g["layout_id"].nunique(), "n_rows": g.size(), "mean_rmse": g["rmse"].mean(),
        "sd_rmse": g["rmse"].std(ddof=1), "mean_heldout_rmse": g["heldout_rmse"].mean(),
        "mean_source_rel_err": g["source_rel_err"].mean(),
        "mean_unmodelled_source_mass": g["unmodelled_source_mass"].mean()}).reset_index()
    summary["sd_label"] = SD_LABEL
    paired_rows, paired = _mismatch_pairs(raw)
    # poor layout: identifiability diagnostics and matched contrast with greedy/spread at the same budget
    b_poor = len(poor)
    contrast = {"poor": poor}
    for meth in ("greedy", "spread"):
        if meth in layouts and b_poor in layouts[meth]:
            contrast[meth] = layouts[meth][b_poor][0]
    diag = _poor_diagnostics(cases_matched, contrast, opts, [s.name for s in st.base.sources])
    mat: Any = raw[raw["condition"] == "matched"]
    prow = []
    for meth in contrast:
        if meth != "poor":
            _, s = paired_comparison(mat, "poor", meth, "rmse")
            s.insert(0, "comparison", f"poor - {meth}")
            prow.append(s)
    poor_paired = pd.concat(prow, ignore_index=True) if prow else pd.DataFrame()
    res = MismatchResult(raw, summary, paired_rows, paired, diag, poor_paired, layouts, tim,
                         time.perf_counter() - t_all, _peak_mb(), run_dir)
    if run_dir is not None:
        _write_mismatch(res, cfg, run_dir, st, m, src if isinstance(src, dict) else str(src), contrast, st.cands)
        res.runtime_s = time.perf_counter() - t_all
        res.peak_memory_mb = _peak_mb()
        write_json(run_dir / "timings.json", {"stages_s": tim, "total_s": res.runtime_s,
                                              "peak_memory_mb": res.peak_memory_mb})
    return res


def _write_mismatch(res: MismatchResult, cfg: dict[str, Any], run_dir: Path, st: _Setup, m: dict[str, Any],
                    src: Any, contrast: dict[str, Layout], cands: Candidates) -> None:
    write_json(run_dir / "config.json", cfg)
    seeds = {k: st.d[k] for k in SEED_KEYS}
    write_json(run_dir / "provenance.json", provenance(cfg, seeds, {
        "experiment": "mismatch", "alpha": st.alpha_info, "backends": st.backend_dict(), "layouts_from": src if isinstance(src, str) else "dict", "mismatch": m,
        "poor_layout": [list(contrast["poor"]), [cands.ids[i] for i in contrast["poor"]]],
        "note": "mismatch differences are descriptive; they need not worsen every draw"}))
    write_csv(run_dir / "raw_metrics.csv", res.raw)
    write_csv(run_dir / "summary.csv", res.summary)
    write_json(run_dir / "summary.json", {"sd_label": SD_LABEL, "alpha": st.alpha_info, "backend": st.backend_dict(),
                                          "summary": res.summary.to_dict("records"),
                                          "paired_mismatch": res.paired.to_dict("records"),
                                          "poor_layout_paired": res.poor_paired.to_dict("records")})
    write_csv(run_dir / "paired_mismatch_rows.csv", res.paired_rows)
    write_csv(run_dir / "paired_mismatch.csv", res.paired)
    write_csv(run_dir / "poor_layout_diagnostics.csv", res.poor_diagnostics)
    write_csv(run_dir / "poor_layout_paired.csv", res.poor_paired)
    write_json(run_dir / "layouts.json", {"candidates_n": int(st.d["candidates_n"]),
                                          "layouts": _jsonable_layouts(res.layouts)})
    figs = run_dir / "figures"
    figs.mkdir(exist_ok=True)
    _save(plot_mismatch_rmse(res.summary), figs / "rmse_by_condition.png", res)
    _save(plot_mismatch_differences(res.paired), figs / "mismatch_minus_matched.png", res)
    _save(plot_poor_layout(res.poor_diagnostics, res.summary, cands, res.layouts, st.base),
          figs / "poor_layout.png", res)


def plot_mismatch_rmse(summary: pd.DataFrame, methods: list[str] | None = None) -> Figure:
    """One panel per condition: mean test RMSE vs sensors (band = +/- 1 s.d. across layouts, cases, reps)."""
    conds = [c for c in ("matched",) + MISMATCH_KINDS if c in set(summary["condition"])]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 4), sharey=True, squeeze=False)
    s = summary.assign(weighting="uniform")
    for ax, c in zip(axes[0], conds):
        plot_error_bands(s, "uniform", methods or ["random", "spread", "greedy"], ax=ax, condition=c)
        ax.set_title(c)
        ax.get_legend().remove() if ax is not axes[0][0] and ax.get_legend() else None
    fig.tight_layout()
    return fig


def plot_mismatch_differences(paired: pd.DataFrame, methods: tuple[str, ...] = ("random", "spread", "greedy")
                              ) -> Figure:
    """Mean (mismatch - matched) RMSE per budget with +/- 1 s.d. of the paired differences (positive: worse)."""
    conds = [c for c in MISMATCH_KINDS if c in set(paired["condition"])]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 4), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], conds):
        for j, mth in enumerate(methods):
            pm: Any = paired[(paired["condition"] == c) & (paired["method"] == mth)]
            g: Any = pm.sort_values("budget")
            ax.errorbar(g["budget"] + 0.08 * (j - 1), g["mean_diff_rmse"], yerr=g["sd_diff_rmse"].fillna(0), capsize=3,
                        marker=_STYLE[mth][1], color=_STYLE[mth][0], label=mth)
        ax.axhline(0, color="k", lw=0.6)
        ax.set(title=c, xlabel="number of sensors")
    axes[0][0].set_ylabel("mismatch - matched RMSE (microgram/m$^3$)\nmean $\\pm$ 1 s.d. of paired differences")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_poor_layout(diag: pd.DataFrame, summary: pd.DataFrame, cands: Candidates,
                     layouts: dict[str, dict[int, list[Layout]]], base: Scenario) -> Figure:
    """Left: sites of the poor / greedy / spread layouts at the poor layout's budget; right: matched mean RMSE."""
    b = int(diag["budget"].iloc[0])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3))
    a1.plot([s.x for s in base.sources if isinstance(s, GaussianSource)], [s.y for s in base.sources if isinstance(s, GaussianSource)], "b*", ms=12, label="sources")
    for mth in diag["method"].unique():
        lay = layouts[mth][b][0]
        a1.plot(cands.x[list(lay)], cands.y[list(lay)], _STYLE[mth][1] if mth != "poor" else "x", ms=9,
                color=_STYLE[mth][0], label=mth, linestyle="")
    a1.set(aspect="equal", xlim=(0, base.domain.lx), ylim=(0, base.domain.ly), xlabel="x (m)", ylabel="y (m)",
           title=f"{b}-sensor layouts")
    a1.legend(fontsize=8)
    s: Any = summary[(summary["condition"] == "matched") & (summary["budget"] == b)]
    s = s[s["method"].isin(list(diag["method"].unique()))]
    a2.bar(s["method"], s["mean_rmse"], yerr=s["sd_rmse"].fillna(0), capsize=4,
           color=[_STYLE[x][0] for x in s["method"]])
    a2.set(ylabel="matched test RMSE (microgram/m$^3$)", title="mean $\\pm$ 1 s.d. across cases x reps")
    fig.tight_layout()
    return fig

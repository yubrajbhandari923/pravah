"""Sensor-placement experiments: scenario ensembles, the design benchmark and the model-mismatch study.

Config sections used (see configs/benchmark.json, configs/mismatch.json):
* ``scenario``, ``q`` (nominal amplitudes), ``solver``, ``inverse.noise_std`` / ``inverse.sensor_interval``;
* ``design``: candidates_n, budgets, n_design, n_test, noise_reps, n_random, weighted, min_separation, heldout_n,
  alpha, t_exclude and the explicit seeds design_seed, test_seed, design_noise_seed, test_noise_seed, layout_seed;
* ``mismatch`` (run_mismatch only): refine, truth_dt_factor, wind_speed_error, wind_rotation_deg, extra_source,
  poor_layout_sites, layouts_from.

Honesty rules.  Layouts are SELECTED only from role='design' cases (design.py enforces it); every reported error
comes from independent role='test' cases with fresh noise.  Random layouts are never selected; their design-stage
score is computed only as a diagnostic.  All numbers are synthetic and idealised.
"""
from __future__ import annotations

import dataclasses
import json
import math
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

from .config import (GaussianSource, Scenario, SolverOptions, Times, Transport, WindSegment, scenario_from_dict,
                     to_dict)
from .design import (Candidates, Constraints, DesignCase, DesignOptions, Layout, assert_disjoint, build_case,
                     candidate_grid, evaluate_layouts, fit_layout, greedy_trace, paired_comparison, random_layouts,
                     score_layout, spread_layout)
from .forward import simulate
from .inverse import ResponseCache
from .io import make_run_dir, provenance, write_csv, write_json
from .metrics import population_weight

Progress = Callable[[str], None] | None

WIND_SPEED_RANGE = (0.6, 1.6)  # m/s
WIND_DIRECTION_SPREAD_DEG = 40.0  # +/- around the nominal direction
Q_FACTOR_RANGE = (0.5, 1.5)
SD_LABEL = "sd of per-row RMSE across layouts x cases x noise reps"
MISMATCH_KINDS = ("refined_truth", "wind_perturbed", "extra_source")
DESIGN_DEFAULTS: dict[str, Any] = {
    "candidates_n": 10, "budgets": [1, 2, 3, 5, 8], "n_design": 6, "n_test": 6, "noise_reps": 3, "n_random": 20,
    "weighted": False, "min_separation": 0.0, "heldout_n": 7, "alpha": 0.0, "t_exclude": 0.0,
    "design_seed": 11, "test_seed": 22, "design_noise_seed": 1000, "test_noise_seed": 5000, "layout_seed": 7}


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
                    design_noise_seed: int, test_noise_seed: int) -> None:
    """Raise ValueError unless seeds differ, the noise-seed ranges are disjoint and no (q, wind) draw is shared."""
    if design_seed == test_seed:
        raise ValueError("design_seed and test_seed must differ")
    d_rng = set(range(design_noise_seed, design_noise_seed + len(design)))
    t_rng = set(range(test_noise_seed, test_noise_seed + len(test)))
    if d_rng & t_rng:
        raise ValueError("design and test noise seeds overlap")
    shared = {_draw_key(s) for s in design} & {_draw_key(s) for s in test}
    if shared:
        raise ValueError("design and test ensembles share a (q, wind) draw")
    if {s.name for s in design} & {s.name for s in test}:
        raise ValueError("design and test case names overlap")


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


def _setup(cfg: dict[str, Any]) -> _Setup:
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
    check_ensembles(ds, ts, int(d["design_seed"]), int(d["test_seed"]), int(d["design_noise_seed"]),
                    int(d["test_noise_seed"]))
    return _Setup(cfg, d, base, SolverOptions(**cfg.get("solver", {})), float(inv.get("noise_std", 0.005)), obs_times,
                  candidate_grid(base.domain, int(d["candidates_n"])), held, ds, ts, population_weight(base.domain))


def _opts(st: _Setup, weighted: bool) -> DesignOptions:
    return DesignOptions(noise_std=st.noise_std, obs_times=st.obs_times, alpha=float(st.d["alpha"]),
                         t_exclude=float(st.d["t_exclude"]), weight=st.weight if weighted else None,
                         eval_sites=st.heldout, solver=st.solver)


def _constraints(st: _Setup) -> Constraints:
    return Constraints(None, float(st.d["min_separation"]))


def _evaluate(flat: dict[str, list[Layout]], cases: list[DesignCase], opts: DesignOptions) -> pd.DataFrame:
    """design.evaluate_layouts; its matrix 'condition' number is renamed 'cond_number' (the word 'condition' tags
    the mismatch kind in this module's tables)."""
    return evaluate_layouts(flat, cases, opts).rename(columns={"condition": "cond_number"})


def _flatten(layouts: dict[str, dict[int, list[Layout]]]) -> dict[str, list[Layout]]:
    return {m: [lay for b in sorted(by_b) for lay in by_b[b]] for m, by_b in layouts.items()}


def _peak_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # bytes on macOS, KiB on Linux


def _say(progress: Progress, msg: str) -> None:
    if progress is not None:
        progress(msg)


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
        truth = simulate(s.scenario, q, st.solver).concentration
        for w in weightings:
            out[w].append(build_case(s.name, role, s.scenario, q, s.scenario, st.cands, _opts(st, w == "population"),
                                     noise_seed0 + i, int(st.d["noise_reps"]), cache, truth_field=truth))
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
                  cache: ResponseCache | None = None, progress: Progress = None) -> BenchmarkResult:
    """Compare random / spread / greedy (and, if design.weighted, greedy_weighted) layouts over several budgets.
    See the module docstring.  Writes a run directory when ``out_dir`` is given (refuses a non-empty one unless
    ``overwrite``)."""
    t_all = time.perf_counter()
    run_dir = None if out_dir is None else make_run_dir(out_dir, overwrite)
    cache = cache or ResponseCache()
    tim: dict[str, float] = {}
    st = _setup(cfg)
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
    seeds = {k_: d[k_] for k_ in ("design_seed", "test_seed", "design_noise_seed", "test_noise_seed", "layout_seed")}
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
        "experiment": "benchmark", "noise_std": st.noise_std,
        "note": "synthetic idealised experiment; RMSE in microgram/m^3 on the inference grid"}))
    write_csv(run_dir / "raw_metrics.csv", res.raw)
    write_csv(run_dir / "summary.csv", res.summary)
    write_json(run_dir / "summary.json", {"sd_label": SD_LABEL, "summary": res.summary.to_dict("records"),
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
        ax.plot([s.x for s in res.base.sources], [s.y for s in res.base.sources], "b*", ms=12, label="sources")
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
        truth = simulate(truth_scn, q, st.solver).concentration
        err = block_average(truth, inf)
    elif kind == "wind_perturbed":
        truth_scn = perturb_wind(inf, float(m["wind_speed_error"]), float(m["wind_rotation_deg"]))
        truth = simulate(truth_scn, q, st.solver).concentration
    elif kind == "extra_source":
        e = m["extra_source"]
        truth_scn = dataclasses.replace(inf, sources=inf.sources + (GaussianSource(
            "X", float(e["x"]), float(e["y"]), float(e["sigma"])),))
        truth_q = np.append(q, float(e["q"]))
        truth = simulate(truth_scn, truth_q, st.solver).concentration
    elif kind != "matched":
        raise ValueError(f"unknown mismatch kind {kind!r}")
    return build_case(spec.name, "test", truth_scn, truth_q, inf, st.cands, opts, seed, reps, cache,
                      error_field=err, truth_field=truth)


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
                 progress: Progress = None) -> MismatchResult:
    """Evaluate FIXED design-stage layouts (from a benchmark run's layouts.json) on the test ensemble regenerated from
    the same config seeds, under matched conditions and three separate mismatches (refined-grid truth, hidden wind
    perturbation, unmodelled 4th source), plus a deliberately poor fixed layout.  ``layouts_from`` defaults to
    cfg['mismatch']['layouts_from']."""
    t_all = time.perf_counter()
    run_dir = None if out_dir is None else make_run_dir(out_dir, overwrite)
    cache = cache or ResponseCache()
    st = _setup(cfg)
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
            matched_truth = simulate(s.scenario, np.array(s.q), st.solver).concentration if kind == "matched" else None
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
    seeds = {k: st.d[k] for k in ("design_seed", "test_seed", "design_noise_seed", "test_noise_seed", "layout_seed")}
    write_json(run_dir / "provenance.json", provenance(cfg, seeds, {
        "experiment": "mismatch", "layouts_from": src if isinstance(src, str) else "dict", "mismatch": m,
        "poor_layout": [list(contrast["poor"]), [cands.ids[i] for i in contrast["poor"]]],
        "note": "mismatch differences are descriptive; they need not worsen every draw"}))
    write_csv(run_dir / "raw_metrics.csv", res.raw)
    write_csv(run_dir / "summary.csv", res.summary)
    write_json(run_dir / "summary.json", {"sd_label": SD_LABEL, "summary": res.summary.to_dict("records"),
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
    a1.plot([s.x for s in base.sources], [s.y for s in base.sources], "b*", ms=12, label="sources")
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

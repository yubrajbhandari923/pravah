# pyright: reportGeneralTypeIssues=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportOperatorIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportReturnType=false, reportAssignmentType=false
"""EXP-017: toy population-aware sensor design on the Phase-1 synthetic domain (synthetic fixtures, synthetic emissions/winds).

    python scripts/population_design.py --profile {smoke,pilot,production} [--config FILE] [--out DIR] [--summary-only]

Objectives (all weighted MSE, selected by greedy on the DESIGN ensemble through DesignOptions.weight): area (beta 0), mixed_<beta>,
population (beta 1) = mixed_weight(counts, 1, beta), capped = min(N, c*mean N)/sum.  Baselines: random (n_random layouts) and spread.
Order: setup -> alpha chosen on the VALIDATION ensemble with the area objective only -> freeze (config hash written to
provenance.json) -> TEST cases built -> one unit per (fixture, objective, budget) (resumable: unit files in <out>/work/units) ->
sanity (uniform fixture layouts == area layouts) -> freeze guard (hash unchanged) -> tables and figures.
Test cases are used only for evaluation.  Per-cell time-averaged squared-error forms of each test case give population, area,
lowest-population-quartile and worst-cell errors for any weight without new solves.  Attribution (local/external) is omitted:
the synthetic scenario has only local sources, open boundary and zero background, so the external component is identically zero.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pravah import design as D
from pravah import population as P
from pravah.config import SolverOptions, _merge, load_config, scenario_from_dict
from pravah.experiments import CaseSpec, check_ensembles, make_ensemble
from pravah.geo.regionio import GridSpec
from pravah.inverse import ResponseCache, Responses, run_forward
from pravah.io import provenance, write_json
from pravah.metrics import quadrature_weights

ROOT = Path(__file__).resolve().parents[1]
STAMP = "EXP-017"
CONFIGS = {"smoke": "configs/population_toy_smoke.json", "pilot": "configs/population_toy.json",
           "production": "configs/population_toy.json"}
OUTS = {"smoke": "results/phase3-population-toy-smoke", "pilot": "results/phase3-population-toy-pilot",
        "production": "results/phase3-population-toy"}
DIFF_METRICS = ("pop_rmse", "area_rmse", "lowq_rmse", "worst_cell_rmse", "heldout_rmse")
BOOT_LABEL = "percentile bootstrap interval over test cases (resampling cases; not a confidence interval for new scenarios)"


def say(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.0f}s] {msg}", flush=True)


T0 = time.perf_counter()


# ------------------------------------------------------------------------------------------------ pure helpers
def beta_name(b: float) -> str:
    return "area" if b == 0 else "population" if b == 1 else f"mixed_{b:g}"


def objective_names(betas: list[float]) -> list[str]:
    return [beta_name(b) for b in betas] + ["capped"]


def objective_weights(counts: np.ndarray, betas: list[float], cap_c: float) -> dict[str, np.ndarray]:
    """name -> (ny, nx) weight summing to 1 (uniform cell area; the whole synthetic domain is the study area)."""
    w = {beta_name(b): P.mixed_weight(counts, 1.0, b) for b in betas}
    w["capped"] = np.minimum(counts, cap_c * counts.mean()) / np.minimum(counts, cap_c * counts.mean()).sum()
    return w


def lowpop_mask(counts: np.ndarray, q: float, support_min: float) -> np.ndarray:
    """Cells with N >= support_min whose N is at or below the q-quantile of N over those cells (ties included)."""
    sup = counts >= support_min
    if not sup.any():
        raise ValueError("no cell has N >= support_min")
    return sup & (counts <= np.quantile(counts[sup], q))


def bootstrap_ci(x: np.ndarray, n_boot: int = 10000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean of x (resampling its entries with replacement)."""
    x = np.asarray(x, float)
    if x.size < 2:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, x.size, (n_boot, x.size))
    lo, hi = np.quantile(x[idx].mean(axis=1), [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def frozen_payload(cfg: dict[str, Any], alpha: float) -> dict[str, Any]:
    """Everything that defines the objectives and the experiment (not run selection, notes or output names)."""
    return {"scenario": cfg["scenario"], "q": cfg["q"], "solver": cfg.get("solver", {}), "inverse": cfg["inverse"],
            "design": cfg["design"], "population": cfg["population"], "alpha": alpha}


def frozen_hash(cfg: dict[str, Any], alpha: float) -> str:
    return hashlib.sha256(json.dumps(frozen_payload(cfg, alpha), sort_keys=True, default=str).encode()).hexdigest()


def assert_frozen(out: Path, cfg: dict[str, Any], alpha: float) -> str:
    """Freeze guard: the hash recorded in provenance.json before test evaluation must equal the live one."""
    rec = json.loads((out / "provenance.json").read_text())["frozen_config_hash"]
    now = frozen_hash(cfg, alpha)
    if rec != now:
        raise RuntimeError(f"freeze guard: frozen config hash changed ({rec} -> {now})")
    return now


def check_uniform_equals_area(layouts: dict[tuple[str, str, int], tuple[int, ...]], budgets: list[int], objs: list[str]) -> None:
    for o in objs:
        for b in budgets:
            if layouts[("uniform", o, b)] != layouts[("uniform", "area", b)]:
                raise RuntimeError(f"SANITY FAILED (stop condition): uniform-fixture layout for {o} budget {b} "
                                   f"{layouts[('uniform', o, b)]} != area layout {layouts[('uniform', 'area', b)]}")


def check_ensemble_seeds(d: dict[str, Any]) -> None:
    """Design/validation/test seeds and noise-seed ranges must be pairwise disjoint (before any solve)."""
    sizes = {"design": d["n_design"], "validation": d["n_validation"], "test": d["n_test"]}
    seeds = [d[f"{r}_seed"] for r in sizes]
    if len(set(seeds)) != 3:
        raise ValueError("design/validation/test seeds must differ")
    rng = [set(range(d[f"{r}_noise_seed"], d[f"{r}_noise_seed"] + n)) for r, n in sizes.items()]
    if rng[0] & rng[1] or rng[0] & rng[2] or rng[1] & rng[2]:
        raise ValueError("design/validation/test noise seeds overlap")


# ------------------------------------------------------------------------------------------------ cell error forms
@dataclass
class CellForm:
    """Time-averaged squared error per cell as a quadratic form in theta: e2(theta) = c0 + 2 h.theta + theta'G theta."""
    c0: np.ndarray  # (ny, nx)
    h: np.ndarray  # (np, ny, nx)
    G: np.ndarray  # (np, np, ny, nx)

    def e2(self, theta: np.ndarray) -> np.ndarray:
        return np.maximum(self.c0 + 2 * np.tensordot(theta, self.h, 1) + np.einsum("j,k,jkyx->yx", theta, theta, self.G), 0.0)


def cell_form(resp: Responses, truth: np.ndarray, t_exclude: float) -> CellForm:
    """Same window and trapezoid time weights as metrics.field_mse (cell weights omitted)."""
    sel, dt, _, _ = quadrature_weights(resp.times, resp.y.size, resp.x.size, None, t_exclude)
    tw = dt / dt.sum()
    d0 = (resp.base - truth)[sel]
    g = resp.g[:, sel]
    return CellForm(np.einsum("t,tyx->yx", tw, d0 ** 2), np.einsum("t,jtyx,tyx->jyx", tw, g, d0),
                    np.einsum("t,jtyx,ktyx->jkyx", tw, g, g))


# ------------------------------------------------------------------------------------------------ setup
@dataclass
class Setup:
    cfg: dict[str, Any]
    d: dict[str, Any]
    pcfg: dict[str, Any]
    solver: SolverOptions
    noise_std: float
    obs_times: tuple[float, ...]
    cands: D.Candidates
    heldout: D.Candidates
    cache: ResponseCache
    counts: dict[str, np.ndarray]
    weights: dict[str, dict[str, np.ndarray]]  # fixture -> objective -> weight
    low: dict[str, np.ndarray]
    specs: dict[str, list[CaseSpec]]
    alpha: float = 0.0

    def opts(self, weight: np.ndarray | None, eval_sites: D.Candidates | None = None) -> D.DesignOptions:
        return D.DesignOptions(noise_std=self.noise_std, obs_times=self.obs_times, alpha=self.alpha,
                               t_exclude=float(self.d["t_exclude"]), weight=weight, eval_sites=eval_sites, solver=self.solver)


def make_setup(cfg: dict[str, Any], work: Path) -> Setup:
    d, pc = cfg["design"], cfg["population"]
    check_ensemble_seeds(d)
    base = scenario_from_dict(cfg["scenario"])
    dom = base.domain
    step = float(cfg["inverse"].get("sensor_interval", 600.0))
    obs = tuple(float(t) for t in np.arange(step, base.times.duration + 1e-9, step))
    n_h = int(d["heldout_n"])
    X, Y = np.meshgrid((np.arange(n_h) + 0.5) * dom.lx / n_h, (np.arange(n_h) + 0.5) * dom.ly / n_h)
    held = D.Candidates(tuple(f"h{k:03d}" for k in range(n_h * n_h)), X.reshape(-1), Y.reshape(-1))
    grid = GridSpec(0.0, 0.0, dom.nx, dom.ny, dom.lx / dom.nx, dom.ly / dom.ny, int(pc["epsg"]))
    if pc["total"] % (dom.nx * dom.ny):
        raise ValueError("population.total must be divisible by the cell count (else the uniform fixture is not constant)")
    counts = {f: P.FIXTURES[f](grid, int(pc["total"]), int(s)) for f, s in pc["fixture_seeds"].items()}
    if not np.all(counts["uniform"] == counts["uniform"].flat[0]):
        raise ValueError("uniform fixture is not constant")
    weights = {f: objective_weights(c, pc["betas"], float(pc["cap_c"])) for f, c in counts.items()}
    low = {f: lowpop_mask(c, float(pc["lowpop_quantile"]), float(pc["support_min"])) for f, c in counts.items()}
    cands = D.candidate_grid(dom, int(d["candidates_n"]))
    q = cfg["q"]
    specs = {"design": make_ensemble(base, int(d["n_design"]), int(d["design_seed"]), q, "d"),
             "validation": make_ensemble(base, int(d["n_validation"]), int(d["validation_seed"]), q, "v"),
             "test": make_ensemble(base, int(d["n_test"]), int(d["test_seed"]), q, "t")}
    check_ensembles(specs["design"], specs["test"], int(d["design_seed"]), int(d["test_seed"]), int(d["design_noise_seed"]),
                    int(d["test_noise_seed"]), specs["validation"], int(d["validation_seed"]), int(d["validation_noise_seed"]))
    # stop condition: population weight must reach the candidate neighbourhood (cells within one candidate spacing)
    r = max(dom.lx, dom.ly) / int(d["candidates_n"])
    cx, cy = np.meshgrid(grid.x_centres(), grid.y_centres())
    for f, w in weights.items():
        near = np.zeros(grid.shape, bool)
        for sx, sy in zip(cands.x, cands.y):
            near |= np.hypot(cx - sx, cy - sy) <= r / 2
        if not (w["population"][near] > 0).any():
            raise RuntimeError(f"STOP: population weight of fixture {f} is zero near every candidate")
    return Setup(cfg, d, pc, SolverOptions(**cfg.get("solver", {})), float(cfg["inverse"]["noise_std"]), obs, cands, held,
                 ResponseCache(work / "responses"), counts, weights, low, specs)


def truth_field(st: Setup, s: CaseSpec) -> np.ndarray:
    """Matched truth = base + sum q_j g_j of the cached responses (the model is affine in q); checked once against FiPy."""
    resp = st.cache.get(s.scenario, st.solver)
    return np.asarray(resp.field(np.array(s.q)).transpose("time", "y", "x").values, float)


def build_cases(st: Setup, role: str, weight: np.ndarray | None, eval_sites: D.Candidates | None = None
                ) -> list[D.DesignCase]:
    n0 = int(st.d[f"{role}_noise_seed"])
    out = []
    for i, s in enumerate(st.specs[role]):
        resp = st.cache.get(s.scenario, st.solver)
        tf = resp.field(np.array(s.q))
        out.append(D.build_case(s.name, role, s.scenario, np.array(s.q), s.scenario, st.cands, st.opts(weight, eval_sites),
                                n0 + i, int(st.d["noise_reps"]), st.cache, truth_field=tf))
    return out


def verify_truth(st: Setup) -> float:
    s = st.specs["design"][0]
    ref = run_forward(D.FIPY, s.scenario, np.array(s.q), st.solver).concentration.transpose("time", "y", "x").values
    mine = truth_field(st, s)
    err = float(np.abs(ref - mine).max() / np.abs(ref).max())
    if err > 1e-6:
        raise RuntimeError(f"affine truth differs from the FiPy truth by {err:.2e} (relative max)")
    return err


# ------------------------------------------------------------------------------------------------ alpha (validation only)
def select_alpha(st: Setup) -> dict[str, Any]:
    """Phase-1 procedure: alpha-independent layouts (random + spread), VALIDATION cases, area weights; lowest mean RMSE wins,
    exact tie -> smaller alpha."""
    vcs = build_cases(st, "validation", None)
    k = D.Constraints()
    lays: list[D.Layout] = []
    for b in st.d["budgets"]:
        lays += D.random_layouts(st.cands, b, int(st.d["n_random"]), k, int(st.d["layout_seed"]) + b) + [D.spread_layout(st.cands, b, k)]
    rows = []
    for a in st.d["alpha_grid"]:
        df = D.evaluate_validation({"v": lays}, vcs, dataclasses.replace(st.opts(None), alpha=float(a)))
        rows.append({"alpha": float(a), "criterion": float(df["rmse"].mean()), "median_rmse": float(df["rmse"].median())})
    best = min(rows, key=lambda r: (r["criterion"], r["alpha"]))
    return {"alpha": best["alpha"], "table": rows, "criterion": "mean regional RMSE (area weights) over budgets, random+spread "
            "layouts, validation cases, noise reps; exact tie -> smaller alpha", "objective": "area"}


# ------------------------------------------------------------------------------------------------ evaluation
def eval_layouts(st: Setup, method: str, fixture: str, budget: int, lays: list[D.Layout], tcs: list[D.DesignCase],
                 forms: dict[str, CellForm], topts: D.DesignOptions) -> pd.DataFrame:
    rows = []
    for k, lay in enumerate(lays):
        sensors = ",".join(st.cands.ids[i] for i in lay)
        for c in tcs:
            for r in range(c.n_noise):
                f = D.fit_layout(c, lay, r, topts)
                err = np.sqrt(forms[c.name].e2(f["q_hat"]))
                area = P.area_weighted_rmse(err, 1.0)
                if abs(area ** 2 - f["mse_w"]) > 1e-6 * f["mse_w"] + 1e-18:
                    raise RuntimeError("per-cell error form disagrees with design.fit_layout mse_w")
                row = {"method": method, "design_fixture": fixture, "budget": budget, "layout_id": f"{method}-{k}",
                       "case": c.name, "rep": r, "sensors": sensors, "area_rmse": area, "worst_cell_rmse": float(err.max()),
                       "heldout_rmse": f["heldout_rmse"]}
                for y, cnt in st.counts.items():
                    row[f"pop_rmse_{y}"] = P.pop_weighted_rmse(err, cnt)
                    row[f"lowq_rmse_{y}"] = P.area_weighted_rmse(err, 1.0, st.low[y].astype(float))
                rows.append(row)
    return pd.DataFrame(rows)


def unit_file(work: Path, fixture: str, method: str, budget: int) -> Path:
    return work / "units" / f"{fixture}__{method}__b{budget}.csv"


def save_unit(path: Path, df: pd.DataFrame, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path.with_suffix(".json"), meta)
    df.to_csv(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").replace(path)  # the csv is the completion marker


# ------------------------------------------------------------------------------------------------ analysis tables
def view(raw: pd.DataFrame, fixture: str) -> pd.DataFrame:
    """Rows seen from `fixture`: its own designed layouts plus the fixture-independent baselines; adds pop_rmse, lowq_rmse."""
    v = raw[raw["design_fixture"].isin([fixture, "-"])].copy()
    v["pop_rmse"], v["lowq_rmse"] = v[f"pop_rmse_{fixture}"], v[f"lowq_rmse_{fixture}"]
    return v


def case_means(v: pd.DataFrame, metrics: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """Mean over noise reps (and over the layouts of `random`) per (method, budget, case)."""
    return v.groupby(["method", "budget", "case"], as_index=False)[list(metrics)].mean()


def paired_table(cm: pd.DataFrame, ref: pd.DataFrame, stats: dict[str, Any], metrics: list[str], extra: dict[str, Any]
                 ) -> list[dict[str, Any]]:
    """Per method != area and budget: per-case difference (method minus the area layout) and its summaries."""
    out = []
    for (m, b), g in cm.groupby(["method", "budget"]):
        if m == "area":
            continue
        r = ref[ref["budget"] == b].set_index("case")
        for met in metrics:
            diff = (g.set_index("case")[met] - r[met]).dropna().to_numpy()
            lo, hi = bootstrap_ci(diff, int(stats["n_boot"]), int(stats["boot_seed"]), float(stats["level"]))
            out.append({**extra, "method": m, "budget": b, "metric": met, "n_cases": diff.size, "mean_diff": diff.mean(),
                        "sd_diff": diff.std(ddof=1) if diff.size > 1 else np.nan, "boot_lo": lo, "boot_hi": hi,
                        "frac_cases_lower": float((diff < 0).mean()), "frac_cases_higher": float((diff > 0).mean()),
                        "mean_area_layout": float(r[met].mean()), "interval": BOOT_LABEL})
    return out


def resolved(row: dict[str, Any], frac: float) -> bool:
    """Pre-registered rule for pop_rmse: interval excludes 0 and the same direction in >= frac of the cases."""
    f = row["frac_cases_lower"] if row["mean_diff"] < 0 else row["frac_cases_higher"]
    return bool((row["boot_lo"] > 0 or row["boot_hi"] < 0) and f >= frac)


def summarise_tables(raw: pd.DataFrame, st: Setup) -> dict[str, pd.DataFrame]:
    stats, fixtures = st.cfg["stats"], sorted(set(raw["design_fixture"]) - {"-"})
    summ, paired, cross, pareto = [], [], [], []
    for fx in fixtures:
        v = view(raw, fx)
        cm = case_means(v, DIFF_METRICS)
        for (m, b), g in cm.groupby(["method", "budget"]):
            row = {"fixture": fx, "method": m, "budget": b, "n_cases": len(g)}
            for met in DIFF_METRICS:
                row[f"{met}_mean"], row[f"{met}_sd_over_cases"] = g[met].mean(), g[met].std(ddof=1)
            summ.append(row)
        ref = cm[cm["method"] == "area"]
        paired += paired_table(cm, ref, stats, list(DIFF_METRICS), {"fixture": fx})
        for (m, b), g in raw[raw["design_fixture"] == fx].groupby(["method", "budget"]):
            r = g[g["method"] == m]
            pareto.append({"fixture": fx, "method": m, "budget": b,
                           "beta": float("nan") if m == "capped" else {beta_name(x): x for x in st.pcfg["betas"]}[m],
                           "test_pop_rmse": r[f"pop_rmse_{fx}"].mean(), "test_area_rmse": r["area_rmse"].mean()})
        for ev in fixtures:  # design on fx, evaluate pop-RMSE on ev, paired against the area layout designed on fx
            sub = raw[raw["design_fixture"] == fx]
            cmx = case_means(sub.assign(pop_rmse=sub[f"pop_rmse_{ev}"]), ["pop_rmse"])
            cross += paired_table(cmx, cmx[cmx["method"] == "area"], stats, ["pop_rmse"], {"design_fixture": fx, "eval_fixture": ev})
    paired_df = pd.DataFrame(paired)
    paired_df["resolved"] = [resolved(r, float(stats["resolved_frac"])) if r["metric"] == "pop_rmse" else np.nan
                             for r in paired_df.to_dict("records")]
    return {"summary": pd.DataFrame(summ), "paired": paired_df, "cross_fixture": pd.DataFrame(cross), "pareto": pd.DataFrame(pareto)}


# ------------------------------------------------------------------------------------------------ figures
def figures(out: Path, st: Setup, tabs: dict[str, pd.DataFrame], layouts: pd.DataFrame, profile: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    ext = (0, float(st.cfg["scenario"]["domain"]["lx"]) / 1e3, 0, float(st.cfg["scenario"]["domain"]["ly"]) / 1e3)
    tag = f"{STAMP} ({profile}, synthetic fixtures)"
    fixtures = sorted(tabs["pareto"]["fixture"].unique(), key=list(st.counts).index)
    objs = objective_names(st.pcfg["betas"])

    def finish(fig: Any, name: str) -> None:
        fig.text(0.995, 0.005, tag, ha="right", va="bottom", fontsize=7, color="0.4")
        fig.savefig(fd / name, dpi=130)
        plt.close(fig)

    # 1: population and weight maps
    cols = [("population", "N (people per cell)"), ("population", "w_pop / w_area (-)"), ("mixed_0.5", "w_mixed(0.5) / w_area (-)"),
            ("capped", "w_capped(c) / w_area (-)")]
    fig, ax = plt.subplots(len(fixtures), 4, figsize=(14, 3.1 * len(fixtures)), squeeze=False)
    for i, fx in enumerate(fixtures):
        for j, (o, lab) in enumerate(cols):
            a = ax[i, j]
            z = st.counts[fx] if j == 0 else st.weights[fx][o] * st.counts[fx].size
            im = a.imshow(z, origin="lower", extent=ext, cmap="viridis")
            fig.colorbar(im, ax=a, label=lab, fraction=0.046)
            a.set(xlabel="x (km)", ylabel="y (km)", title=f"{fx}: {lab.split(' (')[0]}")
    fig.tight_layout()
    finish(fig, "fig1_population_weight_maps.png")

    # 2: layouts per objective for two fixtures
    fb = int(st.cfg["figures"]["layout_budget"])
    lf = [f for f in st.cfg["figures"]["layout_fixtures"] if f in fixtures]
    if lf:
        fig, ax = plt.subplots(len(lf), len(objs), figsize=(2.6 * len(objs), 2.9 * len(lf)), squeeze=False)
        for i, fx in enumerate(lf):
            for j, o in enumerate(objs):
                a = ax[i, j]
                a.imshow(st.counts[fx], origin="lower", extent=ext, cmap="Greys")
                r = layouts[(layouts["fixture"] == fx) & (layouts["method"] == o) & (layouts["budget"] == fb)]
                if len(r):
                    for n, ci in enumerate(json.loads(r.iloc[0]["order"])[:fb]):
                        a.plot(st.cands.x[ci] / 1e3, st.cands.y[ci] / 1e3, "o", color="tab:red", ms=7)
                        a.annotate(str(n + 1), (st.cands.x[ci] / 1e3, st.cands.y[ci] / 1e3), color="w", fontsize=6, ha="center", va="center")
                a.set(title=f"{fx}\n{o}", xlabel="x (km)", ylabel="y (km)")
                a.title.set_fontsize(7)
        fig.suptitle(f"greedy layouts, budget {fb} (numbers = selection order; grey = people per cell)", fontsize=9)
        fig.tight_layout()
        finish(fig, "fig2_layouts.png")

    # 3: Pareto
    par = tabs["pareto"]
    fig, ax = plt.subplots(1, len(fixtures), figsize=(3.3 * len(fixtures), 3.6), squeeze=False)
    for i, fx in enumerate(fixtures):
        a = ax[0, i]
        for b, c in zip(sorted(par["budget"].unique()), ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
            p = par[(par["fixture"] == fx) & (par["budget"] == b)]
            pb = p[p["method"] != "capped"].sort_values("beta")
            a.plot(pb["test_area_rmse"], pb["test_pop_rmse"], "o-", color=c, ms=3, label=f"budget {b}")
            if len(pb):
                a.annotate("0", (pb.iloc[0]["test_area_rmse"], pb.iloc[0]["test_pop_rmse"]), fontsize=6)
                a.annotate("1", (pb.iloc[-1]["test_area_rmse"], pb.iloc[-1]["test_pop_rmse"]), fontsize=6)
            pc = p[p["method"] == "capped"]
            a.plot(pc["test_area_rmse"], pc["test_pop_rmse"], "*", color=c, ms=8)
        a.set(xlabel="test area RMSE (ug/m^3)", ylabel="test population RMSE (ug/m^3)", title=fx)
        a.title.set_fontsize(8)
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=6, title="beta 0 -> 1 (labels); star = capped", title_fontsize=6)
    fig.tight_layout()
    finish(fig, "fig3_pareto.png")

    # 4: cross-fixture matrix
    mo, cf = st.cfg["figures"]["matrix_objective"], tabs["cross_fixture"]
    budgets = sorted(cf["budget"].unique())
    fig, ax = plt.subplots(1, len(budgets), figsize=(3.6 * len(budgets), 3.9), squeeze=False)
    for j, b in enumerate(budgets):
        m = cf[(cf["method"] == mo) & (cf["budget"] == b)].pivot(index="design_fixture", columns="eval_fixture", values="mean_diff")
        m = m.loc[fixtures, fixtures]
        v = max(float(np.nanmax(np.abs(m.to_numpy()))), 1e-12)
        im = ax[0, j].imshow(m.to_numpy(), cmap="RdBu_r", vmin=-v, vmax=v)
        ax[0, j].set(xticks=range(len(fixtures)), yticks=range(len(fixtures)), xlabel="evaluated on fixture", title=f"budget {b}")
        ax[0, j].set_xticklabels([f[:8] for f in fixtures], rotation=60, fontsize=6)
        ax[0, j].set_yticklabels([f[:8] for f in fixtures] if j == 0 else [], fontsize=6)
        if j == 0:
            ax[0, j].set_ylabel("designed on fixture")
        fig.colorbar(im, ax=ax[0, j], fraction=0.046, label="d pop-RMSE vs area layout (ug/m^3)")
    fig.suptitle(f"layout of objective '{mo}' minus area layout, test population RMSE (negative = population design better)", fontsize=8)
    fig.tight_layout()
    finish(fig, "fig4_cross_fixture.png")


# ------------------------------------------------------------------------------------------------ main
def finalise(out: Path, st: Setup, profile: str, budgets: list[int], fixtures: list[str], objs: list[str]) -> None:
    work = out / "work"
    raws, lays = [], []
    for fx in fixtures:
        for o in objs:
            for b in budgets:
                raws.append(pd.read_csv(unit_file(work, fx, o, b)))
                lays.append({"fixture": fx, "method": o, "budget": b, **json.loads(unit_file(work, fx, o, b).with_suffix(".json").read_text())})
    for m in ("random", "spread"):
        for b in budgets:
            raws.append(pd.read_csv(unit_file(work, "-", m, b)))
    raw = pd.concat(raws, ignore_index=True)
    raw.to_csv(out / "test_raw.csv", index=False)
    layouts = pd.DataFrame(lays)
    layouts["layout"] = layouts["layout"].map(json.dumps)
    layouts["order"] = layouts["order"].map(json.dumps)
    layouts.drop(columns=["trace_s", "eval_s"]).to_csv(out / "layouts.csv", index=False)
    tabs = summarise_tables(raw, st)
    for k, t in tabs.items():
        t.to_csv(out / f"{k}.csv", index=False)
    figures(out, st, tabs, layouts, profile)


def run(profile: str, cfg_path: str | None, out_arg: str | None, summary_only: bool) -> int:
    cfg = load_config(ROOT / (cfg_path or CONFIGS[profile]))
    if profile == "pilot":
        cfg = _merge(cfg, cfg["pilot"])
    out = ROOT / (out_arg or OUTS[profile])
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    d = cfg["design"]
    budgets = sorted(int(b) for b in d["budgets"])
    st = make_setup(cfg, work)
    objs_all = objective_names(st.pcfg["betas"])
    fixtures = cfg["run"]["fixtures"] or list(st.counts)
    objs = cfg["run"]["objectives"] or objs_all
    full = fixtures == list(st.counts) and objs == objs_all
    timing: dict[str, Any] = {"profile": profile}
    write_json(out / "config.json", cfg)
    # 1. alpha on validation only
    ap = out / "alpha_selected.json"
    if ap.exists():
        sel = json.loads(ap.read_text())
    else:
        t0 = time.perf_counter()
        say("verifying affine truth against FiPy, selecting alpha on validation")
        timing["truth_check_rel_err"] = verify_truth(st)
        sel = select_alpha(st)
        sel["config_hash_without_alpha"] = frozen_hash(cfg, 0.0)
        write_json(ap, sel)
        timing["alpha_s"] = time.perf_counter() - t0
    if sel["config_hash_without_alpha"] != frozen_hash(cfg, 0.0):
        raise RuntimeError("alpha_selected.json was made with a different frozen config; use a new --out")
    st.alpha = float(sel["alpha"])
    say(f"alpha = {st.alpha:g} (validation, area objective)")
    if summary_only:
        finalise(out, st, profile, budgets, fixtures, objs)
        return 0
    # 2. freeze before any test evaluation
    pp = out / "provenance.json"
    fh = frozen_hash(cfg, st.alpha)
    if pp.exists():
        assert_frozen(out, cfg, st.alpha)
    else:
        write_json(pp, {**provenance(cfg, {k: v for k, v in d.items() if k.endswith("seed")},
                                     {"experiment": STAMP, "profile": profile, "note": "synthetic idealised experiment"}),
                        "frozen_config_hash": fh, "frozen_payload": frozen_payload(cfg, st.alpha),
                        "frozen_objectives": objs_all, "alpha_selected": st.alpha, "frozen_before_test_utc": pd.Timestamp.now("UTC").isoformat()})
    # 3. test cases (evaluation only), after the freeze
    t0 = time.perf_counter()
    topts = st.opts(None, st.heldout)
    tcs = build_cases(st, "test", None, st.heldout)
    forms = {s.name: cell_form(st.cache.get(s.scenario, st.solver), truth_field(st, s), float(d["t_exclude"])) for s in st.specs["test"]}
    dcs0 = build_cases(st, "design", None)
    vcs0 = build_cases(st, "validation", None)
    D.assert_disjoint(dcs0, tcs, vcs0)
    timing["test_cases_s"] = time.perf_counter() - t0
    # 4. units
    k = D.Constraints()
    for m in ("random", "spread"):
        for b in budgets:
            if unit_file(work, "-", m, b).exists():
                continue
            t0 = time.perf_counter()
            lays = (D.random_layouts(st.cands, b, int(d["n_random"]), k, int(d["layout_seed"]) + b) if m == "random"
                    else [D.spread_layout(st.cands, b, k)])
            df = eval_layouts(st, m, "-", b, lays, tcs, forms, topts)
            save_unit(unit_file(work, "-", m, b), df, {"layout": [list(x) for x in lays], "order": [], "trace_s": 0.0, "eval_s": time.perf_counter() - t0})
            say(f"baseline {m} budget {b}")
    memo: dict[tuple[str, int], D.GreedyTrace] = {}
    for fx in fixtures:
        for o in objs:
            todo = [b for b in budgets if not unit_file(work, fx, o, b).exists()]
            if not todo:
                continue
            w = st.weights[fx][o]
            t0 = time.perf_counter()
            key = (hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest(), max(todo))
            if key not in memo:
                dcs = build_cases(st, "design", w)
                memo[key] = D.greedy_trace(st.cands, max(todo), dcs, k, st.opts(w), "total")
            tr = memo[key]
            trace_s = time.perf_counter() - t0
            for b in todo:
                lay = tuple(sorted(tr.order[:b]))
                if cfg["run"]["verify_prefix"]:
                    if D.greedy_trace(st.cands, b, build_cases(st, "design", w), k, st.opts(w), "total").layout != lay:
                        raise RuntimeError("greedy prefix property violated")
                t1 = time.perf_counter()
                df = eval_layouts(st, o, fx, b, [lay], tcs, forms, topts)
                save_unit(unit_file(work, fx, o, b), df, {"layout": list(lay), "order": list(tr.order[:b]),
                                                         "trace_s": trace_s if b == max(todo) else 0.0, "eval_s": time.perf_counter() - t1})
            say(f"{fx} / {o}: greedy to budget {max(todo)} in {trace_s:.1f} s")
    # 5. sanity, freeze guard
    if "uniform" in fixtures and "area" in objs:
        lay = {("uniform", o, b): tuple(json.loads(unit_file(work, "uniform", o, b).with_suffix(".json").read_text())["layout"])
               for o in objs for b in budgets}
        check_uniform_equals_area(lay, budgets, objs)
        say("sanity: uniform-fixture layouts equal area layouts for every objective and budget")
    elif full:
        raise RuntimeError("sanity check could not run")
    assert_frozen(out, cfg, st.alpha)
    # 6. outputs
    finalise(out, st, profile, budgets, fixtures, objs)
    meta = [json.loads(unit_file(work, fx, o, b).with_suffix(".json").read_text()) for fx in fixtures for o in objs for b in budgets]
    tr_s = [m["trace_s"] for m in meta if m["trace_s"] > 0]
    timing.update(n_units=len(meta), trace_s_each=tr_s, eval_s_mean=float(np.mean([m["eval_s"] for m in meta])),
                  total_s=time.perf_counter() - T0, freeze_hash_verified=fh)
    old = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
    write_json(out / "timing.json", {**old, **timing})
    prov = json.loads(pp.read_text())
    write_json(pp, {**prov, "freeze_guard_passed_after_test": True, "sanity_uniform_equals_area": "uniform" in fixtures})
    say(f"done -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=list(CONFIGS), default="smoke")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)
    return run(a.profile, a.config, a.out, a.summary_only)


if __name__ == "__main__":
    sys.exit(main())

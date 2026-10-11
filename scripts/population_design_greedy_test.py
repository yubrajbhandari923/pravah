# pyright: reportGeneralTypeIssues=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportOperatorIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportReturnType=false, reportAssignmentType=false
"""EXP-022: greedy vs exact/best-found layouts of EXP-020 evaluated on the TEST cases (synthetic toy domain).

    python scripts/population_design_greedy_test.py --profile {smoke,production} [--placements 0 1] [--out DIR] [--ref DIR]

Follow-up to EXP-020 (scripts/population_design_exact.py, imported read-only).  Reuses the EXP-020 response-base cache (read-only; a cache
miss raises) and the same frozen config (the hash of config + EXP-020 alphas must equal the one in the EXP-020 provenance.json / run_guard.json).
Layouts are NOT reselected: the exact/best-found layouts are read from the EXP-020 selection.csv; the greedy layouts are recomputed with the
vectorised greedy and must equal the stored ones.  Both are scored on the test cases with the EXP-020 evaluator.  The uniform fixture is added
(selected here, cheap) and the pre-registered check "uniform population/mixed/capped layouts == area layouts" is executed.
Unit of analysis: 4 source placements.  Per-placement means are shown; the bootstrap over 4 placements is descriptive, not a CI.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pravah.config import SolverOptions, load_config
from pravah.forward import FIPY
from pravah.inverse import Q_REF_DEFAULT, ResponseCache, cache_key
from pravah.io import provenance, write_json

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("population_design_exact", ROOT / "scripts/population_design_exact.py")
PE = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["population_design_exact"] = PE
_spec.loader.exec_module(PE)  # type: ignore[union-attr]
PD = PE.PD

STAMP = "EXP-022"
CONFIGS = {"smoke": "configs/population_toy_exact_smoke.json", "production": "configs/population_toy_exact.json"}
REFS = {"smoke": "results/phase3-population-toy-exact-smoke", "production": "results/phase3-population-toy-exact"}
OUTS = {"smoke": "results/phase3-population-toy-exact-greedy-smoke", "production": "results/phase3-population-toy-exact-greedy"}
METRICS = PE.METRICS
BOOT_LABEL = "percentile bootstrap over source placements (4 units in production): descriptive, not a confidence interval"
say = PE.say


# ------------------------------------------------------------------------------------------------ guards
class ReadOnlyCache(ResponseCache):
    """EXP-020 response cache that refuses to compute (and so to write) anything."""

    def get(self, scenario: Any, solver: SolverOptions = SolverOptions(), *a: Any, **k: Any) -> Any:
        if a or set(k) - {"forward"}:
            raise ValueError("ReadOnlyCache supports the default q_ref / inflow_ref only")
        key = cache_key(scenario, solver, Q_REF_DEFAULT, k.get("forward", FIPY))
        if key not in self._mem and not (self.directory / f"{key}.npz").exists():
            raise FileNotFoundError(f"response-base cache miss ({key[:12]}): the EXP-020 cache must cover this run")
        return super().get(scenario, solver, **k)


def verify_reference(cfg: dict[str, Any], guard: dict[str, Any], prov: dict[str, Any], alphas: dict[str, float]) -> str:
    """Frozen config (without and with the EXP-020 alphas) must equal what EXP-020 recorded; returns the with-alpha hash."""
    if PE.frozen_hash(cfg, None) != guard["config_hash_without_alpha"]:
        raise RuntimeError("config differs from the EXP-020 run_guard.json (hash without alpha)")
    fh = PE.frozen_hash(cfg, alphas)
    if fh != prov["frozen_config_hash"]:
        raise RuntimeError(f"config + alphas differ from the EXP-020 provenance.json frozen hash ({fh} vs {prov['frozen_config_hash']})")
    return fh


def uniform_check(sel: pd.DataFrame, path: Path | None = None) -> dict[str, Any]:
    """Pre-registered stop check: on the uniform fixture every objective's layout equals the area layout, per budget and placement."""
    objs = [o for o in sorted(sel["objective"].unique()) if o != "area"]
    res = {"passed": True, "n_compared": 0, "mismatches": []}
    for p, gp in sel.groupby("placement"):
        lay = {("uniform", r.objective, int(r.budget)): tuple(json.loads(r.layout)) for r in gp.itertuples()}
        budgets = sorted({b for _, _, b in lay})
        try:
            PD.check_uniform_equals_area(lay, budgets, objs)
        except RuntimeError as e:
            res["passed"] = False
            res["mismatches"].append({"placement": int(p), "error": str(e)})
        res["n_compared"] += len(objs) * len(budgets)
    if path is not None:
        write_json(path, res)
    if not res["passed"]:
        raise RuntimeError(f"uniform check failed: {res['mismatches']}")
    return res


# ------------------------------------------------------------------------------------------------ layouts and evaluation
def recompute_greedy(pl: Any, alpha: float, bmax: int) -> dict[int, tuple[int, ...]]:
    st = pl.st
    st.alpha = alpha
    forms = [PD.cell_form(st.cache.get(s.scenario, st.solver), PD.truth_field(st, s), float(st.d["t_exclude"])) for s in st.specs["design"]]
    ev = PE.Evaluator(PD.build_cases(st, "design", None), forms, pl.W, alpha, st.noise_std)
    return {c: PE.greedy_order(ev, c, bmax)[0] for c in range(ev.K)}


def check_greedy(sel: pd.DataFrame, pl: Any, order: dict[int, tuple[int, ...]]) -> None:
    for r in sel.itertuples():
        mine = sorted(order[pl.col(r.type, r.draw, r.objective)][:r.budget])
        if mine != json.loads(r.greedy_layout):
            raise RuntimeError(f"recomputed greedy layout differs from EXP-020 selection.csv: {mine} vs {r.greedy_layout} ({r.type}, {r.draw}, {r.objective}, {r.budget})")


def variants(sel: pd.DataFrame) -> pd.DataFrame:
    """Two rows per selection row (objective 'obj|exact' with the selected layout, 'obj|greedy' with the greedy one) for eval_stage."""
    a = sel[["placement", "type", "draw", "objective", "budget", "layout"]].assign(objective=sel["objective"] + "|exact")
    g = sel[["placement", "type", "draw", "objective", "budget", "greedy_layout"]].rename(columns={"greedy_layout": "layout"})
    return pd.concat([a, g.assign(objective=sel["objective"] + "|greedy")], ignore_index=True)


def wide_table(raw: pd.DataFrame, sel: pd.DataFrame) -> pd.DataFrame:
    """One row per setting (type, placement, draw, objective, budget): test metrics (mean over test cases) of both layouts + design gap."""
    raw = raw.assign(method=raw["objective"].str.split("|").str[1], objective=raw["objective"].str.split("|").str[0])
    key = ["type", "placement", "draw", "objective", "budget"]
    m = raw.groupby(key + ["method"], as_index=False)[list(METRICS)].mean().pivot(index=key, columns="method", values=list(METRICS))
    m.columns = [f"{a}_{b}" for a, b in m.columns]
    out = m.reset_index().merge(sel[key + ["layout", "greedy_layout", "method", "value", "greedy_value", "gap_rel"]].rename(
        columns={"layout": "layout_exact", "greedy_layout": "layout_greedy", "method": "selection_method"}), on=key)
    out["same_layout"] = out["layout_exact"] == out["layout_greedy"]
    return out


def paired_greedy_exact(w: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    """Per type / objective / budget / metric: greedy - exact on test.  Per-placement means of the setting-wise differences are shown and
    bootstrapped over placements; frac_greedy_worse is over settings (strictly positive difference)."""
    rows = []
    for (t, o, b), g in w.groupby(["type", "objective", "budget"]):
        for m in METRICS:
            d = g[f"{m}_greedy"] - g[f"{m}_exact"]
            pm = d.groupby(g["placement"]).mean()
            lo, hi = PD.bootstrap_ci(pm.to_numpy(), int(stats["n_boot"]), int(stats["boot_seed"]), float(stats["level"]))
            pe_, pg = g[f"{m}_exact"].groupby(g["placement"]).mean(), g[f"{m}_greedy"].groupby(g["placement"]).mean()
            rows.append({"type": t, "objective": o, "budget": b, "metric": m, "n_settings": len(d), "n_placements": len(pm),
                         "mean_diff": float(pm.mean()), "placement_means": json.dumps([float(x) for x in pm]), "boot_lo": lo, "boot_hi": hi,
                         "frac_greedy_worse": float((d > 0).mean()), "n_identical_layout": int(g["same_layout"].sum()),
                         "mean_exact": float(pe_.mean()), "mean_greedy": float(pg.mean()), "ratio_greedy_over_exact": float(pg.mean() / pe_.mean()),
                         "interval": BOOT_LABEL})
    return pd.DataFrame(rows)


def matched(w: pd.DataFrame) -> pd.DataFrame:
    """Relative test difference greedy/exact - 1 in the metric matching the objective (area -> area RMSE, else population RMSE)."""
    m = np.where(w["objective"] == "area", "area_rmse", "pop_rmse")
    ex = np.where(m == "area_rmse", w["area_rmse_exact"], w["pop_rmse_exact"])
    gr = np.where(m == "area_rmse", w["area_rmse_greedy"], w["pop_rmse_greedy"])
    return w.assign(metric=m, test_rel_diff=gr / ex - 1)


def summarise(w: pd.DataFrame, pg: pd.DataFrame) -> dict[str, Any]:
    wm = matched(w)
    wm = wm[wm["type"] != "uniform"]
    out: dict[str, Any] = {"n_settings": int(len(wm)), "n_identical_layout": int(wm["same_layout"].sum()), "excludes": "uniform fixture"}
    out["frac_greedy_worse_test"] = {m: float(((w[f"{m}_greedy"] - w[f"{m}_exact"]) > 0)[w["type"] != "uniform"].mean()) for m in METRICS}
    out["frac_greedy_worse_matched_by_budget"] = {int(b): float((g["test_rel_diff"] > 0).mean()) for b, g in wm.groupby("budget")}
    a = pg[(pg["objective"] == "area") & (pg["metric"] == "area_rmse") & (pg["type"] != "uniform")]
    out["area_objective_area_rmse_ratio_by_budget_type"] = {f"{int(b)}|{t}": float(r) for (b, t), r in
                                                           a.groupby(["budget", "type"])["ratio_greedy_over_exact"].first().items()}
    out["area_objective_area_rmse_ratio_by_budget"] = {int(b): float(g["ratio_greedy_over_exact"].mean()) for b, g in a.groupby("budget")}
    out["spearman_design_gap_vs_test_rel_diff"] = {"pooled": float(wm["gap_rel"].corr(wm["test_rel_diff"], method="spearman")),
                                                   **{f"budget_{int(b)}": float(g["gap_rel"].corr(g["test_rel_diff"], method="spearman")) for b, g in wm.groupby("budget")}}
    bins = pd.cut(wm["gap_rel"], [-np.inf, 1e-9, 0.1, 1.0, np.inf], labels=["0", "(0,0.1]", "(0.1,1]", ">1"])
    out["test_rel_diff_by_design_gap_bin"] = {str(k): {"n": int(len(g)), "median": float(g["test_rel_diff"].median()), "mean": float(g["test_rel_diff"].mean()),
                                                         "frac_worse": float((g["test_rel_diff"] > 0).mean())} for k, g in wm.groupby(bins, observed=True)}
    return out


# ------------------------------------------------------------------------------------------------ figures
def figures(out: Path, w: pd.DataFrame, profile: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    tag = f"{STAMP} ({profile}, synthetic fixtures)"
    wm = matched(w[w["type"] != "uniform"])
    objs = sorted(wm["objective"].unique())
    budgets = sorted(wm["budget"].unique())

    def finish(fig: Any, name: str) -> None:
        fig.text(0.995, 0.005, tag, ha="right", va="bottom", fontsize=7, color="0.4")
        fig.savefig(fd / name, dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, o in enumerate(objs):
        g = wm[wm["objective"] == o]
        ax.scatter(g["gap_rel"], g["test_rel_diff"], s=9, alpha=0.6, color=f"C{i}", label=f"{o} ({g['metric'].iloc[0]})")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set(xscale="symlog", yscale="symlog", xlabel="design gap: greedy / best design objective - 1", ylabel="test gap: greedy / exact RMSE - 1 (matched metric)")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    ax.set_title("design gap vs test gap, one point per setting", fontsize=8)
    fig.tight_layout()
    finish(fig, "fig1_design_gap_vs_test_gap.png")

    fig, ax = plt.subplots(1, len(budgets), figsize=(3.0 * len(budgets), 3.2), squeeze=False)
    for j, b in enumerate(budgets):
        for i, o in enumerate(objs):
            g = wm[(wm["budget"] == b) & (wm["objective"] == o)]
            x, y = (g["area_rmse_exact"], g["area_rmse_greedy"]) if o == "area" else (g["pop_rmse_exact"], g["pop_rmse_greedy"])
            ax[0, j].scatter(x, y, s=8, alpha=0.6, color=f"C{i}", label=o)
        lim = [min(ax[0, j].get_xlim()[0], ax[0, j].get_ylim()[0]), max(ax[0, j].get_xlim()[1], ax[0, j].get_ylim()[1])]
        ax[0, j].plot(lim, lim, color="0.4", lw=0.8)
        ax[0, j].set(title=f"budget {b}", xscale="log", yscale="log", xlabel="test RMSE, exact/best-found layout", ylabel="test RMSE, greedy layout" if j == 0 else "")
        ax[0, j].title.set_fontsize(8)
    ax[0, 0].legend(fontsize=5)
    fig.suptitle("test RMSE of greedy vs exact layouts (area objective: area RMSE; others: population RMSE); above the line = greedy worse", fontsize=8)
    fig.tight_layout()
    finish(fig, "fig2_test_rmse_greedy_vs_exact.png")


# ------------------------------------------------------------------------------------------------ main
def run(profile: str, cfg_path: str | None, out_arg: str | None, ref_arg: str | None, placements: list[int] | None) -> int:
    cfg = load_config(ROOT / (cfg_path or CONFIGS[profile]))
    ref = ROOT / (ref_arg or REFS[profile])
    guard, prov0 = json.loads((ref / "run_guard.json").read_text()), json.loads((ref / "provenance.json").read_text())
    s0 = guard["selection"]
    alphas = {str(p): float(json.loads((ref / f"alpha_selected_p{p}.json").read_text())["alpha"]) for p in s0["placements"]}
    fh = verify_reference(cfg, guard, prov0, alphas)
    pls = placements if placements is not None else list(s0["placements"])
    if not set(pls) <= set(s0["placements"]):
        raise ValueError("placements must be a subset of the EXP-020 run")
    PE.check_seeds(cfg)
    out = ROOT / (out_arg or OUTS[profile])
    PE.guard_out(out, cfg, {**s0, "placements": pls, "experiment": STAMP})
    ref_work, work = ref / "work", out / "work"
    budgets = sorted(int(b) for b in cfg["design"]["budgets"])
    n_cache0 = len(list((ref_work / "responses").glob("*.npz")))
    write_json(out / "config.json", cfg)
    pp = out / "provenance.json"
    if pp.exists():
        PE.assert_frozen(out, cfg, alphas)
    else:
        write_json(pp, {**provenance(cfg, {k: v for k, v in cfg["design"].items() if k.endswith("seed")},
                                     {"experiment": STAMP, "profile": profile, "note": "synthetic; follow-up to EXP-020, no reselection"}),
                        "frozen_config_hash": fh, "alphas": alphas, "reference": str(ref.relative_to(ROOT)), "reference_frozen_config_hash": prov0["frozen_config_hash"],
                        "cache": "EXP-020 response bases, read-only", "frozen_before_test_utc": pd.Timestamp.now("UTC").isoformat()})
    sel_all, sel_u, raws, ref_diff = [], [], [], 0.0
    for p in pls:
        alpha, t0 = alphas[str(p)], time.perf_counter()
        pl = PE.Placement(cfg, p, ref_work, s0["types"], s0["draws"])
        pu = PE.Placement(cfg, p, ref_work, ["uniform"], [0])
        for x in (pl, pu):
            x.st.cache = ReadOnlyCache(ref_work / "responses")
        sel = pd.concat([pd.read_csv(PE.selection_file(ref_work, p, b)) for b in budgets], ignore_index=True)
        check_greedy(sel, pl, recompute_greedy(pl, alpha, max(budgets)))
        say(f"placement {p}: recomputed greedy layouts equal EXP-020 selection.csv; selecting uniform layouts")
        PE.select_stage(cfg, pu, alpha, work / "uniform", {})
        su = pd.concat([pd.read_csv(PE.selection_file(work / "uniform", p, b)) for b in budgets], ignore_index=True)
        sel_all.append(sel)
        sel_u.append(su)
        for tag, x, s in (("main_v", pl, sel), ("uniform_v", pu, su)):
            for b in budgets:
                f = PE.selection_file(work / tag, p, b)
                f.parent.mkdir(parents=True, exist_ok=True)
                variants(s[s["budget"] == b]).to_csv(f, index=False)
            PE.eval_stage(cfg, x, alpha, work / tag, {})
            raws.append(pd.concat([pd.read_csv(PE.eval_file(work / tag, p, b)) for b in budgets], ignore_index=True))
        # the re-evaluated exact layouts must reproduce the EXP-020 test rows
        old = pd.concat([pd.read_csv(PE.eval_file(ref_work, p, b)) for b in budgets])
        new = raws[-2][raws[-2]["objective"].str.endswith("|exact")].assign(objective=lambda d: d["objective"].str[:-6])
        k = ["placement", "type", "draw", "objective", "budget", "case"]
        mg = old.merge(new, on=k, suffixes=("_old", "_new"))
        if len(mg) != len(old):
            raise RuntimeError("re-evaluated exact layouts do not cover the EXP-020 test rows")
        ref_diff = max(ref_diff, max(float((mg[f"{m}_old"] / mg[f"{m}_new"] - 1).abs().max()) for m in METRICS))
        say(f"placement {p} done in {time.perf_counter() - t0:.0f} s")
    if ref_diff > 1e-8:
        raise RuntimeError(f"exact layouts re-evaluated on test differ from EXP-020 test_raw rows (max rel diff {ref_diff:.2e})")
    uc = uniform_check(pd.concat(sel_u, ignore_index=True), out / "uniform_check.json")
    sel, su = pd.concat(sel_all, ignore_index=True), pd.concat(sel_u, ignore_index=True)
    w = wide_table(pd.concat(raws, ignore_index=True), pd.concat([sel, su], ignore_index=True))
    pg = paired_greedy_exact(w, cfg["stats"])
    w.to_csv(out / "greedy_vs_exact_test.csv", index=False)
    pg.to_csv(out / "paired_greedy_exact.csv", index=False)
    write_json(out / "summary.json", summarise(w, pg))
    figures(out, w, profile)
    if len(list((ref_work / "responses").glob("*.npz"))) != n_cache0:
        raise RuntimeError("EXP-020 cache changed")
    PE.assert_frozen(out, cfg, alphas)
    write_json(pp, {**json.loads(pp.read_text()), "freeze_guard_passed_after_test": True, "exact_reeval_max_rel_diff_vs_exp020": ref_diff,
                    "uniform_check_passed": uc["passed"], "total_s": time.perf_counter() - PE.T0})
    say(f"done -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=list(CONFIGS), default="smoke")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--placements", type=int, nargs="+", default=None)
    a = ap.parse_args(argv)
    return run(a.profile, a.config, a.out, a.ref, a.placements)


if __name__ == "__main__":
    sys.exit(main())

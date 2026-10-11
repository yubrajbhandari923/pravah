"""EXP-019: toy wind-error response and three inference strategies (SYNTHETIC; Phase-1 Domain).

    python scripts/wind_error.py run      --config configs/wind_error[_pilot|_smoke].json [--out DIR] [--overwrite]
    python scripts/wind_error.py report   --config ...            (tables, figures, identifiability demo, README)
    python scripts/wind_error.py estimate --config configs/wind_error.json --from results/phase3-wind-error-toy/pilot

Order of `run` (resumable; every step reads its file in OUT when present, and refuses files from a different config):
  1. validation scenarios (own seeds): alpha chosen by the validation RMSE of the nominal-wind fit B (not tuned for C);
     prior sensitivity is reported for information only (the prior is declared in the config);
  2. design ensemble (own seeds): greedy-under-nominal-wind, spread and random layouts per budget;
  3. FREEZE: hash of grid, prior, alpha, layouts, seeds written to freeze.json; only then are the test cases drawn;
  4. test: one raw CSV per truth wind condition (skipped when present); the freeze hash is re-derived and asserted.
Raw rows (OUT/raw/) are separate from tables/ and figures/."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pravah import design as D  # noqa: E402
from pravah import windunc as W  # noqa: E402
from pravah.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STAMP = "EXP-019"
T0 = time.perf_counter()
COLORS = {"A": "#1b9e77", "B": "#d95f02", "C": "#7570b3"}
NAMES = {"A": "A known wind (oracle)", "B": "B nominal wind", "C": "C wind ensemble"}


def say(msg: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)
    print(f"[{time.perf_counter() - T0:7.0f}s rss {rss:.2f}GB] {msg}", flush=True)


def paths(cfg: dict[str, Any], out: str | None) -> tuple[Path, Path]:
    o = Path(out) if out else ROOT / cfg["out"]
    return o, ROOT / cfg["work"]


def label(cfg: dict[str, Any]) -> str:
    return "EXPLORATORY " + cfg["profile"] if cfg["profile"] != "production" else "production"


def load_json(path: Path, chash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if d.get("cfg_hash") != chash:
        raise SystemExit(f"{path} was written by a different config ({d.get('cfg_hash')} != {chash}); use --overwrite")
    return d


def save_json(path: Path, d: dict[str, Any], chash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**d, "cfg_hash": chash}, indent=1, default=list))


# ------------------------------------------------------------------------------------------------ run
def do_validation(st: W.Study, chash: str, out: Path) -> dict[str, Any]:
    we = st.we
    done = load_json(out / "validation.json", chash)
    if done:
        return done
    sd = we["seeds"]
    cases = W.draw_cases("validation", we["n_validation"], sd["validation"], sd["validation_noise"], st.base_theta,
                         tuple(we["amp_range"]), len(st.cands), len(st.times))
    vk = [W.key(d, 1 + p / 100) for d, p in we["validation"]["winds"]]
    gram = st.gram(vk)
    b = we["validation"]["budget"]
    lays = [D.spread_layout(st.cands, b, st.constraints),
            *D.random_layouts(st.cands, b, we["validation"]["n_random"], st.constraints, we["layout_seed"] + 900)]
    idxs = [tuple(lay) for lay in lays]

    def score(strategy: str) -> float:
        v = [r["rmse_total"] for c in cases for k in vk for i in idxs
             for r in st.evaluate(c, k, i, gram, (strategy,), fcache)]
        return float(np.mean(v))

    fcache: dict[Any, Any] = {}
    table = []
    for a in we["alpha_grid"]:
        st.alpha = float(a)
        fcache.clear()
        table.append({"alpha": a, "rmse_B": score("B"), "rmse_C": score("C"), "rmse_A": score("A")})
        say(f"validation alpha={a}: {table[-1]}")
    best = min(table, key=lambda r: r["rmse_B"])
    st.alpha = float(best["alpha"])
    alt = {}
    full = st.ens_prior
    for name, pr in we["validation"]["alt_priors"].items():
        keys = W.prior_keys(pr["direction_deg"], pr["speed_factor"])
        if not set(keys) <= set(st.prior):
            raise SystemExit(f"alt prior {name!r} must be a subset of the declared prior grid (no new bases)")
        st.ens_prior = W.make_ensemble(st.bank, keys, st.sites, st.times)
        fcache.clear()
        alt[name] = score("C")
    st.ens_prior = full
    fcache.clear()
    alt["declared"] = score("C")
    res = {"alpha": best["alpha"], "criterion": "min validation RMSE of strategy B", "table": table,
           "prior_sensitivity_rmse_C_info_only": alt, "validation_winds": [list(k) for k in vk],
           "label": "validation scenarios (own seeds), not test"}
    save_json(out / "validation.json", res, chash)
    return res


def do_design(st: W.Study, chash: str, out: Path) -> dict[str, Any]:
    we = st.we
    done = load_json(out / "layouts.json", chash)
    if done:
        return done
    sd = we["seeds"]
    dcs = W.draw_cases("design", we["n_design"], sd["design"], sd["design_noise"], st.base_theta,
                       tuple(we["amp_range"]), len(st.cands), len(st.times))
    lays = W.design_layouts(st, dcs, st.alpha, we["budgets"], we["n_random"], we["layout_seed"], we["design_noise_reps"])
    res = {"layouts": {str(b): {"greedy": list(d["greedy"]), "spread": list(d["spread"]),
                                "random": [list(x) for x in d["random"]]} for b, d in lays.items()}}
    save_json(out / "layouts.json", res, chash)
    return res


def run(cfg: dict[str, Any], out: Path, work: Path, overwrite: bool) -> None:
    chash = W.config_hash(cfg)
    if overwrite and out.exists():
        for f in out.rglob("*"):
            if f.is_file() and f.parent.name in ("raw",):
                f.unlink()
        for n in ("validation.json", "layouts.json", "freeze.json", "timing.json"):
            (out / n).unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    timing: dict[str, Any] = {}
    t = time.perf_counter()
    st = W.Study(cfg, work)
    we = st.we
    conds = W.conditions(we["grid"])
    tkeys = list(conds)
    say(f"profile {cfg['profile']}: {len(tkeys)} truth wind conditions, prior members {len(st.prior)}")
    val = do_validation(st, chash, out)
    st.alpha = float(val["alpha"])
    say(f"alpha frozen from validation: {st.alpha}")
    des = do_design(st, chash, out)
    lays = {int(b): {"greedy": tuple(d["greedy"]), "spread": tuple(d["spread"]), "random": [tuple(x) for x in d["random"]]}
            for b, d in des["layouts"].items()}
    timing["setup_validation_design_s"] = time.perf_counter() - t
    # ---- freeze before the test cases exist
    seeds = we["seeds"]
    frozen = st.frozen(tkeys, {str(b): {k: v for k, v in d.items()} for b, d in lays.items()}, seeds)
    fz = out / "freeze.json"
    if fz.exists():
        W.assert_frozen(fz, frozen)
    else:
        say(f"freeze hash {W.write_freeze(fz, frozen)} written before the test cases are drawn")
    # ---- test cases (fresh, never used to choose anything)
    n_c = len(st.cands)
    test = W.draw_cases("test", we["n_test"], seeds["test"], seeds["test_noise"], st.base_theta, tuple(we["amp_range"]), n_c, len(st.times))
    others = [W.draw_cases("validation", we["n_validation"], seeds["validation"], seeds["validation_noise"], st.base_theta,
                           tuple(we["amp_range"]), n_c, len(st.times)),
              W.draw_cases("design", we["n_design"], seeds["design"], seeds["design_noise"], st.base_theta,
                           tuple(we["amp_range"]), n_c, len(st.times))]
    W.assert_disjoint_cases(test, *others)
    table = W.layout_table(lays)
    t = time.perf_counter()
    gram = st.gram(tkeys)
    timing["gram_s"] = time.perf_counter() - t
    fcache: dict[Any, Any] = {}
    n_eval, t_rows = 0, 0.0
    for k in tkeys:
        f = out / "raw" / f"{W.cond_id(k)}.csv"
        if f.exists() and not overwrite:
            if pd.read_csv(f, nrows=1)["cfg_hash"][0] != chash:
                raise SystemExit(f"{f} was written by a different config; use --overwrite")
            say(f"skip {f.name} (done)")
            continue
        t = time.perf_counter()
        st.truth_ens(k)
        rows = []
        for c in test:
            for lid, b, lt, idx in table:
                for r in st.evaluate(c, k, idx, gram, W.STRATEGIES, fcache):
                    rows.append({"cond": W.cond_id(k), "blocks": ",".join(conds[k]), "deg": k[0], "speed_pct": round((k[1] - 1) * 100, 6),
                                 "vec_err": W.vector_error(*k), "case": c.name, "budget": b, "layout_type": lt, "layout_id": lid,
                                 **r, "cfg_hash": chash})
                n_eval += 1
        pd.DataFrame(rows).to_csv(f, index=False)
        t_rows += time.perf_counter() - t
        say(f"{f.name}: {len(rows)} rows in {time.perf_counter() - t:.0f}s")
    W.assert_frozen(fz, st.frozen(tkeys, {str(b): {k: v for k, v in d.items()} for b, d in lays.items()}, seeds))
    say("freeze guard passed after the test")
    timing.update(rows_s=t_rows, n_case_layout_evals=n_eval, n_winds_total=len(st.bank.cache._mem),
                  basis_s=st.bank.solve_seconds, bases_computed=st.bank.cache.misses, bases_hit=st.bank.cache.hits,
                  n_conditions=len(tkeys), n_test=we["n_test"], n_layouts=len(table), profile=cfg["profile"], total_wall_s=time.perf_counter() - T0)
    prev = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
    if prev.get("rows_s") and not n_eval:
        timing = prev  # nothing recomputed: keep the earlier timing
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    (out / "provenance.json").write_text(json.dumps({"experiment": STAMP, "profile": cfg["profile"], "label": label(cfg), "cfg_hash": chash,
                                                     "config": cfg}, indent=1, default=list))
    say("run complete")


# ------------------------------------------------------------------------------------------------ report
def mean_ci(v: np.ndarray, n_boot: int, rng: np.random.Generator, level: float) -> tuple[float, float, float]:
    bs = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(bs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(v.mean()), float(lo), float(hi)


MAIN = ["rmse_total", "rmse_pop", "rmse_heldout", "amp_relerr_l2", "local_relerr_l2", "rmse_local", "rmse_external", "ext_frac_abs"]
EXTRA = ["bias_rel_mean", "chi2_per_dof", "chi2_reject", "cov90", "cov90_spread_only", "width90", "flag_any", "w_max", "n_eff_members",
         "runtime_s", "inflow_abs_abs"]


def report(cfg: dict[str, Any], out: Path, work: Path) -> None:
    we = cfg["wind_error"]
    raw: Any = pd.concat([pd.read_csv(f) for f in sorted((out / "raw").glob("*.csv"))], ignore_index=True)
    raw["ext_frac_abs"] = raw["ext_frac_err"].abs()
    raw["inflow_abs_abs"] = raw["inflow_abs_err"].abs()
    raw["chi2_reject"] = (raw["chi2_p"] < 0.05).astype(float)
    tab, fig = out / "tables", out / "figures"
    tab.mkdir(exist_ok=True)
    fig.mkdir(exist_ok=True)
    lab = label(cfg)
    bs = we["bootstrap"]
    nb, seed, level = int(bs["n"]), int(bs["seed"]), float(bs.get("level", 0.95))
    meta: Any = raw.groupby("cond")[["blocks", "deg", "speed_pct", "vec_err"]].first().reset_index()
    pc: Any = W.per_case(raw, MAIN + EXTRA)
    pc_b: Any = pc.drop(columns="layout_type").groupby(["cond", "strategy", "budget", "case"], as_index=False).mean(numeric_only=True)
    pc_all: Any = pc_b.drop(columns="budget").groupby(["cond", "strategy", "case"], as_index=False).mean(numeric_only=True)
    # ---- paired differences (percentile bootstrap over test cases)
    frames = []
    for (a, b) in (("C", "B"), ("B", "A")):
        for m in MAIN + ["bias_rel_mean", "chi2_per_dof", "cov90"]:
            frames.append(W.paired_bootstrap(pc_all, a, b, m, ["cond"], nb, seed, level).assign(pool="all budgets and layout types"))
            frames.append(W.paired_bootstrap(pc_b, a, b, m, ["cond", "budget"], nb, seed, level).assign(pool="layout types pooled"))
            frames.append(W.paired_bootstrap(pc, a, b, m, ["cond", "budget", "layout_type"], nb, seed, level).assign(pool="by layout type"))
    pd.concat(frames).merge(meta, on="cond").to_csv(tab / "paired_differences.csv", index=False)
    # ---- response curves with bootstrap intervals over cases
    rng = np.random.default_rng(seed)
    cur = []
    for (c, s, b), sub in cast(Any, pc_b.groupby(["cond", "strategy", "budget"])):
        for m in MAIN + EXTRA:
            mu, lo, hi = mean_ci(sub[m].to_numpy(float), nb, rng, level)
            cur.append({"cond": c, "strategy": s, "budget": b, "metric": m, "mean": mu, "ci_lo": lo, "ci_hi": hi, "n_cases": len(sub)})
    curves: Any = pd.DataFrame(cur).merge(meta, on="cond")
    curves.to_csv(tab / "response_curves.csv", index=False)
    # ---- chi2 / flags / coverage calibration
    cal: Any = raw.groupby(["cond", "strategy"], as_index=False).agg(chi2_per_dof=("chi2_per_dof", "mean"), frac_chi2_rejects=("chi2_reject", "mean"),
                                                                 frac_flagged=("flag_any", "mean"), cov90=("cov90", "mean"),
                                                                 cov90_spread_only=("cov90_spread_only", "mean")).merge(meta, on="cond")
    cal.to_csv(tab / "calibration.csv", index=False)
    wt: Any = raw[raw.strategy == "C"].groupby("cond", as_index=False).agg(w_max=("w_max", "mean"), n_eff_members=("n_eff_members", "mean")).merge(meta, on="cond")
    wt.to_csv(tab / "posterior_weight_spread.csv", index=False)
    # ---- figures
    one: Any = curves[curves.blocks.str.contains("direction|speed")]

    def panels(x: str, metrics: list[str], rows: Any, xlabel: str, ylabel: str, name: str, strategies: str = "ABC", hline: float | None = None,
               ylim: tuple[float, float] | None = None) -> None:
        ln = "-" if x != "vec_err" else ""  # direction and speed rows share x on a vector-error axis: markers only
        bud = sorted(rows.budget.unique())
        f, axs = plt.subplots(1, len(bud), figsize=(4.2 * len(bud), 3.6), sharey=True, squeeze=False)
        for ax, b in zip(axs[0], bud):
            for s in strategies:
                for m, ls in zip(metrics, (ln, "--" if ln else "")):
                    d = rows[(rows.strategy == s) & (rows.budget == b) & (rows.metric == m)].sort_values(x)
                    if d.empty:
                        continue
                    ax.errorbar(d[x], d["mean"], yerr=[d["mean"] - d.ci_lo, d.ci_hi - d["mean"]], color=COLORS[s], ls=ls, marker="o", ms=3,
                                capsize=2, lw=1.2, label=f"{NAMES[s]}" + (f" [{m}]" if len(metrics) > 1 else ""))
            if hline is not None:
                ax.axhline(hline, color="k", lw=0.7, ls=":")
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_title(f"budget {b}")
            ax.set_xlabel(xlabel)
        axs[0][0].set_ylabel(ylabel)
        axs[0][-1].legend(fontsize=7)
        f.suptitle(f"{STAMP} ({lab}): {name}; mean over test cases and layouts, bars = percentile bootstrap {int(level * 100)}% over cases", fontsize=8)
        f.tight_layout()
        f.savefig(fig / f"{name}.png", dpi=130)
        plt.close(f)

    rm: Any = one[one.metric == "rmse_total"]
    panels("deg", ["rmse_total"], rm[rm.blocks.str.contains("direction")], "truth wind direction error (deg, speed exact)", "field RMSE (ug/m^3)", "rmse_vs_direction_error")
    panels("speed_pct", ["rmse_total"], rm[rm.blocks.str.contains("speed")], "truth wind speed error (%, direction exact)", "field RMSE (ug/m^3)", "rmse_vs_speed_error")
    panels("vec_err", ["rmse_total"], rm, "vector error |du|/|u| (= 2 sin(dtheta/2) for rotations)", "field RMSE (ug/m^3)", "rmse_vs_vector_error")
    for m, nm, yl in (("amp_relerr_l2", "amplitude_error_vs_vector_error", "relative L2 error of emission amplitudes"),
                      ("rmse_pop", "pop_weighted_rmse_vs_vector_error", "population-weighted RMSE (ug/m^3)"),
                      ("rmse_heldout", "heldout_rmse_vs_vector_error", "held-out sensor RMSE (ug/m^3)")):
        panels("vec_err", [m], one[one.metric == m], "vector error |du|/|u|", yl, nm)
    ex: Any = one[one.metric.isin(["rmse_local", "rmse_external"])]
    panels("vec_err", ["rmse_local", "rmse_external"], ex[ex.strategy == "B"], "vector error |du|/|u|", "RMSE of component field (ug/m^3)", "local_vs_external_rmse_B", strategies="B")
    # chi2/dof with flag-saturation overlay
    cc: Any = cal[cal.blocks.str.contains("direction|speed")]
    f, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for s in "ABC":
        d = cc[cc.strategy == s].sort_values("vec_err")
        axs[0].plot(d.vec_err, d.chi2_per_dof, "o", color=COLORS[s], label=NAMES[s], ms=3)
        axs[1].plot(d.vec_err, d.frac_chi2_rejects, "o", color=COLORS[s], ms=3, label=f"{NAMES[s]}: chi2 rejects (p<0.05)")
        axs[1].plot(d.vec_err, d.frac_flagged, "x", color=COLORS[s], ms=4, label=f"{NAMES[s]}: identifiability flag")
    axs[0].axhline(1, color="k", lw=0.7, ls=":")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("vector error |du|/|u|")
    axs[0].set_ylabel("chi2 per dof (known sigma)")
    axs[1].set_xlabel("vector error |du|/|u|")
    axs[1].set_ylabel("fraction of (case, layout) fits")
    axs[1].legend(fontsize=6)
    axs[0].legend(fontsize=7)
    f.suptitle(f"{STAMP} ({lab}): chi2/dof vs wind error, with fraction flagged by existing identifiability flags", fontsize=8)
    f.tight_layout()
    f.savefig(fig / "chi2_and_flags_vs_vector_error.png", dpi=130)
    plt.close(f)
    # coverage
    f, ax = plt.subplots(figsize=(5.5, 3.8))
    for s in "ABC":
        d = cc[cc.strategy == s].sort_values("vec_err")
        ax.plot(d.vec_err, d.cov90, "o", color=COLORS[s], label=NAMES[s] + (" (naive covariance)" if s != "C" else " (spread + posterior)"), ms=3)
    d = cc[cc.strategy == "C"].sort_values("vec_err")
    ax.plot(d.vec_err, d.cov90_spread_only, "s", color=COLORS["C"], label="C spread only", ms=3)
    ax.axhline(0.9, color="k", lw=0.7, ls=":")
    ax.axvline(W.vector_error(20.0), color="grey", lw=0.7, ls="--")
    ax.text(W.vector_error(20.0), 0.02, " edge of declared direction prior (20 deg)", fontsize=6, color="grey")
    ax.set_xlabel("vector error |du|/|u|")
    ax.set_ylabel("coverage of 90% interval, true field at held-out sensors")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6)
    ax.set_title(f"{STAMP} ({lab}): interval coverage", fontsize=8)
    f.tight_layout()
    f.savefig(fig / "coverage_vs_vector_error.png", dpi=130)
    plt.close(f)
    # C - B by budget
    pdiff: Any = pd.concat(frames)
    pdiff = pdiff[(pdiff.pool == "layout types pooled") & (pdiff.metric == "rmse_total") & (pdiff.contrast == "C-B")].merge(meta, on="cond")
    pdiff = pdiff[pdiff.blocks.str.contains("direction|speed")]
    f, ax = plt.subplots(figsize=(5.5, 3.8))
    for b, col in zip(sorted(pdiff.budget.unique()), ("#4c78a8", "#f58518", "#54a24b")):
        d = pdiff[pdiff.budget == b].sort_values("vec_err")
        ax.errorbar(d.vec_err, d.diff_mean, yerr=[d.diff_mean - d.ci_lo, d.ci_hi - d.diff_mean], color=col, ls="", marker="o", ms=3, capsize=2, label=f"budget {b}")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("vector error |du|/|u|")
    ax.set_ylabel("paired RMSE difference C - B (ug/m^3); < 0 means C better")
    ax.legend(fontsize=7)
    ax.set_title(f"{STAMP} ({lab}): C vs B by budget; bars = bootstrap {int(level * 100)}% over cases", fontsize=8)
    f.tight_layout()
    f.savefig(fig / "C_minus_B_by_budget.png", dpi=130)
    plt.close(f)
    ident(cfg, out, work, wt, lab)
    say(f"report written to {out}")


def ident(cfg: dict[str, Any], out: Path, work: Path, wt: Any, lab: str) -> None:
    st = W.Study(cfg, work)
    lays = json.loads((out / "layouts.json").read_text())["layouts"]
    rots = [d for d in cfg["wind_error"]["grid"]["direction_deg"] if d > 0]
    demo = {}
    for b, d in lays.items():
        for lt in ("greedy", "spread"):
            demo[f"{lt}-{b}"] = W.wind_amplitude_confounding(st, d[lt], st.base_theta, rots)
    (out / "tables" / "identifiability_demo.json").write_text(json.dumps(demo, indent=1))
    rows = []
    for name, d in demo.items():
        for r in d["rotations"]:
            rows.append(f"| {name} | {r['deg']:g} | {r['signal_change_sigma']:.1f} | {r['absorbed_by_amplitudes_sigma']:.1f} | "
                        f"{r['leftover_sigma']:.1f} | {r['implied_chi2_excess']:.0f} / {r['dof']} | "
                        f"{', '.join(f'{x:+.2f}' for x in r['amplitude_shift_rel'])} |")
    sv = "\n".join(f"| {name} | " + ", ".join(f"{x:.3g}" for x in d["singular_values_nominal"]) + " | " + ", ".join(f"{x:.3g}" for x in d["singular_values_augmented"])
                   + f" | {d['tangent_sin_angle_to_span']:.3f} |" for name, d in demo.items())
    wrows = "\n".join(f"| {r.cond} | {r.vec_err:.3f} | {r.w_max:.2f} | {r.n_eff_members:.2f} |" for r in wt.sort_values("vec_err").itertuples())
    (out / "README.md").write_text(f"""# EXP-019 wind-error toy ({lab})

Synthetic Phase-1 Domain (configs/reference.json), 3 local sources plus one external (inflow) basis. Raw rows: `raw/` (one CSV per truth wind
condition); derived: `tables/`; figures: `figures/` (stamped {STAMP}); frozen choices: `freeze.json`; timing: `timing.json`.
Vector error |du|/|u| of a rotation by dtheta at constant speed is 2 sin(dtheta/2) (5/10/15/30 deg = 0.087/0.174/0.261/0.518); a speed error e
is |e|; a joint error is sqrt(1 + f^2 - 2 f cos(dtheta)).

## Identifiability: when amplitude and wind compensate

A rotation of the wind moves a plume centreline at downwind distance L sideways by about L*dtheta, i.e. it acts like a displaced source, and
changes the travel time and dilution, i.e. like a rescaled source. With source positions known and only amplitudes free, the only way the
nominal-wind model (strategy B) can imitate the rotated truth is to change the amplitudes (and the inflow amplitude). The rank/effective-rank/
condition/confounded-pair flags only look at the columns of the NOMINAL operator: they do not depend on the data, and therefore cannot see a wrong
wind. chi2/dof with known sigma can, once the part of the signal change that amplitudes cannot absorb exceeds the noise.

Numerical demonstration (noise-free, from the cached response bases, base amplitudes, frozen layouts). For each layout and rotation: the signal change
(A(phi) - A(0)) theta in units of sigma, the part absorbed by refitting amplitudes on the nominal operator, the leftover (its square is the chi2
excess a noiseless fit would show; the expected noise chi2 is about dof), and the amplitude shift that does the absorbing (relative; emission 1..3,
then inflow). The refit is unconstrained least squares: the q >= 0 bound is NOT applied, so huge negative shifts (spread layouts leave one
direction, essentially source S3, almost unobserved: smallest singular value ~0.01) overstate what a bounded fit can absorb.

| layout | rotation (deg) | signal change (sigma) | absorbed (sigma) | leftover (sigma) | chi2 excess / dof | relative amplitude shift |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Singular values of the whitened, scaled nominal operator, then of the operator augmented with the wind-direction tangent column (+-5 deg central
difference), and the sine of the angle between that tangent and the span of the nominal columns (1 = wind direction fully separable from the
amplitudes, 0 = fully confounded):

| layout | singular values (nominal) | singular values (augmented) | sin(angle) |
|---|---|---|---|
{sv}

Posterior member weights of strategy C (mean over test cases and layouts): the weights concentrate when the data can tell winds apart; the effective
number of members 1/sum(w^2) of the 15-member declared ensemble measures how spread the posterior is.

| condition | vector error | mean max weight | mean effective members |
|---|---|---|---|
{wrows}
""")


# ------------------------------------------------------------------------------------------------ estimate
def estimate(cfg: dict[str, Any], frm: Path) -> None:
    tm = json.loads((frm / "timing.json").read_text())
    we = cfg["wind_error"]
    nc = len(W.conditions(we["grid"]))
    nl = len(we["budgets"]) * (2 + we["n_random"])
    evals = nc * we["n_test"] * nl
    per_eval = tm["rows_s"] / max(tm["n_case_layout_evals"], 1)
    prior = len(W.prior_keys(we["prior"]["direction_deg"], we["prior"]["speed_factor"]))
    winds = nc + len(we["validation"]["winds"]) + prior  # upper bound: upper limit of distinct new winds (overlaps are cached)
    per_wind = tm["basis_s"] / max(tm["bases_computed"], 1)
    print(json.dumps({"case_layout_condition_evals": evals, "seconds_per_eval_measured_on": tm["profile"], "rows_minutes": evals * per_eval / 60,
                      "winds_upper_bound": winds, "seconds_per_wind_basis": per_wind, "bases_minutes_upper_bound": winds * per_wind / 60,
                      "total_minutes_upper_bound(excl. validation/design, which the pilot timing includes in setup_validation_design_s)":
                          (evals * per_eval + winds * per_wind) / 60}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "estimate"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out")
    ap.add_argument("--from", dest="frm")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config if not Path(a.config).is_absolute() else a.config)
    out, work = paths(cfg, a.out)
    if a.cmd == "run":
        run(cfg, out, work, a.overwrite)
    elif a.cmd == "report":
        report(cfg, out, work)
    else:
        estimate(cfg, Path(a.frm))


if __name__ == "__main__":
    main()

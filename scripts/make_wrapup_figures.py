"""Wrap-up figures from STORED outputs only (no simulation, no inference).

    .venv/bin/python scripts/make_wrapup_figures.py

Writes docs/figures/final/wrapup_*.png and appends/replaces the matching entries in docs/figures/final/figure_index.json
(existing entries are kept).  Sources: results/exp027b-hpc/{stage1,stage2}, results/dyn08-v2-hpc/dyn08_all, results/dyn08-hpc/dyn08_all,
data/population/belauri_padded_worldpop_500m.npy, results/geodesign/candidates.csv, and the study polygon via geodesign.load_geography.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures/final"
BLUE, ORANGE, GREY, RED, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#e34948", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
INDEX: list[dict] = []


def rec(name, sources, caption):
    INDEX.append({"figure": name, "file": f"docs/figures/final/{name}.png", "status": "produced", "sources": sources, "caption": caption})


def jl(p):
    return json.loads((ROOT / p).read_text())


def cells(stage):
    rows = list(csv.DictReader(open(ROOT / f"results/exp027b-hpc/{stage}/exp027b/exp027b_cells.csv")))
    return {(int(r["budget"]), float(r["noise"]), r["mesh"]): {k: float(v) for k, v in r.items() if k not in ("mesh",)} for r in rows}


# ------------------------------------------------------------------------------------------------ (1) EXP-027b
def fig_exp027b():
    s1, s2 = cells("stage1"), cells("stage2")
    v1, v2 = jl("results/exp027b-hpc/stage1/exp027b/exp027b_verdict.json"), jl("results/exp027b-hpc/stage2/exp027b/exp027b_verdict.json")
    keys = sorted({(b, n) for (b, n, _) in s2})
    delta = v2["delta"]
    labels = [f"k={b}\nnoise {int(n * 100)} %" for b, n in keys]

    def verdict(c):
        return "INADEQUATE" if c["E_lo95"] > delta else ("ADEQUATE" if c["E_hi95"] < delta else "INCONCLUSIVE")
    # consistency check against the stored counts
    chk = {st: (sum(verdict(c[(b, n, "main")]) == "INADEQUATE" for b, n in keys), sum(verdict(c[(b, n, "main")]) == "ADEQUATE" for b, n in keys))
           for st, c in (("s1", s1), ("s2", s2))}
    stored = {"s1": (v1["main_cells_inadequate"], v1["main_cells_adequate"]), "s2": (v2["main_cells_inadequate"], v2["main_cells_adequate"])}
    print("exp027b per-cell rule (E_lo>delta -> inadequate, E_hi<delta -> adequate) vs stored counts:", chk, stored)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    x = np.arange(len(keys))
    for stage, d, dx, fill in (("stage 1 (48 cases)", s1, -0.06, False), ("stage 2 (72 cases)", s2, 0.06, True)):
        for mesh, col, off in (("main", BLUE, -0.17), ("control", ORANGE, 0.17)):
            e = np.array([d[(b, n, mesh)]["E"] for b, n in keys]); lo = np.array([d[(b, n, mesh)]["E_lo95"] for b, n in keys])
            hi = np.array([d[(b, n, mesh)]["E_hi95"] for b, n in keys])
            ax[0].errorbar(x + off + dx, e, yerr=[e - lo, hi - e], fmt="o" if fill else "s", color=col, mfc=col if fill else "none",
                           capsize=3, lw=1.6 if fill else 1.0, ms=6, alpha=1 if fill else 0.7)
    ax[0].axhline(delta, color=INK, ls="--", lw=1); ax[0].text(len(keys) - 0.45, delta + 0.008, f"margin {delta}", ha="right", fontsize=8)
    for i, (b, n) in enumerate(keys):
        ax[0].text(i - 0.17, 0.655, verdict(s2[(b, n, "main")]), ha="center", fontsize=7.5, color=BLUE, fontweight="bold")
    ax[0].set_xticks(x, labels); ax[0].set_ylabel("E (mesh-induced discrepancy) with 95 % interval"); ax[0].set_ylim(0, 0.85)
    ax[0].set_title("E: 500 m main vs 2 km control", loc="left", fontsize=10)
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", label="500 m main, stage 2"), Line2D([], [], marker="s", mfc="none", color=BLUE, ls="", label="500 m main, stage 1"),
                          Line2D([], [], marker="o", color=ORANGE, ls="", label="2 km control, stage 2"), Line2D([], [], marker="s", mfc="none", color=ORANGE, ls="", label="2 km control, stage 1")],
                 loc="upper left", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0, 1.0))
    for a, col, ttl, ref in ((ax[1], "J_ratio_median", "median J ratio", 1.0), (ax[2], "rho", "rank correlation rho", None)):
        for stage, d, dx, fill in (("s1", s1, -0.06, False), ("s2", s2, 0.06, True)):
            for mesh, c, off in (("main", BLUE, -0.17), ("control", ORANGE, 0.17)):
                a.plot(x + off + dx, [d[(b, n, mesh)][col] for b, n in keys], "o" if fill else "s", color=c, mfc=c if fill else "none", ms=6, alpha=1 if fill else 0.7)
        if ref is not None:
            a.axhline(ref, color=INK, ls=":", lw=1)
        a.set_xticks(x, labels); a.set_title(ttl + " (no interval stored)", loc="left", fontsize=10)
    ax[2].axhline(0, color=INK, ls=":", lw=1)
    fig.suptitle(f"EXP-027b: stage 2 verdict {v2['status'].replace('main ', '').upper()}  (stage 1: {v1['status'].replace('main ', '').upper()}).  "
                 f"Control cells failing the gate: stage 2 = {v2['control_cells_failing']}/{v2['n_cells']}, stage 1 = {v1['control_cells_failing']}/{v1['n_cells']}; "
                 "the control can fail, so the test is not void.", x=0.01, ha="left", fontsize=10)
    fig.text(0.01, 0.005, f"Per-cell verdict label (blue text) = stage-2 rule E_lo > {delta} -> INADEQUATE, E_hi < {delta} -> ADEQUATE, else INCONCLUSIVE (recomputed from the stored intervals; "
             f"stored counts inadequate/adequate stage 2 = {stored['s2']}, stage 1 = {stored['s1']}).  Hollow = stage 1, filled = stage 2.\n"
             "The INADEQUATE call is borderline and interval-method dependent (F-075; F-077, post-hoc).", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94)); fig.savefig(OUT / "wrapup_exp027b_adequacy.png", dpi=160); plt.close(fig)
    rec("wrapup_exp027b_adequacy", ["results/exp027b-hpc/stage1/exp027b/exp027b_cells.csv", "results/exp027b-hpc/stage2/exp027b/exp027b_cells.csv",
        "results/exp027b-hpc/stage1/exp027b/exp027b_verdict.json", "results/exp027b-hpc/stage2/exp027b/exp027b_verdict.json"],
        "EXP-027b (F-075; post-hoc sensitivity F-077): E with 95 % intervals for the 500 m main mesh vs the 2 km control in the 4 (budget, noise) cells, stage 1 (hollow) vs stage 2 (filled), "
        "margin 0.05; J-ratio and rho panels. Stage 2 verdict main INADEQUATE with all 4 control cells failing the gate; stage 1 was main INCONCLUSIVE. "
        "Per-cell labels recomputed from the stored intervals. Synthetic scenarios.")


# ------------------------------------------------------------------------------------------------ (2) DYN-08 v2
CELLS = [(3, 0.02), (3, 0.1), (5, 0.02), (5, 0.1)]
CL = [f"k={k}, noise {int(n * 100)} %" for k, n in CELLS]


def fig_dyn08_forest(S):
    names = [("dyn_opt-random (M2)", "vs random\n(layout population)"), ("dyn_opt-spread (M2) [this layout only]", "vs spread\n(this layout only)"),
             ("dyn_opt-highpop (M2) [this layout only]", "vs highpop\n(this layout only)"), ("dyn_opt-static_opt (M2) [this layout only]", "vs static_opt\n(this layout only)"),
             ("dyn_opt-src_opt (M2) [this layout only]", "vs src_opt\n(this layout only)")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    for a, (k, n), lab in zip(axs, CELLS, CL):
        for i, (c, _) in enumerate(names):
            r = next(r for r in S["summary"] if r["k"] == k and r["noise"] == n and r["contrast"] == c)
            col = BLUE if i == 0 else GREY
            a.errorbar(r["mean_diff"], -i, xerr=[[r["mean_diff"] - r["ci_lo"]], [r["ci_hi"] - r["mean_diff"]]], fmt="o", color=col, capsize=3, lw=1.8 if i == 0 else 1.2)
            a.text(r["ci_hi"], -i + 0.2, f"{r['rel_change'] * 100:+.0f} %", fontsize=8, ha="right", va="bottom", color=MUTED)
        a.axvline(0, color=INK, lw=1); a.set_title(lab, fontsize=10, loc="left"); a.set_xlabel("M2 error difference (dyn_opt - other)\n<0: dyn_opt lower error")
    axs[0].set_yticks([-i for i in range(len(names))], [l for _, l in names])
    fig.suptitle("DYN-08 v2: dyn_opt minus comparison, M2 population error.  Blue: vs random, two-level (layout x episode) bootstrap 95 % CI.  "
                 "\nGrey: single fixed layouts, episode bootstrap CI (not a statement about the design rule).  Numbers = relative change.", x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT / "wrapup_dyn08v2_forest.png", dpi=160); plt.close(fig)
    rec("wrapup_dyn08v2_forest", ["results/dyn08-v2-hpc/dyn08_all/summary.json"],
        "DYN-08 v2 (RUN-0101, F-076): dyn_opt minus random (two-level layout x episode bootstrap) and minus spread/highpop/static_opt/src_opt (single layouts, labelled 'this layout only') "
        "in the 4 (k, noise) cells, with relative change. Synthetic emissions/episodes; real geometry; model-free of empirical validation.")


def fig_dyn08_controls(S):
    rows = []
    for k, n in CELLS:
        for r in S["controls"]:
            if r["k"] == k and r["noise"] == n:
                rows.append((k, n, r))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    spec = [("a'", "a'. spread vs worst-surrogate layout\n(spread must be better; gating)", BLUE),
            ("b'", "b'. random group A vs B\n(must NOT show an effect; gating)", ORANGE)]
    rep = {(r["k"], r["noise"]): r for r in S["reported_layouts"]}
    y = 0; yt, yl = [], []
    for k, n in CELLS:
        for tag, lab, col in spec:
            r = next(r for kk, nn, r in rows if kk == k and nn == n and r["control"].startswith(tag))
            m = r["mean_b"]
            ax.errorbar(r["rel_change"], y, xerr=[[(r["mean_diff"] - r["ci_lo"]) / m], [(r["ci_hi"] - r["mean_diff"]) / m]], fmt="o", color=col, capsize=3, lw=1.6)
            ax.text(0.995, y, "PASS" if r["passed"] else "FAIL", ha="right", va="center", fontsize=8, fontweight="bold", color=INK, transform=ax.get_yaxis_transform())
            yt.append(y); yl.append(f"k={k}, {int(n * 100)} %: {tag}"); y -= 1
        r = rep[(k, n)]; m = r["mean_b"]
        ax.errorbar(r["rel_change"], y, xerr=[[(r["mean_diff"] - r["ci_lo"]) / m], [(r["ci_hi"] - r["mean_diff"]) / m]], fmt="s", mfc="none", color=GREY, capsize=3, lw=1.2)
        ax.text(0.995, y, "reported", ha="right", va="center", fontsize=8, color=MUTED, transform=ax.get_yaxis_transform())
        yt.append(y); yl.append(f"k={k}, {int(n * 100)} %: v1 lowpop (not gating)"); y -= 1.6
    ax.axvline(0, color=INK, lw=1); ax.set_yticks(yt, yl, fontsize=8); ax.set_xlim(-0.9, 0.95)
    ax.set_xlabel("relative M2 change (A - B) / B with 95 % CI; < 0 means A has lower error")
    ax.set_title("DYN-08 v2 controls: 4/4 a' cells met (>= 3 required), b' passed in all cells; v1 lowpop layout reported only", loc="left", fontsize=10)
    ax.text(0.0, -0.13, "a' and b' CIs are expressed relative to the comparison arm mean (mean_b). b' uses the two-level layout x episode bootstrap.", transform=ax.transAxes, fontsize=7.5, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "wrapup_dyn08v2_controls.png", dpi=160); plt.close(fig)
    rec("wrapup_dyn08v2_controls", ["results/dyn08-v2-hpc/dyn08_all/summary.json"],
        "DYN-08 v2 gating controls: a' (spread vs worst-surrogate-risk layout, effect expected) and b' (random group A vs B, no effect expected) in the 4 cells, relative M2 change with 95 % CI, "
        "plus the v1 lowpop-vs-spread layout (reported, not gating). All gates passed.")


def fig_dyn08_modelerr(ME):
    conds = [("matched", "0 deg"), ("rot5", "5 deg"), ("rot10", "10 deg"), ("rot20", "20 deg"), ("speed1.2", "speed\nx1.2"), ("mesh250", "mesh\n250 m")]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    for j, ((k, n), lab) in enumerate(zip(CELLS, CL)):
        for layout, col, off in (("dyn_opt", BLUE, -0.08), ("spread", ORANGE, 0.08)):
            rr = [next(r for r in ME if r["k"] == k and r["noise"] == n and r["layout"] == layout and r["cond"] == c) for c, _ in conds]
            x = np.arange(len(conds)) + off
            for row, (key, ci) in enumerate((("M2_ratio_vs_matched", "M2_ratio_vs_matched_ci95"), ("M2_over_M1", "M2_over_M1_ci95"))):
                v = np.array([r[key] for r in rr]); lo = np.array([r[ci][0] for r in rr]); hi = np.array([r[ci][1] for r in rr])
                axs[row, j].errorbar(x, v, yerr=[v - lo, hi - v], fmt="o-", color=col, capsize=2.5, lw=1.2, ms=5, label=layout)
        axs[0, j].set_title(lab, loc="left", fontsize=10); axs[0, j].set_yscale("log"); axs[0, j].axhline(1, color=INK, lw=0.8)
        axs[1, j].axhline(1, color=INK, ls="--", lw=1)
        axs[1, j].set_xticks(range(len(conds)), [l for _, l in conds], fontsize=8)
    axs[0, 0].set_ylabel("M2 error ratio vs matched model\n(log scale)"); axs[1, 0].set_ylabel("M2 / M1 error ratio\n(<1: M2 better)")
    axs[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("DYN-08 v2 model error: truth wind rotated 0/5/10/20 deg, speed x1.2, or 250 m truth mesh while the inference model stays nominal.\nMean with 95 % CI; dyn_opt vs spread.  "
                 "\nTop: M2 error / matched M2 error.  Bottom: M2 / M1, crossing 1 where the dynamic model stops helping.\nRotation and speed are not exchangeable axes.", x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / "wrapup_dyn08v2_model_error.png", dpi=160); plt.close(fig)
    rec("wrapup_dyn08v2_model_error", ["results/dyn08-v2-hpc/dyn08_all/modelerr_summary.json"],
        "DYN-08 v2 model error: M2 error ratio vs matched under truth rotation 0/5/10/20 deg, speed x1.2 and 250 m mesh for dyn_opt and spread (95 % CI), and the M2/M1 ratio crossing 1. Synthetic episodes.")


# ------------------------------------------------------------------------------------------------ (3) v1 vs v2 controls
def fig_v1_v2(S1, S2):
    fig, ax = plt.subplots(figsize=(13, 4.8)); ax.axis("off"); ax.grid(False)
    cols = ["cell", "v1: lowpop vs spread\n(lowpop must be WORSE)", "v1: random reps 0-4 vs 5-9\n(must show NO effect)", "v2 a': spread vs worst\n(spread must be better)", "v2 b': random A vs B\n(must show NO effect)"]
    xs = [0.0, 0.14, 0.38, 0.62, 0.82]
    for x, c in zip(xs, cols):
        ax.text(x, 0.93, c, fontsize=9, fontweight="bold", va="top", transform=ax.transAxes)
    def cell(r):
        m = r["mean_b"]
        return f"{r['rel_change'] * 100:+.0f} % [{r['ci_lo'] / m * 100:+.0f}, {r['ci_hi'] / m * 100:+.0f}]  " + ("PASS" if r["passed"] else "FAIL")
    for i, (k, n) in enumerate(CELLS):
        y = 0.72 - i * 0.17
        ax.text(xs[0], y, CL[i], fontsize=9, transform=ax.transAxes, va="center")
        v1 = [r for r in S1["controls"] if r["k"] == k and r["noise"] == n]
        v2 = [r for r in S2["controls"] if r["k"] == k and r["noise"] == n]
        items = [next(r for r in v1 if r["control"].startswith("lowpop")), next(r for r in v1 if r["control"].startswith("random")),
                 next(r for r in v2 if r["control"].startswith("a'")), next(r for r in v2 if r["control"].startswith("b'"))]
        for x, r in zip(xs[1:], items):
            ax.text(x, y, cell(r), fontsize=9, transform=ax.transAxes, va="center", color=INK if r["passed"] else RED, fontweight="normal" if r["passed"] else "bold")
    n1 = sum(r["passed"] for r in S1["controls"]); n2 = sum(r["passed"] for r in S2["controls"])
    ax.text(0, 0.0, f"Relative M2 change (A - B)/B [95 % CI], % ; v1 controls passed {n1}/{len(S1['controls'])} (frozen v1 gate: not passed), v2 controls passed {n2}/{len(S2['controls'])} (gate passed). "
            "v1 'PASS/FAIL' is the stored per-control verdict; the stored 'must be > 0' flag for the v1 lowpop control passes on the sign of mean_diff alone (PASS at +2 % [-12, +16]), so it is not evidence that lowpop was worse (F-069).\n"
            "v1 (F-069, stopped at its stop rules): lowpop did not lose to spread in 3 of 4 cells (it did better at k=3, 2 %, -16 %), and random replicate groups differed at k=5 (the control was badly constructed). "
            "v2 (F-076) replaced them with new controls (worst-surrogate layout; random groups A/B with a two-level bootstrap).", fontsize=8, color=MUTED, transform=ax.transAxes, va="bottom")
    fig.suptitle("DYN-08 control checks: v1 (failed) vs v2 (passed)", x=0.01, ha="left", fontsize=11)
    fig.savefig(OUT / "wrapup_dyn08_v1_v2_controls.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    rec("wrapup_dyn08_v1_v2_controls", ["results/dyn08-hpc/dyn08_all/summary.json", "results/dyn08-v2-hpc/dyn08_all/summary.json"],
        "DYN-08 v1 vs v2 control outcomes per cell (relative M2 change with 95 % CI and the stored pass/fail). The v1 controls failed (F-069: lowpop did not lose to spread in 3 of 4 cells; the stored 'must be > 0' flag passes on sign alone at +2 % [-12, +16]; random replicate groups "
        "differed at k=5), which is why D-029 registered v2 with new controls (F-076). Shown unedited.")


# ------------------------------------------------------------------------------------------------ (4) Belauri map
def fig_map():
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    from pravah.studies.belauri import geodesign as G
    L = jl("results/dyn08-v2-hpc/dyn08_all/layouts.json")
    ids = L["candidates"]
    tab = {r["id"]: (float(r["utm_x"]), float(r["utm_y"])) for r in csv.DictReader(open(ROOT / "results/geodesign/candidates.csv"))}
    missing = [i for i in ids if i not in tab]
    if missing:
        print("SKIPPED map: candidate ids missing from candidates.csv", missing[:3]); return False
    xy = np.array([tab[i] for i in ids])
    pad = jl("data/population/belauri_padded_worldpop.json")["500"]["grid"]
    pop = np.load(ROOT / "data/population/belauri_padded_worldpop_500m.npy")
    cfg = jl("configs/geodesign.json"); cfg["region"]["manifest"] = str(ROOT / cfg["region"]["manifest"])
    geo = G.load_geography(cfg)
    ext = (pad["x0"], pad["x0"] + pad["nx"] * pad["dx"], pad["y0"], pad["y0"] + pad["ny"] * pad["dy"])
    # sanity: population of cells whose centres fall in the study polygon should be close to the stored study total (54357)
    import shapely
    cx = pad["x0"] + (np.arange(pad["nx"]) + 0.5) * pad["dx"]; cy = pad["y0"] + (np.arange(pad["ny"]) + 0.5) * pad["dy"]
    X, Y = np.meshgrid(cx, cy)
    inside = shapely.contains_xy(geo.study, X, Y)
    print(f"map check: sum of pop[iy,ix] with centre in polygon = {pop[inside].sum():.0f} (stored fractional-overlap study total 54357); grid total {pop.sum():.0f}")
    lay = L["layouts"]["0.02"]["5"]
    which = [("dyn_opt", lay["dyn_opt"], BLUE, "o"), ("spread", lay["spread"], ORANGE, "s"), ("highpop", lay["highpop"], "#1baf7a", "^"),
             ("worst", lay["worst"], RED, "X"), ("random #0", lay["random"][0], GREY, "D")]
    b = geo.study.bounds; pad_m = 800
    fig, axs = plt.subplots(1, 5, figsize=(17, 4.6))
    for a, (name, idx, col, mk) in zip(axs, which):
        a.imshow(np.where(pop > 0, pop, np.nan), origin="lower", extent=ext, cmap="Greys", vmax=np.percentile(pop[inside], 99), alpha=0.85)
        polys = [geo.study] if geo.study.geom_type == "Polygon" else list(geo.study.geoms)
        for p in polys:
            a.plot(*p.exterior.xy, color=INK, lw=1)
        a.scatter(xy[:, 0], xy[:, 1], s=6, c="#b0aea8", zorder=3)
        sel = xy[list(idx)]
        a.scatter(sel[:, 0], sel[:, 1], s=110, c=col, marker=mk, edgecolors="white", linewidths=1.2, zorder=5)
        a.set_xlim(b[0] - pad_m, b[2] + pad_m); a.set_ylim(b[1] - pad_m, b[3] + pad_m); a.set_aspect("equal"); a.grid(False)
        a.set_xticks([]); a.set_yticks([]); a.set_title(f"{name}  (sites {list(idx)})", fontsize=9, loc="left")
    fig.suptitle("DYN-08 v2 layouts at k=5, noise 2 % (WorldPop 500 m population shading, darker = more people; light dots = 54 candidate sites; UTM 44N). "
                 "Layouts are candidate-site indices from layouts.json -> results/geodesign/candidates.csv. Synthetic study, modelled population.", x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / "wrapup_dyn08v2_layout_map.png", dpi=160); plt.close(fig)
    rec("wrapup_dyn08v2_layout_map", ["results/dyn08-v2-hpc/dyn08_all/layouts.json", "results/geodesign/candidates.csv", "data/population/belauri_padded_worldpop_500m.npy",
                                      "configs/geodesign.json (study polygon via geodesign.load_geography)"],
        "DYN-08 v2 layouts (dyn_opt, spread, highpop, worst, first random replicate) at k=5, noise 2 %, on the Belauri study polygon with WorldPop 500 m population shading; sites mapped from "
        "candidate indices through layouts.json 'candidates' ids to the geodesign candidate table UTM coordinates. Modelled population estimate, synthetic design study.")
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S2 = jl("results/dyn08-v2-hpc/dyn08_all/summary.json"); S1 = jl("results/dyn08-hpc/dyn08_all/summary.json")
    ME = jl("results/dyn08-v2-hpc/dyn08_all/modelerr_summary.json")["model_error"]
    fig_exp027b(); fig_dyn08_forest(S2); fig_dyn08_controls(S2); fig_dyn08_modelerr(ME); fig_v1_v2(S1, S2); fig_map()
    p = OUT / "figure_index.json"
    idx = json.loads(p.read_text())
    new = {r["figure"] for r in INDEX}
    idx = [r for r in idx if r.get("figure") not in new] + INDEX
    p.write_text(json.dumps(idx, indent=1) + "\n")
    print("wrote", [r["figure"] for r in INDEX])


if __name__ == "__main__":
    main()

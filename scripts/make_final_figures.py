"""Figures for docs/FINAL_RESEARCH_REPORT.md and docs/PEER_PRESENTATION.md.

    .venv/bin/python scripts/make_final_figures.py      (run from the repository root; CPU only, < 2 min)

Reads ONLY stored artifacts (.research/runs/RUN-*/, results/, docs/figures/dynamic/tables) with one exception: fig03 and the toy
panel of fig01 re-simulate ONE toy episode (seed 8634, cached FiPy responses, a few seconds) because per-episode source histories and
fields were not stored. Everything quantitative is synthetic. Missing inputs are reported as 'pending' and the figure is skipped
(DYN-08, EXP-027b and Q-013 outputs do not exist yet, so they have no figure).

Palette: Okabe-Ito (colour-blind safe; categorical, fixed order, never cycled); perceptually uniform sequential maps for fields;
direct labels plus legends so identity is never colour-only; light background only (static PNG for print/slides).
"""
from __future__ import annotations

import csv
import json
import shutil
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / ".research" / "runs"
OUT = ROOT / "docs" / "figures" / "final"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 150
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#555555", "black": "#000000"}
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": True, "legend.framealpha": 0.9})
INDEX: list[dict] = []


def pending(name: str, reason: str) -> None:
    print("pending", name, reason)
    INDEX.append({"figure": name, "status": "pending", "reason": reason})


def save(fig, name: str, caption: str, sources: list[str]) -> None:
    import textwrap
    fig.text(0.01, -0.02, "\n".join(textwrap.wrap(caption, 190)), fontsize=6.5, ha="left", va="top", color=OI["grey"])
    fig.savefig(OUT / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    INDEX.append({"figure": name, "file": f"docs/figures/final/{name}.png", "status": "produced", "sources": sources,
                  "caption": caption})
    print("wrote", name)


def js(p: Path):
    return json.loads(p.read_text())


def rows(p: Path) -> list[dict]:
    return list(csv.DictReader(open(p)))


# ---------------------------------------------------------------------------------------------- toy episode (re-simulated)
def toy_episode():
    """One test episode of the DYN-03 v2 test set (wind rotation_a, family activation, seed 8634): truth field/history, and the
    CT / CAUSAL / RETRO estimates at L4, noise 0.10 with the frozen hyper-parameters (results/dyn03v2, hash 3c86b5e668181a3d)."""
    from pravah import dynamic as dyn
    from pravah.studies.toy import dynamic_history as dh
    from pravah.studies.toy import dynamic_study as ds
    from pravah.studies.toy.dynamic_study import LAYOUTS, N_FINE, N_INV, NP, REF, _cells

    fr = js(ROOT / "results/dyn03v2/frozen_hparams.json")
    # choose the median-gain 'activation' episode (CT-CAUSAL jpop at L4, 10 % noise) from RUN-0094: not cherry-picked
    ep_rows = rows(RUNS / "RUN-0094/episodes.csv")
    gain: dict[int, dict] = {}
    for r in ep_rows:
        if r["family"] == "activation" and r["layout"] == "L4" and float(r["noise"]) == 0.1 and r["model"] in ("CT", "CAUSAL"):
            gain.setdefault(int(r["episode"]), {})[r["model"]] = float(r["jpop"])
    order = sorted(gain, key=lambda e: gain[e]["CT"] - gain[e]["CAUSAL"])
    eid = order[len(order) // 2]
    ep = [e for e in dh.episodes_test() if e["id"] == eid][0]
    R = ds.build_responses(ep["wind"])
    hist, valid = ds.make_history(ep["family"], np.random.default_rng([ep["seed"], 0]), ds.live_mask(R))
    tf = ds.truth_field(ep["wind"], hist, np.random.default_rng([ep["seed"], 1]))
    t = np.asarray(R.times)
    s0 = ds.reference_scale(R)
    layout, frac, ni = "L4", 0.1, 1
    iy, ix = _cells(LAYOUTS[layout])
    sd = frac * s0
    y = tf[1:, iy, ix] + np.random.default_rng([ep["seed"], 2, ni]).standard_normal((len(t) - 1, len(iy))) * sd
    A = np.moveaxis(R.g[:, 1:, iy, ix], 0, -1).reshape(-1, R.g.shape[0])
    y = y.reshape(-1)
    sig = np.full(y.size, sd)
    obs_t = np.repeat(t[1:], len(iy))
    cfg = fr["frozen"][f"{layout}|{frac}"]
    a0, _ = cfg["M0"][0]
    prior = dyn.fit_history(A, y, sig, NP, N_INV, "M0", alpha=a0, ref=REF).theta
    est = {m: dh.estimate(m, A, y, sig, obs_t, cfg[m][0][0], cfg[m][0][1], prior, a0) for m in ("CT", "CAUSAL", "RETRO")}
    binned = hist.reshape(NP, N_INV, N_FINE // N_INV).mean(axis=2)
    jp = {m: ds.jpop(R, th, tf, ds.population())[0] for m, th in est.items() if th is not None}
    # consistency with the stored run (printed, not asserted: the stored value is the reference)
    stored = {r["model"]: float(r["jpop"]) for r in ep_rows if int(r["episode"]) == eid and r["layout"] == "L4" and float(r["noise"]) == 0.1}
    print("episode", ep, "resim jpop", jp, "stored RUN-0094", {k: stored.get(k) for k in jp})
    spec = dh.change_spec(ep["family"], hist)
    return dict(ep=ep, R=R, tf=tf, t=t, hist=hist, binned=binned, est=est, jp=jp, stored=stored, spec=spec, REF=REF, ds=ds, LAYOUTS=LAYOUTS)


# ---------------------------------------------------------------------------------------------- fig01 concentration fields
def fig01(E) -> None:
    ds = E["ds"]
    tf, t = E["tf"], E["t"]
    pop = ds.population()
    nc = ROOT / "results/belauri-forward/fields/from_west_southwest.nc"
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    vmax = float(tf.max())
    for ax, k in zip(axes[0], (6, 18, 30)):
        im = ax.imshow(tf[k], origin="lower", extent=(0, 12, 0, 9), cmap="viridis", vmin=0, vmax=vmax)
        ax.contour(np.linspace(0.25, 11.75, 24), np.linspace(0.25, 8.75, 18), pop, levels=[pop.max() * 0.3], colors="w", linewidths=0.8)
        for (x, yv) in ds.SRC_XY:
            ax.scatter(x / 1000, yv / 1000, marker="*", s=110, c=OI["yellow"], edgecolor="k", zorder=5)
        for (ixc, iyc) in E["LAYOUTS"]["L4"]:
            ax.scatter((ixc + .5) * .5, (iyc + .5) * .5, marker="s", s=40, facecolor="none", edgecolor="w", lw=1.4, zorder=6)
        ax.set_title(f"toy, truth, t = {t[k] / 3600:.1f} h", fontsize=9)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.grid(False)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label="concentration (ug/m3, synthetic)")
    try:
        import xarray as xr

        d = xr.open_dataset(nc)
        vm = float(d["combined"].max())
        for ax, (k, var) in zip(axes[1], ((8, "combined"), (32, "combined"), (32, "external"))):
            im2 = ax.imshow(d[var].isel(time=k).values, origin="lower", cmap="viridis", vmin=0, vmax=vm,
                            extent=(0, float(d.x[-1]) / 1000 + .25, 0, float(d.y[-1]) / 1000 + .25))
            ax.set_title(f"Belauri (synthetic WSW wind), {var}, t = {float(d.time[k]) / 3600:.0f} h", fontsize=9)
            ax.set_xlabel("x (km)")
            ax.set_ylabel("y (km)")
            ax.grid(False)
        fig.colorbar(im2, ax=axes[1].tolist(), shrink=0.85, label="concentration (ug/m3, synthetic)")
    except Exception as exc:  # pragma: no cover
        pending("fig01-belauri-panel", str(exc))
    cap = (f"fig01: top, DYN toy truth field of episode {E['ep']['id']} (re-simulated, seed {E['ep']['seed']}, {E['ep']['wind']}/{E['ep']['family']}; white squares = L4 sensors, stars = sources, "
           "white contour = 30 % of peak population). Bottom, stored Belauri forward fields (results/belauri-forward, EXP-forward Phase 1; "
           "500 m mesh; all emissions and winds synthetic). Not a result of an inference experiment.")
    save(fig, "fig01_concentration_fields", cap, ["results/belauri-forward/fields/from_west_southwest.nc", "re-simulation ep 34"])


# ---------------------------------------------------------------------------------------------- fig02 layouts
def fig02(E) -> None:
    ds = E["ds"]
    L = js(ROOT / "results/dyn04/layouts.json")["layouts"]["0.02"]
    pop = ds.population()
    fams = [("dyn_opt", "dynamic-optimal", OI["red"], "o"), ("highpop", "high-population", OI["orange"], "s"),
            ("spread", "spread", OI["sky"], "^"), ("static_opt", "static-optimal", OI["green"], "D"), ("src_opt", "source-optimal", OI["purple"], "v")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, k in zip(axes, ("2", "3", "5")):
        ax.imshow(pop, origin="lower", extent=(0, 12, 0, 9), cmap="Greys", alpha=0.55)
        for (x, yv) in ds.SRC_XY:
            ax.scatter(x / 1000, yv / 1000, marker="*", s=160, c=OI["yellow"], edgecolor="k", zorder=5)
        for j, (key, lab, c, mk) in enumerate(fams):
            cells = L[k][key]
            off = (j - 2) * 0.09
            ax.scatter([(a + .5) * .5 + off for a, _ in cells], [(b + .5) * .5 + off for _, b in cells], marker=mk, s=70, facecolor="none",
                       edgecolor=c, lw=1.8, zorder=6, label=lab)
        ax.set_title(f"k = {k} sensors (designed at 2 % noise)")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.grid(False)
    axes[0].legend(loc="upper left", fontsize=7)
    cap = ("fig02: DYN-04 layouts (results/dyn04/layouts.json, hash 8bdef74d7c30fe2e, frozen before any test episode; F-061, RUN-0090). Grey = synthetic "
           "population, stars = sources; markers are offset slightly so coincident layouts remain visible. Random layouts (10 per budget) omitted.")
    save(fig, "fig02_sensor_layouts", cap, ["results/dyn04/layouts.json"])


# ---------------------------------------------------------------------------------------------- fig03 temporal reconstruction
def fig03(E) -> None:
    REF, binned, hist, est = E["REF"], E["binned"], E["hist"], E["est"]
    p = E["spec"][0] if E["spec"] else 0
    names = ["S1 emission", "S2 emission", "S3 emission", "WSW inflow basis", "ENE inflow basis"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    tfine = np.arange(hist.shape[1] + 1) * 0.5
    ax.step(tfine, np.r_[hist[p], hist[p, -1]] / REF[p], where="post", color=OI["black"], lw=1.2, ls=":", label="truth, 12 fine bins")
    tb = np.arange(binned.shape[1] + 1)
    ax.step(tb, np.r_[binned[p], binned[p, -1]] / REF[p], where="post", color=OI["black"], lw=2.2, label="truth, 6-bin average (inverse basis)")
    sty = {"CT": (OI["orange"], "--", "CT: current-time-only"), "CAUSAL": (OI["blue"], "-", "CAUSAL filter (bin-b estimate)"),
           "RETRO": (OI["green"], "-.", "RETRO smoother (uses future data)")}
    for m, (c, ls, lab) in sty.items():
        th = est[m]
        if th is not None:
            ax.step(tb, np.r_[th[p], th[p, -1]] / REF[p], where="post", color=c, ls=ls, lw=1.8, label=lab)
    ax.set_xlabel("time (h)")
    ax.set_ylabel(f"{names[p]} (units of reference amplitude)")
    ax.set_title(f"episode {E['ep']['id']}: wind {E['ep']['wind']}, family {E['ep']['family']}, L4, 10 % noise", fontsize=9)
    ax.legend(fontsize=7.5, loc="best")
    ax = axes[1]
    ms = ["CT", "CAUSAL", "RETRO"]
    resim = [E["jp"].get(m, np.nan) for m in ms]
    stored = [E["stored"].get(m, np.nan) for m in ms]
    x = np.arange(3)
    ax.bar(x - 0.18, stored, 0.34, color=[sty[m][0] for m in ms], label="stored RUN-0094")
    ax.bar(x + 0.18, resim, 0.34, color="white", edgecolor=[sty[m][0] for m in ms], hatch="//", label="re-simulated here")
    ax.set_xticks(x, ms)
    ax.set_ylabel("sqrt(J_pop) (ug/m3)")
    ax.set_title("population-weighted error of this episode", fontsize=9)
    ax.legend(fontsize=7.5)
    cap = ("fig03: DYN-03 v2 (F-064, RUN-0094, frozen hash 3c86b5e668181a3d). ONE episode (median-gain 'activation' episode at L4/10 % noise, chosen by rule, "
           "not by appearance) re-simulated because per-episode histories were not stored; right panel checks the re-simulation against the stored J_pop. "
           "Synthetic, known wind, matched model. Illustration of mechanism, not evidence.")
    save(fig, "fig03_history_reconstruction", cap, ["RUN-0094/episodes.csv", "results/dyn03v2/frozen_hparams.json"])


# ---------------------------------------------------------------------------------------------- fig04 DYN-03 forest
def fig04() -> None:
    S = js(RUNS / "RUN-0094/summary.json")
    prim = S["primary"]
    grp = S["wind_groups"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.75})
    ax = axes[0]
    cols = {"CAUSAL-CT": OI["blue"], "RETRO-CAUSAL": OI["green"], "M1-CAUSAL": OI["orange"]}
    ylab = []
    for i, r in enumerate(prim):
        ax.errorbar(r["mean_diff"] * 1e3, i, xerr=[[(r["mean_diff"] - r["ci_lo"]) * 1e3], [(r["ci_hi"] - r["mean_diff"]) * 1e3]], fmt="o",
                    color=cols[r["contrast"]], capsize=3)
        
        ylab.append(f"{r['layout']} {int(r['noise'] * 100)}% {r['contrast']} ({100 * r['rel_change']:+.0f}%)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(prim)), ylab, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(-12, 1)
    ax.set_xlabel("paired difference in sqrt(J_pop) (1e-3 ug/m3); negative = first estimator better")
    ax.set_title("DYN-03 v2: 12 primary contrasts, 96 episodes, Bonferroni 98.75 % CI", fontsize=9)
    ax = axes[1]
    sub = [g for g in grp if g["contrast"] == "CAUSAL-CT"]
    for i, g in enumerate(sub):
        c = OI["grey"] if g["group"] == "steady" else OI["red"]
        ax.errorbar(g["mean_diff"] * 1e3, i, xerr=[[(g["mean_diff"] - g["ci_lo"]) * 1e3], [(g["ci_hi"] - g["mean_diff"]) * 1e3]], fmt="o" if g["group"] == "steady" else "s", color=c, capsize=3)
    ax.set_yticks(range(len(sub)), [f"{g['layout']} {int(g['noise'] * 100)}% {g['group']} (n={g['n']})" for g in sub], fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("CAUSAL - CT (1e-3 ug/m3)")
    ax.set_title("registered 'negative control' FAILS: steady winds gain too\n(grey = steady, red = variable; intervals as stored)", fontsize=9)
    cap = ("fig04: F-064 / RUN-0094 summary.json ('primary', 'wind_groups'); percentages in the left labels are relative changes. CT is a deliberately naive baseline; "
           "RETRO and M1 use future data (non-causal). Synthetic, known wind, one fixture. Preliminary, not independently reviewed.")
    save(fig, "fig04_dyn03_forest", cap, ["RUN-0094/summary.json"])


# ---------------------------------------------------------------------------------------------- fig05 DYN-04 forest
def fig05() -> None:
    S = js(RUNS / "RUN-0090/summary.json")
    C = [c for c in S["contrasts"] if c["metric"] == "jpop"]
    base_names = {"random": "random", "spread": "spread", "static": "static-opt", "highpop": "high-pop", "src": "source-opt"}
    def short(n: str) -> str:
        return n.split("-", 1)[1]
    order = ["random", "spread", "static", "highpop", "src"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 6.2), sharey=True)
    for ax, k in zip(axes, (1, 2, 3, 5)):
        y = 0
        ticks, lab = [], []
        for b in order:
            for nz, mk in ((0.02, "o"), (0.1, "s")):
                c = [x for x in C if x["k"] == k and x["noise"] == nz and (short(x["contrast"]).startswith(b))]
                if not c:
                    continue
                c = c[0]
                rel = c["rel_change"] * 100
                lo = c["ci_lo"] / c["mean_base"] * 100
                hi = c["ci_hi"] / c["mean_base"] * 100
                col = OI["blue"] if (c["effect_criterion_met"] and rel < 0) else (OI["red"] if (hi < 0 or lo > 0) and abs(rel) >= 10 and not c["effect_criterion_met"] else OI["grey"])
                if rel > 0 and lo > 0 and abs(rel) >= 10:
                    col = OI["red"]
                ax.errorbar(rel, y, xerr=[[rel - lo], [hi - rel]], fmt=mk, color=col, capsize=2.5, ms=5)
                ticks.append(y)
                lab.append(f"{base_names[b]} {int(nz * 100)}%")
                y += 1
            y += 0.6
        ax.axvline(0, color="k", lw=0.8)
        ax.axvspan(-10, 10, color="0.9", zorder=0)
        ax.set_title(f"k = {k}", fontsize=10)
        ax.set_xlabel("change of sqrt(J_pop), %")
        ax.set_xlim(-100, 60)
        if k == 1:
            ax.set_yticks(ticks, lab, fontsize=7.5)
    axes[0].invert_yaxis()
    fig.suptitle("dynamic-optimal vs baseline; negative = dynamic-optimal better", fontsize=10)
    cap = ("fig05: F-061 / RUN-0090 summary.json 'contrasts' (40 contrasts; 96 test episodes seeds 7800-7895; CI = lenient Bonferroni 0.05/4 as registered, "
           "converted to % of the baseline mean). Blue = effect criterion met (CI excludes 0, >= 10 %, sign persists leave-one-wind-out); grey = tie or "
           "criterion not met; red = baseline better (point estimate >= +10 % with CI above 0; several of these become ties under the strict 0.05/40 family, see F-061). "
           "Grey band = +-10 %. Synthetic, known wind, hyper-parameters tuned on the spread layout (a comparator).")
    save(fig, "fig05_dyn04_forest", cap, ["RUN-0090/summary.json"])


# ---------------------------------------------------------------------------------------------- fig06 DYN-05
def fig06() -> None:
    T = ROOT / "docs/figures/dynamic/tables"
    des = rows(T / "fig06_design_summary.csv")
    ci = js(T / "dyn05_actual_regret_ci.json")
    cells = [(0.02, 2), (0.1, 2), (0.02, 3), (0.1, 3)]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), sharex=True)
    for j, (nz, k) in enumerate(cells):
        ax = axes[0, j]
        rr = sorted([r for r in des if float(r["noise"]) == nz and int(r["k"]) == k], key=lambda r: int(r["N"]))
        N = [int(r["N"]) for r in rr]
        ax.plot(N, [max(float(r["p90_rel_regret"]) * 100, 0.05) for r in rr], "o-", color=OI["blue"], label="p90 relative regret (surrogate)")
        ax.plot(N, [max(float(r["mean_rel_regret"]) * 100, 0.05) for r in rr], "s--", color=OI["orange"], label="mean (0 plotted at 0.05)")
        ax.axhline(5, color=OI["red"], lw=1, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"noise {nz}, k = {k}", fontsize=9)
        if j == 0:
            ax.set_ylabel("relative regret vs reference B (%)\n(red dotted = registered 5 %)")
            ax.legend(fontsize=7)
        ax = axes[1, j]
        cc = sorted([c for c in ci if c[0] == nz and c[1] == k], key=lambda c: c[2])
        N2 = [c[2] for c in cc]
        ax.errorbar(N2, [c[3] * 100 for c in cc], yerr=[[(c[3] - c[4]) * 100 for c in cc], [(c[5] - c[3]) * 100 for c in cc]], fmt="o-", color=OI["red"], capsize=2.5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("design episodes N")
        if j == 0:
            ax.set_ylabel("real estimator: excess J_pop of ensemble layout\nvs reference layout (%), 95 % episode bootstrap")
    cap = ("fig06: DYN-05 (F-062; design stage RUN-0078, design.json hash e3968dd3e90b6902; real-estimator check RUN-0080, 96 episodes seeds 8400-8495; "
           "tables docs/figures/dynamic/tables/fig06_design_summary.csv and dyn05_actual_regret_ci.json derived from those runs). Top: the registered surrogate criterion "
           "(met only in two of four cells). Bottom: descriptive, not the registered criterion; the CI conditions on the 16 ensemble layouts held fixed "
           "(a two-stage bootstrap in F-062 is wider). Reference layout is itself noisy (F-062 split-half). Synthetic, one fixture.")
    save(fig, "fig06_dyn05_sufficiency", cap, ["RUN-0078/design.json", "RUN-0080/test_rows.json", "docs/figures/dynamic/tables"])


# ---------------------------------------------------------------------------------------------- fig07 DYN-06
def fig07() -> None:
    S = js(RUNS / "RUN-0093/summary.json")["summary"]
    d = [2, 4, 8, 16]
    scen = ["S1", "S2", "S4", "S5", "S6"]
    names = {"S1": "S1 local up vs influx", "S2": "S2 shutdown vs cessation", "S4": "S4 opposing", "S5": "S5 influx ends", "S6": "S6 two local sources"}
    col = dict(zip(scen, [OI["blue"], OI["orange"], OI["green"], OI["red"], OI["purple"]]))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for j, lay in enumerate(("L4", "L9", "L4_ref")):
        for s in scen:
            r = sorted([x for x in S if x["scenario"] == s and x["layout"] == lay and x["noise"] == 0.1], key=lambda x: x["target_d"])
            if not r:
                continue
            axes[0, j].plot(d, [x["mean_detect"] for x in r], "o-", color=col[s], label=names[s])
            xs = [(dd, x["pooled_mis"]) for dd, x in zip(d, r) if x["pooled_mis"] is not None]
            if xs:
                axes[1, j].plot(*zip(*xs), "o-", color=col[s], label=names[s])
        axes[0, j].axhline(0.10, color="grey", ls=":", lw=1)
        axes[0, j].set_title(f"layout {lay}, noise 0.10", fontsize=9)
        axes[0, j].set_ylim(0, 1.02)
        axes[1, j].set_ylim(-0.01, 0.66)
        axes[1, j].set_xscale("log", base=2)
        axes[1, j].set_xticks(d, [str(v) for v in d])
        axes[1, j].set_xlabel("distinguishability d (noise sd)")
    axes[0, 0].set_ylabel("mean detection (mean over 6 winds)\n(dotted = ~false-alarm level)")
    axes[1, 0].set_ylabel("pooled wrong-group rate\n(only where detection is clearly above false alarm)")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    cap = ("fig07: DYN-06 v4 (F-063, RUN-0093, protocol D-025: null-calibrated detection; misattribution scored only in (wind, cell) pairs where detection clearly "
           "exceeds false alarm; pooled = wrong/detected). Gaps in the lower row mean no wind was evaluable (d = 2 mostly). No CIs are stored for these rates (a registered "
           "requirement not met). The scoring rule was adopted after F-044's headline was contradicted; under the earlier unweighted mean S1/L4 at d=16 is 0.31 (F-063). "
           "S5 has no observable look-alike (detectability only). Synthetic, known wind.")
    save(fig, "fig07_dyn06_detection_misattribution", cap, ["RUN-0093/summary.json"])


# ---------------------------------------------------------------------------------------------- fig08 DYN-07
def fig08() -> None:
    S = js(RUNS / "RUN-0088/summary.json")["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    ax = axes[0]
    style = {("L4", "M1"): (OI["blue"], "o-"), ("L4", "M2"): (OI["blue"], "s--"), ("L9", "M1"): (OI["red"], "o-"), ("L9", "M2"): (OI["red"], "s--")}
    cond = [("nominal", 0), ("rot5", 5), ("rot10", 10), ("rot20", 20)]
    for (lay, m), (c, ls) in style.items():
        ys = []
        for cn, _ in cond:
            r = [x for x in S if x["layout"] == lay and x["noise"] == 0.1 and x["condition"] == cn][0]
            ys.append(r[f"{m}_degradation_vs_nominal"])
        ax.plot([a for _, a in cond], ys, ls, color=c, label=f"{lay} {m}")
    ax.set_xlabel("wind rotation error (deg)")
    ax.set_ylabel("error / nominal error (10 % noise)")
    ax.legend(fontsize=7)
    ax.set_title("model-error multiplier", fontsize=9)
    ax = axes[1]
    for lay, c in (("L4", OI["blue"]), ("L9", OI["red"])):
        ys = [[x for x in S if x["layout"] == lay and x["noise"] == 0.1 and x["condition"] == cn][0]["M1"] for cn, _ in cond]
        ax.plot([a for _, a in cond], [v * 1e3 for v in ys], "o-", color=c, label=f"{lay} M1 absolute")
    ax.set_xlabel("wind rotation error (deg)")
    ax.set_ylabel("sqrt(J_pop) (1e-3 ug/m3)")
    ax.legend(fontsize=7)
    ax.set_title("absolute error: L9 and L4 curves cross\nbetween 10 and 20 deg (F-056)", fontsize=9)
    ax = axes[2]
    allc = ["nominal", "rot5", "rot10", "rot20", "speed0.8", "speed1.2", "K2", "extra0.1", "extra0.3", "combined"]
    for lay, c, dx in (("L4", OI["blue"], -0.18), ("L9", OI["red"], 0.18)):
        ys = [[x for x in S if x["layout"] == lay and x["noise"] == 0.1 and x["condition"] == cn][0]["rel_M2_vs_M1"] * 100 for cn in allc]
        ax.bar(np.arange(len(allc)) + dx, ys, 0.34, color=c, label=lay)
    ax.set_xticks(range(len(allc)), allc, rotation=60, fontsize=7)
    ax.set_ylabel("M2 vs M1 (%), 10 % noise")
    ax.legend(fontsize=7)
    ax.set_title("TV advantage erodes with model error", fontsize=9)
    cap = ("fig08: DYN-07 v2 (F-056, RUN-0088; the same 96 episodes in 10 paired conditions; nominal inverse, frozen DYN-02 v3 hyper-parameters, never re-tuned). "
           "The 'extra source' conditions are confounded with the 9-sensor centre sensor (354 m from the unmodelled source; F-056) and are shown, not interpreted. "
           "Global wind rotations/scalings only; synthetic.")
    save(fig, "fig08_dyn07_model_error", cap, ["RUN-0088/summary.json"])


# ---------------------------------------------------------------------------------------------- fig09 EXP-027
def fig09() -> None:
    r = rows(ROOT / "results/exp027/paired.csv")
    cells = [("3", "0.05"), ("5", "0.05"), ("3", "0.15"), ("5", "0.15")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True)
    kinds = [("main", "500 m vs 250 m ref", OI["blue"]), ("control", "2 km control vs 250 m ref", OI["orange"]), ("pair_ref", "replicate design pairs (250 m)", OI["grey"])]
    for ax, (b, nz) in zip(axes, cells):
        for i, (k, lab, c) in enumerate(kinds):
            v = [float(x["mean_disp_m"]) for x in r if x["kind"] == k and x["budget"] == b and x["noise_fraction"] == nz]
            ax.boxplot([v], positions=[i], widths=0.5, showfliers=False, medianprops=dict(color=c, lw=2), boxprops=dict(color=c), whiskerprops=dict(color=c), capprops=dict(color=c))
            ax.scatter(np.random.default_rng(0).normal(i, 0.05, len(v)), v, s=10, color=c, alpha=0.6)
        ax.set_xticks(range(3), ["500 m", "2 km\ncontrol", "replicate\npairs"], fontsize=8)
        ax.set_title(f"budget {b}, noise {nz}", fontsize=9)
    axes[0].set_ylabel("mean layout displacement (m)")
    cap = ("fig09: EXP-027 (F-057, RUN-0081; results/exp027/paired.csv; 8 draws x 6 cases, seeds 1101-1108; 28 replicate pairs). VOID: the deliberately coarse 2 km "
           "control is not distinguishable from replicate-design noise (it was required to fail), so the 500 m mesh is only provisionally admissible. Boxes = quartiles of "
           "per-draw values (8, 8, 28); dots are the values. Padded Belauri domain, synthetic population/winds.")
    save(fig, "fig09_exp027_void", cap, ["results/exp027/paired.csv"])


# ---------------------------------------------------------------------------------------------- fig10 DYN-02 by wind
def fig10() -> None:
    r = rows(RUNS / "RUN-0087/episodes.csv")
    d: dict = {}
    for x in r:
        if x["layout"] == "L4" and x["model"] in ("M1", "M2"):
            d.setdefault((float(x["noise"]), x["wind"], int(x["episode"])), {})[x["model"]] = float(x["jpop"])
    winds = ["steady_a", "steady_b", "rotation_a", "rotation_b", "reversal", "slow_reversal"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for nz, c, dx in ((0.02, OI["blue"], -0.18), (0.1, OI["orange"], 0.18)):
        means, se = [], []
        for w in winds:
            v = np.array([m["M2"] - m["M1"] for (n, ww, _), m in d.items() if n == nz and ww == w and "M1" in m and "M2" in m])
            means.append(v.mean() * 1e3)
            se.append(v.std(ddof=1) / np.sqrt(len(v)) * 1e3)
        ax.bar(np.arange(6) + dx, means, 0.34, yerr=se, color=c, capsize=2, label=f"noise {nz}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(6), winds, rotation=25, fontsize=8)
    ax.set_ylabel("mean M2 - M1 sqrt(J_pop) (1e-3 ug/m3), L4\n(error bars: episode s.e. within wind)")
    ax.legend(fontsize=8)
    cap = ("fig10: DYN-02 v3 (F-055, RUN-0087 episodes.csv, 4 sensors, 16 episodes per wind). rotation_a supplies 68 % (2 % noise) / 48 % (10 %) of the pooled effect; "
           "a t-interval over the six wind means includes zero (F-055). Synthetic, known wind, matched model.")
    save(fig, "fig10_dyn02_by_wind", cap, ["RUN-0087/episodes.csv"])


def main() -> None:
    try:
        E = toy_episode()
        fig01(E)
        fig02(E)
        fig03(E)
    except Exception as exc:
        pending("fig01-03", f"toy re-simulation failed: {exc!r}")
    for f in (fig04, fig05, fig06, fig07, fig08, fig09, fig10):
        try:
            f()
        except Exception as exc:
            pending(f.__name__, repr(exc))
    # reuse: DYN-00 verification figure from the earlier script (same RUN-0091 data)
    src = ROOT / "docs/figures/dynamic/fig03_dyn00_verification.png"
    if src.exists():
        shutil.copy(src, OUT / "fig00_dyn00_verification.png")
        INDEX.append({"figure": "fig00_dyn00_verification", "file": "docs/figures/final/fig00_dyn00_verification.png", "status": "reused",
                      "sources": ["RUN-0091/dyn00_checks.json (via scripts/make_dynamic_figures.py)"]})
    for name in ("DYN-08", "EXP-027b", "Q-013"):
        pending(name, "experiment not run / result pending: no figure (placeholder in the report)")
    (OUT / "figure_index.json").write_text(json.dumps(INDEX, indent=1))


if __name__ == "__main__":
    main()

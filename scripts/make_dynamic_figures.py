"""Figures and tables for the DYN-* (dynamic toy study) results.

    .venv/bin/python scripts/make_dynamic_figures.py      (run from the repository root)

Reads only real experiment outputs (results/ and .research/runs/) plus the scenario definition in
pravah.studies.toy.dynamic_study. An input that does not exist is reported as 'pending' and its figure is skipped.
Everything quantitative is synthetic (toy geometry, synthetic emissions/winds/noise).
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / ".research" / "runs"
OUT = ROOT / "docs" / "figures" / "dynamic"
TAB = OUT / "tables"
DPI = 130
# Okabe-Ito colour-blind-safe palette
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#555555"}
plt.rcParams.update({"figure.facecolor": "#FAFAF7", "savefig.facecolor": "#FAFAF7", "axes.facecolor": "#FFFFFF",
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": True, "legend.framealpha": 0.9})

INDEX: list[dict] = []


def pending(fig_id: str, reason: str) -> None:
    print(f"pending {fig_id}: {reason}")
    INDEX.append({"figure": fig_id, "status": "pending", "reason": reason})


def load(rel: str):
    p = RUNS / rel if not rel.startswith("results") and not rel.startswith("docs") else ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text()) if p.suffix == ".json" else p


def run_exp(run: str) -> str:
    meta = json.loads((RUNS / run / "run.json").read_text())
    return str(meta.get("experiment_id", "?"))


def write_csv(name: str, rows: list[dict]) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(TAB / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def finish(fig, name: str, caption: str, runs: list[str], n: str, desc: str) -> None:
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.012, caption, fontsize=7, ha="left", va="bottom")
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    INDEX.append({"figure": name.split(".")[0], "file": f"docs/figures/dynamic/{name}", "status": "produced",
                  "caption": caption, "source_runs": runs, "n": n, "description": desc})
    print("wrote", name)


def cap(exp: str, runs: list[str], n: str, extra: str) -> str:
    return f"{exp} · {' + '.join(runs) if runs else 'no run (scenario definition)'} · n = {n} · synthetic toy geometry · {extra}"


# ---------------------------------------------------------------- fig01 domain, population, layouts
def fig01() -> None:
    from pravah.studies.toy import dynamic_study as ds

    pop = ds.population()  # (ny=18, nx=24) population count per 500 m cell
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    im = ax.imshow(pop, origin="lower", extent=(0, 12, 0, 9), cmap="cividis", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="population per 500 m cell (count)")
    for i, (x, y) in enumerate(ds.SRC_XY):
        ax.scatter(x / 1000, y / 1000, marker="*", s=220, c=OI["yellow"], edgecolor="k", zorder=5,
                   label="local source (600 m Gaussian)" if i == 0 else None)
        ax.annotate(f"S{i + 1}", (x / 1000, y / 1000), xytext=(6, 6), textcoords="offset points", color="k", fontsize=9)
    for (name, cells), mk, col in zip(ds.LAYOUTS.items(), ("s", "D"), (OI["sky"], OI["red"])):
        cx = [(ix + 0.5) * 0.5 for ix, _ in cells]  # cell centres, x = (ix + 0.5) * 500 m
        cy = [(iy + 0.5) * 0.5 for _, iy in cells]
        ax.scatter(cx, cy, marker=mk, s=70, facecolor="none", edgecolor=col, linewidth=2, zorder=6,
                   label=f"{name} sensor cells ({len(cells)})")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("DYN toy domain: population counts, 3 local sources, L4 and L9 sensors")
    ax.legend(loc="upper left", bbox_to_anchor=(1.22, 1.0), fontsize=8)
    cells = sum(len(c) for c in ds.LAYOUTS.values())
    finish(fig, "fig01_domain_population_layouts.png",
           cap("fig01 DYN scenario map (dynamic_study, 12 x 9 km, 24 x 18 cells)", [], f"3 sources, 4 (L4) + 9 (L9) sensors, {cells} cells in total",
               "population: separated_settlements, 100 000 people, seed 0"),
           [], "3 sources; 4 L4 + 9 L9 sensors", "domain, synthetic population counts, sources and sensor cells")
    rows = [{"layout": k, "ix": ix, "iy": iy, "x_km": (ix + 0.5) * 0.5, "y_km": (iy + 0.5) * 0.5} for k, c in ds.LAYOUTS.items() for ix, iy in c]
    write_csv("fig01_sensor_cells.csv", rows)


# ---------------------------------------------------------------- fig02 winds and example histories
def fig02() -> None:
    from pravah.studies.toy import dynamic_study as ds
    from pravah.studies.toy.dynamic_study import build_responses, live_mask, make_history, NP, N_FINE, N_INV, REF

    live = live_mask(build_responses("rotation_a"))
    fig = plt.figure(figsize=(12, 9.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1])
    ax_u, ax_v = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:])
    cols = [OI["blue"], OI["orange"], OI["green"], OI["red"], OI["purple"], OI["sky"]]
    for (name, segs), c in zip(ds.WINDS.items(), cols):
        t = [s[0] / 3600 for s in segs] + [ds.DUR / 3600]
        for ax, k, ls in ((ax_u, 1, "-"), (ax_v, 2, "-")):
            vals = [s[k] for s in segs]
            ax.step(t, vals + [vals[-1]], where="post", color=c, ls=ls, lw=1.8, label=name)
    for ax, lab in ((ax_u, "u (m/s), eastward"), (ax_v, "v (m/s), northward")):
        ax.set_xlabel("time (h)")
        ax.set_ylabel(lab)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-1.2, 1.2)
    ax_u.set_title("nominal wind segments: u")
    ax_v.set_title("nominal wind segments: v")
    ax_v.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
    bins_t = np.arange(N_INV) * (ds.DUR / N_INV) / 3600
    labels = ["S1", "S2", "S3", "B_wsw", "B_ene"]
    for i, fam in enumerate(ds.FAMILIES):
        ax = fig.add_subplot(gs[1 + i // 4, i % 4])
        h, valid = make_history(fam, np.random.default_rng([7200, 0]), live)
        binned = h.reshape(NP, N_INV, N_FINE // N_INV).mean(axis=2) / REF[:, None]
        for p, c in zip(range(NP), (OI["blue"], OI["orange"], OI["green"], OI["red"], OI["purple"])):
            ax.step(bins_t, binned[p], where="post", color=c, lw=1.5, label=labels[p])
        ax.set_title(f"{fam}" + ("" if valid else " (no live window)"), fontsize=8.5)
        ax.set_xlabel("time (h, 6 bins)")
        ax.set_ylabel("value / REF (–)")
        if i == 0:
            ax.legend(fontsize=6.5, ncol=2, loc="upper right")
    n_valid = sum(make_history(f, np.random.default_rng([7200, 0]), live)[1] for f in ds.FAMILIES)
    finish(fig, "fig02_winds_and_example_histories.png",
           cap("fig02 wind segments; EXAMPLE generator draws, not results", [],
               f"6 winds; 8 families x 1 draw ({n_valid} valid)", "6-bin averages in REF units"),
           [], "6 winds; 8 example draws", "wind histories (u,v) and example source-event profiles, generator draws only")
    write_csv("fig02_wind_segments.csv", [{"wind": w, "t_start_s": s[0], "u_ms": s[1], "v_ms": s[2]} for w, segs in ds.WINDS.items() for s in segs])


# ---------------------------------------------------------------- fig03 DYN-00 verification
def dyn00_source() -> Path | None:
    cands = []
    for p in RUNS.glob("RUN-*/dyn00_checks.json"):
        try:
            rows = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        cands.append((int(re.search(r"RUN-(\d+)", str(p)).group(1)), p, len(rows)))
    if not cands:
        return None
    new25 = [c for c in cands if c[2] == 25]  # newer 25-check table, if any
    return max(new25 or cands)[1]


def fig03() -> None:
    src = dyn00_source()
    if src is None:
        return pending("fig03", "no dyn00_checks.json under .research/runs/")
    checks = json.loads(src.read_text())
    run = re.search(r"RUN-\d+", str(src)).group(0)
    exp = run_exp(run)
    mass = [c for c in checks if c["check"].startswith("mass_closure") or c["check"] in
            ("line_source_mass_closure", "superposition", "static_equivalence_rel", "footprint_integral_err")]
    temporal = [c for c in checks if c["check"] in ("closed_box_temporal_order_last", "west_no_import_after_reversal",
                                                   "dt_refinement_monotone", "dx_refinement_monotone")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1.5, 1]})
    floor = 1e-17
    for i, c in enumerate(mass):
        v = max(abs(c["value"]), floor)
        col = OI["green"] if c["pass"] else OI["red"]
        a.barh(i, v, color=col, alpha=0.8)
        a.plot(c["tol"] if c["tol"] > 0 else floor, i, marker="|", ms=14, color="k")
        a.text(v * 1.8, i, f"{'PASS' if c['pass'] else 'FAIL'}  |res|={c['value']:.2g}", va="center", fontsize=7.5)
    a.set_xscale("log")
    a.set_yticks(range(len(mass)))
    a.set_yticklabels([c["check"] for c in mass], fontsize=7.5)
    a.set_xlabel("residual |value| (log scale; black tick = tolerance; zero-valued residuals drawn at 1e-17)")
    a.set_title("mass closure and conservation residuals")
    a.set_xlim(floor, 1e1)
    for i, c in enumerate(temporal):
        ok = c["pass"]
        b.scatter(c["value"], i, marker="o" if ok else "x", s=80, color=OI["green"] if ok else OI["red"])
        b.text(c["value"], i + 0.22, f"{'PASS' if ok else 'FAIL'}; tol {c['tol']:g}", fontsize=7.5, ha="center")
    b.set_yticks(range(len(temporal)))
    b.set_yticklabels([c["check"] for c in temporal], fontsize=7.5)
    b.set_xlabel("observed value (units of each check; see table)")
    b.set_title("temporal order and refinement checks")
    b.set_ylim(-0.6, len(temporal) - 0.2)
    rows = [{"check": c["check"], "value": c["value"], "tol": c["tol"], "pass": c["pass"], "note": c["note"]} for c in checks]
    write_csv("fig03_dyn00_checks.csv", rows)
    n_pass = sum(bool(c["pass"]) for c in checks)
    finish(fig, "fig03_dyn00_verification.png",
           cap(f"fig03 DYN-00 verification ({exp}; {src.relative_to(ROOT)}; {n_pass}/{len(checks)} checks pass)", [run],
               str(len(checks)) + " checks", "mass-closure residuals (log) and temporal-order checks"),
           [run], f"{len(checks)} checks", "DYN-00 transport verification: mass closure (log) and temporal order with pass/fail")


# ---------------------------------------------------------------- fig04 DYN-02 v2 (EXP-030)
def fig04() -> None:
    run = "RUN-0087"
    doc = load(f"{run}/summary.json")
    eps_path = RUNS / run / "episodes.csv"
    if doc is None or not eps_path.exists():
        return pending("fig04", f"{run}/summary.json or episodes.csv missing")
    exp = run_exp(run)
    S = doc["summary"]
    combos = [("L4", 0.02), ("L4", 0.1), ("L9", 0.02), ("L9", 0.1)]
    contrasts = ("M2-M1", "M2-M1z", "M1-M0")
    with open(eps_path) as f:
        eps = list(csv.DictReader(f))
    fig = plt.figure(figsize=(13, 9.4))
    gs = fig.add_gridspec(2, 4)
    ax_a = fig.add_subplot(gs[0, :2])
    rows_a = []
    for j, (c, col) in enumerate(zip(contrasts, (OI["blue"], OI["orange"], OI["green"]))):
        for i, (lay, nz) in enumerate(combos):
            r = next((s for s in S if s["layout"] == lay and s["noise"] == nz and s["contrast"] == c and s["metric"] == "jpop"), None)
            if r is None:
                continue
            x = i + (j - 1) * 0.22
            ax_a.errorbar(x, r["mean_diff"], yerr=[[r["mean_diff"] - r["ci_lo"]], [r["ci_hi"] - r["mean_diff"]]], fmt="o", color=col,
                          capsize=3, label=c if i == 0 else None)
            rows_a.append({"contrast": c, "layout": lay, "noise": nz, "n": r["n"], "mean_diff": r["mean_diff"],
                           "ci_lo_bonf": r["ci_lo"], "ci_hi_bonf": r["ci_hi"], "rel_change": r["rel_change"]})
    ax_a.axhline(0, color="k", lw=0.8)
    ax_a.set_xticks(range(len(combos)))
    ax_a.set_xticklabels([f"{l}, noise {n:g}" for l, n in combos])
    ax_a.set_ylabel("paired mean difference of J_pop (µg/m³)")
    ax_a.set_title("(a) paired contrasts, Bonferroni 98.75% bootstrap CI")
    ax_a.legend(title="contrast (A−B)", fontsize=8)
    ax_b = fig.add_subplot(gs[0, 2:])
    width = 0.38
    rows_b = []
    for k, (lay, nz) in enumerate([("L4", 0.1), ("L4", 0.02)]):
        vals, errs = [], []
        ws = sorted({bw["wind"] for bw in doc["by_wind"] if bw["layout"] == lay and bw["noise"] == nz})
        ev = {}
        for e in eps:
            if e["layout"] != lay or float(e["noise"]) != nz or e["model"] not in ("M1", "M2") or e["valid"] != "True":
                continue
            ev.setdefault((e["wind"], e["episode"]), {})[e["model"]] = float(e["jpop"])
        for w in ws:
            dd = np.array([v["M2"] - v["M1"] for (ww, _), v in ev.items() if ww == w and np.isfinite(v.get("M2", np.inf)) and np.isfinite(v.get("M1", np.inf))])
            bw = next(b for b in doc["by_wind"] if b["layout"] == lay and b["noise"] == nz and b["wind"] == w)
            m = bw["M2_minus_M1_jpop"]
            vals.append(np.nan if m is None else m)
            errs.append(dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else np.nan)
            rows_b.append({"layout": lay, "noise": nz, "wind": w, "n": len(dd), "M2_minus_M1": vals[-1], "se_from_episodes": errs[-1]})
        xs = np.arange(len(ws)) + (k - 0.5) * width
        ax_b.bar(xs, vals, width, yerr=errs, capsize=3, color=[OI["blue"], OI["sky"]][k], label=f"{lay}, noise {nz:g}")
    ax_b.axhline(0, color="k", lw=0.8)
    ax_b.set_xticks(np.arange(len(ws)))
    ax_b.set_xticklabels(ws, rotation=20, fontsize=8)
    ax_b.set_ylabel("M2 − M1 J_pop (µg/m³)")
    ax_b.set_title("(b) by wind (error bars: ±1 SE from episodes.csv)")
    ax_b.legend(fontsize=8)
    # (c) by family, primary (finite only) and sensitivity (nonconverged M2 replaced by M1)
    fams = sorted({e["family"] for e in eps})
    rows_c = []
    for k, (lay, nz) in enumerate([("L4", 0.1), ("L9", 0.1)]):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        ev = {}
        for e in eps:
            if e["layout"] != lay or float(e["noise"]) != nz or e["model"] not in ("M1", "M2") or e["valid"] != "True":
                continue
            ev.setdefault((e["family"], e["episode"]), {})[e["model"]] = float(e["jpop"])
        for vi, (label, col) in enumerate([("primary: nonconverged M2 excluded", OI["blue"]),
                                           ("sensitivity: nonconverged M2 replaced by M1", OI["orange"])]):
            vals, errs = [], []
            for fam in fams:
                pairs = [v for (f, _), v in ev.items() if f == fam and "M1" in v and "M2" in v]
                if vi == 0:
                    d = np.array([v["M2"] - v["M1"] for v in pairs if np.isfinite(v["M2"]) and np.isfinite(v["M1"])])
                else:
                    d = np.array([(v["M2"] - v["M1"]) if np.isfinite(v["M2"]) else 0.0 for v in pairs if np.isfinite(v["M1"])])
                m = float(d.mean()) if len(d) else np.nan
                se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
                vals.append(m)
                errs.append(se)
                rows_c.append({"layout": lay, "noise": nz, "family": fam, "variant": label, "n": len(d), "M2_minus_M1": m, "se": se,
                               "n_nonconverged_M2": sum(1 for v in pairs if not np.isfinite(v["M2"]))})
            xs = np.arange(len(fams)) + (vi - 0.5) * 0.38
            ax.bar(xs, vals, 0.38, yerr=errs, capsize=2, color=col, label=label)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(fams)))
        ax.set_xticklabels(fams, rotation=30, fontsize=7.5)
        ax.set_ylabel("M2 − M1 J_pop (µg/m³)")
        ax.set_title(f"(c{k + 1}) by family, {lay}, noise {nz:g} (±1 SE)")
        ax.legend(fontsize=6.8)
    n_ep = doc["n_episodes"]
    write_csv("fig04a_paired_contrasts.csv", rows_a)
    write_csv("fig04b_by_wind.csv", rows_b)
    write_csv("fig04c_by_family.csv", rows_c)
    finish(fig, "fig04_dyn02v2_inversion.png",
           cap(f"fig04 DYN-02 v2 inversion ({exp}; J_pop = stored RMS; (c) sensitivity: nonconverged M2 replaced by M1)",
               [run], f"{n_ep} episodes", "paired contrasts, by-wind and by-family M2 − M1"),
           [run], f"{n_ep} episodes", "DYN-02 v2 paired contrasts, by-wind and by-family M2-M1")


# ---------------------------------------------------------------- fig05 DYN-07 robustness (EXP-035)
def fig05() -> None:
    run = "RUN-0088"
    doc = load(f"{run}/summary.json")
    if doc is None:
        return pending("fig05", f"{run}/summary.json missing")
    exp = run_exp(run)
    S = doc["summary"]
    conds = ["nominal", "rot5", "rot10", "rot20"]
    angle = {"nominal": 0, "rot5": 5, "rot10": 10, "rot20": 20}
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.4]})
    rows_a = []
    for lay, ls in (("L4", "-"), ("L9", "--")):
        for key, col, lab in (("M1_degradation_vs_nominal", OI["blue"], "M1"), ("M2_degradation_vs_nominal", OI["red"], "M2")):
            pts = []
            for c in conds:
                r = next((s for s in S if s["layout"] == lay and s["noise"] == 0.1 and s["condition"] == c), None)
                if r is None:
                    continue
                pts.append((angle[c], r[key]))
                rows_a.append({"layout": lay, "noise": 0.1, "condition": c, "rotation_deg": angle[c], "estimator": lab,
                               "degradation_vs_nominal": r[key], "n": r["n"]})
            if pts:
                a.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker="o", color=col, label=f"{lab}, {lay}")
    a.set_xlabel("rotation of the true wind relative to the inverse model (deg)")
    a.set_ylabel("degradation ratio: J_pop(cond) / J_pop(nominal)")
    a.set_xticks([0, 5, 10, 20])
    a.set_title("(a) model-error degradation vs rotation (noise 0.1)")
    a.legend(fontsize=8)
    conds_all = list(doc["conditions"].keys())
    x = np.arange(len(conds_all))
    rows_b = []
    for k, nz in enumerate((0.02, 0.1)):
        vals, lo, hi = [], [], []
        for c in conds_all:
            r = next((s for s in S if s["layout"] == "L4" and s["noise"] == nz and s["condition"] == c), None)
            if r is None:
                vals.append(np.nan); lo.append(0); hi.append(0); continue
            m1 = r["M1"]
            # rel = M2_minus_M1 / M1 exactly, so the CI scales by 1 / M1
            vals.append(r["rel_M2_vs_M1"])
            lo.append(r["rel_M2_vs_M1"] - r["ci_lo"] / m1)
            hi.append(r["ci_hi"] / m1 - r["rel_M2_vs_M1"])
            rows_b.append({"layout": "L4", "noise": nz, "condition": c, "rel_M2_vs_M1": r["rel_M2_vs_M1"],
                           "ci_lo_rel": r["ci_lo"] / m1, "ci_hi_rel": r["ci_hi"] / m1, "n": r["n"]})
        b.bar(x + (k - 0.5) * 0.38, vals, 0.38, yerr=[lo, hi], capsize=2, color=[OI["sky"], OI["blue"]][k], label=f"L4, noise {nz:g}")
    b.axhline(0, color="k", lw=0.8)
    b.set_xticks(x)
    b.set_xticklabels(conds_all, rotation=30, fontsize=8)
    b.set_ylabel("relative gain of M2 over M1: (M2 − M1) / M1")
    b.set_title("(b) all conditions, L4 (Bonferroni CI rescaled by M1)")
    b.legend(fontsize=8)
    write_csv("fig05a_degradation.csv", rows_a)
    write_csv("fig05b_rel_M2_vs_M1_L4.csv", rows_b)
    n = max((s["n"] for s in S), default=0)
    finish(fig, "fig05_dyn07_robustness.png",
           cap(f"fig05 DYN-07 robustness to model error ({exp}; test set; wind model error, K, extra source)", [run],
               f"up to {n} episodes per cell", "degradation vs rotation, relative M2 gain over all conditions"),
           [run], f"up to {n} episodes per cell", "DYN-07 degradation ratios and relative M2 gain per condition")


# ---------------------------------------------------------------- fig06 DYN-05 sufficiency (EXP-033)
def fig06() -> None:
    from pravah.studies.toy import dynamic_sufficiency as dsuf

    design_p = ROOT / "results" / "dyn05" / "design.json"
    test_run = "RUN-0080"
    test_p = RUNS / test_run / "test_rows.json"
    if not design_p.exists():
        return pending("fig06", "results/dyn05/design.json missing")
    doc = json.loads(design_p.read_text())
    rows = dsuf.summarise(doc)
    exp = run_exp(test_run) if test_p.exists() else "EXP-033"
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 5.4))
    keys = sorted({(r["noise"], r["k"]) for r in rows})
    cols = {keys[i]: c_ for i, c_ in enumerate([OI["blue"], OI["orange"], OI["green"], OI["red"]])}
    for (nz, k) in keys:
        rr = sorted((r for r in rows if r["noise"] == nz and r["k"] == k), key=lambda r: r["N"])
        Ns = [r["N"] for r in rr]
        a.plot(Ns, [r["p90_rel_regret"] for r in rr], "-o", color=cols[(nz, k)], label=f"p90, noise {nz:g}, k={k}")
        a.plot(Ns, [r["mean_rel_regret"] for r in rr], "--s", color=cols[(nz, k)], alpha=0.6, label=f"mean, noise {nz:g}, k={k}")
        b.plot(Ns, [r["mean_jaccard_between_replicates"] for r in rr], "-o", color=cols[(nz, k)], label=f"noise {nz:g}, k={k}")
    a.axhline(0.05, color="k", ls=":", label="registered 5% tolerance")
    a.set_xscale("log")
    if min(min(r["p90_rel_regret"], r["mean_rel_regret"]) for r in rows) > 0:
        a.set_yscale("log")
    else:
        a.set_yscale("symlog", linthresh=1e-3)
    a.set_xlabel("N (episodes in the design ensemble, log scale)")
    a.set_ylabel("relative regret vs reference layout (–; symlog)")
    a.set_title("(a) regret of the selected layout")
    a.legend(fontsize=6.5, ncol=2)
    b.set_xscale("log")
    b.set_xlabel("N (episodes in the design ensemble, log scale)")
    b.set_ylabel("mean Jaccard overlap between replicate layouts (–)")
    b.set_ylim(0, 1.02)
    b.set_title("(b) stability of the chosen layout")
    b.legend(fontsize=7)
    # (c) actual-estimator J_pop of each chosen layout relative to the reference layout (joined on sorted sites and noise)
    rows_c = []
    if test_p.exists():
        test_rows = json.loads(test_p.read_text())
        by = {}
        for t in test_rows:
            by.setdefault((tuple(sorted(t["sites"])), t["noise"]), []).append(t["jpop"])
        for (nz, k) in keys:
            rec = doc["results"][f"{nz:g}|{k}"]
            ref = tuple(sorted(rec["star"]))
            ref_m = np.mean(by[(ref, nz)]) if (ref, nz) in by else np.nan
            xs, ys, es = [], [], []
            for n_s, sel in rec["selected"].items():
                ratios = [np.mean(by[(tuple(sorted(s["layout"])), nz)]) / ref_m for s in sel if (tuple(sorted(s["layout"])), nz) in by]
                if len(ratios) < 2:
                    continue
                xs.append(int(n_s))
                ys.append(float(np.mean(ratios)))
                es.append(float(np.std(ratios, ddof=1) / np.sqrt(len(ratios))))
                rows_c.append({"noise": nz, "k": k, "N": int(n_s), "mean_ratio_J_chosen_over_ref": ys[-1], "se": es[-1],
                               "n_replicates": len(ratios), "n_test_episodes": len(by[(ref, nz)])})
            c.errorbar(xs, ys, yerr=es, fmt="-o", capsize=3, color=cols[(nz, k)], label=f"noise {nz:g}, k={k}")
        c.axhline(1.0, color="k", ls=":", label="reference layout")
        c.set_xscale("log")
        c.set_xlabel("N (design ensemble size, log scale)")
        c.set_ylabel("mean J_pop(chosen) / J_pop(reference) (–)")
        c.set_title(f"(c) actual M1z estimator, {test_run} test episodes (±1 SE)")
        c.legend(fontsize=7)
    else:
        c.text(0.5, 0.5, f"{test_run}/test_rows.json missing: panel (c) pending", ha="center", transform=c.transAxes)
    write_csv("fig06_design_summary.csv", rows)
    write_csv("fig06c_actual_estimator.csv", rows_c)
    nrep = len(next(iter(next(iter(doc["results"].values()))["selected"].values())))
    finish(fig, "fig06_dyn05_sufficiency.png",
           cap(f"fig06 DYN-05 sufficiency of the design ensemble ({exp}; design hash {doc['hash']}; test {test_run})",
               [test_run] if test_p.exists() else [], f"{nrep} replicate layouts per (noise,k,N); 96 test episodes", "regret vs N, replicate stability, actual estimator"),
           [test_run] if test_p.exists() else [], f"{nrep} replicates per cell", "DYN-05 regret, Jaccard stability and actual J_pop vs design size N")


# ---------------------------------------------------------------- fig07 DYN-06 (newest run with targets)
def newest_dyn06() -> tuple[str, dict] | None:
    best = None
    for p in RUNS.glob("RUN-*/summary.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        rows = d.get("rows") or []
        if "targets" in d and rows and "detect" in rows[0] and "mis" in rows[0]:
            k = int(re.search(r"RUN-(\d+)", str(p)).group(1))
            if best is None or k > best[0]:
                best = (k, d, p)
    return None if best is None else (f"RUN-{best[0]:04d}", best[1])


def fig07() -> None:
    found = newest_dyn06()
    if found is None:
        return pending("fig07", "no DYN-06 summary.json with targets and detect/mis rows")
    run, doc = found
    exp = run_exp(run)
    scen = ["S1", "S2", "S4", "S5", "S6"]
    lays = [("L4", OI["blue"]), ("L9", OI["red"]), ("L4_ref", OI["green"])]
    noise = 0.1
    rows = [r for r in doc["rows"] if r["noise"] == noise and r.get("valid", True)]
    winds = sorted({r["wind"] for r in rows})
    d_vals = sorted(doc["targets"])
    fig, axes = plt.subplots(2, len(scen), figsize=(15, 6.6), sharex=True)
    out_rows = []
    for j, s in enumerate(scen):
        for i, metric in enumerate(("mis", "detect")):
            ax = axes[i, j]
            for lay, col in lays:
                mean, lo, hi = [], [], []
                for d in d_vals:
                    v = [r[metric] for r in rows if r["scenario"] == s and r["layout"] == lay and r["target_d"] == d]
                    per_w = {r["wind"]: r[metric] for r in rows if r["scenario"] == s and r["layout"] == lay and r["target_d"] == d}
                    if not v:
                        mean.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                    mean.append(np.mean(list(per_w.values())))
                    lo.append(min(per_w.values())); hi.append(max(per_w.values()))
                    out_rows.append({"metric": metric, "scenario": s, "layout": lay, "noise": noise, "target_d": d,
                                     "mean_over_winds": mean[-1], "min_over_winds": lo[-1], "max_over_winds": hi[-1], "n_winds": len(per_w)})
                ax.plot(d_vals, mean, "-o", color=col, label=lay)
                ax.fill_between(d_vals, lo, hi, color=col, alpha=0.15)
            ax.set_xscale("log", base=2)
            ax.set_xticks(d_vals)
            ax.set_xticklabels([f"{d:g}" for d in d_vals])
            ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title(f"scenario {s}")
            if j == 0:
                ax.set_ylabel("wrong-group rate (–)" if metric == "mis" else "detection rate (–)")
            if i == 1:
                ax.set_xlabel("target effect size d (registered DYN-06 units)")
    axes[0, 0].legend(fontsize=8)
    write_csv("fig07_rates_noise0.1.csv", out_rows)
    finish(fig, "fig07_dyn06_detection_misattribution.png",
           cap(f"fig07 DYN-06 detection and misattribution vs target d ({exp}; {run}; lines = mean over winds, band = min-max over winds; noise 0.1)",
               [run], f"{len(winds)} winds x 100 replicates per cell", "wrong-group and detection rates"),
           [run], f"{len(winds)} winds x 100 replicates", "DYN-06 wrong-group and detection rate vs d")
    # heatmap at d=16
    d16 = max(d_vals)
    fig2, axh = plt.subplots(1, 2, figsize=(12, 4.8))
    hm_rows = []
    for ax, (lay, _) in zip(axh, lays[:2]):
        M = np.full((len(winds), len(scen)), np.nan)
        for a_, w in enumerate(winds):
            for b_, s in enumerate(scen):
                v = [r["mis"] for r in rows if r["wind"] == w and r["scenario"] == s and r["layout"] == lay and r["target_d"] == d16]
                if v:
                    M[a_, b_] = v[0]
                    hm_rows.append({"layout": lay, "wind": w, "scenario": s, "target_d": d16, "wrong_group_rate": v[0]})
        im = ax.imshow(M, cmap="cividis", vmin=0, vmax=1, aspect="auto")
        for a_ in range(len(winds)):
            for b_ in range(len(scen)):
                if np.isfinite(M[a_, b_]):
                    ax.text(b_, a_, f"{M[a_, b_]:.2f}", ha="center", va="center", fontsize=8,
                            color="k" if M[a_, b_] > 0.5 else "w")
        ax.set_xticks(range(len(scen)))
        ax.set_xticklabels(scen)
        ax.set_yticks(range(len(winds)))
        ax.set_yticklabels(winds, fontsize=8)
        ax.set_xlabel("scenario")
        ax.set_ylabel("wind")
        ax.set_title(f"{lay}, d = {d16:g}, noise 0.1")
        ax.grid(False)
    fig2.colorbar(im, ax=axh, label="wrong-group rate (–)", shrink=0.85)
    write_csv("fig07_heatmap_d16.csv", hm_rows)
    fig2.tight_layout(rect=(0, 0.05, 1, 1))
    fig2.text(0.01, 0.012, cap(f"fig07b DYN-06 wrong-group heatmap at d={d16:g} ({exp}; {run}; noise 0.1)", [run],
                              f"{len(winds)} winds x 100 replicates per cell", "per wind x scenario"), fontsize=7.2)
    fig2.savefig(OUT / "fig07b_dyn06_wrong_group_heatmap.png", dpi=DPI)
    plt.close(fig2)
    INDEX.append({"figure": "fig07b", "file": "docs/figures/dynamic/fig07b_dyn06_wrong_group_heatmap.png", "status": "produced",
                  "caption": f"fig07b DYN-06 wrong-group heatmap at d={d16:g} ({exp}; {run}; noise 0.1)", "source_runs": [run],
                  "n": f"{len(winds)} winds x 100 replicates per cell", "description": "wrong-group rate by wind x scenario at d=16"})
    print("wrote fig07b_dyn06_wrong_group_heatmap.png")


# ---------------------------------------------------------------- fig08 performance (EXP-025, RUN-0076)
def fig08() -> None:
    import ast as _ast

    run = "RUN-0076"
    lap = list((ROOT / "results").glob("**/dyn_bench*.json"))  # laptop numbers only if such a file exists
    p = RUNS / run / "collected" / "dyn_bench.json"
    if p.exists():
        bench = json.loads(p.read_text())
        bench = bench if isinstance(bench, list) else bench.get("rows", [])
        src = p.relative_to(ROOT)
    else:
        logp = RUNS / run / "collected" / "logs" / "stdout.log"
        if not logp.exists():
            return pending("fig08", f"{run} has neither dyn_bench.json nor stdout.log")
        bench = [_ast.literal_eval(line) for line in logp.read_text().splitlines() if line.startswith("{'workers':")]
        src = logp.relative_to(ROOT)
    if not bench:
        return pending("fig08", "no benchmark rows parsed")
    exp = run_exp(run)
    w = np.array([b["workers"] for b in bench], float)
    eps_s = np.array([b["episodes_per_s"] for b in bench])
    eff = np.array([b["efficiency"] for b in bench])
    sp = np.array([b["speedup"] for b in bench])
    n_ep = int(bench[0]["episodes"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8))
    a.plot(w, eps_s, "-o", color=OI["blue"], label="cluster CPU (measured)")
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xticks(w)
    a.set_xticklabels([f"{int(x)}" for x in w])
    a.set_xlabel("workers (processes, cluster CPU)")
    a.set_ylabel("throughput (episodes / s)")
    a.set_title("(a) throughput")
    a.legend(fontsize=8)
    b.plot(w, eff, "-o", color=OI["red"], label="parallel efficiency (speedup / workers)")
    b.axhline(1.0, ls=":", color="k", label="ideal efficiency")
    b.set_xscale("log", base=2)
    b.set_xticks(w)
    b.set_xticklabels([f"{int(x)}" for x in w])
    b.set_ylim(0, 1.1)
    b.set_xlabel("workers (processes, cluster CPU)")
    b.set_ylabel("parallel efficiency (–)")
    b.set_title("(b) efficiency vs workers")
    b.legend(fontsize=8)
    write_csv("fig08_bench.csv", [{"workers": int(x), "episodes_per_s": t, "speedup": s, "efficiency": e, "wall_s": bb["wall_s"]}
                                  for x, t, s, e, bb in zip(w, eps_s, sp, eff, bench)])
    lap_note = "no laptop benchmark file found; laptop numbers not plotted" if not lap else "laptop file found but not plotted"
    print(lap_note)
    finish(fig, "fig08_performance.png",
           cap(f"fig08 DYN bench, cluster CPU only ({exp}; laptop file absent, not plotted)",
               [run], f"{n_ep} episodes per worker count", "throughput and parallel efficiency vs workers"),
           [run], f"{n_ep} episodes per worker count", "throughput and parallel efficiency on cluster CPU (laptop not available)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    for f in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08):
        f()
    index = {"generated_by": "scripts/make_dynamic_figures.py", "dpi": DPI, "figures": INDEX,
             "synthetic_only": True}
    (TAB / "figure_index.json").write_text(json.dumps(index, indent=2, default=str) + "\n")
    print("index written", TAB / "figure_index.json")


if __name__ == "__main__":
    main()

"""Render e03-inversion.csv / e03-estimates.csv / e03-validity.csv / e03-nonidentifiable.json into the markdown fragments
included by chapters/03-inverse.qmd, and the figure ../figures/e03-source-estimates.png. Every number is computed here."""
import csv, json, pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = pathlib.Path(__file__).parent
rd = lambda n: list(csv.DictReader((here / n).open()))
T, EST, VAL = rd("e03-inversion.csv"), rd("e03-estimates.csv"), rd("e03-validity.csv")
NONID = json.loads((here / "e03-nonidentifiable.json").read_text())
for r in T:
    for k in ("K", "n_sensors", "seed", "rank", "rank_with_background", "n_zero_coef", "n_obs"):
        r[k] = int(r[k])
    for k in ("sigma0", "snr_rms", "cond", "crb_fraction", "source_relative_rmse", "source_rmse_ref", "parameter_bias",
              "measurement_fit", "field_rmse", "field_rmse_zero_estimate"):
        r[k] = float(r[k])
LV = ["r=0.01", "r=0.1", "r=1", "prereg"]
g = lambda x: f"{x:.2g}"
md = lambda cols, rows: "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]) + "\n"
sel = lambda **kw: [r for r in T if all(r[k] == v for k, v in kw.items())]
rng = lambda v: f"{g(np.mean(v))} [{g(min(v))}, {g(max(v))}]"

# 1. main table: weighted, background known; mean [min, max] over the three seeds
rows = []
for K in (2, 4, 8):
    for n in (2, 4, 8):
        for lv in LV:
            s = sel(K=K, n_sensors=n, noise_level=lv, estimator="weighted", background="known")
            if not s:
                continue
            fm = defaultdict(int)
            for r in s:
                fm[r["failure_mode"]] += 1
            rows.append([K, n, lv, g(np.mean([r["snr_rms"] for r in s])), rng([r["rank"] for r in s]),
                         rng([r["source_relative_rmse"] for r in s]), rng([r["field_rmse"] / r["field_rmse_zero_estimate"] for r in s]),
                         rng([r["measurement_fit"] for r in s]), g(np.mean([r["crb_fraction"] for r in s])),
                         "; ".join(f"{v} {k}" for k, v in sorted(fm.items()))])
(here / "e03-main.md").write_text(md(["K", "sensors", "noise level", "SNR (rms signal / sigma0)", "rank of weighted G", "source relative RMSE", "field RMSE / zero-estimate RMSE", "measurement fit (rms of residual / sigma)", "CRB fraction", "outcome (3 seeds)"], rows))

# 2. estimator and background comparison: median over cells of the paired difference, same data for every method
rows = []
base = {(r["K"], r["n_sensors"], r["seed"], r["noise_level"]): r for r in sel(estimator="weighted", background="known")}
for lv in LV:
    for lab, kw in (("unweighted vs weighted (bg known)", dict(estimator="unweighted", background="known")),
                    ("ridge vs weighted (bg known)", dict(estimator="weighted_ridge", background="known")),
                    ("bg estimated vs known (weighted)", dict(estimator="weighted", background="estimated"))):
        d, w = [], 0
        for r in sel(noise_level=lv, **kw):
            b = base[(r["K"], r["n_sensors"], r["seed"], lv)]
            if r["failure_mode"] != "estimated":  # identifiable, non-zero cells only
                continue
            d.append(r["source_relative_rmse"] - b["source_relative_rmse"]); w += d[-1] < 0
        rows.append([lv, lab, len(d), g(np.median(d)) if d else "n/a", f"{w} of {len(d)}" if d else "n/a"])
(here / "e03-compare.md").write_text(md(["noise level", "comparison (method minus baseline)", "paired cells (full rank, non-zero)", "median change in source relative RMSE", "cells where method is better"], rows))

# 3. failure tally (all estimators)
tally = defaultdict(lambda: defaultdict(int))
for r in T:
    tally[r["noise_level"]][r["failure_mode"]] += 1
modes = sorted({m for v in tally.values() for m in v})
(here / "e03-failures.md").write_text(md(["noise level"] + modes + ["rows"], [[lv] + [tally[lv][m] for m in modes] + [sum(tally[lv].values())] for lv in LV]))

# 4. non-identifiable case + validity
lab = {"rank": "rank of snapshot sensitivity", "K": "sources K", "n_rows": "observations", "alt_min_coef": "alternative: smallest coefficient (ug/s)",
       "alt_vs_true_rel": "alternative vs truth, relative distance", "alt_data_misfit_rel": "alternative: relative data misfit",
       "alt_field_diff_rel": "alternative: relative field difference at that hour", "nnls_noiseless_source_rel": "noiseless NNLS: source relative error",
       "nnls_noiseless_misfit_rel": "noiseless NNLS: relative data misfit"}
(here / "e03-nonid.md").write_text(md(["quantity", "value"], [[lab[k], g(v)] for k, v in NONID.items()]))
(here / "e03-validity.md").write_text(md(["scenario", "check", "value"], [[r["scenario"], r["check"], g(float(r["value"]))] for r in VAL]))

# 5. generated findings sentences
full = [r for r in sel(estimator="weighted", background="known") if r["failure_mode"] == "estimated"]
parts = []
for lv in LV:
    s = [r for r in sel(estimator="weighted", background="known", noise_level=lv)]
    ok = [r for r in s if r["failure_mode"] == "estimated"]
    parts.append(f"At noise level {lv} (mean SNR {g(np.mean([r['snr_rms'] for r in s]))}), {len(ok)} of {len(s)} weighted known-background fits are full rank "
                 f"with a non-zero estimate" + (f"; their median source relative RMSE is {g(np.median([r['source_relative_rmse'] for r in ok]))} (worst {g(max(r['source_relative_rmse'] for r in ok))})." if ok else "."))
parts.append(f"Over all {len(T)} fits, {sum(r['failure_mode'] == 'non_identifiable' for r in T)} are non-identifiable at relative tolerance 1e-8 and "
             f"{sum(r['failure_mode'] == 'all_zero_estimate' for r in T)} return an all-zero estimate.")
w0 = sel(estimator="weighted", background="known", noise_level="r=1")
parts.append("Rank of the weighted sensitivity matrix (fits per K x sensors, deficit = K - rank): " + "; ".join(
    f"K={K}, n={n}: mean deficit {np.mean([r['K'] - r['rank'] for r in w0 if r['K'] == K and r['n_sensors'] == n]):.2g}" for K in (2, 4, 8) for n in (2, 4, 8)) + ".")
parts.append(f"The largest relative error seen is {g(max(r['source_relative_rmse'] for r in T))}; values far above 1 are noise amplified along a near-null direction, not recovery.")
(here / "e03-findings.md").write_text("\n\n".join(parts) + "\n")

# 6. figure: true vs estimated strength, weighted NNLS with known background, per sensor count, coloured by noise level
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True, sharex=True, sharey=True)
col = dict(zip(LV, ["tab:blue", "tab:green", "tab:orange", "tab:red"]))
floor = 1.0
for j, n in enumerate((2, 4, 8)):
    for r in EST:
        if int(r["n_sensors"]) == n and r["estimator"] == "weighted" and r["background"] == "known":
            ax[j].scatter(float(r["true_ug_s"]), max(float(r["est_ug_s"]), floor), s=14, color=col[r["noise_level"]], alpha=.6,
                          marker="o" if float(r["est_ug_s"]) > 0 else "v")
    ax[j].plot([10, 1e3], [10, 1e3], "k--", lw=.8)
    ax[j].set(xscale="log", yscale="log", title=f"{n} sensors", xlabel="true source strength, ug s$^{-1}$")
ax[0].set_ylabel(f"estimated (zeros drawn at {floor:g}), ug s$^{{-1}}$")
for lv in LV:
    ax[2].scatter([], [], color=col[lv], label=f"noise {lv}")
ax[2].legend(title="noise level; triangle = zero estimate", fontsize=8)
fig.suptitle("Weighted NNLS, known background: estimated vs true source strengths (all K, 3 seeds)")
fig.savefig(here.parent / "figures" / "e03-source-estimates.png", dpi=110)

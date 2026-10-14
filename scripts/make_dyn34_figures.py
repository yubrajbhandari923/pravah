"""Figures for DYN-03 (RUN-0092) and DYN-04 (RUN-0090) from their run summaries; real data only."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("docs/figures/dynamic")
(OUT / "tables").mkdir(parents=True, exist_ok=True)
C = {"random": "#999999", "spread": "#0072B2", "highpop": "#E69F00", "static_opt": "#009E73", "src_opt": "#CC79A7"}

# DYN-04: dyn_opt - baseline, relative change in sqrt(J_pop), by budget, both noise levels
d = json.load(open(".research/runs/RUN-0090/summary.json"))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
rows = []
for ax, nz in zip(axes, (0.02, 0.1)):
    for i, base in enumerate(C):
        xs, ys, lo, hi = [], [], [], []
        for c in d["contrasts"]:
            if c["metric"] == "jpop" and c["noise"] == nz and c["contrast"] == f"dyn_opt-{base}":
                m = c["mean_base"]
                xs.append(c["k"] + (i - 2) * 0.09); ys.append(100 * c["mean_diff"] / m)
                lo.append(100 * (c["mean_diff"] - c["ci_lo"]) / m); hi.append(100 * (c["ci_hi"] - c["mean_diff"]) / m)
                rows.append([nz, c["k"], base, c["mean_diff"], c["ci_lo"], c["ci_hi"], c["rel_change"]])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=5, capsize=3, color=C[base], label=f"vs {base}")
    ax.axhline(0, color="k", lw=0.8); ax.axhline(-10, color="k", lw=0.6, ls=":")
    ax.set(title=f"noise {nz:g}", xlabel="sensor budget k", xticks=[1, 2, 3, 5]); ax.grid(alpha=0.3)
axes[0].set_ylabel("dynamic-optimal minus baseline, % of baseline sqrt(J_pop)\n(negative = dynamic-optimal better)"); axes[0].legend(fontsize=8)
fig.suptitle("DYN-04 held-out comparison (EXP-032, RUN-0090; n = 96 episodes; Bonferroni 98.75% CI; dotted: 10% effect threshold; synthetic)", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "fig09_dyn04_layouts.png", dpi=130); plt.close(fig)
np.savetxt(OUT / "tables/fig09_dyn04_contrasts.csv", np.array(rows, dtype=object), delimiter=",", fmt="%s", header="noise,k,baseline,mean_diff,ci_lo,ci_hi,rel_change")

# DYN-03: contrasts by layout/noise
h = json.load(open(".research/runs/RUN-0092/summary.json"))
fig, ax = plt.subplots(figsize=(9, 4.6))
labels, k = [], 0
cols = {"CAUSAL-CT": "#D55E00", "RETRO-CAUSAL": "#0072B2", "M1-CAUSAL": "#009E73"}
for c in h["primary"]:
    pass
cells = sorted({(s["layout"], s["noise"]) for s in h["primary"]})
for j, cell in enumerate(cells):
    for i, con in enumerate(cols):
        s = [x for x in h["primary"] if (x["layout"], x["noise"]) == cell and x["contrast"] == con][0]
        ax.errorbar(j + (i - 1) * 0.2, 100 * s["rel_change"], yerr=[[100 * (s["rel_change"] - s["ci_lo"] / (s["mean_diff"] / s["rel_change"]))], [100 * (s["ci_hi"] / (s["mean_diff"] / s["rel_change"]) - s["rel_change"])]],
                    fmt="o", color=cols[con], capsize=3, label=con if j == 0 else None)
ax.axhline(0, color="k", lw=0.8); ax.axhline(-10, color="k", lw=0.6, ls=":")
ax.set_xticks(range(len(cells))); ax.set_xticklabels([f"{a}, noise {b:g}" for a, b in cells]); ax.set_ylabel("relative change in sqrt(J_pop), %")
ax.legend(); ax.grid(alpha=0.3)
ax.set_title("DYN-03 value of history (EXP-031, RUN-0092; n = 96 episodes; Bonferroni CI; synthetic)\nCAUSAL-CT: filter vs current-time-only; RETRO-CAUSAL: smoother vs filter; M1-CAUSAL: full-history independent bins vs filter", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig10_dyn03_history.png", dpi=130); plt.close(fig)
print("ok")

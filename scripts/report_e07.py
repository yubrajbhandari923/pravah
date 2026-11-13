"""Report-only summaries of the committed E07 64² sweep tables (no simulation).

Reads reports/quarto/tables/e07-sweep-{confirmatory,rows}.csv and writes
  tables/report-e07-confirmatory.md   confirmatory cells with all three CI variants (gate-07 issue 5)
  tables/report-e07-sourcenear.md     source_near vs dopt paired counts per noise arm (gate-07 issue 2)
  figures/report-e07-confirmatory.png forest plot of the dopt-vs-random relative reduction
Usage: uv run python scripts/report_e07.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

Q = Path(__file__).resolve().parents[1] / "reports" / "quarto"
T, F = Q / "tables", Q / "figures"

c = pd.read_csv(T / "e07-sweep-confirmatory.csv")
c["cell"] = c.wind + " K=" + c.K.astype(str) + " k=" + c.k.astype(str)
f3 = lambda v: f"{v:.2f}"  # noqa: E731
md = ["| training wind | K | k | n | median reduction | 95% bootstrap CI | distribution-free CI | mean [CI] | dopt wins/losses | verdict (bootstrap / distribution-free) |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for r in c.itertuples():
    md.append(f"| {r.wind} | {r.K} | {r.k} | {r.n} | {f3(r.median)} | {f3(r.ci_lo)}–{f3(r.ci_hi)} | "
              f"{f3(r.df_ci_lo)}–{f3(r.df_ci_hi)} | {f3(r.mean)} [{f3(r.mean_ci_lo)}–{f3(r.mean_ci_hi)}] | "
              f"{r.wins}/{r.losses} | {r.verdict} / {r.df_verdict} |")
md.append("")
md.append(f"Cells with bootstrap CI above 10 %: {(c.ci_lo > 0.1).sum()}/{len(c)}; distribution-free: "
          f"{(c.df_ci_lo > 0.1).sum()}/{len(c)}; mean bootstrap: {(c.mean_ci_lo > 0.1).sum()}/{len(c)}.")
(T / "report-e07-confirmatory.md").write_text("\n".join(md) + "\n")

rows = pd.read_csv(T / "e07-sweep-rows.csv")
rows = rows[rows.estimator == "map"]
out = ["| noise arm | paired (scenario, k) | source_near better | median (dopt − source_near)/dopt |", "|---|---|---|---|"]
for arm in ["white", "ar1", "persite"]:
    p = rows[(rows.arm == arm) & rows.strategy.isin(["dopt", "source_near"])].pivot_table(
        index=["wind", "K", "seed", "k"], columns="strategy", values="forecast_rmse")
    out.append(f"| {arm} | {len(p)} | {(p.source_near < p.dopt).sum()} | {f3(((p.dopt - p.source_near) / p.dopt).median())} |")
out += ["", "| k | dopt: scenarios with ≥ 1 rank-deficient 24 h test window | source_near: same |", "|---|---|---|"]
w = rows[rows.arm == "white"]
for k in sorted(w.k.unique()):
    n = {s: int((w[(w.strategy == s) & (w.k == k)].frac_window_rank_lt_K > 0).sum()) for s in ["dopt", "source_near"]}
    tot = len(w[(w.strategy == "dopt") & (w.k == k)])
    out.append(f"| {k} | {n['dopt']}/{tot} | {n['source_near']}/{tot} |")
(T / "report-e07-sourcenear.md").write_text("\n".join(out) + "\n")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
y = range(len(c))[::-1]
ax.errorbar(c["median"], list(y), xerr=[c["median"] - c.ci_lo, c.ci_hi - c["median"]], fmt="o", color="#4c72b0",
            capsize=3, label="median, 95% bootstrap CI")
ax.plot(c.df_ci_lo, [v - 0.2 for v in y], "|", color="#c44e52", ms=10, label="distribution-free lower bound")
ax.axvline(0.10, ls=":", color="k", lw=1)
ax.axvline(0, color="grey", lw=0.5)
ax.set_yticks(list(y), c.cell)
ax.set_xlabel("relative reduction in held-out forecast RMSE, dopt vs random")
ax.text(0.11, len(c) - 0.6, "10 % threshold", fontsize=8)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(F / "report-e07-confirmatory.png", dpi=150)
assert (T / "report-e07-confirmatory.md").stat().st_size > 0
print((T / "report-e07-confirmatory.md").read_text(), (T / "report-e07-sourcenear.md").read_text())

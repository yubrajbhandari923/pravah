"""EXP-026 (RUN-0098) figures from results/exp026-hpc/tables. Synthetic emissions/winds; WorldPop is a modelled estimate. Not validation.
Case-level bootstrap CIs for worldpop_real are recomputed here (registered method: percentile bootstrap over the 12 test cases,
n_boot 10000, seed 0) because the run's paired.csv has NaN intervals for the single worldpop draw."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
T = R / "results/exp026-hpc/tables"
OUT = R / "docs/figures/final"
SYN = ["one_town", "separated_settlements", "dense_core_sparse_surround", "mixed_rural_urban"]
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e3e2dd"
BUD = [2, 3, 4, 5, 8]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

paired = pd.read_csv(T / "paired.csv")
paired = paired[paired.subset == "all"]
tm = pd.read_csv(T / "test_metrics.csv")


def wp_case_ci(frac, budget, obj="population", metric="pop_rmse"):
    w = tm[(tm.type == "worldpop_real") & (tm.kind == "best") & (tm.fraction == frac) & (tm.budget == budget)]
    a = w[w.objective == "area"].set_index("case")[metric]
    x = w[w.objective == obj].set_index("case")[metric].loc[a.index]
    d = (x - a).to_numpy()
    if not d.any():
        return 0.0, 0.0, 0.0, a.mean()
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return d.mean(), *np.quantile(bs, [0.025, 0.975]), a.mean()


def fig1():
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
    for ax, f in zip(axs, (0.05, 0.15)):
        ax.axhline(0, color=MUTED, lw=1)
        for i, b in enumerate(BUD):
            c = paired[(paired.fraction == f) & (paired.budget == b) & (paired.objective == "population") & (paired.metric == "pop_rmse")
                       & paired.type.isin(SYN)]
            rel = 100 * c.mean_diff / c.mean_area_layout
            ax.scatter(np.full(len(rel), i) - 0.12 + np.linspace(-0.04, 0.04, len(rel)), rel, s=22, color=BLUE, zorder=3,
                       label="synthetic concentrated fixture (mean of 6 draws)" if i == 0 else None)
            m, lo, hi, base = wp_case_ci(f, b)
            ax.errorbar(i + 0.15, 100 * m / base, yerr=[[100 * (m - lo) / base], [100 * (hi - m) / base]], fmt="D", color=ORANGE, ms=6,
                        capsize=3, zorder=4, label="WorldPop (95% case bootstrap, 12 cases)" if i == 0 else None)
        ax.set_xticks(range(5)); ax.set_xticklabels(BUD); ax.set_xlabel("sensor budget")
        ax.set_title(f"noise fraction {f}", loc="left", fontsize=10); ax.grid(axis="y", color=GRID, lw=0.6)
    axs[0].set_ylabel("test pop-weighted RMSE, population layout minus\narea layout (% of area-layout RMSE)")
    axs[0].set_ylim(-95, 165)
    axs[0].legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, -0.42), ncol=2)
    axs[0].text(0.02, 0.97, "WorldPop budget 3 (0.05) interval clipped at the top", transform=axs[0].transAxes, fontsize=7, color=MUTED, va="top")
    fig.suptitle("EXP-026: no resolved benefit from population weighting (positive = population layout worse)", x=0.01, ha="left", fontsize=10)
    fig.text(0.01, 0.005, "Synthetic emissions/winds; WorldPop modelled; 500 m mesh provisional. Preliminary, not empirical validation.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.subplots_adjust(bottom=0.3, left=0.1, right=0.98, wspace=0.06); fig.savefig(OUT / "exp026_fig1_pop_rmse_effect.png"); plt.close(fig)


def fig2():
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
    c = paired[(paired.objective == "population") & paired.type.isin(SYN)]
    a = c[c.metric == "area_rmse"].set_index(["fraction", "budget", "type"]); p = c[c.metric == "pop_rmse"].set_index(["fraction", "budget", "type"])
    x = 100 * p.mean_diff / p.mean_area_layout; y = 100 * a.mean_diff / a.mean_area_layout
    nz = (x != 0) | (y != 0)
    ax.scatter(x[nz], y[nz], s=22, color=BLUE, label=f"synthetic concentrated ({int(nz.sum())} non-identical cells)", zorder=3)
    xs, ys = [], []
    for f in (0.05, 0.15):
        for b in BUD:
            m, _, _, base = wp_case_ci(f, b); ma, _, _, ba = wp_case_ci(f, b, metric="area_rmse")
            if m or ma: xs.append(100 * m / base); ys.append(100 * ma / ba)
    ax.scatter(xs, ys, s=40, marker="D", color=ORANGE, label=f"WorldPop ({len(xs)} non-identical cells)", zorder=4)
    ax.set_xlabel("pop-weighted RMSE change (%)"); ax.set_ylabel("area RMSE change (%)")
    ax.text(0.98, 0.02, "upper-left quadrant = true trade-off", transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom", ha="right")
    ax.set_title("Population-layout vs area-layout: the two RMSEs mostly move together", loc="left", fontsize=9.5)
    ax.grid(color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.tight_layout(); fig.savefig(OUT / "exp026_fig2_tradeoff_plane.png"); plt.close(fig)


def fig3():
    gap = pd.read_csv(T / "design_gap.csv"); ds = pd.read_csv(T / "design_split.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    g = gap[gap.type.isin(SYN + ["worldpop_real"])].groupby(["fraction", "budget"]).mean_gap.mean().unstack(0)
    xi = np.arange(5)
    for k, (f, col) in enumerate(zip((0.05, 0.15), (BLUE, AQUA))):
        axs[0].bar(xi + (k - 0.5) * 0.38, 100 * g[f].reindex(BUD), 0.36, color=col, label=f"noise fraction {f}")
    axs[0].set_xticks(xi); axs[0].set_xticklabels(BUD); axs[0].set_xlabel("sensor budget")
    axs[0].set_ylabel("greedy objective above best found (%)"); axs[0].set_title("Greedy is far from the design optimum", loc="left", fontsize=9.5)
    axs[0].legend(frameon=False, fontsize=8); axs[0].grid(axis="y", color=GRID, lw=0.6)
    o = ds.groupby(["fraction", "budget"]).optimism_gap.median().unstack(0)
    xb = np.array([0, 1])
    for k, (f, col) in enumerate(zip((0.05, 0.15), (BLUE, AQUA))):
        axs[1].bar(xb + (k - 0.5) * 0.38, 100 * o[f].reindex([5, 8]), 0.36, color=col, label=f"noise fraction {f}")
    axs[1].set_xticks(xb); axs[1].set_xticklabels([5, 8]); axs[1].set_xlabel("sensor budget")
    axs[1].set_ylabel("other-half minus own-half objective (%), median")
    axs[1].set_title("Design-split optimism (budgets 5, 8; 72 layouts each)", loc="left", fontsize=9.5); axs[1].grid(axis="y", color=GRID, lw=0.6)
    fig.text(0.01, 0.005, "Design-objective quantities (6 design cases); synthetic. Preliminary.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "exp026_fig3_greedy_gap_optimism.png"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1(); fig2(); fig3()

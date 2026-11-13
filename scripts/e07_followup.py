"""E07 gate-07 follow-ups (post-hoc, exploratory; no new PDE solves).

windows (cluster, needs the sweep's cached fields):
  OMP_NUM_THREADS=1 uv run python scripts/e07_followup.py windows --fields DIR/fields --out NEW --workers 40
  For every committed sweep design (reports/quarto/tables/e07-sweep-rows.csv) and arm: per 24 h test window, numerical
  rank and practical rank (eigenvalues >= 1 noise sd^2) of the window information, split by the held-out wind regime of
  the issue hour, and the squared forecast error (same MAP, same 5 noise draws) in each window. -> NEW/e07-followup-windows.csv
report (local): uv run python scripts/e07_followup.py report --excl NEW/e07-sweep-rows.csv --windows NEW/e07-followup-windows.csv
  -> reports/quarto/tables/e07-followup-*.csv, reports/quarto/figures/e07-followup.png
The exclusion-control rows come from: e07_cross_objective.py --sweep --stage 2 --fields DIR/fields --exclude-radius 1200.
"""
import argparse, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src")); sys.path.insert(0, str(root / "scripts"))
import e07_cross_objective as E  # noqa: E402
from pravah import benchmarks as B  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402

tab, fig_dir = root / "reports/quarto/tables", root / "reports/quarto/figures"
RANK_TOL = 1e-10  # numerical rank: eigenvalue > RANK_TOL x the design's largest window eigenvalue


def scenario_windows(job):
    fdir, wind, K, seed, rows = job
    names = [n for n, _ in E.scenario_fields("sweep", wind, K, seed)]
    sc = B.generate_scenario(E.make_cfg("sweep", wind), seed, K, truth_field=np.load(Path(fdir) / f"{names[0]}.npy"))
    F = np.stack([np.load(Path(fdir) / f"{n}.npy") for n in names[1:]])
    G3 = X.site_responses(sc, F)
    a_ref, c_bg, n_h = sc.cfg.sources.reference_rate_ug_s, sc.cfg.sources.background_ug_m3, len(sc.truth.winds)
    issue, t0 = X.issue_times(sc), sc.split_hours("test")[0]
    reg = np.array(sc.truth.wind_regimes["test"])[(issue - t0) // B.BLOCK_H]  # regime of the issue hour's 48 h block
    Fi, Ti = F[:, issue + 1], sc.truth.field[issue + 1] - c_bg
    out = []
    for arm, (kind, ps) in X.ARMS.items():
        sigma = X.site_sigma(sc, ps)
        Y = [X.readings(sc, sigma, kind, r, n_h) for r in range(E.SCALES["sweep"]["R"])]
        for _, d in rows[rows.arm == arm].iterrows():
            S = np.array(d.sites.split(), int)
            A = a_ref * G3[S] / sigma[S][:, None, None]
            ev = X.window_info(A, issue)
            nr, pr = (ev > RANK_TOL * ev.max()).sum(1), (ev >= 1).sum(1)
            se, rm = np.zeros(len(issue)), []
            for y in Y:
                u = X.estimate_series(A, (y[S] - c_bg) / sigma[S][:, None], issue)
                e2 = ((np.einsum("tk,ktxy->txy", u * a_ref, Fi) - Ti) ** 2).mean((1, 2))
                se += e2 / len(Y); rm.append(np.sqrt(e2.mean()))
            key = dict(wind=wind, K=K, seed=seed, arm=arm, k=d.k, strategy=d.strategy, net=d.net,
                       rmse_check=float(np.mean(rm) - d.forecast_rmse))  # must be ~0: same estimator/draws as the sweep
            for g in ["all", *dict.fromkeys(reg)]:
                m = reg == g if g != "all" else np.ones(len(reg), bool)
                out.append(dict(**key, regime=g, n_windows=int(m.sum()), frac_rank_lt_K=float((nr[m] < K).mean()),
                                frac_prank_lt_K=float((pr[m] < K).mean()), prank_mean=float(pr[m].mean()),
                                se_sum=float(se[m].sum()), se_def_sum=float(se[m & (pr < K)].sum())))
    return out


def windows(a):
    rows = pd.read_csv(tab / "e07-sweep-rows.csv")
    rows = rows[(rows.status == "ok") & (rows.estimator == "map") & (rows.k > 0)]
    jobs = [(a.fields, w, K, s, g) for (w, K, s), g in rows.groupby(["wind", "K", "seed"])]
    out = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(scenario_windows, jobs):
            out += r
            print(f"done {r[0]['wind']}:{r[0]['K']}:{r[0]['seed']}", flush=True)
    d = pd.DataFrame(out)
    print("max |rmse_check|", d.rmse_check.abs().max())
    d.to_csv(Path(a.out) / "e07-followup-windows.csv", index=False)


# ---------------------------------------------------------------- report
def paired(rho):
    rho = np.asarray(rho, float)
    lo, hi = X.boot_median_ci(rho); dlo, dhi = X.df_median_ci(rho)
    pos, neg, tie, _ = X.sign_test(rho)
    return dict(n=len(rho), median=float(np.median(rho)), ci_lo=lo, ci_hi=hi, df_ci_lo=dlo, df_ci_hi=dhi, wins=pos, losses=neg, ties=tie)


def comparisons(df, excl):
    """Paired forecast-RMSE contrasts (rho = (RMSE_ref - RMSE_strategy)/RMSE_ref, > 0 = strategy better; undefined = -inf
    as in the sweep) per (arm, K, k) pooled over training winds (n = 20), plus all cells (n = 120) with layout counts."""
    out = []
    for arm in X.ARMS:
        st = E.scenario_table(df, arm, "map")
        v = st.pivot_table(index=["wind", "K", "seed", "k"], columns="strategy", values="forecast_rmse", dropna=False)
        ok = st.pivot_table(index=["wind", "K", "seed", "k"], columns="strategy", values="defined").astype(bool)
        pairs = [(s, "random") for s in v.columns if s not in ("random", "background_only", "climatology")] + [("source_near", "dopt")]
        for s, ref in pairs:
            rho = ((v[ref] - v[s]) / v[ref]).where(ok[s], -np.inf).where(ok[ref], np.inf).rename("rho").reset_index()
            for (K, k), c in rho.groupby(["K", "k"]):
                out.append(dict(exclusion_m=excl, arm=arm, strategy=s, vs=ref, K=K, k=k, **paired(c.rho)))
            lay = rho.groupby(["K", "seed"]).rho.median()  # 20 source layouts (a layout = K, seed; shared by both winds)
            out.append(dict(exclusion_m=excl, arm=arm, strategy=s, vs=ref, K="all", k="all", **paired(rho.rho),
                            layouts_better=int((lay > 0).sum()), layouts_worse=int((lay < 0).sum()), n_layouts=len(lay)))
    return pd.DataFrame(out)


def report(a):
    ex = pd.read_csv(a.excl)
    ex.to_csv(tab / "e07-followup-exclusion-rows.csv", index=False)
    base = pd.read_csv(tab / "e07-sweep-rows.csv")
    cmp_ = pd.concat([comparisons(base, 0), comparisons(ex, 1200)])
    cmp_.to_csv(tab / "e07-followup-exclusion-summary.csv", index=False)
    conf = E.confirmatory(ex)
    conf.to_csv(tab / "e07-followup-exclusion-confirmatory.csv", index=False)
    print("exclusion, dopt vs random (confirmatory-style, exploratory here):", conf.attrs["overall"])
    print(cmp_[(cmp_.K == "all") & (cmp_.arm == "white")].to_string(index=False))

    w = pd.read_csv(a.windows)
    assert w.rmse_check.abs().max() < 1e-6, w.rmse_check.abs().max()
    # one value per (scenario, arm, k, strategy, regime): random = mean over its 10 networks
    g = w.groupby(["wind", "K", "seed", "arm", "k", "strategy", "regime"])
    sc = g[["n_windows", "frac_rank_lt_K", "frac_prank_lt_K", "prank_mean", "se_sum", "se_def_sum"]].mean().reset_index()
    sc["se_share_def"] = sc.se_def_sum / sc.se_sum
    sc.to_csv(tab / "e07-followup-windows-scenario.csv", index=False)
    tot = sc[sc.regime == "all"].set_index(["wind", "K", "seed", "arm", "k", "strategy"]).se_sum
    sc["se_share_of_total"] = sc.se_sum.to_numpy() / tot.loc[list(zip(sc.wind, sc.K, sc.seed, sc.arm, sc.k, sc.strategy))].to_numpy()
    agg = sc.groupby(["arm", "strategy", "k", "regime"]).agg(
        n_scen=("seed", "size"), frac_rank_lt_K=("frac_rank_lt_K", "mean"), frac_prank_lt_K=("frac_prank_lt_K", "mean"),
        scen_any_prank_def=("frac_prank_lt_K", lambda x: int((x > 0).sum())), prank_mean=("prank_mean", "mean"),
        se_share_def_median=("se_share_def", "median"), se_share_of_total_median=("se_share_of_total", "median"),
        se_def_pooled=("se_def_sum", "sum"), se_pooled=("se_sum", "sum")).reset_index()
    agg["se_share_def_pooled"] = agg.se_def_pooled / agg.se_pooled
    agg.drop(columns=["se_def_pooled", "se_pooled"]).to_csv(tab / "e07-followup-windows.csv", index=False)
    pd.set_option("display.width", 250)
    print(agg[(agg.arm == "white") & agg.strategy.isin(["dopt", "source_near", "random"])].round(3).to_string(index=False))
    plot(cmp_, agg)


def plot(cmp_, agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"dopt": "#2a78d6", "source_near": "#eb6834", "random": "#7f7f7f"}
    fg, ax = plt.subplots(1, 2, figsize=(11.5, 5))
    c = cmp_[(cmp_.arm == "white") & (cmp_.K != "all")]
    for (s, ref, lab, hue), off in zip([("source_near", "dopt", "source_near vs dopt", "#eb6834"),
                                        ("dopt", "random", "dopt vs random", "#2a78d6")], (-.12, .12)):
        for ex_, ls, mk in ((0, "-", "o"), (1200, "--", "s")):
            q = c[(c.strategy == s) & (c.vs == ref) & (c.exclusion_m == ex_)]
            x = np.arange(len(q)) + off + (.05 if ex_ else -.05)
            ax[0].errorbar(x, q["median"], yerr=[q["median"] - q.ci_lo, q.ci_hi - q["median"]], fmt=mk, ls="none", color=hue,
                           mfc=hue if ex_ == 0 else "white", ms=6, lw=1.5, label=f"{lab}, {'all sites' if ex_ == 0 else '>1200 m only'}")
    ax[0].set_xticks(range(len(q)), [f"K={K}\nk={k}" for K, k in zip(q.K, q.k)])
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set_ylabel("median relative forecast-RMSE reduction\n(95 % bootstrap CI, n = 20 scenarios)")
    ax[0].set_title("Exclusion control (white noise, MAP)", fontsize=10)
    ax[0].legend(fontsize=7.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.2))
    r = agg[(agg.arm == "white") & (agg.k == 8) & (agg.regime != "all")]
    regs = sorted(r.regime.unique())
    for i, s in enumerate(["random", "dopt", "source_near"]):
        q = r[r.strategy == s].set_index("regime").reindex(regs)
        ax[1].bar(np.arange(len(regs)) + (i - 1) * .27, q.frac_prank_lt_K, .25, color=col[s], label=s, edgecolor="white", lw=1)
    ax[1].set_xticks(range(len(regs)), [x.replace("_", "\n") for x in regs])
    ax[1].set_ylabel("fraction of 24 h test windows with practical rank < K\n(mean over 40 scenarios)")
    ax[1].set_title("Identifiability by held-out wind regime (k = 8, white)", fontsize=10)
    ax[1].legend(fontsize=8, frameon=False)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fg.tight_layout()
    fg.savefig(fig_dir / "e07-followup.png", dpi=130)
    plt.close(fg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("windows", "report"))
    ap.add_argument("--fields"); ap.add_argument("--out", default=".")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count()))
    ap.add_argument("--excl"); ap.add_argument("--windows")
    a = ap.parse_args()
    (windows if a.mode == "windows" else report)(a)

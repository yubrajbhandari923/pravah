"""Gate-07 independent checks on reports/quarto/tables/e07-sweep-rows.csv (no reuse of the driver's analysis code).

uv run python docs/research/gates/gate-07/check_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

root = Path(__file__).resolve().parents[4]
tab = root / "reports/quarto/tables"
df = pd.read_csv(tab / "e07-sweep-rows.csv")
d = df[df.arm != "-"].copy()
d["ok"] = d.status == "ok"
print("rows", len(df), "design rows", len(d), "status counts:\n", d.groupby(["strategy", "ok"]).size().unstack(fill_value=0))

# per scenario x arm x k x strategy (random: mean of 10 network RMSEs)
key = ["arm", "wind", "K", "seed", "k"]
m = d.groupby(key + ["strategy"]).agg(f=("forecast_rmse", "mean"), ok=("ok", "all"), src=("source_rmse", "mean"),
                                      D_test=("D_test", "mean"), fr=("frac_window_rank_lt_K", "mean")).reset_index()
P = m.pivot_table(index=key, columns="strategy", values="f")
OK = m.pivot_table(index=key, columns="strategy", values="ok").astype(bool)


def boot_ci(x, n=10000, seed=4000):
    i = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    md = np.median(x[i], 1)
    return np.quantile(md, .025, method="lower"), np.quantile(md, .975, method="higher")


print("\n== 1. confirmatory re-derivation (white arm, dopt vs mean-of-10-random) ==")
w = P.loc["white"]
out = []
for (wind, K, k), g in w.groupby(level=["wind", "K", "k"]):
    if k < 8:
        continue
    rho = ((g.random - g.dopt) / g.random).to_numpy().copy()
    rho[~OK.loc["white"].loc[g.index, "dopt"].to_numpy()] = -np.inf
    lo, hi = boot_ci(rho)
    # protocol §10.1 literally says paired bootstrap of the (mean) relative reduction; report that too
    i = np.random.default_rng(4000).integers(0, len(rho), (10000, len(rho)))
    mn = rho[i].mean(1)
    out.append(dict(wind=wind, K=K, k=k, n=len(rho), median=np.median(rho), lo=lo, hi=hi, mean=rho.mean(),
                    mean_lo=np.quantile(mn, .025), mean_hi=np.quantile(mn, .975), min=rho.min(), wins=(rho > 0).sum()))
t = pd.DataFrame(out)
print(t.round(3).to_string(index=False))
ref = pd.read_csv(tab / "e07-sweep-confirmatory.csv")
print("max |median diff| vs committed table:", np.abs(t["median"].to_numpy() - ref["median"].to_numpy()).max(),
      " max |ci_lo diff|:", np.abs(t.lo.to_numpy() - ref.ci_lo.to_numpy()).max())
print("cells with median-CI lo > 0.10:", (t.lo > .1).sum(), "/", len(t), "; mean-CI lo > 0.10:", (t.mean_lo > .1).sum())
# pooled over winds (pilot-style cell definition)
for (K, k), g in w[w.index.get_level_values("k") >= 8].groupby(level=["K", "k"]):
    rho = ((g.random - g.dopt) / g.random).to_numpy().copy()
    print(f"pooled winds K={K} k={k}: n={len(rho)} median {np.median(rho):.3f} CI {boot_ci(rho)}")

print("\n== 2. source_near vs dopt, paired per (arm, scenario, k) ==")
for arm in ["white", "ar1", "persite"]:
    a = P.loc[arm]
    okk = OK.loc[arm]
    both = okk.dopt & okk.source_near
    r = ((a.dopt - a.source_near) / a.dopt)[both]
    print(f"{arm}: n={both.sum()} source_near better in {(r > 0).sum()}  median rel. adv {r.median():.3f} "
          f"IQR [{r.quantile(.25):.3f}, {r.quantile(.75):.3f}]  sign p {binomtest(int((r > 0).sum()), len(r)).pvalue:.2g}")
    print("   by k:", {k: f"{(g > 0).sum()}/{len(g)}" for k, g in r.groupby(level="k")},
          " by wind/K:", {f"{wd}{K}": f"{(g > 0).sum()}/{len(g)}" for (wd, K), g in r.groupby(level=["wind", "K"])})
    # seed-level (cluster) view: count scenarios where source_near wins at every k
    sc = (r > 0).groupby(level=["wind", "K", "seed"]).all()
    print(f"   scenarios where source_near wins at all k: {sc.sum()}/{len(sc)}")

print("\n== 3. design-time vs held-out: D_test and window rank deficits (white) ==")
q = m[m.arm == "white"].pivot_table(index=["wind", "K", "seed", "k"], columns="strategy", values=["D_test", "fr", "src"])
for s in ["random", "dopt", "source_near", "hotspot", "coverage"]:
    print(f"{s:12s} median D_test {q['D_test'][s].median():7.2f}  frac windows rank<K {q['fr'][s].median():.3f}  "
          f"source_rmse {q['src'][s].median():.3f}")
print("D_test(source_near) > D_test(dopt) in", (q["D_test"].source_near > q["D_test"].dopt).sum(), "/", len(q))

print("\n== 4. near-source geometry: distance of chosen sites to the nearest true source (white, k=8) ==")
import sys
sys.path.insert(0, str(root / "src"))
from pravah import benchmarks as B  # noqa: E402
from pravah.experiment_config import ExperimentConfig, Winds  # noqa: E402

c = ExperimentConfig.default("EXP-E07")
c.model.grid = (64, 64)
dom = np.array(c.model.domain_m)
ii, jj = np.divmod(np.arange(256), 16)
sxy_site = np.stack([(4 // 2 + 4 * ii + .5) * dom[0] / 64, (4 // 2 + 4 * jj + .5) * dom[1] / 64], 1)
dist = []
for (wind, K, seed), g in d[(d.arm == "white") & (d.k == 8) & d.strategy.isin(["dopt", "source_near"])].groupby(["wind", "K", "seed"]):
    _, _, sxy, _, _, _ = B.scenario_drivers(c, seed, K)
    for _, r in g.iterrows():
        S = np.array(r.sites.split(), int)
        dd = np.linalg.norm(sxy_site[S][:, None] - sxy[None], axis=2).min(1)
        phi = np.exp(-dd ** 2 / (2 * 400 ** 2)) * (dd <= 1200)
        dist.append(dict(strategy=r.strategy, med_d=np.median(dd), n_in_footprint=(dd <= 1200).sum(), mean_phi_rel=phi.mean()))
dist = pd.DataFrame(dist)
print(dist.groupby("strategy").median().round(3))
print("(footprint = emission disc radius 3 sigma = 1200 m; admissible sites are > 400 m = 1 sigma from a source)")

print("\n== 5. superposition error per scenario (compute table) ==")
cp = pd.read_csv(tab / "e07-sweep-compute.csv")
print(cp.groupby(["wind", "K"]).linearity_max_abs.describe().round(4))
print("max linearity_max_abs / sigma(0.25):", (cp.linearity_max_abs.max() / .25).round(3))

print("\n== 6. sensitivity: distribution-free order-statistic 95% CI for the median (n=10: x(2)..x(9), coverage 97.9%) ==")
for (wind, K, k), g in w.groupby(level=["wind", "K", "k"]):
    if k < 8:
        continue
    rho = np.sort(((g.random - g.dopt) / g.random).to_numpy())
    print(f"{wind:8s} K={K} k={k:2d}: x(2) {rho[1]:.3f}  x(9) {rho[8]:.3f}  -> {'above' if rho[1] > .1 else 'straddles/below'}")

print("\n== 7. source_near vs dopt, clustered: per source layout (K, seed; shared by both training winds), white ==")
a = P.loc["white"]
r = ((a.dopt - a.source_near) / a.dopt)
cl = r.groupby(level=["K", "seed"]).median()
print(f"layouts where median advantage > 0: {(cl > 0).sum()}/{len(cl)}; median of layout medians {cl.median():.3f}; "
      f"min {cl.min():.3f}; bootstrap 95% CI of the median over layouts {boot_ci(cl.to_numpy())}")

print("\n== 8. regret matrix (white): median and 90th percentile over (scenario, k); recomputed from the committed regret table ==")
rg = pd.read_csv(tab / "e07-sweep-regret.csv")
rg = rg[rg.arm == "white"]
print(rg.groupby(["selector", "metric"]).regret.median().unstack().round(3))
print(rg.groupby(["selector", "metric"]).regret.quantile(.9).unstack().round(3))
fz = rg[(rg.selector == "forecast_rmse") & (rg.metric == "source_rmse")]
print("forecast-best == source-best in", (fz.regret == 0).sum(), "/", len(fz))

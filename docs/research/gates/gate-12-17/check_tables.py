"""Gate-12-17: claim checks that need only the committed tables and the 64^2 row CSVs (read-only scratch copies).
Independent re-implementations of the cluster statistics (E11), k* (E08) and the claim numbers (E12-E17).
Usage: uv run python check_tables.py SCRATCH > check_tables.out   (SCRATCH holds e08-sweep/, e11-sweep/, e17-sweep/)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
root = Path(__file__).resolve().parents[4]
T = root / "reports/quarto/tables"
SC = Path(sys.argv[1])


def sign_p(x):
    x = np.asarray(x, float)
    p, n = int((x > 1e-12).sum()), int((x < -1e-12).sum())
    return p, n, (binomtest(p, p + n).pvalue if p + n else 1.0)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0
    for i, j in enumerate(o):
        run = max(run, min(1, (m - i) * p[j]))
        adj[j] = run
    return adj


def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


# ------------------------------------------------------------------------------------------------ E08
hdr("E08 map_full: k*(0.2), floor, H5 (64^2 rows)")
for tag in ("", "-mapfull", "-excl1200"):
    r = pd.read_csv(SC / f"e08-sweep/e08-sweep{tag}-rows.csv")
    bg = r[r.strategy == "background_only"].set_index(["K", "seed"]).forecast_rmse
    d = r[(r.strategy != "background_only") & (r.arm == "w1") & (r.schedule == "S4")].copy()
    d["nr"] = d.forecast_rmse.where(d.status == "ok", np.inf).to_numpy() / bg.loc[list(zip(d.K, d.seed))].to_numpy()
    ks = {}
    for (K, s, st), g in d.groupby(["K", "seed", "strategy"]):  # independent k*: smallest k with nr <= .2 at it and all larger k
        g = g.sort_values("k")
        ok = np.nan_to_num(g.nr.to_numpy(), nan=np.inf) <= 0.2
        kk = 64
        for k, o in zip(g.k.to_numpy()[::-1], ok[::-1]):
            if not o:
                break
            kk = int(k)
        ks[(K, s, st)] = kk
    ks = pd.Series(ks).rename("kstar").rename_axis(["K", "seed", "strategy"]).reset_index()
    print(f"\n[{tag or 'reference'}] median k*(0.2) (K x strategy):")
    print(ks.pivot_table(index="K", columns="strategy", values="kstar", aggfunc="median").to_string())
    print("max nRMSE at k = 2 by K x strategy:")
    print(d[d.k == 2].pivot_table(index="K", columns="strategy", values="nr", aggfunc="max").round(4).to_string())
    for k in (2, 8, 32):
        q = d[d.k == k].pivot_table(index=["K", "seed"], columns="strategy", values="forecast_rmse")
        x = np.log(q.dopt) - np.log(q.source_near)
        fresh = ~np.array([(K in (2, 4) and s <= 9) for K, s in x.index])
        xf = x[fresh]
        print(f"H5 k={k} fresh n={len(xf)}: source_near wins {int((xf > 0).sum())}, dopt wins {int((xf < 0).sum())}, "
              f"median log ratio {np.nanmedian(xf):.3f}, n_nonfinite {int((~np.isfinite(xf)).sum())}")
c = pd.read_csv(SC / "e08-sweep/e08-sweep-mapfull-compute.csv") if (SC / "e08-sweep/e08-sweep-mapfull-compute.csv").exists() else None
cr = pd.read_csv(SC / "e08-sweep/e08-sweep-compute.csv")
print("\nn_hours per scenario:", cr.n_hours.unique(), "; train = days 1-30 = 720 h; test issue hours start after day 42")
print("VanLeer superposition |c_bg + sum a F - truth| max (ug/m3): median %.3f, max %.3f" % (cr.linearity_max_abs.median(), cr.linearity_max_abs.max()))
r = pd.read_csv(SC / "e08-sweep/e08-sweep-mapfull-rows.csv")
bg = r[r.strategy == "background_only"].forecast_rmse
print("background-only forecast RMSE median %.3f; map_full dopt k=8 forecast RMSE median %.4f" % (
    bg.median(), r[(r.strategy == "dopt") & (r.k == 8) & (r.arm == "w1") & (r.schedule == "S4")].forecast_rmse.median()))
e = pd.read_csv(SC / "e08-sweep/e08-sweep-excl1200-rows.csv")
bad = e[(e.k == 32) & (e.arm == "w1") & (e.schedule == "S4") & (e.strategy.isin(["dopt", "source_near"]))]
print("excl1200 k=32 non-ok rows (dopt/source_near):")
print(bad[bad.status != "ok"][["K", "seed", "strategy", "status"]].drop_duplicates().to_string(index=False))

# ------------------------------------------------------------------------------------------------ E11
hdr("E11 64^2: independent cluster-level statistics and Holm family")
r = pd.read_csv(SC / "e11-sweep/e11-sweep-rows.csv")
ps = r.groupby(["estimator", "wind", "K", "seed", "k", "design"]).forecast_rmse_h1.mean().reset_index()
com = pd.read_csv(T / "e11-sweep-contrasts.csv")
fam = com[com.holm_family]
mine = []
for _, row in fam.iterrows():
    q = ps[(ps.estimator == row.estimator) & (ps.k == row.k)].pivot_table(index=["wind", "K", "seed"], columns="design",
                                                                         values="forecast_rmse_h1")
    rho = ((q[row.comparator] - q[row.design]) / q[row.comparator]).dropna()
    cl = rho.groupby(level=["K", "seed"]).mean()
    w, l, p = sign_p(cl)
    mine.append(dict(estimator=row.estimator, k=row.k, design=row.design, comparator=row.comparator, n=len(cl),
                     med=float(np.median(cl)), w=w, l=l, p=p, com_med=row.cl_median, com_p=row.cl_sign_p, com_holm=row.cl_holm_p))
mine = pd.DataFrame(mine)
mine["holm72"] = holm(mine.p)
mine["holm24_perk"] = mine.groupby("k").p.transform(lambda x: holm(x.to_numpy()))
print("family size:", len(mine), "(sign-off text says 24; code family = 72 = 2 est x 3 k x 6 designs x 2 comparators)")
print("max |median - committed|:", float((mine.med - mine.com_med).abs().max()), " max |p - committed|:",
      float((mine.p - mine.com_p).abs().max()), " max |holm72 - committed|:", float((mine.holm72 - mine.com_holm).abs().max()))
print("verdict changes 72 vs 24-per-k at 0.05:", int(((mine.holm72 < .05) != (mine.holm24_perk < .05)).sum()))
sel = mine[mine.design.isin(["fc_w24_rob", "fc_w24_rob_noSN", "fc_grow"])]
print(sel[["estimator", "k", "design", "comparator", "n", "med", "w", "l", "p", "holm72", "holm24_perk"]].round(4).to_string(index=False))

# ------------------------------------------------------------------------------------------------ E12
hdr("E12 pilot: claims (k = 8, dopt/source_near)")
s = pd.read_csv(T / "e12-local-summary.csv")
q = s[(s.k == 8) & s.design.isin(["dopt", "source_near"]) & s.estimator.isin(["map24", "map_full"]) & (s.spec != "oracle")]
print("sensor-fit ratio range by spec:")
print(q.groupby("spec").agg(fit_lo=("D_sensor_fit", "min"), fit_hi=("D_sensor_fit", "max"), src_lo=("D_source_rmse", "min"),
                            src_hi=("D_source_rmse", "max"), cov_lo=("med_cover95_a", "min"), cov_hi=("med_cover95_a", "max"),
                            ac1_lo=("med_innov_ac1", "min"), ac1_hi=("med_innov_ac1", "max")).round(3).to_string())
c = pd.read_csv(T / "e12-local-contrasts.csv")
m = c[(c.contrast == "H3_memory_gain_M") & (c.spec == "oracle") & (c.k == 8)]
print("H3 oracle memory gain M (k = 8):", m[["design", "median"]].round(3).to_string(index=False))
r12 = pd.read_csv(T / "e12-local-rows.csv")
b = r12[r12.spec == "baseline"]
print("persistence 1 h RMSE median %.4f; background-only %.4f" % (b[b.design == "persistence"].forecast_rmse_h1.median(),
                                                                b[b.design == "background_only"].forecast_rmse_h1.median()))
o = r12[(r12.spec == "oracle") & (r12.k == 8) & r12.design.isin(["dopt", "source_near"])]
print("oracle forecast_rmse_h1 at k=8 by estimator (median):", o.groupby("estimator").forecast_rmse_h1.median().round(4).to_dict())

# ------------------------------------------------------------------------------------------------ E13
hdr("E13 pilot: background / common-mode bias / silent failure (k = 8)")
s = pd.read_csv(T / "e13-local-summary.csv")
q = s[(s.k == 8) & s.design.isin(["dopt", "source_near"]) & s.spec.isin(["oracle", "bgoff1", "common1", "bgdrift1", "shift400",
                                                                         "shift800", "wide800", "missing1", "ar1", "drift1"])]
print(q[["spec", "design", "estimator", "skill_h1", "skill_lo", "skill_hi", "skill_wins", "med_cover95_a", "D_sensor_fit",
         "med_innov_ac1", "med_source_bias"]].round(3).to_string(index=False))
r13 = pd.read_csv(T / "e13-local-rows.csv")
cm = r13[(r13.spec == "common1") & r13.estimator.str.endswith("_bg") & (r13.k == 8)]
print("common1 _bg: bg_hat median %.3f (truth has no field offset; the forecast adds bg_hat x 1 ug/m3 everywhere)" % cm.bg_hat.median())
pe = r13[(r13.design == "persistence") & (r13.horizon == 1)].groupby("spec").forecast_rmse.median()
print("persistence 1 h RMSE (median) oracle %.4f, common1 %.4f, bgoff1 %.4f" % (pe["oracle"], pe["common1"], pe["bgoff1"]))
print("common1 _bg forecast_rmse_h1 median %.3f  (|offset| = 1 would give RMSE ~ 1)" % cm.forecast_rmse_h1.median())
# post-hoc field-level rule as worded in the lead decision: fit ratio < 1.1, |lag-1| < 0.1, skill < 0
f = s[(s.spec != "oracle")]
rule = f[(f.D_sensor_fit < 1.1) & (f.med_innov_ac1.abs() < 0.1) & (f.skill_h1 < 0)]
print("cells meeting the post-hoc field-level rule (median level):")
print(rule[["spec", "k", "design", "estimator", "skill_h1", "D_sensor_fit", "med_innov_ac1", "med_cover95_a"]].round(3).to_string(index=False))
cx = pd.read_csv(T / "e13-local-counterexamples.csv")
print("pre-declared counterexample counts:", cx.kind.value_counts().to_dict())

# ------------------------------------------------------------------------------------------------ E14
hdr("E14 pilot: AUC scopes, omitted source by design, AR(1)")
u = pd.read_csv(T / "e14-local-units.csv.gz")
ev = u[u.split == "evaluation"]
g = ev[ev.arm.isin(["omit1", "oracle", "noise_ar1"])].groupby(["arm", "design"]).agg(
    L1=("worse_than_persistence", "mean"), nis=("nis", "median"), ac1=("innov_ac1", "median"), fit=("sensor_fit", "median"),
    ratio=("rmse_model", "median"))
print(g.round(3).to_string())
print("validation blocks:", sorted(u[u.split == "validation"].block.unique()), " evaluation blocks:", sorted(ev.block.unique()))
a = pd.read_csv(T / "e14-local-auc.csv")
print(a[a.label == "worse_than_persistence"][["diagnostic", "scope", "auc", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
# omitted source excluding random networks
from sklearn.metrics import roc_auc_score  # noqa: E402  (sanity: same AUC definition as RL.auc)
for des in ("dopt", "source_near", "random"):
    gg = ev[ev.arm.isin(["oracle", "omit1"]) & (ev.design == des)]
    for dg in ("nis", "innov_ac1", "sensor_fit", "sensor_skill"):
        y = gg.worse_than_persistence.astype(bool)
        if y.any() and (~y).any():
            print(f"omit1+oracle, design={des:11s} {dg:12s} pooled AUC {roc_auc_score(y, gg[dg]):.3f}  base rate {y.mean():.2f}")

# ------------------------------------------------------------------------------------------------ E15
hdr("E15 pilot: multiplicity of the contrast table; headline")
c = pd.read_csv(T / "e15-local-contrasts.csv")
print("contrast rows:", len(c), "; rows with 6/0 or 0/6 (p = 0.031):", int(((c.wins == 6) | (c.losses == 6)).sum()))
tail = c[c.metric.isin(["cond_cvar", "cond_worst", "max_regret"]) & c.design.str.startswith("rob_")]
print("robust-design tail rows:", len(tail), "; 6/0 favouring the robust design:", int((tail.wins == 6).sum()),
      "; 0/6 against:", int((tail.losses == 6).sum()))
h = tail[(tail.wins == 6)][["estimator", "k", "metric", "design", "comparator", "median", "ci_lo", "ci_hi"]]
print(h.round(3).to_string(index=False))

# ------------------------------------------------------------------------------------------------ E16
hdr("E16 pilot: src_sparse layouts; bg_hi sigma ratio")
cp = pd.read_csv(T / "e16-local-compute.csv")
lay = cp.pivot_table(index=["K", "seed"], columns="cell", values="source_xy", aggfunc="first")
print("src_sparse layout differs from nominal in", int((lay.src_sparse != lay.nominal).sum()), "of", len(lay), "scenarios")
r16 = pd.read_csv(T / "e16-local-rows.csv")
sg = r16[r16.strategy == "background_only"].pivot_table(index=["K", "seed"], columns="cell", values="sigma")
print("sigma(bg_hi)/sigma(nominal): median %.2f, range %.2f-%.2f" % ((sg.bg_hi / sg.nominal).median(), (sg.bg_hi / sg.nominal).min(),
                                                                    (sg.bg_hi / sg.nominal).max()))

# ------------------------------------------------------------------------------------------------ E17
hdr("E17 64^2: baselines, wrong wind vs persistence/background, clusters share one wind history")
r = pd.read_csv(SC / "e17-sweep/e17-sweep-rows.csv")
bl = r[r.spec == "-"].pivot_table(index=["case", "K", "seed"], columns="strategy", values="forecast_rmse")
x = r[r.spec != "-"].groupby(["case", "spec", "estimator", "k", "strategy", "K", "seed"]).forecast_rmse.mean().reset_index()
x = x.join(bl, on=["case", "K", "seed"])
x["vs_pers"] = x.forecast_rmse / x.persistence
x["vs_bg"] = x.forecast_rmse / x.background_only
t = x[x.k == 8].groupby(["case", "spec", "estimator", "strategy"]).agg(rmse=("forecast_rmse", "median"),
                                                                    vs_pers=("vs_pers", "median"), vs_bg=("vs_bg", "median"),
                                                                    beats_pers=("vs_pers", lambda v: int((v < 1).sum())))
print(t.round(4).to_string())
print("all wrong-wind (coolA) forecasts / persistence, every k and strategy: range of cluster medians",
      x[(x.case == "coolA") & (x.spec != "oracle")].groupby(["spec", "estimator", "k", "strategy"]).vs_pers.median().agg(["min", "max"]).round(2).to_dict())
mon = x[(x.case == "monsoon") & (x.spec == "oracle") & (x.estimator == "map24") & (x.k == 8)].pivot_table(
    index=["K", "seed"], columns="strategy", values="forecast_rmse")
rho = (mon.source_near - mon.dopt) / mon.source_near
w, l, p = sign_p(rho)
print(f"monsoon map24 k=8 dopt vs source_near: median {rho.median():.3f}, {w}/{l}, p {p:.4f}; Holm over the 18 dopt-vs-source_near "
      f"(case x est x k) tests would need p < {0.05 / 18:.4f}")
sc_ = pd.read_csv(SC / "e17-sweep/e17-sweep-scenarios.csv")
print("superposition_rel_max (VanLeer, 64^2): median %.4f, max %.4f" % (sc_.superposition_rel_max.median(), sc_.superposition_rel_max.max()))

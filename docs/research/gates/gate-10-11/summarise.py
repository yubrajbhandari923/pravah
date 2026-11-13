"""Gate-10/11 analysis of check.py outputs + committed tables. uv run python docs/research/gates/gate-10-11/summarise.py
> docs/research/gates/gate-10-11/summarise.out"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest

here = Path(__file__).resolve().parent
tab = here.parents[3] / "reports/quarto/tables"
pd.set_option("display.width", 250), pd.set_option("display.max_columns", 40), pd.set_option("display.max_rows", 400)


def ci(x):
    x, n = np.sort(np.asarray(x, float)), len(x)
    j = max(i for i in range(1, n // 2 + 1) if 1 - 2 * binom.cdf(i - 1, n, .5) >= .95)
    return x[j - 1], x[n - j]


def stat(rho, cl=None):
    rho = pd.Series(rho)
    lo, hi = ci(rho)
    out = dict(n=len(rho), med=round(float(rho.median()), 3), lo=round(lo, 3), hi=round(hi, 3), w=int((rho > 1e-12).sum()),
               l=int((rho < -1e-12).sum()))
    if cl is not None:  # cluster = (K, seed): veering/mixed share layout, strengths, noise and (checked) test winds
        c = rho.groupby(cl).mean()
        out.update(n_cl=len(c), cl_med=round(float(c.median()), 3), cl_min=round(float(c.min()), 3),
                   cl_w=int((c > 0).sum()), cl_p=round(binomtest(int((c > 0).sum()), len(c)).pvalue, 3))
    return out


# ---------------------------------------------------------------- E10 memory competitors
e = pd.read_csv(here / "e10-memory-competitors.csv")
t = pd.read_csv(tab / "e10-local-rows.csv")
key = ["wind", "K", "seed", "k", "design", "variant"]
kf = t[t.estimator.isin(["kf_qsel", "kf_q1e-01", "kf_q1e-02", "map_full", "map24"])].pivot_table(
    index=key, columns="estimator", values="forecast_rmse_h1")
j = e.set_index(key).join(kf)
print("E10 reproduction: max |f1_map_full - table| =", float((j.f1_map_full - j.map_full).abs().max()),
      " max |f1_map24 - table| =", float((j.f1_map24 - j.map24).abs().max()))
rows = []
for (v, k, des), g in j.reset_index().groupby(["variant", "k", "design"]):
    cl = list(zip(g.K, g.seed))
    for comp in ("f1_wsel", "f1_ewsel", "f1_anysel", "f1_oracle_map"):
        rows.append(dict(variant=v, k=k, design=des, kf="kf_qsel", vs=comp,
                         **stat(((g[comp] - g.kf_qsel) / g[comp]).values, pd.Index(cl))))
    rows.append(dict(variant=v, k=k, design=des, kf="f1_wsel", vs="map_full",
                     **stat(((g.f1_map_full - g.f1_wsel) / g.f1_map_full).values, pd.Index(cl))))
r = pd.DataFrame(rows)
print("\nE10: (competitor - kf_qsel)/competitor, 1 h forecast RMSE (>0 = filter better); and wsel vs map_full")
print(r[r.design != "random"].to_string(index=False))
print("\nE10 picks of validation-selected MAP window (dopt/source_near, diurnal+logrw):")
pk = e[(e.design != "random") & (e.variant != "const")]
print(pd.Series(" ".join(pk.pick_anysel).split()).value_counts().to_string())

# ---------------------------------------------------------------- E10 horizon reversal and lag break-even (committed tables)
p = pd.read_csv(tab / "e10-local-paired.csv")
q = p[(p.ref == "map_full") & (p.variant == "diurnal") & (p.design != "random") & (p.estimator == "kf_qsel")]
print("\nE10 diurnal, kf_qsel vs map_full (committed paired table):")
print(q[["k", "design", "metric", "median_rel_reduction", "ci_lo", "ci_hi", "wins", "losses"]].to_string(index=False))
d = np.arange(0, 13)
print("\nDiurnal: RMS(a(t-d) - a(t)) / RMS(a(t) - mean a) = 2 sin(pi d / 24); break-even at d = 4 h:",
      dict(zip(d.tolist(), np.round(2 * np.sin(np.pi * d / 24), 3).tolist())))

# ---------------------------------------------------------------- E11 prior ablation
f = pd.read_csv(here / "e11-prior-ablation.csv")
st = f.assign(lab=np.where(f.prior.isin(["-", "orig8"]), f.design, f.design + ":" + f.prior))
u = pd.read_csv(tab / "e11-local-rows.csv")
uu = u[(u.net == 0) & u.design.isin(["dopt", "source_near", "fc_w24_rob", "fc_grow_rob", "fc_w24", "fc_grow"])]
m = st[st.prior.isin(["-", "orig8"])].merge(uu, on=["wind", "K", "seed", "k", "design", "estimator"])
print("\nE11 reproduction (dopt, source_near, fc_* under both estimators): sites identical",
      bool((m.sites_x == m.sites_y).all()), "max |f1 - table| =", float((m.f1 - m.forecast_rmse_h1).abs().max()),
      "rows", len(m))
out = []
for (est, k), g in st.groupby(["estimator", "k"]):
    for metric in ("f1", "f1_south"):
        piv = g.pivot_table(index=["wind", "K", "seed"], columns="lab", values=metric)
        cl = pd.Index(list(zip(piv.index.get_level_values("K"), piv.index.get_level_values("seed"))))
        for des in [c for c in piv.columns if c.startswith("fc_w24_rob") or c.startswith("fc_grow_rob")]:
            for cmp_ in ("dopt", "source_near"):
                rho = ((piv[cmp_] - piv[des]) / piv[cmp_]).values
                out.append(dict(estimator=est, k=k, metric=metric, design=des, vs=cmp_, **stat(rho, cl)))
o = pd.DataFrame(out)
print("\nE11 prior ablation: (comparator - design)/comparator (>0 favours the robust design)")
print(o[o.design.str.startswith("fc_w24_rob")].to_string(index=False))
print(o[o.design.str.startswith("fc_grow_rob") & (o.metric == "f1")].to_string(index=False))
same = st[st.design == "fc_w24_rob"].pivot_table(index=["wind", "K", "seed", "k"], columns="prior", values="sites", aggfunc="first")
print("\nfc_w24_rob: fraction of (scenario, k) with the same sites as the original 8-direction design")
print(same.apply(lambda c: (c == same["orig8"]).mean()).round(2).to_string())

# ---------------------------------------------------------------- E11 predicted vs realised
f["ratio_prior"], f["ratio_true"] = f.f1 / f.pred_prior, f.f1 / f.pred_true
g = f[f.prior.isin(["-", "orig8"])]
print("\nE11 realised / predicted 1 h RMSE (non-random designs): N(0, I)-prior prediction vs bias-aware with true a")
print(g.groupby("estimator")[["ratio_prior", "ratio_true"]].describe(percentiles=[.1, .5, .9]).T.round(3).to_string())
print("\nScenario dependence (veering vs mixed, same K/seed):")
print(pd.read_csv(here / "scenario-dependence.csv").to_string(index=False))
print("\nk = 2 brute force vs greedy+swap:")
print(pd.read_csv(here / "e11-bruteforce-k2.csv").assign(gap=lambda x: x.brute - x.greedy_swap).to_string(index=False))

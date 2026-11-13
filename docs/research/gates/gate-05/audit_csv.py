"""Gate-05 audit: read-only re-analysis of reports/quarto/tables/e05-duration.csv."""
import numpy as np, pandas as pd
from scipy.stats import binomtest, spearmanr
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
df = pd.read_csv("reports/quarto/tables/e05-duration.csv")
df["rel"] = df.src_rmse / df.true_rms
df["snr"] = df.s_max / (df.sigma_prereg * df.noise_scale) * 1e6  # top whitened singular value x a_ref
print("rows", len(df), "seeds", df.seed.nunique())
print("\n== sigma_prereg range by wind", df.groupby("wind").sigma_prereg.agg(["min", "max"]))

A = df[df.branch == "A"]; P = A[A.noise_scale == 1.0]
print("\n== rel error quantiles (branch A) by noise")
print(A.groupby("noise_scale").rel.describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))
for ns in (1.0, .1):
    a = A[A.noise_scale == ns]
    print(ns, "frac rel>1", (a.rel > 1).mean().round(3), ">10", (a.rel > 10).mean().round(3), ">1e3", (a.rel > 1e3).mean().round(3),
          "share of mean from top 1%", (np.sort(a.rel)[-len(a)//100:].sum() / a.rel.sum()).round(6))
print("\n== rel error vs practical rank at noise (A): median by prank/K")
A2 = A.assign(full=lambda d: d.prank_at_noise / d.K)
print(A2.groupby(["noise_scale", "K", "prank_at_noise"]).rel.agg(["count", "median", lambda x: (x > 1).mean(), "max"]))
print("\n== blow-up vs top whitened s*a_ref (A, noise 1)")
P2 = P.assign(snrbin=pd.cut(np.log10(P.snr + 1e-300), [-400, -6, -2, 0, 1, 2, 10]))
print(P2.groupby("snrbin", observed=True).rel.agg(["count", "median", "max"]))
print("cells with s_max*a_ref/sigma < 1e-6 (no signal at all):", (P.snr < 1e-6).mean().round(3))

# m=1 reading hour check
import sys; sys.path.insert(0, "src")
from pravah.temporal_basis import reading_hours
print("\n== reading_hours(167,W,m=1):", {W: reading_hours(167, W, 1).tolist() for W in (1, 6, 24, 168)})
print("reading_hours(167,168,2):", reading_hours(167, 168, 2).tolist(), "(167,24,4):", reading_hours(167, 24, 4).tolist())

# Claim 1: monotone in W. Fixed m (fair) vs best m.
print("\n== Claim 1: per (K,N), seed-mean practical rank & median rel err vs W at FIXED m=4 (W>=6) and m=1")
for m_ in (1, 4):
    g = P[P.m == m_].groupby(["wind", "K", "N", "W"]).agg(prank=("prank_aref1", "mean"), med_rel=("rel", "median")).unstack("W")
    print(f"m={m_}\n", g.round(2))
# per-seed monotonicity at fixed m=4: W 6 -> 24 -> 168
p4 = P[P.m == 4].pivot_table(index=["wind", "K", "N", "seed"], columns="W", values="prank_aref1")
print("m=4 per-seed: frac prank decreases 6->24:", ((p4[24] < p4[6]).mean()).round(3), " 24->168:", ((p4[168] < p4[24]).mean()).round(3),
      " frac increases 24->168:", ((p4[168] > p4[24]).mean()).round(3), " frac equal 24->168:", ((p4[168] == p4[24]).mean()).round(3))
e4 = P[P.m == 4].pivot_table(index=["wind", "K", "N", "seed"], columns="W", values="rel")
print("m=4 per-seed: frac rel error WORSE 24->168:", (e4[168] > e4[24]).mean().round(3), " 6->24:", (e4[24] > e4[6]).mean().round(3))
# best-m comparison: what m achieves the max?
bm = P.loc[P.groupby(["wind", "K", "N", "W", "seed"]).prank_aref1.idxmax()]
print("readings available per W:", P.groupby("W").m.unique().to_dict())

# Claim 2: K=4 N=4 W=168 vs N=8 snapshot, budget-matched alternative
print("\n== Claim 2 (K=4): compare designs per seed (noise 1), pooled winds")
def cell(N, m_, W, K=4, d=P):
    return d[(d.N == N) & (d.m == m_) & (d.W == W) & (d.K == K)].set_index(["wind", "seed"])
snap8 = cell(8, 1, 1)
for (N, m_, W) in [(4, 16, 168), (4, 2, 168), (4, 2, 24), (4, 1, 168), (8, 1, 1), (4, 1, 1), (8, 2, 168)]:
    c = cell(N, m_, W)
    d = c.prank_aref1 - snap8.prank_aref1
    print(f"N={N} m={m_} W={W} B={N*m_}: mean prank {c.prank_aref1.mean():.2f}, median rel {c.rel.median():.2f};"
          f" vs N=8 snapshot: wins {int((d>0).sum())} losses {int((d<0).sum())} ties {int((d==0).sum())}; "
          f"rel better {(c.rel < snap8.rel).sum()}/{len(c)}")

# Claim 3: matched B, more sensors vs more readings
print("\n== Claim 3 matched budget: median rel err and mean prank by (B, W, N, m), pooled winds, noise 1 and 0.1")
for ns in (1.0, .1):
    a = A[(A.noise_scale == ns) & A.B.isin([8, 16, 32])]
    print("noise", ns); print(a.groupby(["K", "B", "W", "N", "m"]).agg(prank=("prank_at_noise", "mean"), med_rel=("rel", "median")).unstack(["N", "m"]).round(2).to_string())
# paired seed test at B=16, W=168: N=8,m=2 vs N=2,m=8 and N=4 m=4
for K in (2, 4):
    for W in (24, 168):
        x = {N: cell(N, 16 // N, W, K) for N in (2, 4, 8)}
        for lo in (2, 4):
            d = x[8].rel - x[lo].rel
            print(f"K={K} W={W} B=16: N=8 vs N={lo}: rel better {int((d<0).sum())}/{len(d)}, sign p={binomtest(int((d<0).sum()), int((d!=0).sum())).pvalue:.2g}")

# Wind contrast at the seed level (correct unit)
print("\n== wind: seed-level paired difference changing-constant, averaged over (N,m,W) cells")
for br in "AB":
    for ns in (1.0, .1):
        d = df[(df.branch == br) & (df.noise_scale == ns)]
        pv = d.pivot_table(index=["K", "seed", "N", "m", "W"], columns="wind", values="prank_at_noise")
        diff = (pv.changing - pv.constant).groupby(["K", "seed"]).mean()
        for K, s in diff.groupby(level=0):
            print(br, ns, "K", K, "mean diff", s.mean().round(3), "pos/neg/0", int((s > 0).sum()), int((s < 0).sum()), int((s == 0).sum()),
                  "sign p", round(binomtest(int((s > 0).sum()), int((s != 0).sum())).pvalue, 4) if (s != 0).any() else None)
        pr = pv.reset_index(); pr["d"] = pr.changing - pr.constant
        print("   by W mean diff:", pr.groupby("W").d.mean().round(3).to_dict(), " cell-level frac pairs tied:", (pr.d == 0).mean().round(3))

# float rank > rows? / rank ceiling
print("\n== rank <= min(rows,p) violations:", int((df["rank"] > np.minimum(df.rows, df.p)).sum()))
# Branch B
Bb = df[df.branch == "B"]
print("\n== Branch B: median rel(const comp) by W, noise; and float rank/p")
print(Bb.assign(relc=Bb.src_rmse_const / Bb.true_rms).groupby(["noise_scale", "wind", "W"]).agg(rank_p=("rank_over_p", "mean"), prank=("prank_at_noise", "mean"), med_relc=("relc", "median")).round(3))
AK2 = A[A.K == 2].set_index(["wind", "seed", "N", "m", "W", "noise_scale"])
BB = Bb.set_index(["wind", "seed", "N", "m", "W", "noise_scale"])
r = (BB.src_rmse_const / AK2.src_rmse)
print("B const / A error, median by noise:", r.groupby(level="noise_scale").median().round(3).to_dict(),
      "frac B better:", (r < 1).groupby(level="noise_scale").mean().round(3).to_dict())
print("B rank at W=168, N=8, m=16:", Bb[(Bb.W == 168) & (Bb.N == 8) & (Bb.m == 16) & (Bb.noise_scale == 1)]["rank"].describe().round(2).to_dict())
# zero-signal scenarios
z = P[(P.N == 8) & (P.W == 168) & (P.m == 16)]
print("\n== N=8,W=168,m=16 scenarios with prank 0:", int((z.prank_aref1 == 0).sum()), "of", len(z))
print("== N=2 cells with snr<1e-6 by wind:", P[P.N == 2].groupby("wind").apply(lambda g: (g.snr < 1e-6).mean(), include_groups=False).round(3).to_dict())

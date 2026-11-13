"""Gate-06 re-analysis of reports/quarto/tables/e06-criteria.csv (no simulation). Output: audit_csv.out beside this file."""
import sys
from math import comb
from pathlib import Path
import numpy as np, pandas as pd

here = Path(__file__).resolve().parent
df = pd.read_csv(Path.cwd() / "reports/quarto/tables/e06-criteria.csv")
out = []
pr = lambda *a: out.append(" ".join(map(str, a)))
ok = df[df.status == "ok"]
# 1. spacing never binds on the small set: n_feasible == C(n_cand, k) for every non-diversity brute force
sm = ok[(ok.cand_set == "small") & ok.brute_n_feasible.notna() & (ok.design != "diversity")]
pr("small-set brute rows:", len(sm), "| n_feasible == C(n_cand,k) in", int((sm.brute_n_feasible == [comb(int(n), int(k)) for n, k in zip(sm.n_cand, sm.k)]).sum()))
pr("small n_cand range:", ok[ok.cand_set == "small"].n_cand.min(), ok[ok.cand_set == "small"].n_cand.max(),
   "| min spacing of small designs (m):", ok[ok.cand_set == "small"].min_spacing_m.min())
pr("full-set min spacing (m):", ok[ok.cand_set == "full"].min_spacing_m.min(), "(lattice 625 m, min_sep 1250 m)")
# 2. same candidate set / budget per scenario row group
g = df.groupby(["wind", "seed", "K", "cand_set"]).n_cand.nunique()
pr("n_cand unique per (scenario, cand_set):", g.max())
# 3. A-opt on the full set: aopt design A value vs dopt design A value
f = ok[ok.cand_set == "full"].pivot_table(index=["wind", "seed", "K", "k"], columns="design", values="aopt")
d = f.aopt - f.dopt
pr("full set, A(aopt design) - A(dopt design): >0:", int((d > 1e-9).sum()), "=0:", int((d.abs() <= 1e-9).sum()), "<0:", int((d < -1e-9).sum()),
   "| min", f"{d.min():.3g}", "max", f"{d.max():.3g}")
# 4. practical-rank cells: how many scenarios behind each prank entry
n = ok[ok.cand_set == "full"].groupby(["K", "k", "design"]).size().unstack()
pr("scenarios per prank cell (full set):\n" + n.to_string())
# 5. practical rank: source-informed vs blind, paired by scenario at K=4
p = ok[(ok.cand_set == "full")].pivot_table(index=["wind", "seed", "K", "k"], columns="design", values="practical_rank")
pr("paired prank source_near - dopt: mean", f"{(p.source_near - p.dopt).mean():.3g}", "| dopt>source_near", int((p.dopt > p.source_near).sum()),
   "=", int((p.dopt == p.source_near).sum()), "<", int((p.dopt < p.source_near).sum()))
# 6. field_rel_err_val is uninformative: fraction == 1
fe = ok[ok.cand_set == "full"].field_rel_err_val
pr("field_rel_err_val == 1 (to 1e-6) in", int((fe - 1).abs().lt(1e-6).sum()), "of", len(fe), "full-set rows")
# 7. podqr_value of designs computed with r_num (dust) modes: fraction -inf
pv = ok[ok.cand_set == "full"].podqr_value
pr("podqr_value -inf in", int(np.isneginf(pv).sum()), "of", len(pv), "full-set rows; range of finite", f"{pv[np.isfinite(pv)].min():.3g}..{pv[np.isfinite(pv)].max():.3g}")
# 8. diversity: picks all above noise?
dv = ok[ok.design == "diversity"]
pr("diversity rows with a sub-noise site:", int((dv.n_above_noise < dv.k).sum()), "of", len(dv))
txt = "\n".join(out); print(txt); (here / "audit_csv.out").write_text(txt + "\n")

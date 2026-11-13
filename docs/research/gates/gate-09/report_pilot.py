"""Gate-09 analysis of check_pilot.py output.  uv run python docs/research/gates/gate-09/report_pilot.py PILOT_CSV"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root / "src"))
from pravah.experiments import cross_objective as X  # noqa: E402

d = pd.read_csv(sys.argv[1])
key = ["wind", "K", "seed", "k", "design", "net", "arm"]
m = d.groupby(key + ["estimator"])[["forecast_rmse_h1", "cov95_active", "source_rmse", "nees"]].mean().reset_index()

# 1. reproduction of committed E09 rows
e9 = pd.read_csv(root / "reports/quarto/tables/e09-local-rows.csv")
for mine, theirs in [("map24_N0", "map24"), ("full_mm", "kf_aug"), ("full_mm", "map_full")]:
    j = m[m.estimator == mine].merge(e9[e9.estimator == theirs], on=key, suffixes=("", "_e9"))
    print(f"reproduce {theirs:8s} by {mine:8s}: n={len(j)}  max|d forecast_rmse_h1| {np.abs(j.forecast_rmse_h1 - j.forecast_rmse_h1_e9).max():.2e}"
          f"  max|d source_rmse| {np.abs(j.source_rmse - j.source_rmse_e9).max():.2e}"
          f"  max|d cov95_active| {np.abs(j.cov95_active - j.cov95_active_e9).max():.2e}")

# 2. prior vs memory attribution (scenario = unit; random = mean of its 2 networks)
s = m.groupby(["arm", "wind", "K", "seed", "k", "design", "estimator"]).forecast_rmse_h1.mean().unstack("estimator").reset_index()
print("\nrelative 1 h forecast-RMSE reduction vs map24_N0 (E07 frozen): median [bootstrap CI], wins/12")
print("arm   k design       | prior only (map24_mm) | memory only (full_N0) | both (full_mm = kf_aug) | memory given mm prior | prior given full memory")
for (arm, k, des), q in s.groupby(["arm", "k", "design"]):
    cells = []
    for num, den in [("map24_mm", "map24_N0"), ("full_N0", "map24_N0"), ("full_mm", "map24_N0"), ("full_mm", "map24_mm"), ("full_mm", "full_N0")]:
        r = ((q[den] - q[num]) / q[den]).to_numpy()
        lo, hi = X.boot_median_ci(r)
        cells.append(f"{np.median(r):+.2f} [{lo:+.2f},{hi:+.2f}] {(r > 0).sum():2d}")
    print(f"{arm:5s} {k} {des:12s} | " + " | ".join(cells))

# share of the log-RMSE gain: log(map24_N0/full_mm) = memory (at N0) + prior (at full memory)
w = s[s.arm == "white"]
tot = np.log(w.map24_N0 / w.full_mm)
mem = np.log(w.map24_N0 / w.full_N0)
print(f"\nwhite arm, all 96 (scenario, k, design): median share of log-RMSE gain from memory at N(0,I) prior "
      f"{np.median(mem / tot):.2f} (IQR {np.quantile(mem / tot, .25):.2f}-{np.quantile(mem / tot, .75):.2f}); "
      f"pooled sum share {mem.sum() / tot.sum():.2f}")

# 3. dopt vs source_near under each estimator
print("\nsource_near vs dopt, (dopt - sn)/dopt > 0 = source_near better; median, sn wins / 12")
for arm in ["white", "ar1"]:
    for k in (4, 8):
        q = s[(s.arm == arm) & (s.k == k)].set_index(["wind", "K", "seed", "design"])
        out = []
        for e in ["map24_N0", "map24_mm", "full_N0", "full_mm"]:
            p = q[e].unstack("design")
            r = (p.dopt - p.source_near) / p.dopt
            out.append(f"{e} {np.median(r):+.3f} ({(r > 0).sum()}/12)")
        print(arm, k, " | ".join(out))

# 4. calibration
print("\ncalibration (median over scenario x k x design rows): cov95_active (nominal 0.95), source NEES/K (1 if calibrated)")
print(m.groupby(["arm", "estimator"])[["cov95_active", "nees"]].median().round(3).to_string())
print(m.groupby(["arm", "estimator"])["nees"].quantile([.1, .9]).unstack().round(2).to_string())
a = d[(d.arm == "white")]
print("\nNEES/K pooled mean over draws/rows (white):", a.groupby("estimator").nees.mean().round(3).to_dict())

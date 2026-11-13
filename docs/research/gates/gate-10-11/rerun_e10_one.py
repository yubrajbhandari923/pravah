"""Re-run scripts/e10_joint.run_scenario (incl. all filter runs) for one scenario into a scratch cache and compare every
row with the committed e10-local-rows.csv. uv run python docs/research/gates/gate-10-11/rerun_e10_one.py CACHE_DIR"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(root / "src"), str(root / "scripts")]
import numpy as np, pandas as pd  # noqa: E402
import e10_joint as E10  # noqa: E402

E10.cache = Path(sys.argv[1])
rows = pd.DataFrame(E10.run_scenario(("mixed", 2, 2, 32))[0])
t = pd.read_csv(root / "reports/quarto/tables/e10-local-rows.csv")
t = t[(t.wind == "mixed") & (t.K == 2) & (t.seed == 2)]
k = ["k", "design", "estimator", "variant"]
m = rows.merge(t, on=k, suffixes=("", "_tab"))
num = [c for c in rows.columns if c not in k + ["wind", "K", "seed"] and c + "_tab" in m]
d = {c: float(np.nanmax(np.abs(m[c] - m[c + "_tab"]))) if m[c].notna().any() else 0.0 for c in num}
print(f"rows compared {len(m)} of {len(t)}; max abs diff per metric:", d)

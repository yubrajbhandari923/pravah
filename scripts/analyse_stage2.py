"""EXP-015 stage 2: apply the PRE-REGISTERED acceptance criteria to stability.csv / replicate_baseline.csv.

    python scripts/analyse_stage2.py --dir results/phase3-convergence

The rule, quoted from EXP-015 and docs/numerical-methodology.md section 5, is applied verbatim:

    Stable only if mesh-induced displacement and |regret| lie within the replicate-pair
    interquartile range AND decrease monotonically from 500 to 250 m at dt 100.

This script decides nothing else. It introduces no new threshold, and it reports per sensor budget and
per noise fraction separately because both are registered axes. `kind == "pair"` rows are the
replicate-pair distribution (the baseline); `vs_main` rows are a different comparison and are NOT the
registered baseline, so they are reported separately and never mixed in.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MONO = [(500.0, 100.0), (250.0, 100.0)]  # the registered monotonicity axis: 500 -> 250 at dt 100


def iqr(s: "pd.Series[float]") -> tuple[float, float, float]:
    return float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75))


def pick(df: pd.DataFrame, **eq: float) -> pd.DataFrame:
    """Rows where every named column equals its value (np.isclose on floats)."""
    m = np.ones(len(df), dtype=bool)
    for k, v in eq.items():
        m &= np.isclose(df[k].to_numpy(dtype=float), v)
    return df.loc[m]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/phase3-convergence")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    stab = pd.read_csv(d / "stability.csv")
    base = pd.read_csv(d / "replicate_baseline.csv")
    pairs = base[base["kind"] == "pair"]

    verdicts = []
    for f in sorted(stab["noise_fraction"].unique()):
        for b in sorted(stab["budget"].unique()):
            s = pick(stab, noise_fraction=float(f), budget=float(b))
            p = pick(pairs, noise_fraction=float(f), budget=float(b))
            if not len(p):
                print(f"  noise {f} budget {b}: NO replicate baseline rows — cannot judge"); continue
            dq1, dmed, dq3 = iqr(p["mean_displacement_m"].astype(float))
            rq1, rmed, rq3 = iqr(p["regret"].astype(float).abs())
            print(f"\n=== noise fraction {f}, budget {b} ===")
            print(f"  replicate-pair baseline (n={len(p)}): displacement IQR [{dq1:.0f}, {dq3:.0f}] m (median {dmed:.0f})")
            print(f"                                        |regret|     IQR [{rq1:.4g}, {rq3:.4g}] (median {rmed:.4g})")
            print(f"  {'config':14} {'jaccard':>8} {'mean_disp_m':>12} {'in IQR':>7} {'|regret|':>11} {'in IQR':>7} {'test_d':>9}")
            disp_by_mesh = {}
            for _, r in s.sort_values(by=["res_m", "dt_s"], ascending=False).iterrows():
                mesh = (float(r["res_m"]), float(r["dt_s"]))
                disp, reg = float(r["mean_displacement_m"]), abs(float(r["regret"]))
                is_ref = bool(np.isclose(reg, 0.0) and np.isclose(disp, 0.0))
                din = dq1 <= disp <= dq3
                rin = rq1 <= reg <= rq3
                disp_by_mesh[mesh] = (disp, reg)
                tag = "  (reference)" if is_ref else ""
                print(f"  {mesh[0]:.0f}m/dt{mesh[1]:.0f}{'':5} {float(r['jaccard']):8.3f} {disp:12.0f} "
                      f"{str(din):>7} {reg:11.4g} {str(rin):>7} {float(r['test_rmse_delta']):9.4g}{tag}")
            # the registered monotonicity check
            have = [m for m in MONO if m in disp_by_mesh]
            mono_d = mono_r = None
            if len(have) == len(MONO):
                ds = [disp_by_mesh[m][0] for m in MONO]
                rs = [disp_by_mesh[m][1] for m in MONO]
                mono_d = all(x >= y for x, y in zip(ds, ds[1:]))
                mono_r = all(x >= y for x, y in zip(rs, rs[1:]))
                print(f"  monotonic 500->250 at dt100: displacement {ds[0]:.0f} -> {ds[1]:.0f} m = {mono_d}; "
                      f"|regret| {rs[0]:.4g} -> {rs[1]:.4g} = {mono_r}")
            # the registered claim: coarsest config (500/100) judged against both conditions
            c = pick(s, res_m=500.0, dt_s=100.0)
            stable = None
            if len(c) and mono_d is not None:
                row = c.iloc[0]
                din = dq1 <= float(row["mean_displacement_m"]) <= dq3
                rin = rq1 <= abs(float(row["regret"])) <= rq3
                stable = bool(din and rin and mono_d and mono_r)
                print(f"  --> 500 m / dt 100 ADMISSIBLE per the registered rule: {stable} "
                      f"(disp in IQR {din}, |regret| in IQR {rin}, monotonic disp {mono_d} regret {mono_r})")
            verdicts.append({"noise_fraction": float(f), "budget": int(b),
                             "baseline_n_pairs": int(len(p)),
                             "disp_iqr": [dq1, dq3], "disp_median": dmed,
                             "regret_iqr": [rq1, rq3], "regret_median": rmed,
                             "mono_displacement": mono_d, "mono_regret": mono_r,
                             "stable_500_dt100": stable,
                             "tie_fraction": float(c.iloc[0]["tie_fraction"]) if len(c) else None,
                             "eps_set_size": int(c.iloc[0]["eps_set_size"]) if len(c) else None})

    print("\n=== summary of the registered verdict, per budget and noise fraction ===")
    for v in verdicts:
        print(f"  noise {v['noise_fraction']} budget {v['budget']}: 500m/dt100 admissible = {v['stable_500_dt100']}"
              f"  (tie_fraction {v['tie_fraction']}, eps_set {v['eps_set_size']})")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(verdicts, indent=1) + "\n")
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

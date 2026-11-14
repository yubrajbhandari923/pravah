"""Gate-12-17, E15 + E16 at 32^2 (caches of other worktrees are read only; everything new goes to OUT).
E15: (1) CVaR vs the Rockafellar-Uryasev minimisation, brute force; (2) the closed-form robust loss vs Monte Carlo of the
     MAP error; (3) declared ensembles contain no 0 deg / 180 deg block hour; (4) re-run cluster (K, seed) stage 2 from
     symlinked fields/blocks and compare with e15-local-rows.csv.
E16: (5) bg_hi reuse: drivers identical to the nominal cell, and generate_scenario(nominal truth + 15) equals the
     truth that the bg_hi config would produce (same anomaly + 20) to rounding; (6) re-run nominal and bg_hi for (K, seed)
     and compare with e16-local-rows.csv.
Usage: OMP_NUM_THREADS=1 uv run python e15_e16_check32.py OUT E15_CACHE E16_CACHE [K seed]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

root = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(root / "scripts"), str(root / "src")]
import e15_robust_placement as E15  # noqa: E402
import e16_generalization as E16  # noqa: E402
from pravah.experiments import forecast_design as FD  # noqa: E402
from pravah.experiments import robust_design as RD  # noqa: E402

T = root / "reports/quarto/tables"


def link_dir(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for f in Path(src).iterdir():
        if f.suffix in (".npy", ".json") and not (dst / f.name).exists():
            os.symlink(f, dst / f.name)


def main():
    out, c15, c16 = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    K, seed = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (2, 0)
    rng = np.random.default_rng(0)
    # (1) CVaR
    worst = 0.0
    for _ in range(300):
        n = int(rng.integers(2, 50))
        L = rng.gamma(2, 1, n)
        a = float(rng.choice([0.5, 0.8, 0.9, 0.95]))
        ru = minimize_scalar(lambda t: t + np.maximum(L - t, 0).mean() / (1 - a), bounds=(L.min() - 1, L.max() + 1),
                             method="bounded", options=dict(xatol=1e-12)).fun
        worst = max(worst, abs(RD.cvar(L, a) - ru) / ru)
    print(f"(1) CVaR vs Rockafellar-Uryasev min_t t + E[(L-t)+]/(1-a), 300 random cases: max rel diff {worst:.1e}")
    # (2) loss formula: E||Q^1/2 (a_hat - a)||^2 with a ~ m2 (second moment), e ~ N(0, I)
    Kk, n = 3, 40
    A = rng.normal(size=(n, Kk)) * 0.7
    M = A.T @ A
    Sig = np.linalg.inv(np.eye(Kk) + M)
    Q = np.diag([1.0, .5, .2])
    m2 = FD.loguniform_m2(0.5, 5.0, Kk)
    an = Q @ Sig @ (m2 + M) @ Sig
    ch = np.linalg.cholesky(m2)
    N = 200000
    a = rng.normal(size=(N, Kk)) @ ch.T
    y = a @ A.T + rng.normal(size=(N, n))
    e = y @ A @ Sig - a
    mc = np.einsum("ni,ij,nj->n", e, Q, e).mean()
    print(f"(2) robust loss tr(Q Sig (m2 + M) Sig) {np.trace(an):.5f} vs Monte Carlo {mc:.5f} (N = {N})")
    # (3) directions in the ensembles
    for e_, labs in RD.ENSEMBLES.items():
        th = np.concatenate([np.degrees(np.arctan2(-w[:, 0], -w[:, 1])) % 360 for w in (RD.block_winds(*b) for b in labs)])
        print(f"(3) {e_}: block hours from 0 deg: {int(np.isclose(th % 360, 0, atol=.5).sum())}, from 180 deg: "
              f"{int(np.isclose(th, 180, atol=.5).sum())}; distinct FROM directions {sorted(set(np.round(th).astype(int) % 360))}")
    # (4) E15 cluster rerun
    o15 = out / "e15"
    link_dir(c15 / "fields", o15 / "fields")
    link_dir(c15 / "blocks", o15 / "blocks")
    rows, _ = E15.run_cluster(("pilot", K, seed, str(o15), None))
    mine = pd.DataFrame(rows)
    c = pd.read_csv(T / "e15-local-rows.csv")
    c = c[(c.K == K) & (c.seed == seed)]
    key = ["wind", "k", "design", "net", "spec", "estimator"]
    m = mine.merge(c, on=key, suffixes=("", "_c"))
    cols = ["forecast_rmse_h1", "forecast_rmse_h6", "source_rmse", "ins_cvar_w24_noSN6", "ins_worst_full_train3"]
    d = {x: float(np.nanmax(np.abs(m[x] - m[x + "_c"]) / np.maximum(np.abs(m[x + "_c"]), 1e-12))) for x in cols}
    print(f"(4) E15 K{K} s{seed}: rows mine {len(mine)} committed {len(c)} matched {len(m)}; sites identical "
          f"{bool((m.sites.astype(str) == m.sites_c.astype(str)).all())}; max rel diff {d}")
    # (5) bg_hi reuse
    cn, cb = E16.cfg_of("local", "nominal"), E16.cfg_of("local", "bg_hi")
    dn, db = E16.B.scenario_drivers(cn, seed, K), E16.B.scenario_drivers(cb, seed, K)
    same = all(np.array_equal(np.asarray(x, dtype=object if isinstance(x, (list, dict)) else None),
                              np.asarray(y, dtype=object if isinstance(y, (list, dict)) else None)) for x, y in zip(dn, db))
    print(f"(5) bg_hi drivers (sources, strengths, xy, winds, regimes, hours) identical to nominal: {same}")
    o16 = out / "e16"
    link_dir(c16 / "fields", o16 / "fields")
    fl = E16.E7.scenario_fields(E16.scale_name("local", "nominal"), "mixed", K, seed)
    tn = np.load(o16 / "fields" / f"{fl[0][0]}.npy")
    anom = tn - cn.sources.background_ug_m3
    print("    nominal truth - 5 == the solved anomaly; bg_hi truth by reuse (truth + 15) vs anomaly + 20: max diff %.1e"
          % np.abs((tn + 15.0) - (anom + 20.0)).max())
    # (6) E16 rerun of nominal and bg_hi
    c = pd.read_csv(T / "e16-local-rows.csv")
    for cell in ("nominal", "bg_hi"):
        rows, _ = E16.run_scenario(("local", cell, K, seed, str(o16)))
        mi = pd.DataFrame(rows)
        cc = c[(c.cell == cell) & (c.K == K) & (c.seed == seed)]
        m = mi.merge(cc, on=["k", "strategy", "estimator"], suffixes=("", "_c"))
        d = {x: float(np.nanmax(np.abs(m[x] - m[x + "_c"]) / np.maximum(np.abs(m[x + "_c"]), 1e-12)))
             for x in ("forecast_rmse", "analysis_rmse", "source_rmse", "sigma")}
        print(f"(6) E16 {cell} K{K} s{seed}: rows mine {len(mi)} committed {len(cc)} matched {len(m)}; max rel diff {d}")


if __name__ == "__main__":
    main()

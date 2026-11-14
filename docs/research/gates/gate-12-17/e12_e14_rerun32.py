"""Gate-12-17, E12 + E14: re-run one 32^2 pilot cluster (K, seed) and compare with the committed rows.
Truth/oracle/sweep-spec perturbed fields are read (read-only) from another worktree's E15 cache, which holds the same
content-hashed E07/E12 solves; the three pilot-only specs (spd0.8, lag2, ar15) are solved into OUT.
Also: an independent re-derivation of the E12 wind perturbations (polar convention) and of the persistence baseline.
Usage: OMP_NUM_THREADS=1 uv run python e12_e14_rerun32.py OUT READONLY_FIELDS [K seed]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(root / "scripts"), str(root / "src")]
import e12_wind_mismatch as E12  # noqa: E402
import e14_reliability as E14  # noqa: E402
from pravah import uncertainty as U  # noqa: E402


def main():
    out, ro = Path(sys.argv[1]), Path(sys.argv[2])
    K, seed = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (2, 0)
    (out / "fields").mkdir(parents=True, exist_ok=True)
    names = E12.field_names("pilot", K, seed, U.PILOT_SPECS)
    for v in names.values():
        for n, sp in v:
            if not (ro / f"{n}.npy").exists():
                E12.run_field(((n, sp), str(out)))
    T = root / "reports/quarto/tables"
    # ---- E12
    rows, _, tm = E12.run_scenario(("pilot", K, seed, str(out), str(ro)))
    mine = pd.DataFrame(rows)
    c = pd.read_csv(T / "e12-local-rows.csv")
    c = c[(c.K == K) & (c.seed == seed)]
    key = ["spec", "k", "design", "net", "estimator"]
    m = mine.merge(c, on=key, suffixes=("", "_c"))
    cols = ["forecast_rmse_h1", "forecast_rmse_h6", "analysis_rmse", "source_rmse", "sensor_fit", "innov_ac1", "cover95_a",
            "rel_bias_a"]
    d = {x: float(np.nanmax(np.abs(m[x] - m[x + "_c"]) / np.maximum(np.abs(m[x + "_c"]), 1e-12))) for x in cols}
    print(f"E12 K{K} s{seed}: rows mine {len(mine)} committed {len(c)} matched {len(m)}; max rel diff {d}")
    print("E12 linearity_max_abs (upwind):", tm["linearity_max_abs"])
    # independent wind-perturbation check: dir +30 rotates the FROM direction clockwise by 30 deg, keeps the speed
    w = E12.B.scenario_drivers(E12.E7.make_cfg("pilot", "mixed"), seed, K)[3]
    wm = U.perturb_winds(w, U.WindSpec("dir", 30), seed)
    s0, s1 = np.hypot(*w.T), np.hypot(*wm.T)
    th0 = np.degrees(np.arctan2(-w[:, 0], -w[:, 1])) % 360
    th1 = np.degrees(np.arctan2(-wm[:, 0], -wm[:, 1])) % 360
    print("dir30: speed max diff %.1e; FROM-direction shift (deg) min/max %.6f/%.6f" % (
        np.abs(s1 - s0).max(), ((th1 - th0) % 360).min(), ((th1 - th0) % 360).max()))
    wl = U.perturb_winds(w, U.WindSpec("lag", 6), seed)
    print("lag6: model wind at h equals true wind at h-6 for h>=6:", bool(np.array_equal(wl[6:], w[:-6])),
          "(the model lags the truth: a 6 h late forecast wind)")
    # ---- E14 (needs every field in one directory)
    agg = out / "agg" / "fields"
    agg.mkdir(parents=True, exist_ok=True)
    for v in names.values():
        for n, _ in v:
            src = (ro if (ro / f"{n}.npy").exists() else out / "fields") / f"{n}.npy"
            if not (agg / f"{n}.npy").exists():
                os.symlink(src, agg / f"{n}.npy")
    u = pd.DataFrame(E14.run_cluster((K, seed, str(out / "agg"))))
    cu = pd.read_csv(T / "e14-local-units.csv.gz")
    cu = cu[(cu.K == K) & (cu.seed == seed)]
    key = ["arm", "k", "design", "net", "estimator", "draw", "block"]
    m = u.merge(cu, on=key, suffixes=("", "_c"))
    cols = list(E14.RL.DIAGS) + ["rmse_model", "rmse_persist", "cov90", "crps"]
    d = {x: float(np.nanmax(np.abs(m[x] - m[x + "_c"]) / np.maximum(np.abs(m[x + "_c"]), 1e-6))) for x in cols}
    lab = bool((m.worse_than_persistence == m.worse_than_persistence_c).all() and (m.coverage_failure == m.coverage_failure_c).all())
    print(f"E14 K{K} s{seed}: units mine {len(u)} committed {len(cu)} matched {len(m)}; labels identical {lab}")
    print("  max rel diff (committed CSV stores 6 significant digits):", {k: f"{v:.1e}" for k, v in d.items()})


if __name__ == "__main__":
    main()

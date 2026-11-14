"""Gate-12-17, E08: re-run the 32^2 pilot scenario (K, seed) for the reference and the map_full variant, compare with the
committed rows, and check the all-past-data estimator independently:
  (1) per issue time t, a direct MAP solve from readings at hours <= t only (no cumulative sums) vs estimate_series;
  (2) poisoning: every reading after the last issue hour + 1 set to 1e6 must not change any estimate (no future data);
  (3) how many readings the map_full estimator has at the first test issue time.
Usage: OMP_NUM_THREADS=1 uv run python e08_rerun32.py OUT [K seed]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(root / "scripts"), str(root / "src")]
import e08_sufficiency as E8  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402
from pravah.experiments import sufficiency as S  # noqa: E402


def main():
    out = Path(sys.argv[1])
    K, seed = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (2, 0)
    (out / "fields").mkdir(parents=True, exist_ok=True)
    for t in E8.E7.scenario_fields("e08pilot", "mixed", K, seed):
        E8.run_field((t, str(out)))
    T = root / "reports/quarto/tables"
    for full, ref in ((False, "e08-local-rows.csv"), (True, "e08-local-mapfull-rows.csv")):
        rows, _, _ = E8.run_scenario(("e08pilot", K, seed, str(out), 0, full))
        mine = pd.DataFrame(rows)
        c = pd.read_csv(T / ref)
        c = c[(c.K == K) & (c.seed == seed)]
        key = ["arm", "schedule", "k", "strategy"]
        m = mine.merge(c, on=key, suffixes=("", "_c"))
        num = [x for x in ("forecast_rmse", "analysis_rmse", "source_rmse", "lam_min_med", "tti_med") if x in m]
        d = {x: float(np.nanmax(np.abs(m[x] - m[x + "_c"]) / np.maximum(np.abs(m[x + "_c"]), 1e-300))) for x in num}
        print(f"{ref}: rows mine {len(mine)} committed {len(c)} matched {len(m)}; max rel diff {d}")
    # independent estimator checks on the dopt k=2 design, reference noise draw 0
    sc, F, G3, _ = E8.E7.load_scenario("e08pilot", "mixed", K, seed, str(out))
    sigma = X.site_sigma(sc, False)
    p = X.problem(X.design_input(sc, G3, sigma))
    Sx, _ = X.make_design("dopt", p, 2, len(sc.split_hours("train")), sc.cfg.seeds.streams(seed)["placement"])
    a_ref, cbg = sc.cfg.sources.reference_rate_ug_s, sc.cfg.sources.background_ug_m3
    issue, n_h = X.issue_times(sc), len(sc.truth.winds)
    A = a_ref * G3[Sx] / sigma[Sx][:, None, None]
    z = (X.readings(sc, sigma, "white", 0, n_h) - cbg)[Sx] / sigma[Sx][:, None]
    u = X.estimate_series(A, z, issue, None)
    err = 0.0
    for i, t in enumerate(issue[::7]):
        At = A[:, :t + 1].reshape(-1, K)
        zt = z[:, :t + 1].reshape(-1)
        direct = np.linalg.solve(np.eye(K) + At.T @ At, At.T @ zt)
        err = max(err, float(np.abs(direct - u[::7][i]).max() / np.abs(direct).max()))
    zp = z.copy()
    zp[:, issue.max() + 2:] = 1e6
    up = X.estimate_series(A, zp, issue, None)
    print(f"direct per-issue MAP vs estimate_series (map_full): max rel diff {err:.2e}")
    print(f"future readings poisoned (hours > {issue.max() + 1}): max |change| {float(np.abs(up - u).max()):.2e}")
    print(f"calendar {n_h} h; training hours {len(sc.split_hours('train'))}; first issue hour {issue.min()} -> map_full uses "
          f"{issue.min() + 1} hourly readings per sensor vs 24 for map24")
    print("true a / a_ref:", sc.truth.strengths_ug_s / a_ref, " map_full estimate at first/last issue:", u[0], u[-1])


if __name__ == "__main__":
    main()

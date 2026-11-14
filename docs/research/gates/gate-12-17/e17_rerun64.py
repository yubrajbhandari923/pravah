"""Gate-12-17: re-run one E17 64^2 sweep scenario (coolA, K=2, seed 0) locally from raw ERA5, compare every row with the
cluster's e17-sweep-rows.csv, and compute the transport floors the sweep did not record:
  - forecast RMSE with the TRUE strengths under each model wind (oracle -> VanLeer superposition floor; point0/dir30 ->
    pure wrong-wind transport error, the part no estimator or design can remove).
Usage: PRAVAH_E17_RAW=<main>/.cache/e17/raw OMP_NUM_THREADS=1 uv run python e17_rerun64.py OUT SWEEP_ROWS_CSV [K seed]
"""
import importlib.util, sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import numpy as np, pandas as pd

root = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(root / "scripts")]
import e17_regional as E  # noqa: E402  (importable by name so spawned workers can unpickle run_field)
X = E.X


def main():
    out, ref = Path(sys.argv[1]), sys.argv[2]
    K, seed = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (2, 0)
    job = ("sweep", "coolA", K, seed)
    todo = [((n, t), str(out)) for n, t in E.field_tasks(*job) if not (out / "fields" / f"{n}.npy").exists()]
    print(len(todo), "solves", flush=True)
    with ProcessPoolExecutor(4, mp_context=get_context("spawn")) as ex:
        for r in ex.map(E.run_field, todo):
            print(r, flush=True)
    rows, tm = E.evaluate(job + (str(out),))
    mine = pd.DataFrame(rows)
    cl = pd.read_csv(ref)
    cl = cl[(cl.case == "coolA") & (cl.K == K) & (cl.seed == seed)].copy()
    key = ["spec", "estimator", "k", "strategy", "net"]
    mine["net"], cl["net"] = mine.net.fillna(0), cl.net.fillna(0)
    m = mine.merge(cl, on=key, suffixes=("", "_c"))
    print("rows mine/cluster/matched", len(mine), len(cl), len(m))
    print("sites identical:", (m.sites.fillna("").astype(str) == m.sites_c.fillna("").astype(str)).all())
    for c in ("forecast_rmse", "analysis_rmse", "popw_rmse", "source_rmse"):
        print(c, "max rel diff", float(np.nanmax(np.abs(m[c] - m[c + "_c"]) / np.abs(m[c + "_c"]))))
    print("superposition_rel_max", tm["superposition_rel_max"])
    # transport floors: true strengths through each model wind's unit responses
    ft = E.field_tasks(*job)
    cfg, drv = E.drivers(*job)
    sc = E.B.generate_scenario(cfg, seed, K, truth_field=np.load(out / "fields" / f"{ft[0][0]}.npy"), drivers=drv)
    issue, a_ref, cbg = X.issue_times(sc), cfg.sources.reference_rate_ug_s, cfg.sources.background_ug_m3
    wpop = E.masks(cfg.model.grid)[0]
    wpop = wpop / wpop.sum()
    u_true = np.tile(sc.truth.strengths_ug_s / a_ref, (len(issue), 1))
    res = []
    for sp in E.specs_for("coolA"):
        F = np.stack([np.load(out / "fields" / f"{n}.npy") for n, t in ft[1:] if t[6] == sp.name])
        s = X.score(u_true, F, sc.truth.field, issue, sc.truth.strengths_ug_s, a_ref, cbg, wpop)
        res.append(dict(spec=sp.name, true_strength_forecast_rmse=s["forecast_rmse"]))
    pers = sc.truth.field[issue] - sc.truth.field[issue + 1]
    bg = X.score(np.zeros_like(u_true), F, sc.truth.field, issue, sc.truth.strengths_ug_s, a_ref, cbg, wpop)["forecast_rmse"]
    fl = pd.DataFrame(res).assign(persistence=float(np.sqrt(np.mean(pers ** 2))), background_only=bg)
    print(fl.to_string(index=False))
    fl.to_csv(Path(__file__).with_name(f"e17_floor64_K{K}_s{seed}.csv"), index=False)
    print(m.groupby(["spec", "estimator", "k", "strategy"]).forecast_rmse.mean().unstack("strategy").round(4).to_string())


if __name__ == "__main__":
    main()

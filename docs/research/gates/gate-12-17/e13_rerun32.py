"""Gate-12-17, E13: re-run one 32^2 pilot cluster (K, seed) (nominal fields read-only from the E15 cache; the 2K+... truth-side
solves into OUT), compare with the committed rows, and check the common-mode-bias mechanism independently:
the readings of `bgoff1` and `common1` are identical at every sensor (so every truth-free diagnostic is identical), while
the scored truths differ by exactly the 1 ug/m3 field offset.
Usage: OMP_NUM_THREADS=1 uv run python e13_rerun32.py OUT READONLY_FIELDS [K seed]
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(root / "scripts"), str(root / "src")]
import e13_model_mismatch as E13  # noqa: E402
from pravah import model_mismatch as M  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402


def main():
    out, ro = Path(sys.argv[1]), Path(sys.argv[2])
    K, seed = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (2, 0)
    (out / "fields").mkdir(parents=True, exist_ok=True)
    names = E13.field_names("pilot", K, seed, M.SPECS)
    todo = [((n, sp), str(out)) for v in names.values() for n, sp in v if not (ro / f"{n}.npy").exists()]
    with ProcessPoolExecutor(int(os.environ.get("GATE_WORKERS", "1")), mp_context=get_context("fork")) as ex:
        list(ex.map(E13.run_field, todo))
    rows, tm = E13.run_scenario(("pilot", K, seed, str(out), str(ro)))
    mine = pd.DataFrame(rows)
    c = pd.read_csv(root / "reports/quarto/tables/e13-local-rows.csv")
    c = c[(c.K == K) & (c.seed == seed)]
    key = ["spec", "k", "design", "net", "estimator", "horizon"]
    m = mine.fillna({"horizon": -1}).merge(c.fillna({"horizon": -1}), on=key, suffixes=("", "_c"))
    cols = ["forecast_rmse_h1", "forecast_rmse_h6", "field_rmse", "source_rmse", "source_bias", "cover95_a", "sensor_fit",
            "innov_ac1", "forecast_rmse"]
    d = {x: float(np.nanmax(np.abs(m[x] - m[x + "_c"]) / np.maximum(np.abs(m[x + "_c"]), 1e-12))) for x in cols if x in m}
    print(f"E13 K{K} s{seed}: rows mine {len(mine)} committed {len(c)} matched {len(m)}; max rel diff {d}")
    # mechanism check
    dirs = [out / "fields", ro]
    sc = E13.B.generate_scenario(E13.E7.make_cfg("pilot", "mixed"), seed, K,
                                 truth_field=np.load(E13._find(names["truth"][0][0], dirs)))
    tfields = {sp[4:]: np.load(E13._find(n, dirs)) for n, sp in names["e13"]}
    Gs = {k: X.site_responses(sc, v[None])[:, :, 0] for k, v in tfields.items()}
    sigma = X.site_sigma(sc, False)
    tb, yb = M.world(sc, "bgoff1", tfields, Gs, sigma, 0)
    tc, yc = M.world(sc, "common1", tfields, Gs, sigma, 0)
    to, _ = M.world(sc, "oracle", tfields, Gs, sigma, 0)
    print("readings bgoff1 vs common1: max |diff| %.1e ug/m3 (rounding only; identical up to the order of additions)"
          % np.abs(yb - yc).max())
    print("truth(bgoff1) - truth(oracle): min/max %.3f/%.3f ; truth(common1) - truth(oracle): max |.| %.1e" % (
        (tb - to).min(), (tb - to).max(), np.abs(tc - to).max()))


if __name__ == "__main__":
    main()

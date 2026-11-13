"""Gate-09: does the 64² growing-window reversal survive the VanLeer superposition (model-error) floor?

  OMP_NUM_THREADS=1 uv run python docs/research/gates/gate-09/check_floor.py FIELD_DIR OUT_CSV

Per scenario: floor = 1 h forecast RMSE with the TRUE strengths (c_bg + sum a F vs the joint VanLeer truth). Then dopt and
source_near (k = 4/8/16, committed sites) under the growing-window N(0, I) MAP, scored (i) as E07 (VanLeer truth and
readings) and (ii) on a LINEAR truth (truth := c_bg + sum a F; readings = linear site truth + the same white noise draws).
"""
import multiprocessing as mp, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root / "src")); sys.path.insert(0, str(root / "scripts")); sys.path.insert(0, str(Path(__file__).parent))
argv = sys.argv[1:]
sys.argv = sys.argv[:1]
import e07_cross_objective as E  # noqa: E402
from check_sweep import ROWS, map_cum  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402
from pravah.observations import bilinear_sample  # noqa: E402


def scen(job):
    fdir, wind, K, seed = job
    rows = pd.read_csv(ROWS)
    sub = rows[(rows.wind == wind) & (rows.K == K) & (rows.seed == seed) & (rows.arm == "white")
               & rows.strategy.isin(["dopt", "source_near"])]
    sc, F, G3, _ = E.load_scenario("sweep", wind, K, seed, None, fdir)
    a_ref, c_bg, a = sc.cfg.sources.reference_rate_ug_s, sc.cfg.sources.background_ug_m3, sc.truth.strengths_ug_s
    tf, n_h, issue = sc.truth.field, len(sc.truth.winds), X.issue_times(sc)
    sigma = X.site_sigma(sc, False)
    lin = c_bg + np.einsum("k,khxy->hxy", a, F)
    floor = float(np.sqrt(np.mean((lin[issue + 1] - tf[issue + 1]) ** 2)))
    lin_sites = np.stack([bilinear_sample(f, sc.site_xy, tuple(sc.cfg.model.domain_m)) for f in lin], 1)
    Y = [X.readings(sc, sigma, "white", r, n_h) for r in range(5)]
    out = []
    for _, r0 in sub.iterrows():
        S = np.array(r0.sites.split(), int)
        A = a_ref * G3[S] / sigma[S][:, None, None]
        v, vl = [], []
        for y in Y:
            for yy, truth, acc in ((y, tf, v), (lin_sites + (y - sc.truth.site_truth), lin, vl)):
                ah, _ = map_cum(A, (yy[S] - c_bg) / sigma[S][:, None], issue, None, np.zeros(K), 1.0)
                e = c_bg + np.einsum("tk,ktxy->txy", ah * a_ref, F[:, issue + 1]) - truth[issue + 1]
                acc.append(np.sqrt(np.mean(e ** 2)))
        out.append(dict(wind=wind, K=K, seed=seed, k=int(r0.k), strategy=r0.strategy, floor=floor,
                        full_N0=float(np.mean(v)), full_N0_linear=float(np.mean(vl))))
    print("done", wind, K, seed, flush=True)
    return out


if __name__ == "__main__":
    fdir, csv = argv
    jobs = [(fdir, wd, K, s) for wd in ("veering", "mixed") for K in (2, 4) for s in range(10)]
    with ProcessPoolExecutor(6, mp_context=mp.get_context("fork")) as ex:
        d = pd.DataFrame([r for x in ex.map(scen, jobs) for r in x])
    d.to_csv(csv, index=False)
    print(f"floor (forecast RMSE with true strengths) over 40 scenarios: median {d.floor.median():.5f}, max {d.floor.max():.5f}")
    for col in ("full_N0", "full_N0_linear"):
        p = d.pivot_table(index=["wind", "K", "seed", "k"], columns="strategy", values=col)
        r = (p.dopt - p.source_near) / p.dopt
        lay = r.groupby(level=["K", "seed"]).mean()
        print(f"{col}: median sn_vs_dopt {r.median():+.3f}, source_near wins {(r > 0).sum()}/{len(r)}, "
              f"layouts favouring source_near {(lay > 0).sum()}/20; by k: "
              + ", ".join(f"k={k} {np.median(r.xs(k, level='k')):+.3f} ({(r.xs(k, level='k') > 0).sum()}/40)" for k in (4, 8, 16)))
    p = d.pivot_table(index=["wind", "K", "seed", "k"], columns="strategy", values="full_N0")
    print(f"dopt RMSE / floor: median {np.median(p.dopt / d.groupby(['wind', 'K', 'seed', 'k']).floor.first()):.2f}; "
          f"source_near RMSE / floor: median {np.median(p.source_near / d.groupby(['wind', 'K', 'seed', 'k']).floor.first()):.2f}")

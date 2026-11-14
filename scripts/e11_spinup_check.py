"""E11 64² design-prior spin-up check (gate-10/11 condition 1): is a 12 h steady spin-up enough for a per-direction 24 h
block, compared with the 48 h used by the truth?

  OMP_NUM_THREADS=1 uv run python scripts/e11_spinup_check.py [--workers 5] [--K 4 --seed 0 --src 0]

Under a steady wind the spin-up replays the block's own wind, so the block after s hours of spin-up is hours [s, s + 24)
of ONE steady solve from zero. One 72 h solve per direction therefore gives the 0, 12, 24 and 48 h spin-up blocks.
Compares them with the 48 h block on the full field, the block-mean field weight Q and the candidate-site responses.
Writes reports/quarto/tables/e11-local-v2-spinup.csv.
"""
import argparse, multiprocessing as mp, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from pravah import benchmarks as B  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402
from pravah.experiments import forecast_design as FD  # noqa: E402
from pravah.observations import bilinear_sample  # noqa: E402

sys.path.insert(0, str(root / "scripts"))
import e11_forecast_design as E11  # noqa: E402

SPINS, HOURS = (0, 12, 24, 48), 72


def solve(job):
    K, seed, src, d = job
    cfg = E11.E7.make_cfg("sweep", "mixed")
    cfg.model.spinup_h = 0
    xy = B.scenario_drivers(cfg, seed, K)[2][src]
    w0, c0 = time.time(), time.process_time()
    f = X.source_field(cfg, xy, np.repeat(FD.design_winds()[d * FD.BLOCK_H][None], HOURS, 0), HOURS)
    return d, f, time.time() - w0, time.process_time() - c0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", type=int, default=0)
    a = ap.parse_args()
    m = E11.E7.make_cfg("sweep", "mixed").model
    dom, (nx, ny), L = tuple(m.domain_m), tuple(m.grid), 16  # candidate sites: generate_scenario's 16 x 16 lattice
    ii, jj = np.divmod(np.arange(L * L), L)
    site_xy = (np.stack([nx // L // 2 + nx // L * ii, ny // L // 2 + ny // L * jj], 1) + .5) * (dom[0] / nx, dom[1] / ny)
    rows = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as ex:
        for d, f, wall, cpu in ex.map(solve, [(a.K, a.seed, a.src, d) for d in range(FD.N_DIRS)]):
            ref = f[48:72]
            for s in SPINS:
                b = f[s:s + FD.BLOCK_H]
                r = dict(K=a.K, seed=a.seed, src=a.src, direction_deg=d * 360 / FD.N_DIRS, spinup_h=s,
                         rel_l2_field=float(np.linalg.norm(b - ref) / np.linalg.norm(ref)),
                         rel_Q=float(abs((b ** 2).mean() - (ref ** 2).mean()) / (ref ** 2).mean()),
                         wall_s_72h=wall, cpu_s_72h=cpu)
                g, g0 = (np.stack([bilinear_sample(x, site_xy, dom) for x in y]) for y in (b, ref))
                r["rel_l2_sites"] = float(np.linalg.norm(g - g0) / np.linalg.norm(g0))
                rows.append(r)
            print(f"dir {d}: {wall:.0f}s wall for {HOURS} h", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(root / "reports/quarto/tables/e11-local-v2-spinup.csv", index=False)
    print(df.groupby("spinup_h")[["rel_l2_field", "rel_Q", "rel_l2_sites"]].agg(["median", "max"]).to_string())


if __name__ == "__main__":
    main()

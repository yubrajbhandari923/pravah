"""CPU scaling of the dominant DYN workload: independent truth episodes (one FiPy run + inverse fits each).

    python scripts/dyn_bench.py --workers 1 2 4 8 16 --episodes 32 [--out DIR]

Reports episodes/s, speed-up, efficiency and peak RSS per worker count; also times one fit so a GPU case can be judged from
numbers (a 30x30 ADMM fit is milliseconds: far below GPU launch/transfer overheads).  Synthetic; no scientific content.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def one(seed: int) -> float:
    import numpy as np

    from pravah import dynamic as dyn
    from pravah.studies.toy import dynamic_study as ds
    rng = np.random.default_rng(seed)
    t = time.perf_counter()
    h = ds.make_history(ds.FAMILIES[seed % len(ds.FAMILIES)], rng)
    ds.truth_field("steady", h)
    A = rng.random((72, 30))
    dyn.fit_history(A, A @ rng.random(30), np.full(72, 0.1), 5, 6, "M2", alpha=1e-4, beta=10.0)
    return time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--out", default=os.environ.get("RESEARCH_RUN_DIR", "results/dyn_bench"))
    a = ap.parse_args()
    for k in THREADS:
        os.environ[k] = "1"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, t1 = [], None
    for w in a.workers:
        t = time.perf_counter()
        with ProcessPoolExecutor(w, mp_context=mp.get_context("spawn")) as ex:
            per = list(ex.map(one, range(a.episodes)))
        wall = time.perf_counter() - t
        t1 = t1 or wall
        rows.append({"workers": w, "episodes": a.episodes, "wall_s": wall, "episodes_per_s": a.episodes / wall,
                     "speedup": t1 / wall, "efficiency": t1 / wall / w, "mean_episode_s": sum(per) / len(per),
                     "peak_rss_gb_children": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1e9 if os.uname().sysname == "Darwin" else 1e6)})
        print(rows[-1], flush=True)
    (out / "dyn_bench.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

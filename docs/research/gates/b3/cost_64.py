"""Serial wall time per simulated hour at 64^2 (idle machine, OMP_NUM_THREADS=1, best of 3), study BC.

Run: OMP_NUM_THREADS=1 uv run python docs/research/gates/b3/cost_64.py  -> vanleer_split/cost_64.csv
"""
import csv, platform, sys, time
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[3] / "src"))
from pravah.forward import simulate  # noqa: E402

rows = []
for scheme, dt in (("upwind", 300.0), ("upwind", 30.0), ("vanleer", 30.0), ("vanleer", 15.0)):
    best = min(
        (lambda t0: (simulate(nx=64, ny=64, Lx=1e4, Ly=1e4, dt=dt, n_steps=round(3600 / dt), u=(5.0, 0.0), K=50.0,
                              lam=1 / 86400, c_bg=5.0, c0=np.full((64, 64), 5.0), bc="study", scheme=scheme),
                     time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(3))
    rows.append(dict(scheme=scheme, n=64, dt=dt, wall_s_per_sim_hour=best, wall_h_per_1488h_run=best * 1488 / 3600,
                     cpu=platform.processor() or platform.machine()))
    print(rows[-1])
with open(here / "vanleer_split" / "cost_64.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0]))
    w.writeheader()
    w.writerows(rows)

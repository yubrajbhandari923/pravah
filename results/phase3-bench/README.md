# resolution_cost.csv (scripts/bench_resolution.py)
One production-like Belauri forward solve (configs/geodesign.json, design draw d0: 3 groups + bg basis, 1 wind segment,
T=28800 s, standard domain, FiPy LinearLUSolver tol 1e-12, exponential convection), fresh subprocess per row, 8 GB laptop.
mode=forward: one solve; mode=basis: compute_responses = n_params+1 = 5 solves = one source-response basis (the cache unit).
wall_s excludes imports/region load; peak_rss_gb is the child's ru_maxrss. Nothing hit the 10 min / 5 GB limits.
Cost scales ~ cells (x4 per halving) at fixed dt and ~ 1/dt at fixed cells: 500 m dt/2 costs 1.8x.
geo125/ holds a throw-away re-preparation of the Belauri grid with an extra 125 m level (same raw data, offline);
data/belauri has only 500 and 250 m.  The 125 m dt=100 s has Courant up to 2 at 2.5 m/s (exponential scheme is implicit/stable, accuracy untested).
Rerun: .venv/bin/python scripts/bench_resolution.py [--mode basis]   (skips rows already present)

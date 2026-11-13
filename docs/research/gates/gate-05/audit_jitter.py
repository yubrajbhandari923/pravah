"""Gate-05 audit spot check: (a) reproduce two CSV scenarios from HEAD code; (b) 'constant' wind with the per-block
direction offset removed (speed factor kept, rng stream unchanged) to see whether the W=168 rank gain under 'constant'
comes from block-to-block direction drift. Branch A, K=4, preregistered noise only."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.argv = [sys.argv[0], "A"]
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from pravah import benchmarks as Bm

_orig = Bm.make_winds
def no_dir_jitter(regimes, n_hours, seed):
    """Same rng draws as make_winds, but the N(0,10 deg) direction offset is dropped for steady_westerly."""
    if not set(regimes) <= {"steady_westerly", "veering"}:
        return _orig(regimes, n_hours, seed)  # held-out split winds: untouched
    rng = np.random.default_rng(seed); out = []
    for b in range(-(-n_hours // Bm.BLOCK_H)):
        reg = regimes[b % len(regimes)]
        assert reg in ("steady_westerly", "veering"), reg
        h = np.arange(Bm.BLOCK_H)
        sp = np.full(Bm.BLOCK_H, 3.0)
        th = np.full(Bm.BLOCK_H, 270.0) if reg == "steady_westerly" else 270.0 + 45.0 * np.minimum(h, 12) / 12
        sp = np.clip(sp * rng.uniform(.8, 1.2), .2, 5.0)
        off = rng.normal(0, 10)
        th = np.radians(th + (0.0 if reg == "steady_westerly" else off))
        out.append(np.stack([-sp * np.sin(th), -sp * np.cos(th)], 1))
    return np.concatenate(out)[:n_hours], [regimes[b % len(regimes)] for b in range(-(-n_hours // Bm.BLOCK_H))]

if os.environ.get("AUDIT_PATCH") == "1":
    Bm.make_winds = no_dir_jitter
import e05_duration as D  # noqa: E402

def job(seed):
    rows, _ = D.run_scenario(("constant", seed, 4))
    return rows

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    seeds = [int(s) for s in os.environ.get("AUDIT_SEEDS", "0,1,2,3,4,5,6,7").split(",")]
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = sum(ex.map(job, seeds), [])
    tag = "nojitter" if os.environ.get("AUDIT_PATCH") == "1" else "repro"
    pd.DataFrame(rows).to_csv(f"{os.path.dirname(__file__)}/audit_{tag}.csv", index=False)
    print("wrote", tag, len(rows))

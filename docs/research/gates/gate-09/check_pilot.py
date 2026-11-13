"""Gate-09 checks on the E09 32² pilot (regenerated locally; same code path as scripts/e09_filter.py for the world,
designs and noise draws). Estimator algebra is re-implemented here with an explicit per-issue-time loop.

  OMP_NUM_THREADS=1 uv run python docs/research/gates/gate-09/check_pilot.py OUT_CSV [--workers 6]

Per scenario x k x design x arm x draw, four static MAP estimators of the strengths (a_ref units), forecast by F a:
  map24_N0   24 h window, prior N(0, I)              (E07 frozen map24)
  map24_mm   24 h window, prior N(abar 1, s^2 I)     (prior effect only)
  full_N0    all past hours, prior N(0, I)           (memory effect only)
  full_mm    all past hours, prior N(abar 1, s^2 I)  (= kf_aug / map_full of E09)
Scores: 1 h forecast RMSE (all cells, E07 issue times), 95 % coverage over active cells (sd >= 0.01 sigma), source
NEES/K = (a_hat - a)^T C^-1 (a_hat - a)/K averaged over issue times. Truth is read only to score.
"""
import multiprocessing as mp, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root / "src")); sys.path.insert(0, str(root / "scripts"))
argv = sys.argv[1:]
sys.argv = sys.argv[:1]
import e09_filter as E9  # noqa: E402  (config, constants, designs list only)
from pravah import benchmarks as B  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402

ABAR, S2 = E9.ABAR, E9.S2
EST = dict(map24_N0=(24, 0.0, 1.0), map24_mm=(24, ABAR, S2), full_N0=(None, 0.0, 1.0), full_mm=(None, ABAR, S2))


def my_map(A, z, issue, W, mu, v):
    """A (k, T, K) whitened, a_ref-scaled; z (k, T). Explicit loop. Returns a_hat (n, K), C (n, K, K)."""
    K = A.shape[2]
    a, C = np.empty((len(issue), K)), np.empty((len(issue), K, K))
    for n, t in enumerate(issue):
        lo = 0 if W is None else t - W + 1
        G = A[:, lo:t + 1].reshape(-1, K)
        Prec = np.eye(K) / v + G.T @ G
        C[n] = np.linalg.inv(Prec)
        a[n] = C[n] @ (G.T @ z[:, lo:t + 1].ravel() + mu / v)
    return a, C


def scen(job):
    wind, K, seed = job
    cfg = E9.make_cfg(wind)
    sc = B.generate_scenario(cfg, seed, K)
    F, G3 = X.responses(sc)
    a_ref, cbg = cfg.sources.reference_rate_ug_s, cfg.sources.background_ug_m3
    T = len(sc.truth.winds)
    N = F.shape[2] * F.shape[3]
    Fr = a_ref * F.reshape(K, T, N)
    tf = sc.truth.field.reshape(T, N) - cbg
    a_true = sc.truth.strengths_ug_s / a_ref
    issue = X.issue_times(sc)
    sigma = X.site_sigma(sc, False)
    sig = float(sigma[0])
    p = X.problem(X.design_input(sc, G3, sigma))
    seedp, T_train = sc.cfg.seeds.streams(seed)["placement"], len(sc.split_hours("train"))
    Yall = {arm: [X.readings(sc, sigma, arm, r, T) for r in range(E9.R)] for arm in E9.ARMS}
    tgt = tf[issue + 1]
    rows = []
    for k in E9.KSENS:
        for name, net in E9.DESIGNS:
            S = X.make_design(name, p, k, T_train, seedp, net)[0]
            A = G3[S] * a_ref / sig
            for arm in E9.ARMS:
                for r, y in enumerate(Yall[arm]):
                    z = (y[S] - cbg) / sig
                    for e, (W, mu, v) in EST.items():
                        ah, C = my_map(A, z, issue, W, np.full(K, mu), v)
                        fc = np.einsum("ktn,tk->tn", Fr[:, issue + 1], ah)
                        sd = np.sqrt(np.maximum(np.einsum("ktn,tkl,ltn->tn", Fr[:, issue + 1], C, Fr[:, issue + 1]), 0))
                        act = sd >= E9.SD_ACTIVE * sig
                        zz = np.abs(fc - tgt)[act] / sd[act]
                        d = ah - a_true
                        nees = np.einsum("tk,tkl,tl->t", d, np.linalg.inv(C), d) / K
                        rows.append(dict(wind=wind, K=K, seed=seed, k=k, design=name, net=net, arm=arm, draw=r, estimator=e,
                                         forecast_rmse_h1=float(np.sqrt(np.mean((fc - tgt) ** 2))),
                                         cov95_active=float(np.mean(zz <= norm.ppf(.975))),
                                         source_rmse=float(np.sqrt(np.mean(d ** 2))), nees=float(nees.mean()),
                                         nees_test_last=float(nees[-1]), sites=" ".join(map(str, S))))
    print("done", job, flush=True)
    return rows


if __name__ == "__main__":
    w = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 6
    jobs = [(wd, K, s) for K in E9.KS for wd in E9.WINDS for s in E9.SEEDS]
    with ProcessPoolExecutor(min(w, 6), mp_context=mp.get_context("fork")) as ex:
        res = [r for x in ex.map(scen, jobs) for r in x]
    pd.DataFrame(res).to_csv(argv[0], index=False)

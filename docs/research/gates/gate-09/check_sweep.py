"""Gate-09 check 3/4 on the cached E07 64² sweep fields (job 204894, code dd2c51c; copied read-only from the cluster).

  OMP_NUM_THREADS=1 uv run python docs/research/gates/gate-09/check_sweep.py run FIELD_DIR OUT_CSV [--workers 6]
  uv run python docs/research/gates/gate-09/check_sweep.py report OUT_CSV

`run`: per scenario, the committed white-arm designs (sites read from e07-sweep-rows.csv: random x10, dopt, source_near;
k = 4/8/16) and the sweep's own R = 5 paired white noise draws (X.readings). Four static MAP estimators of the strengths
on the cached unit responses (superposition, as E07):
  map24_N0  E07 frozen (computed through X.estimate_series + X.score, must equal the committed forecast_rmse)
  map24_mm  24 h window, moment-matched prior N(abar, s^2 I)
  full_N0   growing window (all hours 0..t), prior N(0, I)
  full_mm   growing window, moment-matched prior (= E09 kf_aug / map_full)
Also for the growing-window estimators: source NEES/K and 95 % forecast coverage on active cells (sd >= 0.01 sigma).
`report`: paired contrasts source_near vs dopt and dopt vs mean-of-10-random per estimator.
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
import e07_cross_objective as E  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402

LO, HI = 0.5, 5.0
ABAR = (HI - LO) / np.log(HI / LO)
S2 = (HI ** 2 - LO ** 2) / (2 * np.log(HI / LO)) - ABAR ** 2
ROWS = root / "reports/quarto/tables/e07-sweep-rows.csv"


def map_cum(A, z, issue, W, mu, v):
    """Cumulative-sum MAP (cross-checked against an explicit loop on the first scenario). A (k,T,K), z (k,T)."""
    K = A.shape[2]
    cP = np.concatenate([np.zeros((1, K, K)), np.cumsum(np.einsum("shk,shl->hkl", A, A), 0)])
    cb = np.concatenate([np.zeros((1, K)), np.cumsum(np.einsum("shk,sh->hk", A, z), 0)])
    lo = np.zeros_like(issue) if W is None else issue + 1 - W
    Nt = np.eye(K) / v + cP[issue + 1] - cP[lo]
    bt = mu / v + cb[issue + 1] - cb[lo]
    return np.linalg.solve(Nt, bt[..., None])[..., 0], np.linalg.inv(Nt)


def scen(job):
    fdir, wind, K, seed = job
    rows = pd.read_csv(ROWS)
    sub = rows[(rows.wind == wind) & (rows.K == K) & (rows.seed == seed) & (rows.arm == "white")
               & rows.strategy.isin(["random", "dopt", "source_near"])]
    sc, F, G3, _ = E.load_scenario("sweep", wind, K, seed, None, fdir)
    a_ref, c_bg, a_true = sc.cfg.sources.reference_rate_ug_s, sc.cfg.sources.background_ug_m3, sc.truth.strengths_ug_s
    tf, n_h = sc.truth.field, len(sc.truth.winds)
    issue = X.issue_times(sc)
    w = X.pop_weight(tuple(sc.cfg.model.grid), tuple(sc.cfg.model.domain_m))
    sigma = X.site_sigma(sc, False)
    sig = float(sigma[0])
    Y = [X.readings(sc, sigma, "white", r, n_h) for r in range(5)]
    Fi = F[:, issue + 1].reshape(K, len(issue), -1)            # (K, n_issue, N) forecast targets' responses
    tgt = (tf[issue + 1] - c_bg).reshape(len(issue), -1)
    out = []
    for _, r0 in sub.iterrows():
        S = np.array(r0.sites.split(), int)
        A = a_ref * G3[S] / sigma[S][:, None, None]
        res = {e: [] for e in ("map24_N0", "map24_mm", "full_N0", "full_mm")}
        ext = {e: [] for e in ("full_N0", "full_mm", "map24_N0")}
        for y in Y:
            z = (y[S] - c_bg) / sigma[S][:, None]
            u = X.estimate_series(A, z, issue)                   # E07 frozen code path
            res["map24_N0"].append(X.score(u, F, tf, issue, a_true, a_ref, c_bg, w)["forecast_rmse"])
            for e, (W, mu, v) in dict(map24_mm=(24, ABAR, S2), full_N0=(None, 0.0, 1.0), full_mm=(None, ABAR, S2),
                                      map24_N0=(24, 0.0, 1.0)).items():
                ah, C = map_cum(A, z, issue, W, np.full(K, mu), v)
                if e == "map24_N0":
                    assert np.abs(ah - u).max() < 1e-9
                elif e == "full_mm" and not out:  # explicit-loop cross-check at a few issue times
                    for n in (0, len(issue) // 2, len(issue) - 1):
                        G = A[:, :issue[n] + 1].reshape(-1, K)
                        ref = np.linalg.solve(np.eye(K) / S2 + G.T @ G, G.T @ z[:, :issue[n] + 1].ravel() + ABAR / S2)
                        assert np.abs(ref - ah[n]).max() < 1e-8 * max(1, np.abs(ref).max()), (ref, ah[n])
                    res[e].append(X.score(ah, F, tf, issue, a_true, a_ref, c_bg, w)["forecast_rmse"])
                else:
                    res[e].append(X.score(ah, F, tf, issue, a_true, a_ref, c_bg, w)["forecast_rmse"])
                if e in ext:
                    fc = np.einsum("ktn,tk->tn", Fi, ah * a_ref)
                    sd = np.sqrt(np.maximum(np.einsum("ktn,tkl,ltn->tn", Fi, C, Fi), 0)) * a_ref
                    act = sd >= 0.01 * sig
                    d = ah - a_true / a_ref
                    ext[e].append((float(np.mean(np.abs(fc - tgt)[act] / sd[act] <= norm.ppf(.975))), float(act.mean()),
                                   float(np.mean(np.einsum("tk,tkl,tl->t", d, np.linalg.inv(C), d)) / K)))
        base = dict(wind=wind, K=K, seed=seed, k=int(r0.k), strategy=r0.strategy, net=int(r0.net) if r0.strategy == "random" else 0,
                    committed=float(r0.forecast_rmse))
        for e, v in res.items():
            m = dict(**base, estimator=e, forecast_rmse=float(np.mean(v)))
            if e in ext:
                m.update(cov95_active=np.mean([x[0] for x in ext[e]]), frac_active=np.mean([x[1] for x in ext[e]]),
                         nees_k=np.mean([x[2] for x in ext[e]]))
            out.append(m)
    print("done", wind, K, seed, flush=True)
    return out


def report(csv):
    from scipy.stats import binom
    d = pd.read_csv(csv)
    m = d[d.estimator == "map24_N0"]
    print(f"reproduction: max |map24_N0 - committed forecast_rmse| = {np.abs(m.forecast_rmse - m.committed).max():.3e} "
          f"over {len(m)} rows")
    g = d.groupby(["wind", "K", "seed", "k", "strategy", "estimator"]).forecast_rmse.mean().unstack("strategy").reset_index()
    g["sn_vs_dopt"] = (g.dopt - g.source_near) / g.dopt          # > 0: source_near better (E07/gate-07 convention)
    g["dopt_vs_rand"] = (g.random - g.dopt) / g.random          # > 0: dopt better than mean of 10 random

    def dfci(x):
        return X.df_median_ci(x)

    print("\nestimator  k   n   median sn_vs_dopt [DF 95% CI]  sn wins | median dopt_vs_rand [DF CI] dopt wins | median RMSE dopt / sn / rand")
    for e in ["map24_N0", "map24_mm", "full_N0", "full_mm"]:
        for k in ["all", 4, 8, 16]:
            q = g[g.estimator == e] if k == "all" else g[(g.estimator == e) & (g.k == k)]
            a, b = q.sn_vs_dopt.to_numpy(), q.dopt_vs_rand.to_numpy()
            ca, cb = dfci(a), dfci(b)
            print(f"{e:9s} {k!s:3s} {len(q):3d}  {np.median(a):+.3f} [{ca[0]:+.3f}, {ca[1]:+.3f}]  {(a > 0).sum():3d}/{len(a)} | "
                  f"{np.median(b):+.3f} [{cb[0]:+.3f}, {cb[1]:+.3f}] {(b > 0).sum():3d}/{len(b)} | "
                  f"{q.dopt.median():.4f} / {q.source_near.median():.4f} / {q.random.median():.4f}")
    print("\nlayout level (20 layouts = (K, seed), both winds and all k averaged): median of layout means [DF CI], layouts favouring")
    for e in ["map24_N0", "map24_mm", "full_N0", "full_mm"]:
        L = g[g.estimator == e].groupby(["K", "seed"])[["sn_vs_dopt", "dopt_vs_rand"]].mean()
        for c in L:
            ci = dfci(L[c])
            print(f"{e:9s} {c:12s} {L[c].median():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  {(L[c] > 0).sum()}/20")
        Lk = g[(g.estimator == e) & (g.k == 8)].groupby(["K", "seed"]).sn_vs_dopt.mean()
        print(f"{e:9s} sn_vs_dopt k=8 only: layouts favouring source_near {(Lk > 0).sum()}/20")
    print("\nby wind x K (k = 8), median sn_vs_dopt, sn wins / 10")
    for e in ["map24_N0", "full_N0", "full_mm"]:
        q = g[(g.estimator == e) & (g.k == 8)]
        print(e, {f"{w}K{K}": f"{x.sn_vs_dopt.median():+.3f} ({(x.sn_vs_dopt > 0).sum()}/10)" for (w, K), x in q.groupby(["wind", "K"])})
    print("\nmemory vs prior attribution (median over all 1440 (scenario, k, design) rows: relative RMSE reduction vs map24_N0)")
    p = d.pivot_table(index=["wind", "K", "seed", "k", "strategy", "net"], columns="estimator", values="forecast_rmse")
    for s in ["random", "dopt", "source_near"]:
        q = p.xs(s, level="strategy")
        print(s, {e: f"{np.median((q.map24_N0 - q[e]) / q.map24_N0):+.3f}" for e in ["map24_mm", "full_N0", "full_mm"]})
    print("\ncalibration of the growing-window estimators (median over rows; 95 % nominal; NEES/K = 1 if calibrated)")
    print(d[d.estimator.isin(["map24_N0", "full_N0", "full_mm"])].groupby(["estimator", "strategy"])[["cov95_active", "frac_active", "nees_k"]]
          .median().round(3).to_string())
    print(d[d.estimator.isin(["map24_N0", "full_N0", "full_mm"])].groupby(["estimator"])[["cov95_active", "nees_k"]]
          .quantile([.1, .5, .9]).round(3).to_string())


if __name__ == "__main__":
    if argv[0] == "report":
        report(argv[1])
    else:
        fdir, out = argv[1], argv[2]
        w = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 6
        jobs = [(fdir, wd, K, s) for wd in ("veering", "mixed") for K in (2, 4) for s in range(10)]
        if "--only" in argv:
            jobs = jobs[:int(argv[argv.index("--only") + 1])]
        with ProcessPoolExecutor(min(w, 6), mp_context=mp.get_context("fork")) as ex:
            res = [r for x in ex.map(scen, jobs) for r in x]
        pd.DataFrame(res).to_csv(out, index=False)

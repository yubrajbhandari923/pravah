"""Gate-10/11 independent checks (E10 joint assimilation, E11 forecast placement), 32² pilot world, local only.

  OMP_NUM_THREADS=1 uv run python docs/research/gates/gate-10-11/check.py [--workers 5] [--cache DIR]
  ... --summarise        # tables only, from the per-scenario cache

Per scenario (same world/seeds as scripts/e10_joint.py and scripts/e11_forecast_design.py; E09/E07 make_cfg differ only
in experiment_id, checked: drivers and seed streams identical):
 E10: rebuild truth variants and readings exactly as e10_joint.run_scenario (no filter runs), reproduce map_full / map24
      rows, and add memory competitors the pilot did not have: (a) MAP window chosen per data column on validation
      innovations (same criterion as kf_qsel) over {6,12,24,72,all}; (b) exponential-forgetting MAP, lambda in LAMS, chosen
      the same way; (c) oracle best window on the test (upper bound for the MAP family).
 E11: reproduce dopt / fc_w24_rob rows, then redo the robust designs with design-wind priors that leave out the test
      directions (S = 180 deg, block 4; N = 0 deg, block 0, the 'front' of stagnation_then_front), a non-test direction
      control (NE), all from independent per-direction block solves; also the bias-aware predicted RMSE with TRUE a
      (diagnostic for the predicted-vs-realised gap) and a k = 2 brute-force check of greedy+swap.
"""
import argparse, importlib.util, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

root = Path(__file__).resolve().parents[4]
here = Path(__file__).resolve().parent
sys.path[:0] = [str(root / "src"), str(root / "scripts")]
import e09_filter as E9  # noqa: E402
import e10_joint as E10  # noqa: E402
from pravah import benchmarks as B  # noqa: E402
from pravah import filter as KF  # noqa: E402
from pravah import placement as P  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402
from pravah.experiments import forecast_design as FD  # noqa: E402

WINS = {"map6": 6, "map12": 12, "map24": 24, "map72": 72, "map_full": None}
LAMS = (0.995, 0.98, 0.95, 0.9, 0.8)
PRIORS = {"all8_indep": [0, 1, 2, 3, 4, 5, 6, 7], "noS": [0, 1, 2, 3, 5, 6, 7], "noN": [1, 2, 3, 4, 5, 6, 7],
          "noSN": [1, 2, 3, 5, 6, 7], "noNE": [0, 2, 3, 4, 5, 6, 7]}


def ew_map(A, z, hours, lam):
    """MAP with prior N(0, I) and exponentially forgotten data: N_t = I + sum_s lam^(t-s) A_s'A_s; at each t in hours."""
    K = A.shape[2]
    I = np.einsum("shk,shl->hkl", A, A)
    b = np.einsum("shk,shc->hkc", A, z)
    want, out, C = set(int(h) for h in hours), {}, {}
    Nn, bn = np.zeros((K, K)), np.zeros(b.shape[1:])
    for t in range(max(want) + 1):
        Nn, bn = lam * Nn + I[t], lam * bn + b[t]
        if t in want:
            M = np.eye(K) + Nn
            out[t], C[t] = np.linalg.solve(M, bn), np.linalg.inv(M)
    return np.stack([out[int(h)] for h in hours]), np.stack([C[int(h)] for h in hours])


def run(job):
    wind, K, seed, cache = job
    f = Path(cache) / f"{wind}_K{K}_s{seed}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    w0 = time.time()
    cfg = E9.make_cfg(wind, (32, 32))
    sc = B.generate_scenario(cfg, seed, K)
    F, G3 = X.responses(sc)
    a_ref, cbg = cfg.sources.reference_rate_ug_s, cfg.sources.background_ug_m3
    grid, dom = tuple(cfg.model.grid), tuple(cfg.model.domain_m)
    T, N = len(sc.truth.winds), grid[0] * grid[1]
    Fr = a_ref * F.reshape(K, T, N)
    winds = [tuple(w) for w in sc.truth.winds]
    issue, val = X.issue_times(sc), sc.split_hours("validation")
    sigma = X.site_sigma(sc, False)
    sig = float(sigma[0])
    key = dict(wind=wind, K=K, seed=seed)
    meta = dict(**key, test_winds=sc.truth.winds[sc.split_hours("test")].copy(), a_true=sc.truth.strengths_ug_s / a_ref)
    # ------------------------------------------------ E10 (as e10_joint.run_scenario up to Y; no filters)
    ops = E9.Ops(cfg, sc.truth.source_xy)
    Bh = [None] + [Fr[:, h].T - ops.A(winds[h]) @ Fr[:, h - 1].T for h in range(1, T)]
    av = E10.strengths(sc.truth.strengths_ug_s / a_ref, T, seed, K)
    xt = {}
    for v, a in av.items():
        x = np.empty((T, N))
        x[0] = Fr[:, 0].T @ a[0]
        for h in range(1, T):
            x[h] = ops.A(winds[h]) @ x[h - 1] + Bh[h] @ a[h - 1]
        xt[v] = x
    xt["const"] = sc.truth.field.reshape(T, N) - cbg
    del ops, Bh
    R, V = E10.R, E10.VARIANTS
    E = [X.readings(sc, sigma, "white", r, T) - np.asarray(sc.truth.site_truth)[:, :T] for r in range(R)]
    p = X.problem(X.design_input(sc, G3, sigma))
    seedp, T_train = sc.cfg.seeds.streams(seed)["placement"], len(sc.split_hours("train"))
    e10 = []
    for k in E10.KSENS:
        for name, net in E10.DESIGNS:
            S = X.make_design(name, p, k, T_train, seedp, net)[0]
            H = KF.observation_matrix(sc.site_xy[S], grid, dom)
            Y = np.stack([xt[v] @ H.T + E[r][S].T for v in V for r in range(R)], 2)
            Aw = G3[S] * a_ref / sigma[S][:, None, None]
            z = Y.transpose(1, 0, 2) / sig
            arms = {}
            for nm, win in WINS.items():
                arms[nm] = (E9.map_series(Aw, z, issue, np.zeros(K), 1.0, win)[0],
                            E9.map_series(Aw, z, val - 1, np.zeros(K), 1.0, win)[0])
            for lam in LAMS:
                arms[f"ew{lam}"] = (ew_map(Aw, z, issue, lam)[0], ew_map(Aw, z, val - 1, lam)[0])
            inn = {nm: np.mean([((Y[h] - H @ Fr[:, h].T @ av_[i]) ** 2).mean(0) for i, h in enumerate(val)], 0)
                   for nm, (_, av_) in arms.items()}
            tgt = Fr[:, issue + 1]                                          # (K, n_issue, N)
            fc1 = {nm: np.einsum("ktn,tkc->tnc", tgt, a) for nm, (a, _) in arms.items()}
            for vi, v in enumerate(V):
                cols = range(vi * R, (vi + 1) * R)
                tf, at = xt[v][issue + 1], av[v][issue]
                rm = {nm: [np.sqrt(np.mean((fc1[nm][:, :, c] - tf) ** 2)) for c in cols] for nm in arms}
                sr = {nm: [np.sqrt(np.mean((arms[nm][0][..., c] - at) ** 2)) for c in cols] for nm in arms}
                row = dict(**key, k=k, design=name, variant=v)
                for nm in arms:
                    row[f"f1_{nm}"], row[f"src_{nm}"] = np.mean(rm[nm]), np.mean(sr[nm])
                for lab, pool in (("wsel", list(WINS)), ("ewsel", [f"ew{l}" for l in LAMS] + ["map_full"]),
                                  ("anysel", list(arms))):
                    pick = [pool[int(np.argmin([inn[nm][c] for nm in pool]))] for c in cols]
                    row[f"f1_{lab}"] = np.mean([rm[nm][j] for j, nm in enumerate(pick)])
                    row[f"src_{lab}"] = np.mean([sr[nm][j] for j, nm in enumerate(pick)])
                    row[f"pick_{lab}"] = " ".join(pick)
                row["f1_oracle_map"] = min(np.mean(rm[nm]) for nm in arms)   # test-chosen, upper bound
                e10.append(row)
    # ------------------------------------------------ E11
    dw = FD.design_winds()
    Frob8 = np.stack([X.source_field(cfg, xy, dw, len(dw)) for xy in sc.truth.source_xy])
    blocks = np.stack([np.stack([X.source_field(cfg, xy, dw[d * 24:(d + 1) * 24], 24) for d in range(8)])
                       for xy in sc.truth.source_xy])                     # (K, 8, 24, nx, ny), independent blocks
    di = X.design_input(sc, G3, sigma)
    designs = {}
    obj8 = FD.Objective(p, FD.forecast_input(di, F, X.site_responses(sc, Frob8), Frob8))
    for k in (2, 4, 8):
        for name in ("dopt", "source_near"):
            designs[(k, name, "-")] = X.make_design(name, p, k, T_train, seedp, 0)[0]
        for name in FD.FC:
            designs[(k, name, "orig8")] = FD.make_fc_design(name, p, obj8, k)[0]
    for pr, dirs in PRIORS.items():
        Fb = blocks[:, dirs].reshape(K, len(dirs) * 24, *grid)
        ob = FD.Objective(p, FD.forecast_input(di, F, X.site_responses(sc, Fb), Fb))
        for k in (2, 4, 8):
            for name in ("fc_w24_rob", "fc_grow_rob"):
                designs[(k, name, pr)] = FD.make_fc_design(name, p, ob, k)[0]
    # brute force k = 2 (greedy + swap vs exhaustive) on the original objectives
    bf = []
    for name in FD.FC:
        v = obj8.value(name)
        best, Sb, nf = P.brute_force(p, 2, v)
        bf.append(dict(**key, objective=name, greedy_swap=v(p, designs[(2, name, "orig8")]), brute=best, n_feasible=nf,
                       same_set=sorted(Sb) == sorted(designs[(2, name, "orig8")])))
    tf = xt["const"]
    a_true = sc.truth.strengths_ug_s / a_ref
    last = int(sc.split_hours("test")[-1])
    te0, regs = int(sc.split_hours("test")[0]), sc.truth.wind_regimes["test"]
    Z = np.stack([X.readings(sc, sigma, "white", r, T) for r in range(5)], 2)
    Qall = FD.field_weights(F, a_ref)
    e11 = []
    for (k, name, pr), S in designs.items():
        A = a_ref * G3[S] / sigma[S][:, None, None]
        z = (Z[S] - cbg) / sig
        Cc = FD._cum(A).sum(0)
        for est, win in (("map24", 24), ("map_grow", None)):
            u = FD.estimate(A, z, issue, win)
            r = dict(**key, k=k, design=name, prior=pr, estimator=est, sites=" ".join(map(str, S)))
            r["f1"] = np.mean([FD.score(u[..., j], Fr, tf, issue, a_true, last, (1,))["forecast_rmse_h1"] for j in range(5)])
            sub = np.array([t for t in issue if regs[(t + 1 - te0) // 48] == "steady_southerly"])
            us = u[np.searchsorted(issue, sub)]
            r["f1_south"] = np.mean([FD.score(us[..., j], Fr, tf, sub, a_true, last, (1,))["forecast_rmse_h1"]
                                     for j in range(5)])
            r["pred_prior"] = np.sqrt(FD.expected_mse(Cc, Qall, issue, (1,), win, last))
            # bias-aware expected MSE with the TRUE a: a'Sig Q Sig a + tr(Q Sig M Sig) (perfect linear model)
            t = issue[issue + 1 <= last]
            lo = np.zeros_like(t) if win is None else t + 1 - win
            M = Cc[t + 1] - Cc[lo]
            Sg = np.linalg.inv(np.eye(K) + M)
            Qt = Qall[t + 1]
            bias = np.einsum("k,tkl,tlm,tmn,n->t", a_true, Sg, Qt, Sg, a_true)
            var = np.einsum("tkl,tlm,tmn,tnk->t", Qt, Sg, M, Sg)
            r["pred_true"] = float(np.sqrt(np.mean(bias + var)))
            e11.append(r)
    res = dict(meta=meta, e10=e10, e11=e11, bf=bf, wall_s=time.time() - w0)
    Path(cache).mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps(res))
    os.replace(tmp, f)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--cache", default=str(root / ".cache/gate-10-11"))
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    jobs = [(w, K, s, a.cache) for K in (4, 2) for w in E9.WINDS for s in (0, 1, 2)]
    if a.only:
        sel = {tuple(x.split(":")) for x in a.only.split(",")}
        jobs = [j for j in jobs if (j[0], str(j[1]), str(j[2])) in sel]
    res = []
    with ProcessPoolExecutor(min(a.workers, 5), mp_context=get_context("fork")) as ex:
        futs = {ex.submit(run, j): j for j in jobs}
        for f in as_completed(futs):
            res.append(f.result())
            print("done", futs[f][:3], f"{res[-1]['wall_s']:.0f}s", flush=True)
    if a.only:
        return
    pd.DataFrame([r for x in res for r in x["e10"]]).to_csv(here / "e10-memory-competitors.csv", index=False)
    pd.DataFrame([r for x in res for r in x["e11"]]).to_csv(here / "e11-prior-ablation.csv", index=False)
    pd.DataFrame([r for x in res for r in x["bf"]]).to_csv(here / "e11-bruteforce-k2.csv", index=False)
    tw = {(x["meta"]["wind"], x["meta"]["K"], x["meta"]["seed"]): x["meta"]["test_winds"] for x in res}
    same = [dict(K=K, seed=s, test_winds_identical=bool(np.array_equal(tw[("veering", K, s)], tw[("mixed", K, s)])))
            for K in (2, 4) for s in (0, 1, 2) if ("veering", K, s) in tw and ("mixed", K, s) in tw]
    pd.DataFrame(same).to_csv(here / "scenario-dependence.csv", index=False)


if __name__ == "__main__":
    main()

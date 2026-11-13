"""E09 causal filter replay, local and exploratory (design: docs/research/filter-design.md §5-8).

  OMP_NUM_THREADS=1 uv run python scripts/e09_filter.py [--workers 6] [--only veering:2:0]   # run + report
  uv run python scripts/e09_filter.py --report                                                # tables/figure only
  OMP_NUM_THREADS=1 uv run python scripts/e09_filter.py --cost                                # per-hour cost 16/32/64^2

Per scenario (E07 pilot world): truth and unit responses (E07 code), E07 designs from training hours only, white and
AR(1) readings (R draws each, shared by every design and estimator), then the filters and baselines, scored on the
held-out test hours. Truth is read only by `score_*` and the reconstruction check. Per-scenario results are cached in
.cache/e09/ (a rerun resumes). Writes reports/quarto/tables/e09-local-*.{csv,json} and figures/e09-local-traces.png.
"""
import argparse, json, os, pickle, platform, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from pravah import benchmarks as B  # noqa: E402
from pravah import filter as KF  # noqa: E402
from pravah.experiment_config import ExperimentConfig, Split, Winds  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402
from pravah.forward import simulate  # noqa: E402

WINDS = {"veering": ["veering"], "mixed": ["steady_westerly", "veering", "weak_variable"]}
KS, SEEDS, KSENS, R = (2, 4), (0, 1, 2), (4, 8), 3
DESIGNS = [("random", 0), ("random", 1), ("dopt", 0), ("source_near", 0)]
ARMS = ("white", "ar1")
RHO, Q0_REL, HORIZONS, SD_ACTIVE = 0.8, 1e-3, (1, 6, 24), 1e-2
LO, HI = 0.5, 5.0  # benchmark LogUniform strength range (x a_ref); prior is moment-matched (design note §3.3)
ABAR = (HI - LO) / np.log(HI / LO)
S2 = (HI ** 2 - LO ** 2) / (2 * np.log(HI / LO)) - ABAR ** 2
tab, fig_dir, cache = root / "reports/quarto/tables", root / "reports/quarto/figures", root / ".cache/e09"


def make_cfg(wind, grid=(32, 32)):
    """E07 pilot world (scripts/e07_cross_objective.py SCALES['pilot'])."""
    c = ExperimentConfig.default("EXP-E09")
    c.model.grid, c.model.spinup_h, c.model.dt_s, c.model.advection_scheme = grid, 12, 300, "upwind"
    c.design.n_sources = KS
    c.split = Split((1, 4), (5, 5), (6, 11))
    c.winds = Winds(train=WINDS[wind], heldout=c.winds.heldout)
    c.validate()
    return c


class Ops:
    """Transport operators per distinct hourly wind (known winds, known source locations; no strengths, no truth)."""

    def __init__(self, cfg, source_xy):
        m = cfg.model
        self.kw = dict(nx=m.grid[0], ny=m.grid[1], Lx=m.domain_m[0], Ly=m.domain_m[1], dt=m.dt_s, K=m.diffusivity_m2_s,
                       lam=m.removal_per_s)
        self.steps, self.a_ref = 3600 // m.dt_s, cfg.sources.reference_rate_ug_s
        self.S = [B.source_basis(tuple(m.grid), tuple(m.domain_m), xy, cfg.sources.footprint_sigma_m) / m.mixing_depth_m
                  for xy in source_xy]
        self.lu, self.Bc, self.Ac, self.t_op = {}, {}, {}, 0.0

    def LU(self, w):
        if w not in self.lu:
            t = time.time()
            self.lu[w] = (M := KF.step_operator(u=w, **self.kw), splu(M.tocsc()))
            self.t_op += time.time() - t
        return self.lu[w]

    def A(self, w):
        # all distinct winds kept: ~90 x 8 MB at 32^2, reused by the 24 filter runs of a scenario
        if w not in self.Ac:
            t = time.time()
            self.Ac[w] = KF.hour_operator(self.LU(w)[0], self.steps)
            self.t_op += time.time() - t
        return self.Ac[w]

    def B(self, w):
        """(N, K): one-hour response from zero to each unit source, per a_ref (direct solver calls)."""
        if w not in self.Bc:
            self.Bc[w] = self.a_ref * np.stack([simulate(n_steps=self.steps, c0=np.zeros((self.kw["nx"], self.kw["ny"])),
                                                         u=w, c_bg=0.0, bc="study", source=s, **self.kw).ravel()
                                                for s in self.S], 1)
        return self.Bc[w]

    def apply_hour(self, w, x):
        for _ in range(self.steps):
            x = self.LU(w)[1].solve(x)
        return x


def filters(ops, winds, Fr, H, sig):
    """{name: (m0, P0, transition, H, R)} for the declared filters (design note §3.3). Forcing B_h = F_h - A_h F_{h-1}
    from the unit responses Fr (K, T, N) (known locations, no strengths), so the filters use exactly the response model
    of map24/map_full (deviation 1: direct one-hour solves differ by FiPy LU tolerance, ~1e-5 relative)."""
    Fr0, (K, _, N), k = Fr[:, 0], Fr.shape, len(H)
    Bh_ = lambda h: (Fr[:, h] - (ops.A(tuple(winds[h])) @ Fr[:, h - 1].T).T).T
    w = lambda h: tuple(winds[h])
    abar = np.full(K, ABAR)
    Z = np.zeros
    q0 = (Q0_REL * sig) ** 2

    def tr_field(h):
        Bh = Bh_(h)
        return ops.A(w(h)), Bh @ abar, S2 * Bh @ Bh.T + q0 * np.eye(N)

    def tr_aug(h):
        return np.block([[ops.A(w(h)), Bh_(h)], [Z((K, N)), np.eye(K)]]), None, None

    def tr_ar1(h):
        A = np.block([[ops.A(w(h)), Bh_(h), Z((N, k))], [Z((K, N)), np.eye(K), Z((K, k))],
                      [Z((k, N + K)), RHO * np.eye(k)]])
        return A, None, np.diag(np.r_[Z(N + K), np.full(k, (1 - RHO ** 2) * sig ** 2)])

    Pa = S2 * np.block([[Fr0.T @ Fr0, Fr0.T], [Fr0, np.eye(K)]])
    ma = np.r_[Fr0.T @ abar, abar]
    return {
        "kf_field": (Fr0.T @ abar, S2 * Fr0.T @ Fr0 + q0 * np.eye(N), tr_field, H, sig ** 2),
        "kf_aug": (ma, Pa, tr_aug, np.hstack([H, Z((k, K))]), sig ** 2),
        "kf_aug_ar1": (np.r_[ma, Z(k)], np.block([[Pa, Z((N + K, k))], [Z((k, N + K)), sig ** 2 * np.eye(k)]]), tr_ar1,
                       np.hstack([H, Z((k, K)), np.eye(k)]), 1e-10 * sig ** 2),
    }


def map_series(A, z, issue, prior_mean, prior_var, window):
    """MAP of a (a_ref units) at each issue time from the `window` hours ending there (None = all past hours), with
    posterior covariances. Same algebra as cross_objective.estimate_series (which fixes prior N(0, I))."""
    K = A.shape[2]
    cP = np.concatenate([np.zeros((1, K, K)), np.cumsum(np.einsum("shk,shl->hkl", A, A), 0)])
    cb = np.concatenate([np.zeros((1, K) + z.shape[2:]), np.cumsum(np.einsum("shk,sh...->hk...", A, z), 0)])
    lo = np.zeros_like(issue) if window is None else issue + 1 - window
    Nt = np.eye(K) / prior_var + cP[issue + 1] - cP[lo]
    bt = (prior_mean / prior_var)[:, None] + cb[issue + 1] - cb[lo]
    return np.linalg.solve(Nt, bt), np.linalg.inv(Nt)


# ---------------------------------------------------------------- scoring (the only place truth enters)
def score_rows(est, tf, issue, a_true, n_draws):
    """est: {name: dict(fc={hz: (n_issue, N, c)}, an=(n_issue, N, c)?, sd1=(n_issue, N)?, a=(n_issue, K, c)?,
    nis=(T, c)?, kobs, white=(T, k, c)?)}. Returns rows per (estimator, arm) with metrics averaged over draws."""
    T = len(tf)
    test_h = np.arange(issue[0] + 1, issue[-1] + 2)
    rows = []
    for name, e in est.items():
        for ai, arm in enumerate(ARMS):
            cols = range(ai * n_draws, (ai + 1) * n_draws)
            r = dict(estimator=name, arm=arm)
            for hz, fc in e["fc"].items():
                ok = issue + hz <= T - 1
                r[f"n_issue_h{hz}"] = int(ok.sum())
                r[f"forecast_rmse_h{hz}"] = float(np.mean([np.sqrt(np.mean((fc[ok, :, c] - tf[issue[ok] + hz]) ** 2))
                                                           for c in cols]))
            if "an" in e:
                r["analysis_rmse"] = float(np.mean([np.sqrt(np.mean((e["an"][:, :, c] - tf[issue]) ** 2)) for c in cols]))
            if "a" in e:
                r["source_rmse"] = float(np.mean([np.sqrt(np.mean((e["a"][:, :, c] - a_true) ** 2)) for c in cols]))
            if "sd1" in e:
                sd = np.maximum(e["sd1"], 1e-12)
                act = e["sd1"] >= e["sd_active"]
                r["frac_active"] = float(act.mean())
                cv, cva, cr = [], [], []
                for c in cols:
                    err, mu = tf[issue + 1], e["fc"][1][:, :, c]
                    cv.append(KF.coverage(mu, sd, err))
                    cva.append(KF.coverage(mu[act], sd[act], err[act]))
                    cr.append(float(KF.gaussian_crps(mu, sd, err).mean()))
                for lv in (0.5, 0.8, 0.95):
                    r[f"cov{int(lv * 100)}_all"] = float(np.mean([x[lv] for x in cv]))
                    r[f"cov{int(lv * 100)}_active"] = float(np.mean([x[lv] for x in cva]))
                r["crps"] = float(np.mean(cr))
            if "nis" in e:
                k = np.maximum(e["kobs"], 1)[:, None]
                nk = e["nis"][:, list(cols)] / k
                r["nis_all"], r["nis_test"] = float(np.nanmean(nk)), float(np.nanmean(nk[test_h]))
                wt = e["white"][test_h][..., list(cols)]
                r["innov_lag1"] = float(np.nanmean(wt[1:] * wt[:-1]) / np.nanmean(wt ** 2))
            rows.append(r)
    return rows


# ---------------------------------------------------------------- one scenario
def run_scenario(job):
    wind, K, seed, n = job
    f = cache / f"g{n}_{wind}_K{K}_s{seed}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    w0 = time.time()
    cfg = make_cfg(wind, (n, n))
    sc = B.generate_scenario(cfg, seed, K)
    F, G3 = X.responses(sc)
    t_solve = time.time() - w0
    m, a_ref, cbg = cfg.model, cfg.sources.reference_rate_ug_s, cfg.sources.background_ug_m3
    grid, dom = tuple(m.grid), tuple(m.domain_m)
    T, N = len(sc.truth.winds), grid[0] * grid[1]
    Fr = a_ref * F.reshape(K, T, N)                           # per a_ref
    tf = sc.truth.field.reshape(T, N) - cbg                   # truth anomaly: scorer and checks only
    a_true = sc.truth.strengths_ug_s / a_ref
    issue, winds = X.issue_times(sc), sc.truth.winds
    sigma = X.site_sigma(sc, False)
    sig = float(sigma[0])
    ops = Ops(cfg, sc.truth.source_xy)
    # check: truth[h] = A_h truth[h-1] + B_h a_true for every hour (operators vs the truth's own solver run)
    rec = max(float(np.abs(ops.apply_hour(tuple(winds[h]), tf[h - 1]) + ops.B(tuple(winds[h])) @ a_true - tf[h]).max())
              for h in range(1, T))
    p = X.problem(X.design_input(sc, G3, sigma))
    seedp, T_train = sc.cfg.seeds.streams(seed)["placement"], len(sc.split_hours("train"))
    Yall = {arm: [X.readings(sc, sigma, arm, r, T) for r in range(R)] for arm in ARMS}
    key = dict(wind=wind, K=K, seed=seed)
    rows, traces, runtime, checks = [], [], [], []
    # design-free baselines (same for every design)
    base = dict(
        background_only=dict(fc={hz: np.zeros((len(issue), N, 1)) for hz in HORIZONS}),
        climatology=dict(fc={hz: np.repeat(tf[sc.split_hours("train")].mean(0)[None, :, None], len(issue), 0) for hz in HORIZONS}),
        persistence_oracle=dict(fc={hz: tf[issue][..., None] for hz in HORIZONS}),
        open_loop=dict(fc={hz: np.einsum("ktn,k->tn", Fr[:, np.minimum(issue + hz, T - 1)], np.full(K, ABAR))[..., None]
                           for hz in HORIZONS}, an=np.einsum("ktn,k->tn", Fr[:, issue], np.full(K, ABAR))[..., None],
                       sd1=np.sqrt(S2 * (Fr[:, issue + 1] ** 2).sum(0)), sd_active=SD_ACTIVE * sig),
    )
    for name, e in base.items():  # single data-free column: broadcast to every draw of both arms
        e["fc"] = {hz: np.repeat(v, R * len(ARMS), 2) for hz, v in e["fc"].items()}
        if "an" in e:
            e["an"] = np.repeat(e["an"], R * len(ARMS), 2)
    rows += [{**key, "k": 0, "design": "-", "net": 0, **r} for r in score_rows(base, tf, issue, a_true, R)]
    for k in KSENS:
        for name, net in DESIGNS:
            S = X.make_design(name, p, k, T_train, seedp, net)[0]
            dk = {**key, "k": k, "design": name, "net": net, "sites": " ".join(map(str, S))}
            Y = np.stack([y[S] - cbg for arm in ARMS for y in Yall[arm]], 2).transpose(1, 0, 2)  # (T, k, 2R)
            H = KF.observation_matrix(sc.site_xy[S], grid, dom)
            est = {}
            # E07 frozen estimator (prior N(0, I), 24 h window) and the growing-window MAP with the filter's prior
            Aw = G3[S] * a_ref / sigma[S][:, None, None]
            z = (Y.transpose(1, 0, 2)) / sig
            for nm, (pm, pv, win) in dict(map24=(np.zeros(K), 1.0, X.WINDOW_H), map_full=(np.full(K, ABAR), S2, None)).items():
                ah, C = map_series(Aw, z, issue, pm, pv, win)
                if nm == "map24":  # identical to the frozen E07 code path
                    ref = np.stack([X.estimate_series(Aw, z[..., c], issue) for c in range(z.shape[2])], 2)
                    checks.append(dict(**dk, check="map24_vs_e07_code", value=float(np.abs(ah - ref).max())))
                est[nm] = dict(fc={hz: np.einsum("ktn,tkc->tnc", Fr[:, np.minimum(issue + hz, T - 1)], ah) for hz in HORIZONS},
                               an=np.einsum("ktn,tkc->tnc", Fr[:, issue], ah), a=ah,
                               sd1=np.sqrt(np.maximum(np.einsum("ktn,tkl,ltn->tn", Fr[:, issue + 1], C, Fr[:, issue + 1]), 0)),
                               sd_active=SD_ACTIVE * sig)
            for fn, (m0, P0, tr, Hf, Rf) in filters(ops, winds, Fr, H, sig).items():
                t0, c0 = time.time(), time.process_time()
                o = KF.run(m0, P0, tr, Hf, Rf, Y, issue=issue, horizons=HORIZONS)
                runtime.append(dict(**dk, filter=fn, n_state=len(m0), wall_s=time.time() - t0, cpu_s=time.process_time() - c0,
                                    hours=T))
                est[fn] = dict(fc={hz: v[:, :N] for hz, v in o["fc"].items()}, an=o["am"][issue, :N],
                               sd1=np.sqrt(np.maximum(o["fv"][issue + 1, :N], 0)), sd_active=SD_ACTIVE * sig,
                               nis=o["nis"], kobs=o["kobs"], white=o["white"])
                checks.append(dict(**dk, check=f"{fn}_min_var", value=float(min(o["fv"].min(), o["av"].min()))))
                if fn != "kf_field":
                    est[fn]["a"] = o["am"][issue, N:N + K]
                if fn == "kf_aug":
                    est["persistence_analysis"] = dict(fc={hz: o["am"][issue, :N] for hz in HORIZONS})
            d = {hz: np.abs(est["kf_aug"]["fc"][hz] - est["map_full"]["fc"][hz]) for hz in HORIZONS}
            checks.append(dict(**dk, check="kf_aug_vs_map_full_fc_maxabs",
                               value=float(max(np.nanmax(v[issue + hz <= T - 1]) for hz, v in d.items()))))
            rows += [{**dk, **r} for r in score_rows(est, tf, issue, a_true, R)]
            print(f"  {wind}:{K}:{seed} k={k} {name}{net} {time.time() - w0:.0f}s", flush=True)
            if k == 8 and name in ("dopt", "source_near"):  # per-hour traces, white arm, mean over draws
                for nm in ("open_loop", "map24", "kf_field", "kf_aug", "persistence_oracle", "climatology"):
                    fc = (base | est)[nm]["fc"][1][:, :, :R]
                    rm = np.sqrt(((fc - tf[issue + 1][..., None]) ** 2).mean(1)).mean(1)
                    nis = est[nm]["nis"][issue + 1, :R].mean(1) / k if "nis" in est.get(nm, {}) else np.full(len(issue), np.nan)
                    traces += [dict(**key, design=name, estimator=nm, hour=int(t + 1), forecast_rmse=float(v), nis=float(q))
                               for t, v, q in zip(issue, rm, nis)]
    meta = dict(**key, sigma=sig, a_true=" ".join(f"{x:.4g}" for x in a_true), recon_max_abs=rec,
                truth_anom_max=float(np.abs(tf).max()), n_distinct_winds=len({tuple(x) for x in winds}),
                op_s=ops.t_op, solve_s=t_solve, total_s=time.time() - w0, n_issue=len(issue), hours=T,
                regimes=json.dumps(sc.truth.wind_regimes), host=platform.node(), pid=os.getpid())
    res = (rows, traces, runtime, checks, meta)
    cache.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps(res))
    os.replace(tmp, f)
    return res


# ---------------------------------------------------------------- cost scaling (engineering only)
def cost():
    out = []
    for n, dt in ((16, 300), (32, 300), (64, 300), (64, 30)):
        kw = dict(nx=n, ny=n, Lx=1e4, Ly=1e4, dt=dt, u=(2.0, -1.0), K=50.0, lam=1 / 43200)
        t = time.time(); M = KF.step_operator(**kw); t_probe = time.time() - t
        t = time.time(); A = KF.hour_operator(M, 3600 // dt); t_hour = time.time() - t
        N = n * n
        L = np.random.default_rng(0).standard_normal((N, 32)) / 6
        P, m = L @ L.T + np.eye(N), np.zeros((N, 6))
        H = np.eye(N)[np.linspace(0, N - 1, 8).astype(int)]
        t = time.time(); m, P = KF.predict(m, P, A, None, np.eye(N) * 1e-6); t_pred = time.time() - t
        t = time.time(); KF.update(m, P, np.zeros((8, 6)), H, 1.0); t_upd = time.time() - t
        out.append(dict(grid=n, dt_s=dt, N=N, steps_per_hour=3600 // dt, probe_s=t_probe, hour_operator_s=t_hour,
                        predict_s=t_pred, update_s=t_upd, threads=os.environ.get("OMP_NUM_THREADS"), host=platform.node()))
        print(out[-1], flush=True)
    pd.DataFrame(out).to_csv(tab / "e09-local-cost.csv", index=False)


# ---------------------------------------------------------------- report
EST_ORDER = ["background_only", "climatology", "persistence_oracle", "open_loop", "map24", "map_full", "kf_field",
             "kf_aug", "kf_aug_ar1", "persistence_analysis"]


def boot_ci(x, n=10000, seed=4000):
    x = np.asarray(x, float)
    mm = np.median(x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))], 1)
    return float(np.quantile(mm, .025)), float(np.quantile(mm, .975))


def report():
    df = pd.read_csv(tab / "e09-local-rows.csv")
    # design-free baselines (k = 0 rows) are repeated under every (k, design) so they pair with the estimators
    b = df[df.k == 0].drop(columns=["k", "design", "net"])
    keys = df[df.k > 0][["wind", "K", "seed", "k", "design", "net"]].drop_duplicates()
    d = pd.concat([df[df.k > 0], keys.merge(b, on=["wind", "K", "seed"])], ignore_index=True)
    metrics = [c for c in ["forecast_rmse_h1", "forecast_rmse_h6", "forecast_rmse_h24", "analysis_rmse", "source_rmse",
                           "cov50_active", "cov80_active", "cov95_active", "cov95_all", "crps", "nis_all", "nis_test",
                           "innov_lag1"] if c in d]
    summ = d.groupby(["arm", "k", "design", "estimator"])[metrics].median().reset_index()
    summ.insert(4, "n_rows", d.groupby(["arm", "k", "design", "estimator"]).size().values)
    summ.to_csv(tab / "e09-local-summary.csv", index=False)
    # paired relative reduction vs the frozen E07 estimator, per scenario (random = mean of its 2 networks)
    sc = d.groupby(["arm", "wind", "K", "seed", "k", "design", "estimator"])[metrics].mean().reset_index()
    out = []
    for (arm, k, des), c in sc.groupby(["arm", "k", "design"]):
        for m in ["forecast_rmse_h1", "forecast_rmse_h6", "forecast_rmse_h24", "analysis_rmse"]:
            piv = c.pivot_table(index=["wind", "K", "seed"], columns="estimator", values=m)
            for e in ["open_loop", "map_full", "kf_field", "kf_aug", "kf_aug_ar1", "persistence_analysis"]:
                if e not in piv:
                    continue
                rho = ((piv["map24"] - piv[e]) / piv["map24"]).to_numpy()
                lo, hi = boot_ci(rho)
                out.append(dict(arm=arm, k=k, design=des, metric=m, estimator=e, n=len(rho), median_rel_reduction_vs_map24=float(np.median(rho)),
                                ci_lo=lo, ci_hi=hi, wins=int((rho > 1e-12).sum()), losses=int((rho < -1e-12).sum())))
    pd.DataFrame(out).to_csv(tab / "e09-local-paired.csv", index=False)
    plot()


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = pd.read_csv(tab / "e09-local-traces.csv")
    pr = pd.read_csv(tab / "e09-local-paired.csv")
    meta = pd.read_csv(tab / "e09-local-scenarios.csv")
    col = {"open_loop": "#8c8c8c", "map24": "#d1495b", "kf_field": "#edae49", "kf_aug": "#00798c",
           "persistence_oracle": "#30638e", "climatology": "#b5b5b5", "kf_aug_ar1": "#6a4c93", "map_full": "#00798c",
           "persistence_analysis": "#3d5a80"}
    sel = tr[(tr.wind == "mixed") & (tr.K == 2) & (tr.seed == 0) & (tr.design == "dopt")]
    reg = json.loads(meta[(meta.wind == "mixed") & (meta.K == 2) & (meta.seed == 0)].regimes.iloc[0])["test"]
    fg, ax = plt.subplots(3, 1, figsize=(10, 10.5), gridspec_kw=dict(height_ratios=[1.3, .8, 1.2]))
    t0 = sel.hour.min() - 7  # first test hour
    for nm in ["climatology", "open_loop", "persistence_oracle", "map24", "kf_field", "kf_aug"]:
        q = sel[sel.estimator == nm]
        ax[0].plot(q.hour, q.forecast_rmse, color=col[nm], lw=1.6 if nm.startswith(("kf", "map")) else 1,
                   ls="--" if nm in ("persistence_oracle", "climatology") else "-", label=nm)
        if nm in ("kf_field", "kf_aug"):
            ax[1].plot(q.hour, q.nis, color=col[nm], lw=1.2, label=nm)
    for a in ax[:2]:
        for b, r in enumerate(reg):
            a.axvspan(t0 + 48 * b, t0 + 48 * (b + 1), color=("#f2f2f2" if b % 2 else "white"), zorder=0)
            if a is ax[0]:
                a.text(t0 + 48 * b + 2, 0.97, r.replace("_", " "), transform=a.get_xaxis_transform(), fontsize=8, va="top")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("1 h forecast RMSE (µg m⁻³)")
    ax[0].set_title("Causal replay, scenario mixed K=2 seed 0, dopt k=8, white noise (mean of 3 draws); held-out test hours", fontsize=10)
    ax[0].legend(fontsize=8, ncol=3, loc="lower right")
    ax[1].axhline(1, color="k", lw=.8, ls=":")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("NIS / k (1 = calibrated)")
    ax[1].set_xlabel("hour after spin-up (forecast target)")
    ax[1].legend(fontsize=8)
    p = pr[(pr.arm == "white") & (pr.metric == "forecast_rmse_h1")]
    ests = ["open_loop", "kf_field", "kf_aug", "persistence_analysis"]
    labels = []
    for i, (k, des) in enumerate(p[["k", "design"]].drop_duplicates().itertuples(index=False)):
        labels.append(f"{des}\nk={k}")
        for j, e in enumerate(ests):
            q = p[(p.k == k) & (p.design == des) & (p.estimator == e)]
            x = i + (j - 1.5) * 0.18
            r, lo, hi = 1 - q.median_rel_reduction_vs_map24, 1 - q.ci_hi, 1 - q.ci_lo  # RMSE ratio to map24
            ax[2].errorbar(x, r, yerr=[r - lo, hi - r], fmt="o", color=col[e], ms=5, capsize=2, label=e if i == 0 else None)
    ax[2].axhline(1, color="k", lw=.8)
    ax[2].set_yscale("log")
    ax[2].set_xticks(range(len(labels)), labels, fontsize=8)
    ax[2].set_ylabel("RMSE / RMSE_map24 (log)")
    ax[2].set_title("1 h forecast RMSE relative to E07 frozen MAP-24h, white noise: median over 12 scenarios, 95 % bootstrap CI", fontsize=10)
    ax[2].legend(fontsize=8, ncol=4, loc="upper left")
    fg.tight_layout()
    fg.savefig(fig_dir / "e09-local-traces.png", dpi=130)
    plt.close(fg)


def provenance():
    git = lambda *c: subprocess.run(["git", *c], cwd=root, capture_output=True, text=True).stdout.strip()
    return dict(command="OMP_NUM_THREADS=1 uv run python scripts/e09_filter.py " + " ".join(sys.argv[1:]),
                git_rev=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain", "--", "src", "scripts")),
                host=platform.node(), python=platform.python_version(), numpy=np.__version__,
                design_note="docs/research/filter-design.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None, help="comma list wind:K:seed (smoke; writes no tables)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--grid", type=int, default=32, help="16 for a quick smoke (with --only)")
    a = ap.parse_args()
    if a.report:
        return report()
    if a.cost:
        return cost()
    jobs = [(w, K, s, a.grid) for K in sorted(KS, reverse=True) for w in WINDS for s in SEEDS]
    if a.only:
        sel = {tuple(x.split(":")) for x in a.only.split(",")}
        jobs = [j for j in jobs if (j[0], str(j[1]), str(j[2])) in sel]
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(min(a.workers, 6)) as ex:  # shared machine: at most 6 processes
        futs = {ex.submit(run_scenario, j): j for j in jobs}
        for f in as_completed(futs):
            res.append(f.result())
            mt = res[-1][4]
            print(f"done {futs[f]} total={mt['total_s']:.0f}s op={mt['op_s']:.0f}s recon={mt['recon_max_abs']:.2e}", flush=True)
    if a.only:
        return
    pd.DataFrame([r for x in res for r in x[0]]).to_csv(tab / "e09-local-rows.csv", index=False)
    pd.DataFrame([r for x in res for r in x[1]]).to_csv(tab / "e09-local-traces.csv", index=False)
    pd.DataFrame([r for x in res for r in x[2]]).to_csv(tab / "e09-local-runtime.csv", index=False)
    pd.DataFrame([r for x in res for r in x[3]]).to_csv(tab / "e09-local-checks.csv", index=False)
    pd.DataFrame([x[4] for x in res]).to_csv(tab / "e09-local-scenarios.csv", index=False)
    meta = dict(**provenance(), winds=WINDS, Ks=KS, seeds=SEEDS, ks=KSENS, designs=DESIGNS, R=R, rho=RHO, q0_rel=Q0_REL,
                horizons=HORIZONS, prior=dict(mean=ABAR, var=S2), sd_active_rel=SD_ACTIVE, workers=min(a.workers, 6),
                total_wall_s=time.time() - t0)
    (tab / "e09-local-meta.json").write_text(json.dumps(meta, indent=1, default=str))
    report()


if __name__ == "__main__":
    main()

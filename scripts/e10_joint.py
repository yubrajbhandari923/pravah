"""E10 joint source-strength and field inference, 32² local pilot (design: docs/research/joint-assimilation-design.md).

  OMP_NUM_THREADS=1 uv run python scripts/e10_joint.py [--workers 5] [--only mixed:2:0] [--grid 16]  # run + report
  uv run python scripts/e10_joint.py --report                                                      # tables/figure

E09 pilot world (scripts/e09_filter.py: make_cfg, Ops, map_series reused). Per scenario: three truth variants of the
strengths (const / diurnal / logrw), one set of white-noise draws shared by all variants and designs, the MAP family
(24 h, 72 h, all past; prior N(0, I)) and the random-walk augmented filter over the q grid, each also with the
truncated-mode nonnegativity constraint (suffix _nn). Truth enters only the scorer and the recursion check.
Per-scenario results cached in .cache/e10/. Writes reports/quarto/tables/e10-local-* and figures/e10-local-traces.png.
"""
import argparse, json, os, pickle, platform, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binom  # noqa: E402

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "src"), str(root / "scripts")]
import e09_filter as E9  # noqa: E402
from pravah import benchmarks as B  # noqa: E402
from pravah import filter as KF  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402

WINDS, KS, SEEDS, KSENS, R = E9.WINDS, (2, 4), (0, 1, 2), (4, 8), 3
DESIGNS = [("random", 0), ("dopt", 0), ("source_near", 0)]
QS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)             # a_ref^2 per hour; q = 0 is map_full (filter run once as a check)
MAPW = {"map6": 6, "map12": 12, "map24": 24, "map72": 72, "map_full": None}
VARIANTS = ("const", "diurnal", "logrw")
HORIZONS, SD_ACTIVE, PHI, SD_LOG = (1, 6), 1e-2, np.exp(-1 / 72), 0.5
tab, fig_dir, cache = root / "reports/quarto/tables", root / "reports/quarto/figures", root / ".cache/e10"


def qname(q):
    return "map_full" if q == 0 else f"kf_q{q:.0e}"


def strengths(a0, T, seed, K):
    """(variant -> (T, K)) a_ref units; a[h] is emitted during hour h -> h+1 (design note §2, §4)."""
    rng = np.random.default_rng([10_000 + seed, K])          # E10-only truth stream
    h = np.arange(T)[:, None]
    tk = rng.uniform(0, 24, K)
    z = np.empty((T, K))
    z[0] = SD_LOG * rng.standard_normal(K)
    for t in range(1, T):
        z[t] = PHI * z[t - 1] + SD_LOG * np.sqrt(1 - PHI ** 2) * rng.standard_normal(K)
    return dict(const=np.tile(a0, (T, 1)), diurnal=a0 * (1 + 0.5 * np.sin(2 * np.pi * (h - tk) / 24)), logrw=a0 * np.exp(z))


def ci_median(x, level=0.95):
    """Distribution-free order-statistic CI for the median: (x_(j), x_(n+1-j)), largest j with P(Bin < j) <= alpha/2."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    j = max([j for j in range(1, n // 2 + 1) if binom.cdf(j - 1, n, .5) <= (1 - level) / 2], default=None)
    return (np.nan, np.nan) if j is None else (float(x[j - 1]), float(x[n - j]))


# ---------------------------------------------------------------- one scenario
def run_scenario(job):
    wind, K, seed, n = job
    f = cache / f"g{n}_{wind}_K{K}_s{seed}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    w0 = time.time()
    cfg = E9.make_cfg(wind, (n, n))
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
    ops = E9.Ops(cfg, sc.truth.source_xy)
    Bh = [None] + [Fr[:, h].T - ops.A(winds[h]) @ Fr[:, h - 1].T for h in range(1, T)]       # (N, K)
    # ---- truth variants (scorer only): same linear recursion as the filter; const uses the solver truth
    av = strengths(sc.truth.strengths_ug_s / a_ref, T, seed, K)
    xt = {}
    for v, a in av.items():
        x = np.empty((T, N))
        x[0] = Fr[:, 0].T @ a[0]
        for h in range(1, T):
            x[h] = ops.A(winds[h]) @ x[h - 1] + Bh[h] @ a[h - 1]
        xt[v] = x
    tf = sc.truth.field.reshape(T, N) - cbg
    recon = float(np.abs(xt["const"] - tf).max())
    xt["const"] = tf
    V, nc = len(VARIANTS), len(VARIANTS) * R                           # data columns ordered (variant, draw)
    E = [X.readings(sc, sigma, "white", r, T) - np.asarray(sc.truth.site_truth)[:, :T] for r in range(R)]
    p = X.problem(X.design_input(sc, G3, sigma))
    seedp, T_train = sc.cfg.seeds.streams(seed)["placement"], len(sc.split_hours("train"))
    key = dict(wind=wind, K=K, seed=seed)
    rows, traces, checks, runtime = [], [], [], []
    for k in KSENS:
        for name, net in DESIGNS:
            S = X.make_design(name, p, k, T_train, seedp, net)[0]
            dk = {**key, "k": k, "design": name}
            H = KF.observation_matrix(sc.site_xy[S], grid, dom)
            Y = np.stack([xt[v] @ H.T + E[r][S].T for v in VARIANTS for r in range(R)], 2)     # (T, k, nc)
            checks.append(dict(**dk, check="H_vs_site_truth",
                               value=float(np.abs(tf @ H.T + cbg - np.asarray(sc.truth.site_truth)[S, :T].T).max())))
            est = {}
            # ---- MAP family (prior N(0, I), a_ref units), information form on solver site responses
            Aw = G3[S] * a_ref / sigma[S][:, None, None]
            z = Y.transpose(1, 0, 2) / sig
            for nm, win in MAPW.items():
                ah, C = E9.map_series(Aw, z, issue, np.zeros(K), 1.0, win)
                ann = np.stack([KF.nonneg_mode(ah[i], C[i]) for i in range(len(issue))])
                fc = {hz: np.einsum("ktn,tkc->tnc", Fr[:, np.minimum(issue + hz, T - 1)], ah) for hz in HORIZONS}
                sd1 = np.sqrt(np.maximum(np.einsum("ktn,tkl,ltn->tn", Fr[:, issue + 1], C, Fr[:, issue + 1]), 0))
                est[nm] = dict(fc=fc, an=np.einsum("ktn,tkc->tnc", Fr[:, issue], ah), a=ah, C=C, sd1=sd1)
                est[nm + "_nn"] = dict(fc={1: np.einsum("ktn,tkc->tnc", Fr[:, issue + 1], ann)}, a=ann)
            # map_full 1 h innovations on validation hours (for q selection)
            ahv, _ = E9.map_series(Aw, z, val - 1, np.zeros(K), 1.0, None)
            inn = {0.0: np.mean([((Y[h] - H @ Fr[:, h].T @ ahv[i]) ** 2).mean(0) for i, h in enumerate(val)], 0)}
            # ---- random-walk augmented filters
            P0 = np.block([[Fr[:, 0].T @ Fr[:, 0], Fr[:, 0].T], [Fr[:, 0], np.eye(K)]])
            Hf = np.hstack([H, np.zeros((k, K))])
            for q in QS + ((0.0,) if (k, name) == (8, "dopt") else ()):
                Qa = np.diag(np.r_[np.zeros(N), np.full(K, q)]) if q else None
                tr = lambda h: (np.block([[ops.A(winds[h]), Bh[h]], [np.zeros((K, N)), np.eye(K)]]), None, Qa)
                t0 = time.time()
                o = KF.run(np.zeros(N + K), P0, tr, Hf, sig ** 2, Y, issue=issue, horizons=HORIZONS,
                           keep=np.arange(N, N + K))
                runtime.append(dict(**dk, q=q, wall_s=time.time() - t0))
                ma, C, Pxa = o["am"][issue, N:], o["aPk"][issue][:, N:], o["aPk"][issue][:, :N]
                if q == 0:
                    d = max(float(np.abs(o["fc"][1][:, :N] - est["map_full"]["fc"][1]).max()),
                            float(np.abs(ma - est["map_full"]["a"]).max()))
                    checks.append(dict(**dk, check="kf_q0_vs_map_full_maxabs", value=d))
                    continue
                inn[q] = ((Y[val] - np.einsum("kn,hnc->hkc", H, o["fm"][val, :N])) ** 2).mean((0, 1))
                fc1n, ann = o["fc"][1][:, :N].copy(), ma.copy()
                for i, t in enumerate(issue):
                    an = KF.nonneg_mode(ma[i], C[i])
                    da = an - ma[i]
                    if np.any(da):
                        dx = Pxa[i] @ np.linalg.solve(C[i], da)
                        fc1n[i] += ops.A(winds[t + 1]) @ dx + Bh[t + 1] @ da
                        ann[i] = an
                nm = qname(q)
                est[nm] = dict(fc={hz: v[:, :N] for hz, v in o["fc"].items()}, an=o["am"][issue, :N], a=ma, C=C,
                               sd1=np.sqrt(np.maximum(o["fv"][issue + 1, :N], 0)), nis=o["nis"], kobs=o["kobs"],
                               white=o["white"])
                est[nm + "_nn"] = dict(fc={1: fc1n}, a=ann)
            # ---- data-driven q (validation innovations only), per data column
            qs = sorted(inn)
            pick = np.argmin(np.stack([inn[q] for q in qs]), 0)
            sel = {}
            for part in ("", "_nn"):
                src = [est[qname(qs[j]) + part] for j in pick]
                sel["kf_qsel" + part] = dict(fc={hz: np.stack([s["fc"][hz][..., c] for c, s in enumerate(src)], 2)
                                                 for hz in src[0]["fc"]},
                                             a=np.stack([s["a"][..., c] for c, s in enumerate(src)], 2))
            sel["kf_qsel"]["sd1"] = np.stack([est[qname(qs[j])]["sd1"] for j in pick], 2)
            est |= sel
            checks += [dict(**dk, check="qsel_column", variant=VARIANTS[c // R], draw=c % R, value=qs[j])
                       for c, j in enumerate(pick)]
            rows += [{**dk, **r} for r in score(est, xt, av, issue, sig)]
            if k == 8 and name in ("dopt", "source_near"):          # source traces, draw 0 of each variant
                for nm in ("map24", "map_full", "kf_q1e-03", "kf_q1e-02", "kf_q1e-01"):
                    e = est[nm]
                    for vi, v in enumerate(VARIANTS):
                        for s in range(K):
                            traces += [dict(**dk, variant=v, estimator=nm, source=s, hour=int(t), a_true=float(av[v][t, s]),
                                            mean=float(e["a"][i, s, vi * R]), sd=float(np.sqrt(e["C"][i, s, s])))
                                       for i, t in enumerate(issue)]
            print(f"  {wind}:{K}:{seed} k={k} {name} {time.time() - w0:.0f}s", flush=True)
    meta = dict(**key, sigma=sig, a_true=" ".join(f"{x:.4g}" for x in av["const"][0]), recon_const_max_abs=recon,
                truth_anom_max=float(np.abs(tf).max()), total_s=time.time() - w0, n_issue=len(issue), hours=T,
                host=platform.node(), pid=os.getpid())
    res = (rows, traces, checks, runtime, meta)
    cache.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps(res))
    os.replace(tmp, f)
    return res


def score(est, xt, av, issue, sig):
    """Rows per (estimator, variant): metrics vs that variant's truth, averaged over the R draws."""
    rows, T = [], len(next(iter(xt.values())))
    for nm, e in est.items():
        for vi, v in enumerate(VARIANTS):
            cols, tf, a = range(vi * R, (vi + 1) * R), xt[v], av[v][issue]
            r = dict(estimator=nm, variant=v)
            for hz, fc in e["fc"].items():
                ok = issue + hz <= T - 1
                r[f"forecast_rmse_h{hz}"] = float(np.mean([np.sqrt(np.mean((fc[ok, :, c] - tf[issue[ok] + hz]) ** 2))
                                                           for c in cols]))
            err = np.stack([e["a"][..., c] - a for c in cols])                 # (R, n_issue, K)
            r["source_rmse"] = float(np.mean(np.sqrt((err ** 2).mean((1, 2)))))
            r["neg_frac"] = float(np.mean(e["a"][..., list(cols)] < 0))
            if "an" in e:
                r["analysis_rmse"] = float(np.mean([np.sqrt(np.mean((e["an"][:, :, c] - tf[issue]) ** 2)) for c in cols]))
            if "C" in e:
                sd = np.sqrt(np.einsum("tkk->tk", e["C"]))
                r["src_cov95"] = float(np.mean(np.abs(err) <= 1.96 * sd))
                r["src_nees_k"] = float(np.median(np.einsum("rtk,tkl,rtl->rt", err, np.linalg.inv(e["C"]), err) / a.shape[1]))
            if "sd1" in e:      # sd1: (n_issue, N), or (n_issue, N, n_cols) for kf_qsel (q differs per data column)
                sd1s = [e["sd1"][..., c] if e["sd1"].ndim == 3 else e["sd1"] for c in cols]
                cv = []
                for c, s1 in zip(cols, sd1s):
                    act = s1 >= SD_ACTIVE * sig
                    cv.append(KF.coverage(e["fc"][1][:, :, c][act], np.maximum(s1, 1e-12)[act], tf[issue + 1][act]))
                for lv in (0.5, 0.8, 0.95):
                    r[f"cov{int(lv * 100)}_active"] = float(np.mean([x[lv] for x in cv]))
                r["crps"] = float(np.mean([KF.gaussian_crps(e["fc"][1][:, :, c], np.maximum(s1, 1e-12), tf[issue + 1]).mean()
                                           for c, s1 in zip(cols, sd1s)]))
            if "nis" in e:
                th = np.arange(issue[0] + 1, issue[-1] + 2)
                r["nis_test"] = float(np.nanmean(e["nis"][th][:, list(cols)] / np.maximum(e["kobs"][th], 1)[:, None]))
                wt = e["white"][th][..., list(cols)]
                r["innov_lag1"] = float(np.nanmean(wt[1:] * wt[:-1]) / np.nanmean(wt ** 2))
            rows.append(r)
    return rows


# ---------------------------------------------------------------- report
def paired(d, ref, metric):
    out = []
    for (v, k, des), c in d.groupby(["variant", "k", "design"]):
        piv = c.pivot_table(index=["wind", "K", "seed"], columns="estimator", values=metric)
        for e in piv.columns:
            if e == ref:
                continue
            rho = ((piv[ref] - piv[e]) / piv[ref]).dropna().to_numpy()
            lo, hi = ci_median(rho)
            out.append(dict(variant=v, k=k, design=des, metric=metric, ref=ref, estimator=e, n=len(rho),
                            median_rel_reduction=float(np.median(rho)), ci_lo=lo, ci_hi=hi,
                            wins=int((rho > 1e-12).sum()), losses=int((rho < -1e-12).sum())))
    return out


def report():
    d = pd.read_csv(tab / "e10-local-rows.csv")
    metrics = [c for c in ["forecast_rmse_h1", "forecast_rmse_h6", "analysis_rmse", "source_rmse", "neg_frac",
                           "cov50_active", "cov80_active", "cov95_active", "crps", "src_cov95", "src_nees_k", "nis_test",
                           "innov_lag1"] if c in d]
    g = d.groupby(["variant", "k", "design", "estimator"])
    summ = g[metrics].median().reset_index()
    summ.insert(4, "n_scen", g.size().values)
    summ.to_csv(tab / "e10-local-summary.csv", index=False, float_format="%.5g")
    out = []
    for ref in ("map_full", "map24"):
        for m in ("forecast_rmse_h1", "forecast_rmse_h6", "source_rmse"):
            out += paired(d, ref, m)
    pd.DataFrame(out).to_csv(tab / "e10-local-paired.csv", index=False, float_format="%.4g")
    # design contrasts per estimator: (ref_design - other)/ref_design > 0 means `other` is better
    out = []
    for (v, k, e), c in d.groupby(["variant", "k", "estimator"]):
        piv = c.pivot_table(index=["wind", "K", "seed"], columns="design", values="forecast_rmse_h1")
        for a, b in (("source_near", "dopt"), ("random", "dopt"), ("random", "source_near")):
            rho = ((piv[a] - piv[b]) / piv[a]).to_numpy()
            lo, hi = ci_median(rho)
            out.append(dict(variant=v, k=k, estimator=e, contrast=f"{b}_vs_{a}", n=len(rho), median_rel_gain=float(np.median(rho)),
                            ci_lo=lo, ci_hi=hi, wins=int((rho > 0).sum())))
    pd.DataFrame(out).to_csv(tab / "e10-local-designs.csv", index=False, float_format="%.4g")
    ch = pd.read_csv(tab / "e10-local-checks.csv")
    ch[ch.check == "qsel_column"].groupby(["variant", "value"]).size().rename("n").reset_index().to_csv(
        tab / "e10-local-qsel.csv", index=False)
    plot()


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = pd.read_csv(tab / "e10-local-traces.csv")
    pr = pd.read_csv(tab / "e10-local-paired.csv")
    col = {"map24": "#d1495b", "map_full": "#30638e", "kf_q1e-03": "#edae49", "kf_q1e-02": "#00798c", "kf_q1e-01": "#6a4c93"}
    fg, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    for j, v in enumerate(("diurnal", "logrw")):
        s = tr[(tr.wind == "mixed") & (tr.K == 2) & (tr.seed == 0) & (tr.design == "dopt") & (tr.variant == v) & (tr.source == 0)]
        a = ax[0, j]
        t0 = s[s.estimator == "map_full"]
        a.plot(t0.hour, t0.a_true, color="k", lw=2, label="truth")
        for nm, c in col.items():
            q = s[s.estimator == nm]
            a.plot(q.hour, q["mean"], color=c, lw=1.2, label=nm)
            if nm in ("kf_q1e-02", "map_full"):
                a.fill_between(q.hour, q["mean"] - 1.96 * q.sd, q["mean"] + 1.96 * q.sd, color=c, alpha=.15, lw=0)
        a.set_title(f"{v}: source 1 strength, mixed K=2 seed 0, dopt k=8, draw 0", fontsize=9)
        a.set_xlabel("hour (test issue times)")
        a.set_ylabel("a / a_ref")
        a.legend(fontsize=7, ncol=3)
    qs = [0.0, *QS]
    for j, (m, lab) in enumerate((("forecast_rmse_h1", "1 h forecast RMSE"), ("source_rmse", "source RMSE"))):
        a = ax[1, j]
        for v, ls in zip(VARIANTS, ("-", "--", ":")):
            for des, c in (("dopt", "#00798c"), ("source_near", "#d1495b"), ("random", "#8c8c8c")):
                p = pr[(pr.ref == "map_full") & (pr.metric == m) & (pr.variant == v) & (pr.design == des) & (pr.k == 8)]
                y = [1.0] + [1 - p[p.estimator == qname(q)].median_rel_reduction.iloc[0] for q in QS]
                a.plot(range(len(qs)), y, ls=ls, color=c, marker="o", ms=3, label=f"{v} {des}")
                m24 = 1 - p[p.estimator == "map24"].median_rel_reduction.iloc[0]
                a.plot(len(qs), m24, ls="", marker="s", color=c, ms=4)
        a.axhline(1, color="k", lw=.7)
        a.set_yscale("log")
        a.set_xticks(range(len(qs) + 1), [f"{q:g}" for q in qs] + ["map24"], fontsize=8)
        a.set_xlabel("source random-walk variance q (a_ref² h⁻¹); q = 0 is map_full")
        a.set_ylabel(f"{lab} / map_full (median over 12)")
        a.set_title(f"{lab} relative to growing-window MAP, k = 8", fontsize=9)
        if j == 0:
            a.legend(fontsize=6, ncol=3)
    fg.tight_layout()
    fg.savefig(fig_dir / "e10-local-traces.png", dpi=130)
    plt.close(fg)


def provenance():
    git = lambda *c: subprocess.run(["git", *c], cwd=root, capture_output=True, text=True).stdout.strip()
    return dict(command="OMP_NUM_THREADS=1 uv run python scripts/e10_joint.py " + " ".join(sys.argv[1:]),
                git_rev=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain", "--", "src", "scripts")),
                host=platform.node(), python=platform.python_version(), numpy=np.__version__,
                design_note="docs/research/joint-assimilation-design.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--only", default=None, help="comma list wind:K:seed (smoke; writes no tables)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--grid", type=int, default=32)
    a = ap.parse_args()
    if a.report:
        return report()
    jobs = [(w, K, s, a.grid) for K in sorted(KS, reverse=True) for w in WINDS for s in SEEDS]
    if a.only:
        sel = {tuple(x.split(":")) for x in a.only.split(",")}
        jobs = [j for j in jobs if (j[0], str(j[1]), str(j[2])) in sel]
    t0, res = time.time(), []
    with ProcessPoolExecutor(min(a.workers, 5), mp_context=get_context("fork")) as ex:
        futs = {ex.submit(run_scenario, j): j for j in jobs}
        for f in as_completed(futs):
            res.append(f.result())
            print(f"done {futs[f]} total={res[-1][4]['total_s']:.0f}s recon={res[-1][4]['recon_const_max_abs']:.2e}", flush=True)
    if a.only:
        return
    for i, nm in enumerate(("rows", "traces", "checks", "runtime")):
        df = pd.DataFrame([r for x in res for r in x[i]])
        if nm == "traces":  # full traces are ~15 MB (all scenarios stay in .cache/e10); commit the figure's scenario only
            df = df[(df.wind == "mixed") & (df.K == 2) & (df.seed == 0)]
        df.to_csv(tab / f"e10-local-{nm}.csv", index=False)
    pd.DataFrame([x[4] for x in res]).to_csv(tab / "e10-local-scenarios.csv", index=False)
    meta = dict(**provenance(), winds=WINDS, Ks=KS, seeds=SEEDS, ks=KSENS, designs=DESIGNS, R=R, qs=QS, map_windows=MAPW,
                variants=VARIANTS, phi=PHI, sd_log=SD_LOG, prior="N(0, I) a_ref units", horizons=HORIZONS,
                workers=min(a.workers, 5), total_wall_s=time.time() - t0)
    (tab / "e10-local-meta.json").write_text(json.dumps(meta, indent=1, default=str))
    report()


if __name__ == "__main__":
    main()

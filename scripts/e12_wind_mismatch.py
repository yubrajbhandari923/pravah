"""E12 wind mismatch (design: docs/research/wind-mismatch-design.md). The truth uses the true wind; the estimators and
forecasts use unit responses computed with a perturbed MODEL wind (pravah.uncertainty); the oracle uses the true wind.

Pilot (32², local):  OMP_NUM_THREADS=1 uv run python scripts/e12_wind_mismatch.py --pilot --workers 5
Sweep (64², one job): OMP_NUM_THREADS=1 uv run python scripts/e12_wind_mismatch.py --sweep --fields E07OUT/fields \
                         --out DIR --workers 64 --stage2-workers 32
Check reuse first:    ... --sweep --fields E07OUT/fields --out DIR --dry-run   (expect truth 0, src 0, pert 300)
Report only:          uv run python scripts/e12_wind_mismatch.py --report

Stage 1 solves (cached atomically in OUT/fields): the truth and oracle responses use E07's content-hashed names
(e07_cross_objective.scenario_fields), so the E07 sweep cache is reused read-only via --fields; each perturbed response is
'<E07 src name>_<spec>_r<seed>'. Stage 2 (one task per scenario) evaluates every spec x design x k x estimator.
"""
import argparse, json, os, pickle, platform, resource, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "src"), str(root / "scripts")]
import e07_cross_objective as E7  # noqa: E402
import e09_filter as E9  # noqa: E402
from pravah import benchmarks as B  # noqa: E402
from pravah import uncertainty as U  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402
from pravah.experiments import forecast_design as FD  # noqa: E402

WIND = "mixed"  # veering shares layout, strengths, noise and test winds with mixed: no extra cluster (design §7)
SCALES = {"pilot": dict(specs=U.PILOT_SPECS, ks=(2, 4, 8), seeds=(0, 1, 2)),
          "sweep": dict(specs=U.SWEEP_SPECS, ks=(4, 8, 16), seeds=tuple(range(10)))}
KS_SRC, R, N_RAND = (2, 4), 3, 5
ESTS = {"map24": 24, "map72": 72, "map_full": None}
METRICS = ["forecast_rmse_h1", "forecast_rmse_h6", "analysis_rmse", "source_rmse", "sensor_fit", "innov_rms", "innov_ac1",
           "cover95_a", "rel_bias_a"]
tab, fig_dir = root / "reports/quarto/tables", root / "reports/quarto/figures"


def field_names(scale, K, seed, specs):
    """{spec name: [(cache name, task)]}: oracle = E07 truth + src names; others = one perturbed solve per source."""
    fl = E7.scenario_fields(scale, WIND, K, seed)
    out = {"truth": [fl[0]], "oracle": fl[1:]}
    for sp in specs:
        if not sp.oracle:
            out[sp.name] = [(f"{n}_{sp.name}_r{seed}", ("pert", scale, WIND, K, seed, k, sp)) for k, (n, _) in enumerate(fl[1:])]
    return out


def run_field(task):
    (name, spec), out = task
    if spec[0] != "pert":
        return E7.run_field(((name, spec), out))
    f = Path(out) / "fields" / f"{name}.npy"
    if f.exists():
        return None
    _, scale, wind, K, seed, k, sp = spec
    w0, c0 = time.time(), time.process_time()
    cfg = E7.make_cfg(scale, wind)
    _, _, sxy, winds, _, _ = B.scenario_drivers(cfg, seed, K)
    wm = U.perturb_winds(winds, sp, seed)
    field = X.source_field(cfg, sxy[k], wm, len(wm))
    tm = dict(name=name, kind=sp.name, wind=wind, K=K, seed=seed, k=k, wall_s=time.time() - w0, cpu_s=time.process_time() - c0,
              peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, host=platform.node(), pid=os.getpid())
    E7._save_atomic(f.with_suffix(".json"), lambda fh: fh.write(json.dumps(tm).encode()))
    E7._save_atomic(f, lambda fh: np.save(fh, field))
    return tm


def _find(name, dirs):
    for d in dirs:
        if (Path(d) / f"{name}.npy").exists():
            return Path(d) / f"{name}.npy"
    raise FileNotFoundError(name)


# ---------------------------------------------------------------- stage 2
def run_scenario(job):
    scale, K, seed, out, fields = job
    f = Path(out) / "scen" / f"e12_{WIND}_K{K}_s{seed}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    w0, c0 = time.time(), time.process_time()
    dirs = [Path(out) / "fields"] + ([Path(fields)] if fields else [])
    specs = SCALES[scale]["specs"]
    names = field_names(scale, K, seed, specs)
    sc = B.generate_scenario(E7.make_cfg(scale, WIND), seed, K, truth_field=np.load(_find(names["truth"][0][0], dirs)))
    Fm = {sp.name: np.stack([np.load(_find(n, dirs)) for n, _ in names[sp.name]]) for sp in specs}
    res = evaluate(scale, K, seed, sc, Fm, specs)
    res[2].update(total_wall_s=time.time() - w0, total_cpu_s=time.process_time() - c0,
                  peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    f.parent.mkdir(exist_ok=True)
    E7._save_atomic(f, lambda fh: fh.write(pickle.dumps(res)))
    return res


def evaluate(scale, K, seed, sc, Fm, specs):
    """Designs from ORACLE training responses (E07), plus dopt_m from each model wind; every estimator on the model
    responses of each spec; truth read only for scoring and the superposition check."""
    ks = SCALES[scale]["ks"]
    a_ref, cbg = sc.cfg.sources.reference_rate_ug_s, sc.cfg.sources.background_ug_m3
    T, last, issue = len(sc.truth.winds), sc.split_hours("test")[-1], X.issue_times(sc)
    test_h = sc.split_hours("test")
    N = int(np.prod(sc.cfg.model.grid))
    tf = sc.truth.field.reshape(T, N) - cbg
    a_true = sc.truth.strengths_ug_s / a_ref
    sigma = X.site_sigma(sc, False)
    Y = np.stack([X.readings(sc, sigma, "white", r, T) for r in range(R)], 2)  # (sites, T, R), E07 draws
    T_train, seedp = len(sc.split_hours("train")), sc.cfg.seeds.streams(seed)["placement"]
    G3 = {n: X.site_responses(sc, F) for n, F in Fm.items()}
    tm = dict(scale=scale, wind=WIND, K=K, seed=seed, a_true=" ".join(f"{x:.4g}" for x in a_true),
              linearity_max_abs=float(np.abs(np.einsum("k,khn->hn", a_true * a_ref, Fm["oracle"].reshape(K, T, N)) - tf).max()))
    p0 = X.problem(X.design_input(sc, G3["oracle"], sigma))
    designs = {}
    for k in ks:
        for name, net in [("random", j) for j in range(N_RAND)] + [("dopt", 0), ("source_near", 0)]:
            designs[(k, name, net)] = X.make_design(name, p0, k, T_train, seedp, net)[0]
    base = dict(scale=scale, wind=WIND, K=K, seed=seed, spec="baseline", family="baseline", magnitude=0.0, k=0, net=0, estimator="-")
    for h in (1, 6):  # model-free baselines on the same issue times (persistence: c(t) for c(t+h); background: anomaly 0)
        ok = issue + h <= last
        rows_b = {"persistence": tf[issue[ok] + h] - tf[issue[ok]], "background_only": tf[issue[ok] + h]}
        for d, e in rows_b.items():
            base.setdefault(d, {})[f"forecast_rmse_h{h}"] = float(np.sqrt(np.mean(e ** 2)))
    rows = [dict({k_: v for k_, v in base.items() if k_ not in ("persistence", "background_only")}, design=d, **base[d])
            for d in ("persistence", "background_only")]
    for sp in specs:
        wm = U.perturb_winds(sc.truth.winds, sp, seed)
        derr, sratio = U.wind_error(sc.truth.winds[test_h], wm[test_h])
        Fr = a_ref * Fm[sp.name].reshape(K, T, N)
        G = G3[sp.name]
        des = dict(designs)
        pm = X.problem(X.design_input(sc, G, sigma))  # the misinformed designer (wind-free designs are unchanged)
        for k in ks:
            des[(k, "dopt_m", 0)] = X.make_design("dopt", pm, k, T_train, seedp)[0]
        for (k, name, net), S in des.items():
            A = a_ref * G[S] / sigma[S][:, None, None]
            z = (Y[S] - cbg) / sigma[S][:, None, None]          # (k, T, R)
            for est, win in ESTS.items():
                u, C = E9.map_series(A, z, issue, np.zeros(K), 1.0, win)    # (n_issue, K, R), (n_issue, K, K)
                sd = np.sqrt(np.einsum("tkk->tk", C))[..., None]
                res = z[:, issue] - np.einsum("sik,ikr->sir", A[:, issue], u)
                inn = z[:, issue + 1] - np.einsum("sik,ikr->sir", A[:, issue + 1], u)
                ic = inn - inn.mean(1, keepdims=True)
                ac1 = (ic[:, 1:] * ic[:, :-1]).sum((0, 1)) / (ic ** 2).sum((0, 1))
                m = [FD.score(u[..., r], Fr, tf, issue, a_true, last, horizons=(1, 6)) for r in range(R)]
                rows.append(dict(scale=scale, wind=WIND, K=K, seed=seed, spec=sp.name, family=sp.family, magnitude=sp.magnitude,
                                 dir_err_deg=derr, speed_ratio=sratio, k=k, design=name, net=net, estimator=est,
                                 forecast_rmse_h1=np.mean([x["forecast_rmse_h1"] for x in m]),
                                 forecast_rmse_h6=np.mean([x["forecast_rmse_h6"] for x in m]),
                                 analysis_rmse=np.mean([x["analysis_rmse"] for x in m]),
                                 source_rmse=np.mean([x["source_rmse"] for x in m]),
                                 sensor_fit=float(np.sqrt((res ** 2).mean())), innov_rms=float(np.sqrt((inn ** 2).mean())),
                                 innov_ac1=float(ac1.mean()),
                                 cover95_a=float((np.abs(u - a_true[:, None]) <= 1.96 * sd).mean()),
                                 rel_bias_a=float(((u.mean(2) - a_true) / a_true).mean()),
                                 max_abs_rel_bias_a=float(np.abs((u.mean((0, 2)) - a_true) / a_true).max()),
                                 sites=" ".join(map(str, S))))
    return rows, [], tm


# ---------------------------------------------------------------- analysis (reads only the row table)
def scen_table(df):
    """Random networks averaged per scenario (design 'random'); cluster = (K, seed)."""
    keys = ["K", "seed", "spec", "family", "magnitude", "k", "design", "estimator"]
    return df.groupby(keys, as_index=False)[METRICS + ["dir_err_deg", "speed_ratio"]].mean()


def baseline_contrasts(st, bl):
    """rho = (persistence - model) / persistence, 1 h forecast, per cluster (> 0: the model beats persistence)."""
    pe = bl[bl.design == "persistence"].set_index("cluster").forecast_rmse_h1
    out = []
    for (spec, k, des, est), g in st.groupby(["spec", "k", "design", "estimator"]):
        g = g.set_index("cluster").forecast_rmse_h1
        out.append(dict(contrast="vs_persistence_h1", spec=spec, k=k, design=des, estimator=est,
                        **FD.paired_summary((pe.loc[g.index] - g) / pe.loc[g.index])))
    return out


def summarise(df):
    bl = df[df.spec == "baseline"].copy()
    bl["cluster"] = bl.K.astype(str) + ":" + bl.seed.astype(str)
    st = scen_table(df[df.spec != "baseline"])
    st["cluster"] = st.K.astype(str) + ":" + st.seed.astype(str)
    idx = ["cluster", "k", "design", "estimator"]
    o = st[st.spec == "oracle"].set_index(idx)
    summ, con, cex = [], [], []
    for spec, g in st.groupby("spec"):
        g = g.set_index(idx)
        D = {m: np.log(g[m] / o.loc[g.index, m]) for m in ("forecast_rmse_h1", "forecast_rmse_h6", "analysis_rmse",
                                                          "source_rmse", "sensor_fit", "innov_rms")}
        for (k, des, est), ix in g.groupby(["k", "design", "estimator"]).groups.items():
            s = FD.paired_summary(D["forecast_rmse_h1"].loc[ix])
            row = dict(spec=spec, family=g.family.iloc[0], magnitude=g.magnitude.iloc[0], k=k, design=des, estimator=est,
                       n_clusters=s["n"], dir_err_deg=float(g.loc[ix, "dir_err_deg"].median()),
                       D_forecast_h1=float(np.exp(s["median"])), D_ci_lo=float(np.exp(s["ci_lo"])), D_ci_hi=float(np.exp(s["ci_hi"])),
                       n_worse=s["wins"])
            for m in D:
                row[f"D_{m}"] = float(np.exp(np.median(D[m].loc[ix])))
            for m in ("forecast_rmse_h1", "source_rmse", "sensor_fit", "innov_ac1", "cover95_a", "rel_bias_a"):
                row[f"med_{m}"] = float(g.loc[ix, m].median())
            summ.append(row)
            if spec == "oracle":
                continue
            # H2 compensation gap: log D(source) - log D(sensor_fit) and log D(forecast) - log D(sensor_fit)
            for tgt in ("source_rmse", "forecast_rmse_h1"):
                con.append(dict(contrast=f"H2_gap_{tgt}_vs_fit", spec=spec, k=k, design=des, estimator=est,
                                **FD.paired_summary(D[tgt].loc[ix] - D["sensor_fit"].loc[ix])))
            fit_ok = np.exp(D["sensor_fit"].loc[ix]) < 1.1
            bad_fc = np.exp(D["forecast_rmse_h1"].loc[ix]) > 1.5
            for i in ix[(fit_ok & bad_fc).to_numpy()]:
                cex.append(dict(kind="deceptive_fit", spec=spec, cluster=i[0], k=k, design=des, estimator=est,
                                D_fit=float(np.exp(D["sensor_fit"][i])), D_forecast=float(np.exp(D["forecast_rmse_h1"][i]))))
            for i in ix[(D["forecast_rmse_h1"].loc[ix] < 0).to_numpy()]:
                cex.append(dict(kind="wrong_wind_better", spec=spec, cluster=i[0], k=k, design=des, estimator=est,
                                D_forecast=float(np.exp(D["forecast_rmse_h1"][i]))))
    # H3 memory: M = (map24 - map_full)/map24 per spec; contrast vs the oracle's M
    pv = st.pivot_table(index=["cluster", "k", "design", "spec"], columns="estimator", values="forecast_rmse_h1")
    M = ((pv["map24"] - pv["map_full"]) / pv["map24"]).rename("M").reset_index()
    Mo = M[M.spec == "oracle"].set_index(["cluster", "k", "design"]).M
    for (spec, k, des), g in M.groupby(["spec", "k", "design"]):
        g = g.set_index(["cluster", "k", "design"]).M
        con.append(dict(contrast="H3_memory_gain_M", spec=spec, k=k, design=des, estimator="map24->map_full", **FD.paired_summary(g)))
        if spec != "oracle":
            con.append(dict(contrast="H3_M_minus_oracleM", spec=spec, k=k, design=des, estimator="map24->map_full",
                            **FD.paired_summary(g - Mo.loc[g.index])))
            rev = g.index[((g < 0) & (Mo.loc[g.index] > 0)).to_numpy()]
            cex += [dict(kind="memory_reversal", spec=spec, cluster=i[0], k=k, design=des, M=float(g[i]), M_oracle=float(Mo[i]))
                    for i in rev]
    # H4 placement robustness: log D(dopt) - log D(source_near) (> 0: source_near more robust), and absolute rho
    for spec, g in st[st.spec != "oracle"].groupby("spec"):
        g = g.set_index(idx)
        lD = np.log(g.forecast_rmse_h1 / o.loc[g.index, "forecast_rmse_h1"])
        for (k, est), _ in g.groupby(["k", "estimator"]):
            sl = lambda d: lD.xs((k, d, est), level=["k", "design", "estimator"])
            fc = lambda d: g.forecast_rmse_h1.xs((k, d, est), level=["k", "design", "estimator"])
            for a_, b_ in (("dopt", "source_near"), ("random", "source_near"), ("random", "dopt"), ("dopt", "dopt_m")):
                con.append(dict(contrast=f"H4_robust_{a_}_minus_{b_}", spec=spec, k=k, design=f"{a_}|{b_}", estimator=est,
                                **FD.paired_summary(sl(a_) - sl(b_))))
            con.append(dict(contrast="H4_rho_dopt_vs_source_near", spec=spec, k=k, design="dopt|source_near", estimator=est,
                            **FD.paired_summary((fc("dopt") - fc("source_near")) / fc("dopt"))))
    con += baseline_contrasts(st, bl)
    return pd.DataFrame(summ), pd.DataFrame(con), pd.DataFrame(cex)


def report(pre="local"):
    df = pd.read_csv(tab / f"e12-{pre}-rows.csv")
    summ, con, cex = summarise(df)
    summ.to_csv(tab / f"e12-{pre}-summary.csv", index=False)
    con.to_csv(tab / f"e12-{pre}-contrasts.csv", index=False)
    cex.to_csv(tab / f"e12-{pre}-counterexamples.csv", index=False)
    plot(summ, pre, df[(df.spec == "baseline") & (df.design == "persistence")].forecast_rmse_h1.median())


def plot(summ, pre, persist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = [f for f in U.FAMILIES if f != "oracle"]
    k = max(summ.k)
    cols = {"source_near": "C0", "dopt": "C1", "dopt_m": "C3", "random": "C7"}
    fg, ax = plt.subplots(2, len(fams), figsize=(3.6 * len(fams), 6.8), sharey="row")
    for j, fam in enumerate(fams):
        for des, c in cols.items():
            for est, ls in (("map24", "--"), ("map_full", "-")):
                q = summ[(summ.design == des) & (summ.estimator == est) & (summ.k == k) &
                         ((summ.family == fam) | (summ.spec == "oracle"))].copy()
                q["x"] = np.where(q.spec == "oracle", 0 if fam != "spd" else 1, q.magnitude)
                q = q.sort_values("x")
                ax[0, j].errorbar(q.x, q.med_forecast_rmse_h1, fmt="o" + ls, color=c, ms=3, lw=1,
                                  label=f"{des} {est}" if j == 0 else None)
                if est == "map_full":
                    ax[1, j].errorbar(q.x, q.D_sensor_fit, fmt="s:", color=c, ms=3, lw=1)
                    ax[1, j].errorbar(q.x, q.D_source_rmse, fmt="^-", color=c, ms=3, lw=1)
        ax[0, j].set_title({"dir": "direction bias (deg)", "spd": "speed factor", "lag": "model lag (h)",
                            "ar": "AR(1) direction sd (deg)"}[fam], fontsize=9)
        ax[0, j].axhline(persist, color="k", lw=.8, ls=":", label="persistence (median)" if j == 0 else None)
        ax[1, j].axhline(1, color="k", lw=.6)
        ax[1, j].set_yscale("log")
        if fam == "spd":
            for a in ax[:, j]:
                a.invert_xaxis()
    ax[0, 0].set_ylabel(f"median held-out 1 h forecast RMSE (µg m⁻³), k={k}")
    ax[1, 0].set_ylabel("ratio to oracle (map_full)\n▲ source RMSE   ■ sensor fit")
    ax[0, 0].legend(fontsize=6)
    what = "32² pilot (6 clusters" if pre == "local" else "64² sweep (20 clusters, VanLeer"
    fg.suptitle(f"E12 {what}): wrong-wind inverse, medians. Solid map_full, dashed map24", fontsize=10)
    fg.tight_layout()
    fg.savefig(fig_dir / f"e12-{pre}-bias.png", dpi=130)
    plt.close(fg)


def provenance():
    git = lambda *c: subprocess.run(["git", *c], cwd=root, capture_output=True, text=True).stdout.strip()
    return dict(command="OMP_NUM_THREADS=1 uv run python scripts/e12_wind_mismatch.py " + " ".join(sys.argv[1:]),
                git_rev=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain", "--", "src", "scripts")),
                host=platform.node(), python=platform.python_version(), numpy=np.__version__,
                slurm_job=os.environ.get("SLURM_JOB_ID"), design_note="docs/research/wind-mismatch-design.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--stage2-workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fields", default=None, help="read-only cache with truth/oracle fields (e.g. the E07 sweep's)")
    ap.add_argument("--only", default=None, help="comma list K:seed")
    ap.add_argument("--dry-run", action="store_true", help="print the solve counts (e.g. check E07 cache reuse) and exit")
    a = ap.parse_args()
    scale = "sweep" if a.sweep else "pilot"
    pre = "local" if scale == "pilot" else "sweep"
    if a.report:
        return report(pre)
    out = Path(a.out or root / f".cache/e12/{scale}")
    (out / "fields").mkdir(parents=True, exist_ok=True)
    s = SCALES[scale]
    jobs = [(scale, K, sd) for K in sorted(KS_SRC, reverse=True) for sd in s["seeds"]]
    if a.only:
        sel = {tuple(map(int, x.split(":"))) for x in a.only.split(",")}
        jobs = [j for j in jobs if (j[1], j[2]) in sel]
    dirs = [out / "fields"] + ([Path(a.fields)] if a.fields else [])
    tasks = {n: sp for j in jobs for v in field_names(*j, s["specs"]).values() for n, sp in v}
    todo = [(n, sp) for n, sp in tasks.items() if not any((d / f"{n}.npy").exists() for d in dirs)]
    t0, ctx = time.time(), get_context("fork")
    print(f"stage 1: {len(tasks)} fields, {len(todo)} to solve on {a.workers} workers", flush=True)
    if a.dry_run:  # a sweep reusing the E07 cache must show only perturbed solves here (60 per spec)
        return print({k: sum(sp[0] == k or (k == "pert" and sp[0] == "pert") for _, sp in todo) for k in ("truth", "src", "pert")})
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex:
        futs = [ex.submit(run_field, (t, str(out))) for t in todo]
        for i, f in enumerate(as_completed(futs)):
            if (tm := f.result()) is not None:
                print(f"[{i + 1}/{len(todo)}] {tm['kind']} K{tm['K']} s{tm['seed']} k={tm['k']} wall={tm['wall_s']:.0f}s", flush=True)
    rows, comp = [], []
    with ProcessPoolExecutor(a.stage2_workers or a.workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_scenario, (*j, str(out), a.fields)): j for j in jobs}
        for f in as_completed(futs):
            r, _, tm = f.result()
            rows += r
            comp.append(tm)
            print(f"stage 2 done {futs[f][1:]} wall={tm['total_wall_s']:.0f}s", flush=True)
    st1 = [json.loads((d / f"{n}.json").read_text()) for n in tasks for d in dirs if (d / f"{n}.json").exists()]
    meta = dict(**provenance(), scale=scale, specs=[sp.name for sp in s["specs"]], ks=s["ks"], seeds=s["seeds"], Ks=KS_SRC,
                R=R, n_random=N_RAND, estimators=ESTS, wind=WIND, out=str(out), fields=a.fields, workers=a.workers,
                total_wall_s=time.time() - t0, stage1_cpu_s=sum(x["cpu_s"] for x in st1),
                stage1_cpu_s_by_kind={k: sum(x["cpu_s"] for x in st1 if x["kind"] == k) for k in {x["kind"] for x in st1}})
    dest = tab if scale == "pilot" and not a.only else out
    pd.DataFrame(rows).to_csv(dest / f"e12-{pre}-rows.csv", index=False)
    pd.DataFrame(comp).to_csv(dest / f"e12-{pre}-compute.csv", index=False)
    pd.DataFrame(st1).to_csv(dest / f"e12-{pre}-solves.csv", index=False)
    (dest / f"e12-{pre}-meta.json").write_text(json.dumps(meta, indent=1, default=str))
    if dest == tab:
        report(pre)


if __name__ == "__main__":
    main()

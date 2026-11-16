"""E13 non-wind model mismatch (design: docs/research/model-mismatch-design.md). The inverse/forecast model is the fixed
nominal E07 model; each spec changes the WORLD (truth emissions/background, sensor readings) via pravah.model_mismatch.

Pilot (32², local):  OMP_NUM_THREADS=1 uv run python scripts/e13_model_mismatch.py --pilot --workers 4
Sweep (64², one job): OMP_NUM_THREADS=1 uv run python scripts/e13_model_mismatch.py --sweep --fields E07OUT/fields \
                         --out DIR --workers 64 --stage2-workers 32
Check reuse first:    ... --sweep --fields E07OUT/fields --out DIR --dry-run   (expect truth 0, src 0, e13 140)
Report only:          uv run python scripts/e13_model_mismatch.py --report

Stage 1 (cached atomically in OUT/fields): truth and nominal responses use E07's content-hashed names (reused read-only
via --fields); each truth-side solve is '<E07 src name>_e13_<key>_<k>_K<K>_s<seed>'. Stage 2: one task per scenario.
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
from pravah import benchmarks as B  # noqa: E402
from pravah import model_mismatch as M  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402
from pravah.experiments import forecast_design as FD  # noqa: E402

WIND = "mixed"
SCALES = {"pilot": dict(specs=M.SPECS, ks=(2, 8), seeds=(0, 1, 2)),
          "sweep": dict(specs=M.SWEEP_SPECS, ks=(4, 8, 16), seeds=tuple(range(10)))}
KS_SRC, R, N_RAND = (2, 4), 3, 5
ESTS = {"map24": (24, False), "map_full": (None, False), "map24_bg": (24, True), "map_full_bg": (None, True)}
METRICS = ["forecast_rmse_h1", "forecast_rmse_h6", "field_rmse", "source_rmse", "source_bias", "cover95_a", "false_conf",
           "sensor_fit", "innov_rms", "innov_ac1"]
tab, fig_dir = root / "reports/quarto/tables", root / "reports/quarto/figures"


def field_names(scale, K, seed, specs):
    """{'truth': [...], 'oracle': [...], 'e13': [...]}: E07 names for the nominal world, one truth-side solve per key."""
    fl = E7.scenario_fields(scale, WIND, K, seed)
    cfg = E7.make_cfg(scale, WIND)
    sxy = B.scenario_drivers(cfg, seed, K)[2]
    keys = M.truth_solves(cfg, seed, K, sxy, specs)
    return {"truth": [fl[0]], "oracle": fl[1:],
            "e13": [(f"{fl[1][0]}_e13_{key}_{k}_K{K}_s{seed}", ("e13", scale, K, seed, key, k)) for key, k in keys]}


def run_field(task):
    (name, spec), out = task
    if spec[0] != "e13":
        return E7.run_field(((name, spec), out))
    f = Path(out) / "fields" / f"{name}.npy"
    if f.exists():
        return None
    _, scale, K, seed, key, k = spec
    w0, c0 = time.time(), time.process_time()
    cfg = E7.make_cfg(scale, WIND)
    _, _, sxy, winds, _, _ = B.scenario_drivers(cfg, seed, K)
    field = M.unit_field(cfg, *M.truth_solves(cfg, seed, K, sxy, SCALES[scale]["specs"])[(key, k)], winds)
    tm = dict(name=name, kind=key, wind=WIND, K=K, seed=seed, k=k, wall_s=time.time() - w0, cpu_s=time.process_time() - c0,
              peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, host=platform.node(), pid=os.getpid())
    E7._save_atomic(f.with_suffix(".json"), lambda fh: fh.write(json.dumps(tm).encode()))
    E7._save_atomic(f, lambda fh: np.save(fh, field))
    return tm


def _find(name, dirs):
    for d in dirs:
        if (Path(d) / f"{name}.npy").exists():
            return Path(d) / f"{name}.npy"
    raise FileNotFoundError(name)


def run_scenario(job):
    scale, K, seed, out, fields = job
    f = Path(out) / "scen" / f"e13_{WIND}_K{K}_s{seed}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    w0, c0 = time.time(), time.process_time()
    dirs = [Path(out) / "fields"] + ([Path(fields)] if fields else [])
    specs = SCALES[scale]["specs"]
    names = field_names(scale, K, seed, specs)
    sc = B.generate_scenario(E7.make_cfg(scale, WIND), seed, K, truth_field=np.load(_find(names["truth"][0][0], dirs)))
    F = np.stack([np.load(_find(n, dirs)) for n, _ in names["oracle"]])
    tfields = {sp[4:]: np.load(_find(n, dirs)) for n, sp in names["e13"]}
    rows, tm = evaluate(scale, K, seed, sc, F, tfields, specs)
    tm.update(total_wall_s=time.time() - w0, total_cpu_s=time.process_time() - c0,
              peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    f.parent.mkdir(exist_ok=True)
    E7._save_atomic(f, lambda fh: fh.write(pickle.dumps((rows, tm))))
    return rows, tm


def evaluate(scale, K, seed, sc, F, tfields, specs):
    """Designs and estimators on the NOMINAL responses F (identical for every spec); the spec's world (M.world) enters
    only as readings (estimator input) and as the scored truth."""
    ks = SCALES[scale]["ks"]
    a_ref, cbg = sc.cfg.sources.reference_rate_ug_s, sc.cfg.sources.background_ug_m3
    T, last, issue = len(sc.truth.winds), sc.split_hours("test")[-1], X.issue_times(sc)
    N = int(np.prod(sc.cfg.model.grid))
    a_true = sc.truth.strengths_ug_s / a_ref
    sigma = X.site_sigma(sc, False)
    G3 = X.site_responses(sc, F)
    Gs = {k: X.site_responses(sc, v[None])[:, :, 0] for k, v in tfields.items()}
    Fr = a_ref * F.reshape(K, T, N)
    Fr_bg = np.concatenate([Fr, np.full((1, T, N), M.BG_PRIOR_SD)])
    tm = dict(scale=scale, wind=WIND, K=K, seed=seed, a_true=" ".join(f"{x:.4g}" for x in a_true),
              linearity_max_abs=float(np.abs(np.einsum("k,khn->hn", a_true, Fr) - (sc.truth.field.reshape(T, N) - cbg)).max()))
    p0 = X.problem(X.design_input(sc, G3, sigma))
    T_train, seedp = len(sc.split_hours("train")), sc.cfg.seeds.streams(seed)["placement"]
    designs = {(k, n, j): X.make_design(n, p0, k, T_train, seedp, j)[0]
               for k in ks for n, j in [("random", j) for j in range(N_RAND)] + [("dopt", 0), ("source_near", 0)]}
    key = dict(scale=scale, wind=WIND, K=K, seed=seed)
    rows = []
    for name in specs:
        w = [M.world(sc, name, tfields, Gs, sigma, r, specs) for r in range(R)]
        tf, Y = w[0][0], np.stack([x[1] for x in w], 2)          # (T, N), (sites, T, R)
        for h in (1, 6):  # model-free references on the spec's own truth
            ok = issue + h <= last
            for d, e in (("persistence", tf[issue[ok] + h] - tf[issue[ok]]), ("background_only", tf[issue[ok] + h])):
                rows.append(dict(**key, spec=name, k=0, design=d, net=0, estimator="-", horizon=h,
                                 forecast_rmse=float(np.sqrt(np.mean(e ** 2)))))
        for (k, dname, net), S in designs.items():
            A = a_ref * G3[S] / sigma[S][:, None, None]
            z = (Y[S] - cbg) / sigma[S][:, None, None]
            obs = np.isfinite(z[..., 0])
            for est, (win, bg) in ESTS.items():
                bcol = M.BG_PRIOR_SD / sigma[S] if bg else None
                u, C = M.map_masked(A, z, issue, win, bcol)
                Ae = np.concatenate([A, np.broadcast_to(bcol[:, None, None], A.shape[:2] + (1,))], 2) if bg else A
                Fe = Fr_bg if bg else Fr
                ae = np.r_[a_true, 0.0] if bg else a_true
                m = [FD.score(u[..., r], Fe, tf, issue, ae, last, horizons=(1, 6)) for r in range(R)]
                ua, sd = u[:, :K], np.sqrt(np.einsum("tkk->tk", C))[:, :K, None]
                err = ua - a_true[:, None]
                wrong = np.abs(err) > 1.96 * sd
                zz = np.where(np.isfinite(z), z, 0.0)
                res = (zz[:, issue] - np.einsum("sik,ikr->sir", Ae[:, issue], u))[obs[:, issue]]
                inn = np.where(obs[:, issue + 1, None], zz[:, issue + 1] - np.einsum("sik,ikr->sir", Ae[:, issue + 1], u), np.nan)
                ic = inn - np.nanmean(inn, 1, keepdims=True)
                both = np.isfinite(ic[:, 1:]) & np.isfinite(ic[:, :-1])
                ac1 = np.nansum(np.where(both, ic[:, 1:] * ic[:, :-1], 0)) / np.nansum(ic ** 2)
                rows.append(dict(**key, spec=name, k=k, design=dname, net=net, estimator=est,
                                 forecast_rmse_h1=np.mean([x["forecast_rmse_h1"] for x in m]),
                                 forecast_rmse_h6=np.mean([x["forecast_rmse_h6"] for x in m]),
                                 field_rmse=np.mean([x["analysis_rmse"] for x in m]),
                                 source_rmse=float(np.sqrt((err ** 2).mean((0, 1))).mean()),
                                 source_bias=float((err / a_true[:, None]).mean()),
                                 cover95_a=float((~wrong).mean()),
                                 false_conf=float((wrong & (1.96 * sd < .25 * np.abs(ua))).mean()),
                                 sensor_fit=float(np.sqrt((res ** 2).mean())), innov_rms=float(np.sqrt(np.nanmean(inn ** 2))),
                                 innov_ac1=float(ac1), obs_frac=float(obs.mean()),
                                 bg_hat=float(u[:, -1].mean()) if bg else np.nan, sites=" ".join(map(str, S))))
    return rows, tm


# ---------------------------------------------------------------- analysis (reads only the row table)
def summarise(df):
    df = df.copy()
    df["cluster"] = df.K.astype(str) + ":" + df.seed.astype(str)
    bl = df[df.estimator == "-"]
    pe = bl[(bl.design == "persistence") & (bl.horizon == 1)].set_index(["spec", "cluster"]).forecast_rmse
    st = df[df.estimator != "-"].groupby(["spec", "cluster", "k", "design", "estimator"], as_index=False)[METRICS].mean()
    idx = ["cluster", "k", "design", "estimator"]
    o = st[st.spec == "oracle"].set_index(idx)
    ac_max = o.groupby(["k", "design", "estimator"]).innov_ac1.max()
    summ, con, cex = [], [], []
    for (spec, k, des, est), g in st.groupby(["spec", "k", "design", "estimator"]):
        g = g.set_index("cluster")
        oo = o.xs((k, des, est), level=["k", "design", "estimator"]).loc[g.index]
        p = pe.loc[spec].loc[g.index]
        sk = FD.paired_summary((p - g.forecast_rmse_h1) / p)
        D = {m: g[m] / oo[m] for m in ("forecast_rmse_h1", "field_rmse", "source_rmse", "sensor_fit")}
        row = dict(spec=spec, k=k, design=des, estimator=est, n_clusters=len(g),
                   skill_h1=sk["median"], skill_lo=sk["ci_lo"], skill_hi=sk["ci_hi"], skill_wins=sk["wins"],
                   **{f"med_{m}": float(g[m].median()) for m in METRICS},
                   **{f"lo_{m}": float(g[m].min()) for m in ("cover95_a", "innov_ac1", "source_bias")},
                   **{f"hi_{m}": float(g[m].max()) for m in ("cover95_a", "innov_ac1", "source_bias")},
                   **{f"D_{m}": float(np.median(v)) for m, v in D.items()})
        summ.append(row)
        # post hoc field-level rule (2026-11-16): fit ratio < 1.1, |lag-1| < 0.1, worse than persistence, AND >1.5x the matched
        # oracle forecast (so cells already worse than persistence under the nominal world are not flagged); cell-median level
        if (spec != "oracle" and row["D_sensor_fit"] < 1.1 and abs(row["med_innov_ac1"]) < .1
                and row["skill_h1"] < 0 and row["D_forecast_rmse_h1"] > 1.5):   # skill_h1 = median paired skill vs persistence
            cex.append(dict(kind="field_silent_failure", spec=spec, k=k, design=des, estimator=est,
                            D_fit=row["D_sensor_fit"], D_forecast=row["D_forecast_rmse_h1"], innov_ac1=row["med_innov_ac1"]))
        con.append(dict(contrast="skill_vs_persistence_h1", spec=spec, k=k, design=des, estimator=est, **sk))
        if spec == "oracle":
            continue
        warned = (D["sensor_fit"] > 1.1) | (g.innov_ac1 > ac_max[(k, des, est)])
        for c in g.index:
            if D["sensor_fit"][c] < 1.1 and (D["forecast_rmse_h1"][c] > 1.5 or g.cover95_a[c] < .5):
                cex.append(dict(kind="deceptive_fit", spec=spec, cluster=c, k=k, design=des, estimator=est,
                                D_fit=D["sensor_fit"][c], D_forecast=D["forecast_rmse_h1"][c], cover95=g.cover95_a[c]))
            if not warned[c] and g.cover95_a[c] < .5:
                cex.append(dict(kind="silent_failure", spec=spec, cluster=c, k=k, design=des, estimator=est,
                                D_fit=D["sensor_fit"][c], innov_ac1=g.innov_ac1[c], cover95=g.cover95_a[c]))
    sti = st.set_index(["spec", "cluster", "k", "design", "estimator"])
    for (spec, k, des), g in st.groupby(["spec", "k", "design"]):
        f = lambda e: sti.xs((spec, k, des, e), level=["spec", "k", "design", "estimator"]).forecast_rmse_h1
        for a_, b_ in (("map_full", "map_full_bg"), ("map24", "map24_bg"), ("map24", "map_full")):
            con.append(dict(contrast=f"rho_{b_}_vs_{a_}_h1", spec=spec, k=k, design=des, estimator=f"{a_}->{b_}",
                            **FD.paired_summary((f(a_) - f(b_)) / f(a_))))
    s = pd.DataFrame(summ)
    best = s.groupby(["spec", "k", "design"]).skill_h1.max()
    cex += [dict(kind="no_estimator_beats_persistence", spec=sp, k=k, design=d, best_skill_h1=v)
            for (sp, k, d), v in best.items() if v <= 0]
    return s, pd.DataFrame(con), pd.DataFrame(cex)


def report(pre="local"):
    df = pd.read_csv(tab / f"e13-{pre}-rows.csv")
    summ, con, cex = summarise(df)
    summ.to_csv(tab / f"e13-{pre}-summary.csv", index=False)
    con.to_csv(tab / f"e13-{pre}-contrasts.csv", index=False)
    cex.to_csv(tab / f"e13-{pre}-counterexamples.csv", index=False)
    plot(summ, pre)


def plot(summ, pre):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    specs = [s for s in SCALES["pilot" if pre == "local" else "sweep"]["specs"]]
    k = max(summ.k)
    x = np.arange(len(specs))
    panels = [("skill_h1", "skill vs persistence, 1 h\n(median, cluster range)", ("skill_lo", "skill_hi")),
              ("med_cover95_a", "95 % coverage of a_k", ("lo_cover95_a", "hi_cover95_a")),
              ("D_sensor_fit", "sensor-fit ratio\nto oracle", None),
              ("med_innov_ac1", "lag-1 innovation\nautocorrelation", ("lo_innov_ac1", "hi_innov_ac1")),
              ("med_source_bias", "relative source bias", ("lo_source_bias", "hi_source_bias"))]
    ests = {"map24": "C0", "map_full": "C1", "map24_bg": "C2", "map_full_bg": "C3"}
    fg, ax = plt.subplots(len(panels), 1, figsize=(12, 13), sharex=True)
    for i, (col, lab, rng) in enumerate(panels):
        for j, (est, c) in enumerate(ests.items()):
            for dm, (des, mk) in enumerate((("dopt", "o"), ("source_near", "s"))):
                q = summ[(summ.k == k) & (summ.design == des) & (summ.estimator == est)].set_index("spec").reindex(specs)
                xx = x + (j * 2 + dm - 3.5) * .08
                yerr = None if rng is None else np.abs(np.stack([q[col] - q[rng[0]], q[rng[1]] - q[col]]))
                ax[i].errorbar(xx, q[col], yerr=yerr, fmt=mk, color=c, ms=3.5, lw=.7, mfc=c if dm == 0 else "white",
                               label=f"{est} / {des}" if i == 0 else None)
        ax[i].set_ylabel(lab, fontsize=8)
        ax[i].grid(axis="y", lw=.3)
        ax[i].axhline({"skill_h1": 0, "med_cover95_a": .95, "D_sensor_fit": 1, "med_innov_ac1": 0, "med_source_bias": 0}[col],
                      color="k", lw=.7, ls=":")
    ax[2].set_yscale("log")
    ax[0].set_ylim(max(-3, summ.skill_lo.min() - .1), 1.05)
    ax[0].legend(fontsize=6, ncol=4, loc="lower left")
    ax[-1].set_xticks(x, specs, rotation=45, ha="right", fontsize=8)
    fg.suptitle(f"E13 {'32² pilot' if pre == 'local' else '64²'}: non-wind model mismatch, k = {k} "
                f"(median over {summ.n_clusters.max()} clusters; bars = cluster min–max)", fontsize=10)
    fg.tight_layout()
    fg.savefig(fig_dir / f"e13-{pre}-uncertainty.png", dpi=130)
    plt.close(fg)


def provenance():
    git = lambda *c: subprocess.run(["git", *c], cwd=root, capture_output=True, text=True).stdout.strip()
    return dict(command="OMP_NUM_THREADS=1 uv run python scripts/e13_model_mismatch.py " + " ".join(sys.argv[1:]),
                git_rev=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain", "--", "src", "scripts")),
                host=platform.node(), python=platform.python_version(), numpy=np.__version__,
                slurm_job=os.environ.get("SLURM_JOB_ID"), design_note="docs/research/model-mismatch-design.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage2-workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fields", default=None, help="read-only cache with truth/oracle fields (e.g. the E07 sweep's)")
    ap.add_argument("--only", default=None, help="comma list K:seed")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    scale = "sweep" if a.sweep else "pilot"
    pre = "local" if scale == "pilot" else "sweep"
    if a.report:
        return report(pre)
    out = Path(a.out or root / f".cache/e13/{scale}")
    (out / "fields").mkdir(parents=True, exist_ok=True)
    s = SCALES[scale]
    jobs = [(scale, K, sd) for K in sorted(KS_SRC, reverse=True) for sd in s["seeds"]]
    if a.only:
        sel = {tuple(map(int, x.split(":"))) for x in a.only.split(",")}
        jobs = [j for j in jobs if (j[1], j[2]) in sel]
    dirs = [out / "fields"] + ([Path(a.fields)] if a.fields else [])
    tasks = {n: sp for j in jobs for v in field_names(*j, s["specs"]).values() for n, sp in v}
    todo = [(n, sp) for n, sp in tasks.items() if not any((d / f"{n}.npy").exists() for d in dirs)]
    print(f"stage 1: {len(tasks)} fields, {len(todo)} to solve on {a.workers} workers", flush=True)
    if a.dry_run:
        return print({k: sum(sp[0] == k for _, sp in todo) for k in ("truth", "src", "e13")})
    t0, ctx = time.time(), get_context("fork")
    todo.sort(key=lambda t: t[1][0] != "truth")
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex:
        futs = [ex.submit(run_field, (t, str(out))) for t in todo]
        for i, f in enumerate(as_completed(futs)):
            if (tm := f.result()) is not None:
                print(f"[{i + 1}/{len(todo)}] {tm['kind']} K{tm['K']} s{tm['seed']} k={tm['k']} wall={tm['wall_s']:.0f}s", flush=True)
    rows, comp = [], []
    with ProcessPoolExecutor(a.stage2_workers or a.workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_scenario, (*j, str(out), a.fields)): j for j in jobs}
        for f in as_completed(futs):
            r, tm = f.result()
            rows += r
            comp.append(tm)
            print(f"stage 2 done {futs[f][1:]} wall={tm['total_wall_s']:.0f}s", flush=True)
    st1 = [json.loads((d / f"{n}.json").read_text()) for n in tasks for d in dirs if (d / f"{n}.json").exists()]
    meta = dict(**provenance(), scale=scale, specs={n: list(p) for n, p in s["specs"].items()}, ks=s["ks"], seeds=s["seeds"],
                Ks=KS_SRC, R=R, n_random=N_RAND, estimators=ESTS, wind=WIND, out=str(out), fields=a.fields,
                workers=a.workers, total_wall_s=time.time() - t0, stage1_cpu_s=sum(x["cpu_s"] for x in st1),
                stage1_cpu_s_by_kind={k: sum(x["cpu_s"] for x in st1 if x["kind"] == k) for k in {x["kind"] for x in st1}})
    dest = tab if scale == "pilot" and not a.only else out
    pd.DataFrame(rows).to_csv(dest / f"e13-{pre}-rows.csv", index=False)
    pd.DataFrame(comp).to_csv(dest / f"e13-{pre}-compute.csv", index=False)
    pd.DataFrame(st1).to_csv(dest / f"e13-{pre}-solves.csv", index=False)
    (dest / f"e13-{pre}-meta.json").write_text(json.dumps(meta, indent=1, default=str))
    if dest == tab:
        report(pre)


if __name__ == "__main__":
    main()

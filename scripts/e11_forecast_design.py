"""E11 placement for causal forecasting (design: docs/research/forecast-design-placement.md).

Pilot (32^2, local):  OMP_NUM_THREADS=1 uv run python scripts/e11_forecast_design.py --workers 5
Report only:          uv run python scripts/e11_forecast_design.py --report [--rows CSV --prefix e11-sweep]
64^2 (stage 2 on cached E07 sweep fields + design-prior solves):
                      ... --sweep --fields E07OUT/fields --out DIR --workers 64

v2 (64^2 addenda A1-A5 of the design note, gate-10/11): stage 1 reuses the E07 driver (truth and unit responses, cached
by content hash) and adds one INDEPENDENT design-prior block per source x direction (FD.prior_block: 24 h steady wind after
its own spin-up, cached by content hash in OUT/rob), so every subset prior (FD.PRIORS: all8 and the out-of-sample noSN) is
free. Stage 2 per scenario: E07 baselines and the forecast designs (multi-start 1-swap from greedy, dopt, source_near) from
training information only, then map24 and the growing-window MAP on R paired white-noise draws, scored on the held-out test
hours at h = 1, 6, 24. Truth enters only through FD.score and the diagnostic pred_true columns. Statistics per k with
cluster = (K, seed). Writes reports/quarto/tables/e11-local-v2-* (pilot); the v1 pilot tables e11-local-* are kept.
"""
import argparse, hashlib, importlib.util, json, multiprocessing as mp, os, pickle, resource, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from pravah import benchmarks as B  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402
from pravah.experiments import forecast_design as FD  # noqa: E402

_spec = importlib.util.spec_from_file_location("e07_driver", root / "scripts" / "e07_cross_objective.py")
E7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E7)

WINDS = ("veering", "mixed")
# rob_spinup: spin-up of each design-prior block (A1; 12 h at 64^2 from scripts/e11_spinup_check.py)
SCALE = {"pilot": dict(Ks=(2, 4), seeds=(0, 1, 2), ks=(2, 4, 8), R=5, rob_spinup=12),
         "sweep": dict(Ks=(2, 4), seeds=tuple(range(10)), ks=(4, 8, 16), R=5, rob_spinup=12)}
BASE = [("random", j) for j in range(X.N_RANDOM)] + [("coverage", 0), ("source_near", 0), ("dopt", 0)]
EST = {"map24": X.WINDOW_H, "map_grow": None}
tab, fig_dir = root / "reports/quarto/tables", root / "reports/quarto/figures"


# ---------------------------------------------------------------- stage 1: design-prior blocks (source x direction)
def rob_fields(scale, K, seed):
    """[(cache name, (scale, xy, d))] for every source of (K, seed) and every direction d, source-major."""
    cfg = E7.make_cfg(scale, WINDS[0])  # model/sources identical across wind histories; source layout depends on (K, seed)
    cfg.model.spinup_h = SCALE[scale]["rob_spinup"]
    _, _, sxy, _, _, _ = B.scenario_drivers(cfg, seed, K)
    base = json.dumps({"model": asdict(cfg.model), "sources": asdict(cfg.sources)}, sort_keys=True).encode()
    dw = FD.design_winds()
    h = lambda xy, d: hashlib.sha256(base + dw[d * FD.BLOCK_H:(d + 1) * FD.BLOCK_H].tobytes() + xy.tobytes()).hexdigest()[:20]
    return [(f"robd_{h(xy, d)}", (scale, xy, d)) for xy in sxy for d in range(FD.N_DIRS)]


def run_field(task):
    return E7.run_field(task)  # module-level wrapper: E7 is loaded from a path, so its functions do not pickle


def run_rob(task):
    (name, (scale, xy, d)), out = task
    f = Path(out) / "rob" / f"{name}.npy"
    if f.exists():
        return None
    w0, c0 = time.time(), time.process_time()
    field = FD.prior_block(E7.make_cfg(scale, WINDS[0]), xy, d, SCALE[scale]["rob_spinup"])
    tm = dict(name=name, direction=d, wall_s=time.time() - w0, cpu_s=time.process_time() - c0)
    E7._save_atomic(f.with_suffix(".json"), lambda fh: fh.write(json.dumps(tm).encode()))
    E7._save_atomic(f, lambda fh: np.save(fh, field))
    return tm


# ---------------------------------------------------------------- stage 2
def run_scenario(job, fields=None):
    scale, wind, K, seed, out = job
    fl = E7.scenario_fields(scale, wind, K, seed)
    f = Path(out) / "scen" / f"v2_{wind}_K{K}_s{seed}_{fl[0][0]}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    w0 = time.time()
    sc, F, G3, tm = E7.load_scenario(scale, wind, K, seed, out, fields)
    rb = [np.load(Path(out) / "rob" / f"{n}.npy") for n, _ in rob_fields(scale, K, seed)]
    blocks = np.stack(rb).reshape((K, FD.N_DIRS) + rb[0].shape)  # (K, dirs, 24, nx, ny)
    res = evaluate(scale, wind, K, seed, sc, F, G3, blocks, tm, w0)
    f.parent.mkdir(exist_ok=True)
    E7._save_atomic(f, lambda fh: fh.write(pickle.dumps(res)))
    return res


def evaluate(scale, wind, K, seed, sc, F, G3, blocks, tm, w0):
    s = SCALE[scale]
    a_ref, c_bg = sc.cfg.sources.reference_rate_ug_s, sc.cfg.sources.background_ug_m3
    n_h, T = len(sc.truth.winds), len(sc.split_hours("train"))
    issue, last = X.issue_times(sc), int(sc.split_hours("test")[-1])
    sigma = X.site_sigma(sc, False)
    sig = float(sigma[0])
    key = dict(scale=scale, wind=wind, K=K, seed=seed)
    # designs: training information only (DesignInput + ForecastInput)
    w1 = time.time()
    di = X.design_input(sc, G3, sigma)
    p = X.problem(di)
    objs = {}
    for pr, dirs in FD.PRIORS.items():
        Fp = FD.prior_fields(blocks, dirs)
        objs[pr] = FD.Objective(p, FD.forecast_input(di, F, X.site_responses(sc, Fp), Fp))
    obj = objs["all8"]
    fc = [(n, n, "all8") for n in FD.FC] + [(f"{n}_{pr}", n, pr) for pr in FD.PRIORS if pr != "all8" for n in FD.ROB]
    seedp = sc.cfg.seeds.streams(seed)["placement"]
    designs = {}
    for k in s["ks"]:
        for name, net in BASE:
            designs[(k, name, net)] = X.make_design(name, p, k, T, seedp, net)
        starts = {n: designs[(k, n, 0)][0] for n in ("dopt", "source_near")}
        for lab, name, pr in fc:
            designs[(k, lab, 0)] = FD.make_fc_design(name, p, objs[pr], k, starts)
    tm["design_wall_s"] = time.time() - w1
    # evaluation (truth enters only via FD.score and the per-regime split of test hours)
    w2 = time.time()
    Fr = a_ref * F.reshape(K, n_h, -1)
    tf = sc.truth.field.reshape(n_h, -1) - c_bg
    a_true = sc.truth.strengths_ug_s / a_ref
    Qall = FD.field_weights(F, a_ref)
    te0 = int(sc.split_hours("test")[0])
    regs = sc.truth.wind_regimes["test"]
    reg_of = lambda t: regs[(t - te0) // 48]
    Z = np.stack([X.readings(sc, sigma, "white", r, n_h) for r in range(s["R"])], 2)  # (sites, hours, R): paired draws
    m2 = FD.loguniform_m2(*di.prior_loguniform, K)
    rows = []
    for (k, name, net), (S, info) in designs.items():
        A = a_ref * G3[S] / sigma[S][:, None, None]
        z = (Z[S] - c_bg) / sig
        C = FD._cum(A).sum(0)
        d = dict(**key, k=k, design=name, net=net, sites=" ".join(map(str, S)),
                 **{f"J_{n}": obj.value(n)(p, S) for n in FD.FC}, J_dopt=X.P.dopt(p, S),
                 **{f"J_{n}_{pr}": objs[pr].value(n)(p, S) for pr in FD.PRIORS if pr != "all8" for n in FD.ROB},
                 **{c: info.get(c) for c in ("start", "greedy_sites", "greedy_refined_value", "refined_value")})
        for est, win in EST.items():
            u = FD.estimate(A, z, issue, win)  # (n_issue, K, R)
            sc_r = [FD.score(u[..., r], Fr, tf, issue, a_true, last) for r in range(s["R"])]
            r = {m: float(np.mean([x[m] for x in sc_r])) for m in sc_r[0]}
            for g in sorted(set(regs)):  # h = 1 per held-out regime (issue t whose target t+1 lies in the block)
                sub = np.array([t for t in issue if reg_of(t + 1) == g])
                if len(sub):
                    us = u[np.searchsorted(issue, sub)]
                    r[f"rmse_h1_{g}"] = float(np.mean([FD.score(us[..., j], Fr, tf, sub, a_true, last, (1,))["forecast_rmse_h1"]
                                                       for j in range(s["R"])]))
            # data-free expected forecast RMSE of this network on the realised calendar (descriptive check of §2): N(0, I)
            # as the objective, the declared prior's second moment (A3) and, diagnostic only, the true strengths
            for h in FD.HORIZONS:
                r[f"pred_rmse_h{h}"] = float(np.sqrt(FD.expected_mse(C, Qall, issue, (h,), win, last)))
                r[f"pred_m2_rmse_h{h}"] = float(np.sqrt(FD.expected_mse(C, Qall, issue, (h,), win, last, m2)))
                r[f"pred_true_rmse_h{h}"] = float(np.sqrt(FD.expected_mse(C, Qall, issue, (h,), win, last,
                                                                          np.outer(a_true, a_true))))
            rows.append({**d, "estimator": est, **r})
    tm.update(key, eval_wall_s=time.time() - w2, total_wall_s=time.time() - w0,
              peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              swaps=sum(i.get("n_swaps", 0) for _, i in designs.values()))
    return rows, tm


# ---------------------------------------------------------------- analysis (reads only the rows table)
METRICS = ["forecast_rmse_h1", "forecast_rmse_h6", "forecast_rmse_h24", "analysis_rmse", "source_rmse"]
DESIGNS = list(FD.FC) + [f"{n}_noSN" for n in FD.ROB]
CONTRASTS = [(d, c) for d in DESIGNS for c in ("random", "coverage", "source_near", "dopt")] + \
    [("fc_w24_rob", "fc_w24"), ("fc_grow_rob", "fc_grow"), ("fc_grow", "fc_w24"), ("fc_w24_rob", "fc_grow_rob"),
     ("fc_w24_rob_noSN", "fc_w24_rob"), ("fc_grow_rob_noSN", "fc_grow_rob"), ("dopt", "source_near")]


def in_holm_family(r):
    """A2: the exploratory Holm family, declared before the 64^2 run: 1 h forecast, both estimators, every k, the six
    forecast designs vs dopt and vs source_near; cluster-level sign-test p."""
    return r.metric == "forecast_rmse_h1" and r.design in DESIGNS and r.comparator in ("dopt", "source_near")


def per_scenario(df):
    reg = [c for c in df.columns if c.startswith("rmse_h1_")]
    return df.groupby(["estimator", "wind", "K", "seed", "k", "design"])[METRICS + reg].mean().reset_index()


def summarise(df):
    """Per k only (A5): the scenario-level summary (n = scenarios) and the cluster-level one (cl_*; cluster = (K, seed),
    A2), whose verdict is the reported one. No pooled-k row."""
    st, out = per_scenario(df), []
    reg = [c for c in st.columns if c.startswith("rmse_h1_")]
    for (est, k), c in st.groupby(["estimator", "k"]):
        for m in METRICS + reg:
            piv = c.pivot_table(index=["wind", "K", "seed"], columns="design", values=m)
            for d, cmp_ in CONTRASTS:
                q = ((piv[cmp_] - piv[d]) / piv[cmp_]).dropna()
                if len(q) >= 6:
                    cl = q.index.to_frame()[["K", "seed"]].to_numpy()
                    out.append(dict(estimator=est, k=k, metric=m, design=d, comparator=cmp_, **FD.paired_summary(q),
                                    **{f"cl_{a}": b for a, b in FD.cluster_summary(q.to_numpy(), cl).items()}))
    out = pd.DataFrame(out)
    out["holm_family"] = out.apply(in_holm_family, axis=1)
    out.loc[out.holm_family, "cl_holm_p"] = X.holm(out.loc[out.holm_family, "cl_sign_p"].to_numpy())
    return out


def reproduce(df, old_src):
    """Against the committed v1 pilot rows: per design, how many rows keep their sites, and the largest metric change on
    those rows (must be 0 up to float noise: same sites, same readings, same scorer)."""
    key = ["wind", "K", "seed", "k", "design", "net", "estimator"]
    m = pd.read_csv(old_src).merge(df, on=key, suffixes=("_v1", ""))
    cols = METRICS + [f"pred_rmse_h{h}" for h in FD.HORIZONS]
    m["same_sites"] = m.sites_v1 == m.sites
    m["max_abs_diff"] = np.max([np.abs(m[c + "_v1"] - m[c]) for c in cols], 0)
    m.loc[~m.same_sites, "max_abs_diff"] = np.nan
    return m.groupby("design").agg(rows=("same_sites", "size"), same_sites=("same_sites", "sum"),
                                   max_abs_metric_diff=("max_abs_diff", "max")).reset_index()


def report(src, prefix="e11-local-v2", out_dir=tab, old_src=None):
    df = pd.read_csv(src)
    summ = summarise(df)
    summ.to_csv(out_dir / f"{prefix}-contrasts.csv", index=False)
    st = per_scenario(df)
    st.groupby(["estimator", "k", "design"])[METRICS].median().reset_index().to_csv(out_dir / f"{prefix}-medians.csv",
                                                                                   index=False)
    # §2 / A3 check: realised / data-free expected RMSE under N(0, I) (the objective's), the declared prior's second
    # moment, and (diagnostic) the true strengths; all designs and non-random designs
    rat = {f"{w}ratio_h{h}": df[f"forecast_rmse_h{h}"] / df[f"pred_{w}rmse_h{h}"] for w in ("", "m2_", "true_")
           for h in FD.HORIZONS}
    chk = df.assign(**rat)
    desc = lambda d, lab: d.groupby("estimator")[list(rat)].describe(percentiles=[.1, .5, .9]).T.assign(subset=lab)
    pd.concat([desc(chk, "all"), desc(chk[chk.design != "random"], "nonrandom")]).to_csv(out_dir / f"{prefix}-predcheck.csv")
    # A4: which start won, and the mean relative objective gain over greedy + swap
    fcd = df[df.design.isin(DESIGNS) & (df.estimator == "map24")]
    ms = fcd.groupby(["k", "design", "start"]).size().unstack(fill_value=0)
    ms["mean_rel_gain"] = ((fcd.refined_value - fcd.greedy_refined_value) / fcd.greedy_refined_value.abs()) \
        .groupby([fcd.k, fcd.design]).mean()
    ms.reset_index().to_csv(out_dir / f"{prefix}-multistart.csv", index=False)
    if old_src is not None:
        rep = reproduce(df, old_src)
        rep.to_csv(out_dir / f"{prefix}-reproduce.csv", index=False)
        print(rep.to_string(index=False))
    h1 = summ[(summ.metric == "forecast_rmse_h1") & summ.comparator.isin(["dopt", "source_near"]) & summ.design.isin(DESIGNS)]
    print(h1[["estimator", "k", "design", "comparator", "median", "wins", "losses", "cl_n", "cl_median", "cl_ci_lo",
              "cl_ci_hi", "cl_wins", "cl_losses", "cl_sign_p", "cl_holm_p", "cl_verdict"]].to_string(index=False))
    print(ms.to_string())
    plot(summ, fig_dir / f"{prefix}-forecast-design.png", sorted(df.k.unique()))


def plot(summ, path, ks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fg, ax = plt.subplots(2, 2, figsize=(10, 7.5), sharey="row")
    for i, cmp_ in enumerate(("dopt", "source_near")):
        for j, est in enumerate(EST):
            a = ax[i, j]
            for n, d in enumerate(DESIGNS):
                q = summ[(summ.estimator == est) & (summ.metric == "forecast_rmse_h1") & (summ.design == d)
                         & (summ.comparator == cmp_)]
                x = np.log2(q.k.astype(int)) + (n - 2.5) * .05
                a.errorbar(x, q.cl_median, [q.cl_median - q.cl_ci_lo, q.cl_ci_hi - q.cl_median], fmt="o", ms=4,
                           capsize=2, label=d, ls="--" if d.endswith("noSN") else "-")
            a.axhline(0, color="k", lw=.8)
            a.axhline(-.05, color="k", lw=.6, ls=":")
            a.set_xticks(np.log2(ks), [str(k) for k in ks])
            a.set_title(f"estimator {est}: vs {cmp_}", fontsize=9)
            a.set_xlabel("sensors k")
        ax[i, 0].set_ylabel(f"(RMSE_{cmp_} − RMSE_design) / RMSE_{cmp_}\nheld-out 1 h forecast")
    ax[0, 0].legend(fontsize=7)
    fg.suptitle("E11 v2 (exploratory): forecast designs vs dopt / source_near; median of (K, seed) cluster means, "
                "distribution-free 95 % CI; dashed = prior without S and N; dotted = −5 % margin", fontsize=8)
    fg.tight_layout()
    fg.savefig(path, dpi=130)
    plt.close(fg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rows", default=None, help="--report: rows CSV (default: the v2 pilot rows)")
    ap.add_argument("--prefix", default="e11-local-v2", help="--report: table/figure prefix")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fields", default=None, help="read-only E07 field cache (stage 2 only for truth/responses)")
    ap.add_argument("--only", default=None, help="comma list wind:K:seed")
    a = ap.parse_args()
    scale = "sweep" if a.sweep else "pilot"
    if a.report:
        pilot = a.prefix == "e11-local-v2"
        return report(Path(a.rows or tab / "e11-local-v2-rows.csv"), a.prefix,
                      old_src=tab / "e11-local-rows.csv" if pilot else None)
    out = Path(a.out or root / f".cache/e11/{scale}")
    for d in ("fields", "rob"):
        (out / d).mkdir(parents=True, exist_ok=True)
    s = SCALE[scale]
    jobs = [(scale, w, K, sd, str(out)) for K in sorted(s["Ks"], reverse=True) for w in WINDS for sd in s["seeds"]]
    if a.only:
        sel = {tuple(x.split(":")) for x in a.only.split(",")}
        jobs = [j for j in jobs if (j[1], str(j[2]), str(j[3])) in sel]
    t0 = time.time()
    ctx = mp.get_context("fork")  # as E08: forkserver workers lost their semaphores on the cluster
    tasks = [] if a.fields else [(t, str(out)) for t in {n: sp for j in jobs for n, sp in E7.scenario_fields(*j[:4])}.items()]
    tasks = [t for t in tasks if not (out / "fields" / f"{t[0][0]}.npy").exists()]
    rob = [(t, str(out)) for t in {n: sp for j in jobs for n, sp in rob_fields(scale, j[2], j[3])}.items()]
    rob = [t for t in rob if not (out / "rob" / f"{t[0][0]}.npy").exists()]
    print(f"stage 1: {len(tasks)} field solves, {len(rob)} design-prior block solves on {a.workers} workers", flush=True)
    rob_cpu = 0.0
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex:
        futs = [ex.submit(run_field, t) for t in tasks] + [ex.submit(run_rob, t) for t in rob]
        for i, f in enumerate(as_completed(futs)):
            tm = f.result()
            rob_cpu += tm["cpu_s"] if tm and "direction" in tm else 0
            print(f"[{i + 1}/{len(futs)}] {tm and tm['name']} wall={tm and round(tm['wall_s'])}s", flush=True)
    t1 = time.time()
    rows, comp = [], []
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_scenario, j, a.fields): j for j in jobs}
        for f in as_completed(futs):
            r, tm = f.result()
            rows += r; comp.append(tm)
            print(f"done {futs[f][1:4]} design={tm['design_wall_s']:.0f}s eval={tm['eval_wall_s']:.0f}s", flush=True)
    if a.only:
        return
    meta = dict({**E7.provenance(), "design_note": "docs/research/forecast-design-placement.md (v2: addenda A1-A5)"},
                scale=scale, world=E7.SCALES[scale], e11=s, winds=WINDS, priors=FD.PRIORS,
                design_prior=dict(n_dirs=FD.N_DIRS, block_h=FD.BLOCK_H, speed=FD.PRIOR_SPEED, spinup_h=s["rob_spinup"],
                                  independent_blocks=True),
                workers=a.workers, stage1_wall_s=t1 - t0, stage2_wall_s=time.time() - t1, rob_blocks_solved=len(rob),
                rob_cpu_s=rob_cpu)
    meta["command"] = "OMP_NUM_THREADS=1 uv run python scripts/e11_forecast_design.py " + " ".join(sys.argv[1:])
    pre, od = ("e11-sweep", out) if scale != "pilot" else ("e11-local-v2", tab)
    pd.DataFrame(rows).to_csv(od / f"{pre}-rows.csv", index=False)
    pd.DataFrame(comp).to_csv(od / f"{pre}-compute.csv", index=False)
    (od / f"{pre}-meta.json").write_text(json.dumps(meta, indent=1, default=str))
    if scale == "pilot":
        report(tab / "e11-local-v2-rows.csv", old_src=tab / "e11-local-rows.csv")


if __name__ == "__main__":
    main()

"""E15 nominal vs robust (scenario-distribution) placement (design: docs/research/robust-placement-design.md).

Pilot (32², local):  OMP_NUM_THREADS=1 uv run python scripts/e15_robust_placement.py --workers 4
Report only:         uv run python scripts/e15_robust_placement.py --report
64² (later; §10 of the design note): ... --sweep --fields E07OUT/fields[,E12OUT/fields] --out DIR --workers 32 --stage2-workers 16

Stage 1 (cached atomically under OUT): E07 truth + unit responses (veering, mixed; E07 content hashes), E12 perturbed
responses (mixed; E12 names) and the declared design-prior blocks per source (RD.block_field; content-hashed in OUT/blocks).
Stage 2 per (K, seed): every design from TRAINING information + declared blocks only; evaluation on the held-out test hours
with the oracle wind (veering, mixed) and the E12 wind errors (mixed). Truth enters only through FD.score.
"""
import argparse, hashlib, json, os, pickle, resource, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "src"), str(root / "scripts")]
import e07_cross_objective as E7  # noqa: E402
import e12_wind_mismatch as E12  # noqa: E402
from pravah import benchmarks as B  # noqa: E402
from pravah import uncertainty as U  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402
from pravah.experiments import forecast_design as FD  # noqa: E402
from pravah.experiments import robust_design as RD  # noqa: E402

WINDS = ("veering", "mixed")
SPECS = U.SWEEP_SPECS  # oracle, dir10, dir30, spd0.6, lag6, ar30
SCALE = {"pilot": dict(Ks=(2, 4), seeds=(0, 1, 2), ks=(2, 4, 8), R=5, spinup=12),
         "sweep": dict(Ks=(2, 4), seeds=tuple(range(10)), ks=(4, 8, 16), R=5, spinup=12)}
EST = {"map24": X.WINDOW_H, "map_full": None}
BASE = [("random", j) for j in range(X.N_RANDOM)] + [(n, 0) for n in ("coverage", "uniform", "podqr", "source_near", "dopt")]
E11D = [("fc_w24", "fc_w24", "all8"), ("fc_grow", "fc_grow", "all8"), ("fc_w24_rob", "fc_w24_rob", "all8"),
        ("fc_w24_rob_noSN", "fc_w24_rob", "noSN"), ("fc_grow_rob_noSN", "fc_grow_rob", "noSN")]
ROBUST = [f"rob_{c}_{m}_{e}" for e in RD.ENSEMBLES for m in RD.MEMORIES for c in RD.CRITERIA]
ALL_BLOCKS = list(dict.fromkeys([b for v in RD.ENSEMBLES.values() for b in v] + RD.EXTRA))
tab, fig_dir = root / "reports/quarto/tables", root / "reports/quarto/figures"


# ---------------------------------------------------------------- stage 1
def block_names(scale, K, seed):
    """[(cache name, (scale, xy, block))] for every source of (K, seed) and every declared block."""
    cfg = E7.make_cfg(scale, WINDS[0])
    _, _, sxy, _, _, _ = B.scenario_drivers(cfg, seed, K)
    base = json.dumps({"model": asdict(cfg.model), "sources": asdict(cfg.sources), "spinup": SCALE[scale]["spinup"]},
                      sort_keys=True).encode()
    h = lambda xy, b: hashlib.sha256(base + RD.block_winds(*b).tobytes() + xy.tobytes()).hexdigest()[:20]
    return [(f"blk_{h(xy, b)}", (scale, xy, b)) for xy in sxy for b in ALL_BLOCKS]


def run_block(task):
    (name, (scale, xy, b)), out = task
    f = Path(out) / "blocks" / f"{name}.npy"
    if f.exists():
        return None
    c0 = time.process_time()
    field = RD.block_field(E7.make_cfg(scale, WINDS[0]), xy, b, SCALE[scale]["spinup"])
    tm = dict(name=name, kind="block", block=str(b), cpu_s=time.process_time() - c0)
    E7._save_atomic(f.with_suffix(".json"), lambda fh: fh.write(json.dumps(tm).encode()))
    E7._save_atomic(f, lambda fh: np.save(fh, field))
    return tm


def run_field(task):
    return E12.run_field(task)  # E07 truth/src or E12 perturbed solve


# ---------------------------------------------------------------- stage 2
def run_cluster(job):
    scale, K, seed, out, fields = job
    f = Path(out) / "scen" / f"e15_K{K}_s{seed}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    w0 = time.time()
    names = block_names(scale, K, seed)
    by_src = {}  # names are source-major, so each block's list is in source order
    for n, (_, xy, b) in names:
        by_src.setdefault(b, []).append(np.load(Path(out) / "blocks" / f"{n}.npy"))
    blk = {b: np.stack(v) for b, v in by_src.items()}  # (K, BLOCK_H, nx, ny)
    rows, comp = [], []
    for wind in WINDS:
        fl = fields.split(",") if fields else []  # first: E07 truth/src cache; further: e.g. E12 perturbed cache
        sc, F, G3, tm = E7.load_scenario(scale, wind, K, seed, out, fl[0] if fl else None)
        Fm = {}
        if wind == "mixed":
            dirs = [Path(out) / "fields"] + [Path(d) for d in fl]
            nm = E12.field_names(scale, K, seed, SPECS)
            Fm = {sp: np.stack([np.load(E12._find(n, dirs)) for n, _ in nm[sp.name]]) for sp in SPECS if not sp.oracle}
        r, c = evaluate(scale, wind, K, seed, sc, F, G3, blk, Fm)
        rows += r
        comp.append(dict(c, **{k: v for k, v in tm.items()}))
    comp[-1]["cluster_wall_s"] = time.time() - w0
    comp[-1]["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    res = (rows, comp)
    f.parent.mkdir(exist_ok=True)
    E7._save_atomic(f, lambda fh: fh.write(pickle.dumps(res)))
    return res


def designs_for(scale, sc, F, G3, blk):
    """All designs (training information + declared blocks only) and their in-sample ensemble losses."""
    s = SCALE[scale]
    sigma = X.site_sigma(sc, False)
    T = len(sc.split_hours("train"))
    di = X.design_input(sc, G3, sigma)
    p = X.problem(di)
    K = F.shape[0]
    m2 = FD.loguniform_m2(*di.prior_loguniform, K)
    site = lambda Fb: X.site_responses(sc, Fb)
    robj = {}
    for e, labs in RD.ENSEMBLES.items():
        ens = RD.Ensemble.build([(b, blk[b]) for b in labs], site, di.a_ref, m2)
        for m in RD.MEMORIES:
            robj[(e, m)] = RD.RobustObjective(p, ens, m)
    # E11 objectives from the 3 m/s steady blocks (identical to FD.prior_block)
    fobj = {}
    for pr, dirs in FD.PRIORS.items():
        Fp = np.concatenate([blk[("steady", d, 3.0)] for d in dirs], 1)
        fobj[pr] = FD.Objective(p, FD.forecast_input(di, F, site(Fp), Fp))
    seedp = sc.cfg.seeds.streams(sc.replicate)["placement"]
    D, cost = {}, {}
    for k in s["ks"]:
        for name, net in BASE:
            t0 = time.process_time()
            try:
                D[(k, name, net)] = X.make_design(name, p, k, T, seedp, net)[0]
            except ValueError:  # podqr undefined for k > r*: recorded by absence (as E07)
                continue
            cost[(k, name)] = cost.get((k, name), 0) + time.process_time() - t0
        starts = {n: D[(k, n, 0)] for n in ("dopt", "source_near")}
        for lab, n, pr in E11D:
            t0 = time.process_time()
            D[(k, lab, 0)] = FD.make_fc_design(n, p, fobj[pr], k, starts)[0]
            cost[(k, lab)] = time.process_time() - t0
        for lab in ROBUST:
            _, c, m, e = lab.split("_")
            t0 = time.process_time()
            D[(k, lab, 0)] = RD.make_robust_design(p, robj[(e, m)], c, k, starts)[0]
            cost[(k, lab)] = time.process_time() - t0
    insample = {key: {f"ins_{c}_{m}_{e}": RD.AGG[c](robj[(e, m)].losses(S)) for e in RD.ENSEMBLES for m in RD.MEMORIES
                      for c in RD.CRITERIA} for key, S in D.items()}
    return D, cost, insample, sigma


def evaluate(scale, wind, K, seed, sc, F, G3, blk, Fm):
    s = SCALE[scale]
    w0 = time.time()
    D, cost, insample, sigma = designs_for(scale, sc, F, G3, blk)
    design_s = time.time() - w0
    a_ref, cbg = sc.cfg.sources.reference_rate_ug_s, sc.cfg.sources.background_ug_m3
    n_h, N = len(sc.truth.winds), int(np.prod(sc.cfg.model.grid))
    issue, last = X.issue_times(sc), int(sc.split_hours("test")[-1])
    te0, regs = int(sc.split_hours("test")[0]), sc.truth.wind_regimes["test"]
    tf = sc.truth.field.reshape(n_h, N) - cbg
    a_true = sc.truth.strengths_ug_s / a_ref
    Z = np.stack([X.readings(sc, sigma, "white", r, n_h) for r in range(s["R"])], 2)
    models = {"oracle": F, **{sp.name: f for sp, f in Fm.items()}}
    G = {n: (G3 if n == "oracle" else X.site_responses(sc, f)) for n, f in models.items()}
    reg_t = {g: np.array([t for t in issue if regs[(t + 1 - te0) // 48] == g]) for g in sorted(set(regs))}
    rows = []
    for (k, name, net), S in D.items():
        z = (Z[S] - cbg) / sigma[S][:, None, None]
        for spec, Fx in models.items():
            Fr = a_ref * Fx.reshape(K, n_h, N)
            A = a_ref * G[spec][S] / sigma[S][:, None, None]
            for est, win in EST.items():
                u = FD.estimate(A, z, issue, win)  # (n_issue, K, R)
                sr = [FD.score(u[..., r], Fr, tf, issue, a_true, last, (1, 6)) for r in range(s["R"])]
                r = {m: float(np.mean([x[m] for x in sr])) for m in ("forecast_rmse_h1", "forecast_rmse_h6", "source_rmse")}
                for g, sub in reg_t.items():
                    us = u[np.searchsorted(issue, sub)]
                    r[f"rmse_h1_{g}"] = float(np.mean([FD.score(us[..., j], Fr, tf, sub, a_true, last, (1,))["forecast_rmse_h1"]
                                                       for j in range(s["R"])]))
                rows.append(dict(scale=scale, wind=wind, K=K, seed=seed, k=k, design=name, net=net, spec=spec, estimator=est,
                                 sites=" ".join(map(str, S)), design_cpu_s=cost.get((k, name)), **insample[(k, name, net)], **r))
    return rows, dict(scale=scale, wind=wind, K=K, seed=seed, design_wall_s=design_s, eval_wall_s=time.time() - w0 - design_s)


# ---------------------------------------------------------------- analysis (reads only the rows table)
DESIGNS = ["dopt", "source_near", "coverage", "uniform", "podqr", "random", "fc_w24", "fc_grow", "fc_w24_rob",
           "fc_w24_rob_noSN", "fc_grow_rob_noSN"] + ROBUST
COMPARATORS = ("dopt", "source_near", "fc_w24_rob_noSN")
AGGS = ["nominal", "nominal_h6", "source_rmse", "cond_mean", "cond_cvar", "cond_worst", "max_regret"]


def cluster_table(df):
    """One row per (cluster, k, design, estimator) with the §6 aggregates; random nets averaged."""
    reg = [c for c in df.columns if c.startswith("rmse_h1_")]
    g = df.groupby(["wind", "K", "seed", "k", "design", "spec", "estimator"])[
        ["forecast_rmse_h1", "forecast_rmse_h6", "source_rmse"] + reg].mean().reset_index()
    o = g[g.spec == "oracle"].groupby(["K", "seed", "k", "design", "estimator"])
    out = o[["forecast_rmse_h1", "forecast_rmse_h6", "source_rmse"]].mean().rename(
        columns={"forecast_rmse_h1": "nominal", "forecast_rmse_h6": "nominal_h6"})
    c = g[g.wind == "mixed"].melt(["K", "seed", "k", "design", "spec", "estimator"], reg, "regime", "rmse").dropna()
    best = c[c.design != "random"].groupby(["K", "seed", "k", "estimator", "spec", "regime"]).rmse.min().rename("best")
    c = c.join(best, on=["K", "seed", "k", "estimator", "spec", "regime"])
    a = c.groupby(["K", "seed", "k", "design", "estimator"]).agg(
        cond_mean=("rmse", "mean"), cond_cvar=("rmse", lambda x: float(RD.cvar(x.to_numpy()))), cond_worst=("rmse", "max"),
        n_cond=("rmse", "size"))
    a["max_regret"] = (c.rmse - c.best).groupby([c.K, c.seed, c.k, c.design, c.estimator]).max()
    ins = df[(df.spec == "oracle") & (df.wind == "mixed")].groupby(["K", "seed", "k", "design", "estimator"])[
        [x for x in df.columns if x.startswith("ins_")] + ["design_cpu_s"]].mean()
    return out.join(a).join(ins).reset_index()


def summarise(ct):
    out = []
    for (est, k), g in ct.groupby(["estimator", "k"]):
        for m in AGGS:
            piv = g.pivot_table(index=["K", "seed"], columns="design", values=m)
            for cmp_ in COMPARATORS:
                for d in DESIGNS:
                    if d == cmp_ or d not in piv or cmp_ not in piv:
                        continue
                    q = ((piv[cmp_] - piv[d]) / piv[cmp_]).dropna()
                    if m == "max_regret":  # regret can be 0: use the difference in RMSE units, > 0 favours d
                        q = (piv[cmp_] - piv[d]).dropna()
                    if len(q) >= 6:  # distribution-free 95 % CI needs n >= 6
                        out.append(dict(estimator=est, k=k, metric=m, design=d, comparator=cmp_, **FD.paired_summary(q)))
    return pd.DataFrame(out)


def report(src=None, pre="e15-local"):
    df = pd.read_csv(src or tab / f"{pre}-rows.csv")
    ct = cluster_table(df)
    ct.to_csv(tab / f"{pre}-clusters.csv", index=False)
    summ = summarise(ct)
    summ.to_csv(tab / f"{pre}-contrasts.csv", index=False)
    med = ct.groupby(["estimator", "k", "design"])[AGGS + [c for c in ct.columns if c.startswith("ins_")] + ["design_cpu_s"]]
    med.median().reset_index().to_csv(tab / f"{pre}-medians.csv", index=False)
    key = summ[(summ.comparator == "dopt") & summ.metric.isin(["nominal", "cond_cvar", "cond_worst"])
               & summ.design.isin(["source_near", "fc_w24_rob_noSN"] + ROBUST)]
    print(key.pivot_table(index=["estimator", "k", "design"], columns="metric", values="median").round(3).to_string())
    what = "32² pilot" if df.scale.iloc[0] == "pilot" else "64² sweep"
    n = df.groupby(["K", "seed"]).ngroups
    plot_pareto(summ, fig_dir / ("e15-pareto-local.png" if pre == "e15-local" else f"{pre}-pareto.png"), f"{what} (exploratory): median over {n}")
    plot_sites(df, fig_dir / f"{pre.replace('-local', '')}-sites.png", what)


def plot_pareto(summ, path, what="32² pilot (exploratory): median over 6"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = sorted(summ.k.unique())
    fg, ax = plt.subplots(2, len(ks), figsize=(4.2 * len(ks), 8), sharex=True, sharey=True)
    sty = {"source_near": ("C0", "s"), "fc_w24_rob_noSN": ("C2", "D"), "fc_w24_rob": ("C2", "x"), "fc_w24": ("C5", "v"), "fc_grow": ("C5", "^")}
    for i, est in enumerate(EST):
        for j, k in enumerate(ks):
            a = ax[i, j]
            q = summ[(summ.estimator == est) & (summ.k == k) & (summ.comparator == "dopt")]
            x, y = (q[q.metric == m].set_index("design")["median"] for m in ("nominal", "cond_cvar"))
            for d in x.index.intersection(y.index):
                if d.startswith("rob_"):
                    _, c, m, e = d.split("_")
                    col = {"mean": "C1", "cvar": "C3", "worst": "C4"}[c]
                    mk = ("o" if m == "w24" else "^") if e == "noSN6" else ("$t$" if m == "w24" else "$T$")
                    a.scatter(x[d], y[d], color=col, marker=mk, s=40, label=d if (i, j) == (0, 0) else None)
                elif d in sty:
                    a.scatter(x[d], y[d], color=sty[d][0], marker=sty[d][1], s=40, label=d if (i, j) == (0, 0) else None)
            a.axhline(0, color="k", lw=.7)
            a.axvline(0, color="k", lw=.7)
            a.set_title(f"{est}, k = {k}", fontsize=9)
            a.set_xlabel("nominal gain vs dopt (oracle wind, all test)")
        ax[i, 0].set_ylabel("tail gain vs dopt: CVaR0.8 over\n6 wind specs × 3 test regimes")
    fg.legend(fontsize=6, loc="lower center", ncol=6)
    fg.suptitle(f"E15 {what} (K, seed) clusters of (L_dopt − L_design)/L_dopt; "
                "> 0 favours the design (random/coverage off-scale, in the table). o/^ = noSN6 w24/full; t/T = train3 w24/full", fontsize=8)
    fg.tight_layout(rect=(0, .08, 1, 1))
    fg.savefig(path, dpi=130)
    plt.close(fg)


def plot_sites(df, path, what="32² pilot"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K, seed, k = 4, 0, 8
    cfg = E7.make_cfg(df.scale.iloc[0], "mixed")
    sc = B.generate_scenario(cfg, seed, K, truth_field=np.zeros((len(B.scenario_drivers(cfg, seed, K)[3]),) + tuple(cfg.model.grid)))  # layout only
    ds = ["dopt", "source_near", "fc_w24_rob_noSN", "rob_cvar_w24_noSN6", "rob_worst_full_noSN6", "rob_mean_full_train3"]
    fg, ax = plt.subplots(1, len(ds), figsize=(3 * len(ds), 3.2))
    q = df[(df.wind == "mixed") & (df.K == K) & (df.seed == seed) & (df.k == k) & (df.spec == "oracle")]
    for a, d in zip(ax, ds):
        S = [int(i) for i in q[q.design == d].sites.iloc[0].split()]
        a.scatter(*sc.site_xy[sc.admissible_sites()].T / 1e3, s=2, color="0.8")
        a.scatter(*sc.truth.source_xy.T / 1e3, marker="*", s=120, color="C3")
        a.scatter(*sc.site_xy[S].T / 1e3, s=25, color="C0")
        a.set_title(d, fontsize=8)
        a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
    fg.suptitle(f"E15 {what} sites (mixed training wind, K = {K}, seed {seed}, k = {k}); ★ sources", fontsize=9)
    fg.tight_layout()
    fg.savefig(path, dpi=130)
    plt.close(fg)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage2-workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fields", default=None, help="comma list of read-only caches: E07 truth/src first, then e.g. E12 perturbed fields")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--specs", default=None, help="comma list of wind-error specs to keep (oracle always kept), e.g. dir30,lag6")
    a = ap.parse_args()
    global SPECS
    if a.specs:  # pre-declared 64^2 reduction (design note §10); fork workers inherit it
        SPECS = tuple(sp for sp in U.SWEEP_SPECS if sp.oracle or sp.name in a.specs.split(","))
    scale = "sweep" if a.sweep else "pilot"
    pre = "e15-local" if scale == "pilot" else "e15-sweep"
    if a.report:
        return report(pre=pre)
    out = Path(a.out or root / f".cache/e15/{scale}")
    for d in ("fields", "blocks"):
        (out / d).mkdir(parents=True, exist_ok=True)
    s = SCALE[scale]
    clusters = [(K, sd) for K in sorted(s["Ks"], reverse=True) for sd in s["seeds"]]
    dirs = [out / "fields"] + [Path(d) for d in (a.fields.split(",") if a.fields else [])]
    ft = {}
    for K, sd in clusters:
        for w in WINDS:
            ft.update(dict(E7.scenario_fields(scale, w, K, sd)))
        for sp, v in E12.field_names(scale, K, sd, SPECS).items():
            if sp not in ("truth", "oracle"):
                ft.update(dict(v))
    todo = [(n, sp) for n, sp in ft.items() if not any((d / f"{n}.npy").exists() for d in dirs)]
    bt = {n: sp for K, sd in clusters for n, sp in block_names(scale, K, sd)}
    btodo = [(n, sp) for n, sp in bt.items() if not (out / "blocks" / f"{n}.npy").exists()]
    print(f"stage 1: {len(todo)}/{len(ft)} fields, {len(btodo)}/{len(bt)} blocks to solve on {a.workers} workers", flush=True)
    if a.dry_run:
        return
    t0, ctx = time.time(), get_context("fork")
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex:
        futs = [ex.submit(run_field, (t, str(out))) for t in todo] + [ex.submit(run_block, (t, str(out))) for t in btodo]
        for i, f in enumerate(as_completed(futs)):
            tm = f.result()
            if i % 25 == 0 or i == len(futs) - 1:
                print(f"[{i + 1}/{len(futs)}] {tm and tm.get('name')} {time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    rows, comp = [], []
    with ProcessPoolExecutor(a.stage2_workers or a.workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_cluster, (scale, K, sd, str(out), a.fields)): (K, sd) for K, sd in clusters}
        for f in as_completed(futs):
            r, c = f.result()
            rows += r
            comp += c
            print(f"stage 2 done {futs[f]} {time.time() - t1:.0f}s", flush=True)
    st1 = [json.loads(p.read_text()) for d in (out / "fields", out / "blocks") for p in d.glob("*.json")]
    meta = dict(E12.provenance(), design_note="docs/research/robust-placement-design.md", scale=scale, world=E7.SCALES[scale],
                e15=s, winds=WINDS, specs=[sp.name for sp in SPECS], ensembles={e: [list(map(str, b)) for b in v]
                                                                              for e, v in RD.ENSEMBLES.items()},
                alpha=RD.ALPHA, workers=a.workers, stage1_wall_s=t1 - t0, stage2_wall_s=time.time() - t1,
                stage1_cpu_s=sum(x.get("cpu_s", 0) for x in st1),
                stage1_cpu_s_blocks=sum(x.get("cpu_s", 0) for x in st1 if x.get("kind") == "block"))
    meta["command"] = "OMP_NUM_THREADS=1 uv run python scripts/e15_robust_placement.py " + " ".join(sys.argv[1:])
    dest = tab if scale == "pilot" else out
    pd.DataFrame(rows).to_csv(dest / f"{pre}-rows.csv", index=False)
    pd.DataFrame(comp).to_csv(dest / f"{pre}-compute.csv", index=False)
    (dest / f"{pre}-meta.json").write_text(json.dumps(meta, indent=1, default=str))
    if dest == tab:
        report(pre=pre)


if __name__ == "__main__":
    main()

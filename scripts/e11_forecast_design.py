"""E11 placement for causal forecasting (design: docs/research/forecast-design-placement.md).

Pilot (32^2, local):  OMP_NUM_THREADS=1 uv run python scripts/e11_forecast_design.py --workers 5
Report only:          uv run python scripts/e11_forecast_design.py --report
64^2 (stage 2 on cached E07 sweep fields + design-prior solves):
                      ... --sweep --fields E07OUT/fields --out DIR --workers 64

Stage 1 reuses the E07 driver (truth and unit responses, cached by content hash) and adds one design-prior solve per
source (FD.design_winds, cached by content hash in OUT/rob). Stage 2 per scenario: E07 baselines and the four forecast
designs from training information only, then map24 and the growing-window MAP on R paired white-noise draws, scored on the
held-out test hours at h = 1, 6, 24. Truth enters only through FD.score. Writes reports/quarto/tables/e11-local-* (pilot).
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
SCALE = {"pilot": dict(Ks=(2, 4), seeds=(0, 1, 2), ks=(2, 4, 8), R=5),
         "sweep": dict(Ks=(2, 4), seeds=tuple(range(10)), ks=(4, 8, 16), R=5)}
BASE = [("random", j) for j in range(X.N_RANDOM)] + [("coverage", 0), ("source_near", 0), ("dopt", 0)]
EST = {"map24": X.WINDOW_H, "map_grow": None}
tab, fig_dir = root / "reports/quarto/tables", root / "reports/quarto/figures"


# ---------------------------------------------------------------- stage 1: design-prior solves (one per source)
def rob_fields(scale, K, seed):
    cfg = E7.make_cfg(scale, WINDS[0])  # model/sources identical across wind histories; source layout depends on (K, seed)
    _, _, sxy, _, _, _ = B.scenario_drivers(cfg, seed, K)
    base = json.dumps({"model": asdict(cfg.model), "sources": asdict(cfg.sources)}, sort_keys=True).encode()
    h = lambda xy: hashlib.sha256(base + FD.design_winds().tobytes() + xy.tobytes()).hexdigest()[:20]
    return [(f"rob_{h(xy)}", (scale, xy)) for xy in sxy]


def run_field(task):
    return E7.run_field(task)  # module-level wrapper: E7 is loaded from a path, so its functions do not pickle


def run_rob(task):
    (name, (scale, xy)), out = task
    f = Path(out) / "rob" / f"{name}.npy"
    if f.exists():
        return None
    w0 = time.time()
    dw = FD.design_winds()
    field = X.source_field(E7.make_cfg(scale, WINDS[0]), xy, dw, len(dw))
    E7._save_atomic(f, lambda fh: np.save(fh, field))
    return dict(name=name, wall_s=time.time() - w0)


# ---------------------------------------------------------------- stage 2
def run_scenario(job, fields=None):
    scale, wind, K, seed, out = job
    fl = E7.scenario_fields(scale, wind, K, seed)
    f = Path(out) / "scen" / f"{wind}_K{K}_s{seed}_{fl[0][0]}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    w0 = time.time()
    sc, F, G3, tm = E7.load_scenario(scale, wind, K, seed, out, fields)
    Frob = np.stack([np.load(Path(out) / "rob" / f"{n}.npy") for n, _ in rob_fields(scale, K, seed)])
    res = evaluate(scale, wind, K, seed, sc, F, G3, Frob, tm, w0)
    f.parent.mkdir(exist_ok=True)
    E7._save_atomic(f, lambda fh: fh.write(pickle.dumps(res)))
    return res


def evaluate(scale, wind, K, seed, sc, F, G3, Frob, tm, w0):
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
    obj = FD.Objective(p, FD.forecast_input(di, F, X.site_responses(sc, Frob), Frob))
    seedp = sc.cfg.seeds.streams(seed)["placement"]
    designs = {}
    for k in s["ks"]:
        for name, net in BASE:
            designs[(k, name, net)] = X.make_design(name, p, k, T, seedp, net)
        for name in FD.FC:
            designs[(k, name, 0)] = FD.make_fc_design(name, p, obj, k)
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
    rows = []
    for (k, name, net), (S, info) in designs.items():
        A = a_ref * G3[S] / sigma[S][:, None, None]
        z = (Z[S] - c_bg) / sig
        C = FD._cum(A).sum(0)
        d = dict(**key, k=k, design=name, net=net, sites=" ".join(map(str, S)),
                 **{f"J_{n}": obj.value(n)(p, S) for n in FD.FC}, J_dopt=X.P.dopt(p, S))
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
            # data-free expected forecast RMSE of this network on the realised calendar (descriptive check of §2)
            for h in FD.HORIZONS:
                r[f"pred_rmse_h{h}"] = float(np.sqrt(FD.expected_mse(C, Qall, issue, (h,), win, last)))
            rows.append({**d, "estimator": est, **r})
    tm.update(key, eval_wall_s=time.time() - w2, total_wall_s=time.time() - w0,
              peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              swaps=sum(i.get("n_swaps", 0) for _, i in designs.values()))
    return rows, tm


# ---------------------------------------------------------------- analysis (reads only the rows table)
METRICS = ["forecast_rmse_h1", "forecast_rmse_h6", "forecast_rmse_h24", "analysis_rmse", "source_rmse"]
CONTRASTS = [(d, c) for d in FD.FC for c in ("random", "coverage", "source_near", "dopt")] + \
    [("fc_w24_rob", "fc_w24"), ("fc_grow_rob", "fc_grow"), ("fc_grow", "fc_w24"), ("fc_w24_rob", "fc_grow_rob"),
     ("dopt", "source_near")]


def per_scenario(df):
    reg = [c for c in df.columns if c.startswith("rmse_h1_")]
    return df.groupby(["estimator", "wind", "K", "seed", "k", "design"])[METRICS + reg].mean().reset_index()


def summarise(df):
    st, out = per_scenario(df), []
    reg = [c for c in st.columns if c.startswith("rmse_h1_")]
    for (est, k), c in list(st.groupby(["estimator", "k"])) + [((e, "all"), c) for e, c in st.groupby("estimator")]:
        for m in METRICS + reg:
            piv = c.pivot_table(index=["wind", "K", "seed"] + ([] if k != "all" else ["k"]), columns="design", values=m)
            for d, cmp_ in CONTRASTS:
                rho = ((piv[cmp_] - piv[d]) / piv[cmp_]).dropna().to_numpy()
                if len(rho) >= 6:
                    out.append(dict(estimator=est, k=k, metric=m, design=d, comparator=cmp_, **FD.paired_summary(rho)))
    return pd.DataFrame(out)


def report(src):
    df = pd.read_csv(src)
    summ = summarise(df)
    summ.to_csv(tab / "e11-local-contrasts.csv", index=False)
    st = per_scenario(df)
    med = st.groupby(["estimator", "k", "design"])[METRICS].median().reset_index()
    med.to_csv(tab / "e11-local-medians.csv", index=False)
    # §2 check: realised RMSE vs data-free expected RMSE (perfect model; noise and superposition make the gap)
    chk = df.assign(**{f"ratio_h{h}": df[f"forecast_rmse_h{h}"] / df[f"pred_rmse_h{h}"] for h in FD.HORIZONS})
    chk.groupby("estimator")[[f"ratio_h{h}" for h in FD.HORIZONS]].describe(percentiles=[.1, .5, .9]).T.to_csv(
        tab / "e11-local-predcheck.csv")
    h1 = summ[(summ.metric == "forecast_rmse_h1") & (summ.k != "all")]
    print(h1[h1.comparator.isin(["dopt", "source_near"]) | h1.design.isin(["dopt"])][
        ["estimator", "k", "design", "comparator", "median", "ci_lo", "ci_hi", "wins", "losses", "verdict"]].to_string(index=False))
    plot(summ)


def plot(summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fg, ax = plt.subplots(2, 2, figsize=(10, 7.5), sharey="row")
    for i, cmp_ in enumerate(("dopt", "source_near")):
        for j, est in enumerate(EST):
            a = ax[i, j]
            for n, d in enumerate(FD.FC):
                q = summ[(summ.estimator == est) & (summ.metric == "forecast_rmse_h1") & (summ.design == d)
                         & (summ.comparator == cmp_) & (summ.k != "all")]
                x = np.log2(q.k.astype(int)) + (n - 1.5) * .06
                a.errorbar(x, q["median"], [q["median"] - q.ci_lo, q.ci_hi - q["median"]], fmt="o-", ms=4, capsize=2, label=d)
            a.axhline(0, color="k", lw=.8)
            a.axhline(-.05, color="k", lw=.6, ls=":")
            a.set_xticks([1, 2, 3], ["2", "4", "8"])
            a.set_title(f"estimator {est}: vs {cmp_}", fontsize=9)
            a.set_xlabel("sensors k")
        ax[i, 0].set_ylabel(f"(RMSE_{cmp_} − RMSE_design) / RMSE_{cmp_}\nheld-out 1 h forecast")
    ax[0, 0].legend(fontsize=7)
    fg.suptitle("E11 32² pilot (exploratory): forecast-variance designs vs dopt / source_near; median over 12 scenarios, "
                "distribution-free 95 % CI; dotted = −5 % margin", fontsize=9)
    fg.tight_layout()
    fg.savefig(fig_dir / "e11-local-forecast-design.png", dpi=130)
    plt.close(fg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fields", default=None, help="read-only E07 field cache (stage 2 only for truth/responses)")
    ap.add_argument("--only", default=None, help="comma list wind:K:seed")
    a = ap.parse_args()
    scale = "sweep" if a.sweep else "pilot"
    if a.report:
        return report(tab / "e11-local-rows.csv")
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
    print(f"stage 1: {len(tasks)} field solves, {len(rob)} design-prior solves on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex:
        futs = [ex.submit(run_field, t) for t in tasks] + [ex.submit(run_rob, t) for t in rob]
        for i, f in enumerate(as_completed(futs)):
            tm = f.result()
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
    meta = dict({**E7.provenance(), "design_note": "docs/research/forecast-design-placement.md"}, scale=scale, world=E7.SCALES[scale],
                e11=s, winds=WINDS, design_prior=dict(n_dirs=FD.N_DIRS, block_h=FD.BLOCK_H, speed=FD.PRIOR_SPEED),
                workers=a.workers, stage1_wall_s=t1 - t0, stage2_wall_s=time.time() - t1)
    meta["command"] = "OMP_NUM_THREADS=1 uv run python scripts/e11_forecast_design.py " + " ".join(sys.argv[1:])
    if scale != "pilot":
        pd.DataFrame(rows).to_csv(out / "e11-sweep-rows.csv", index=False)
        pd.DataFrame(comp).to_csv(out / "e11-sweep-compute.csv", index=False)
        (out / "e11-sweep-meta.json").write_text(json.dumps(meta, indent=1, default=str))
        return
    pd.DataFrame(rows).to_csv(tab / "e11-local-rows.csv", index=False)
    pd.DataFrame(comp).to_csv(tab / "e11-local-compute.csv", index=False)
    (tab / "e11-local-meta.json").write_text(json.dumps(meta, indent=1, default=str))
    report(tab / "e11-local-rows.csv")


if __name__ == "__main__":
    main()

"""E07 cross-objective placement ablation (design: docs/research/cross-objective-design.md).

Pilot (32^2, local):  OMP_NUM_THREADS=1 uv run python scripts/e07_cross_objective.py --pilot [--workers 12]
Sweep (64^2, HPC), one job:  ... --sweep --out DIR --workers 64   (stage 1 then stage 2; rerun to resume)
  or split over jobs (optional; --shard i/n):
  stage 1 (solves):   ... --sweep --stage 1 --shard 0/2 --out DIR --workers 64
  stage 2 (designs):  ... --sweep --stage 2 --out DIR --workers 64      (after every stage-1 shard has finished)
Exclusion control (gate-07, post hoc; no solves): ... --sweep --stage 2 --fields OLD/fields --exclude-radius 1200 --out NEW
Preflight (64^2):     ... --preflight --out DIR [--throughput-n 1,16,32,64]
Report only:          uv run python scripts/e07_cross_objective.py --pilot --report

Stage 1: the unit of work is ONE simulation, the truth or one unit-source response of a scenario. Each is written
atomically to DIR/fields/<kind>_<hash>.npy (+ .json timing), the hash covering model, sources, the scenario's winds and
the source layout/strengths, so a changed config can never reuse a stale field. Stage 2 (one task per scenario: designs
and evaluation) loads them and writes DIR/scen/<scenario>_<truth hash>.pkl; results equal the single-process path bit
for bit (tests/test_e07_sweep.py). Without --stage both stages run in turn (pilot default). Reruns skip finished work.
The pilot also writes reports/quarto/tables/e07-local-*.{csv,json} and reports/quarto/figures/e07-local-crossobjective.png.
"""
import argparse, hashlib, json, os, pickle, platform, resource, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from pravah import benchmarks as B  # noqa: E402
from pravah import placement as P  # noqa: E402
from pravah.experiment_config import ExperimentConfig, Split, Winds  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402
from pravah.sensitivity import anomaly_fields  # noqa: E402

WINDS = {"constant": ["constant_westerly"], "veering": ["veering"], "mixed": ["steady_westerly", "veering", "weak_variable"]}
SCALES = {
    "pilot": dict(grid=(32, 32), dt=300, scheme="upwind", spinup=12, split=((1, 4), (5, 5), (6, 11)), Ks=(2, 4), ks=(2, 4, 8), seeds=(0, 1, 2), R=5),
    "sweep": dict(grid=(64, 64), dt=30, scheme="vanleer", spinup=48,  # amendment A2 (max wind 3.6 m/s -> Courant 0.69)
                  split=None, Ks=(2, 4), ks=(4, 8, 16), seeds=tuple(range(10)), R=5,  # design note §12 deviation 6
                  winds=("veering", "mixed"), estimators=("map",),
                  strategies=("uniform", "coverage", "source_near", "hotspot", "dopt", "diversity", "podqr")),
}
ESTIMATORS = {"map": False, "nnls": True}


def scale_lists(scale):
    """(winds, strategies, estimators) of a scale; the pilot (no keys) keeps everything."""
    s = SCALES[scale]
    return (list(s.get("winds", WINDS)), list(s.get("strategies", X.STRATEGIES)),
            {e: ESTIMATORS[e] for e in s.get("estimators", ESTIMATORS)})
tab, fig_dir = root / "reports/quarto/tables", root / "reports/quarto/figures"


def make_cfg(scale, wind):
    s = SCALES[scale]
    c = ExperimentConfig.default("EXP-E07")
    c.model.grid, c.model.spinup_h = s["grid"], s["spinup"]
    c.model.dt_s, c.model.advection_scheme = s["dt"], s["scheme"]
    c.design.n_sources = s["Ks"]
    if s["split"]:
        c.split = Split(*s["split"])
    c.winds = Winds(train=WINDS[wind], heldout=c.winds.heldout)
    c.validate()
    return c


def design_descriptors(p, S, G3, sigma, test_h, a_ref):
    A = a_ref * G3[S][:, test_h] / sigma[S][:, None, None]
    ev = np.linalg.eigvalsh(np.einsum("shk,shl->kl", A, A))
    return dict(D_train=P.dopt(p, S), A_train=P.aopt(p, S), E_train=P.eopt(p, S), fieldvar_train=P.fieldvar(p, S),
                coherence=P.coherence(p, S), D_test=float(np.log1p(ev).sum()), prank_test=int((ev >= 1).sum()))


# ---------------------------------------------------------------- stage 1: one simulation per task, cached on disk
def scenario_fields(scale, wind, K, seed):
    """[(cache name, spec)] for the truth and the K unit-source solves of one scenario. Names hash the content that
    determines the field (model, sources config, winds, and source xy / strengths), so stale caches cannot match."""
    cfg = make_cfg(scale, wind)
    _, a, sxy, winds, _, _ = B.scenario_drivers(cfg, seed, K)
    base = json.dumps({"model": asdict(cfg.model), "sources": asdict(cfg.sources)}, sort_keys=True).encode() + winds.tobytes()
    h = lambda *b: hashlib.sha256(base + b"|".join(b)).hexdigest()[:20]
    return [(f"truth_{h(b'truth', sxy.tobytes(), a.tobytes())}", ("truth", scale, wind, K, seed, -1))] + \
        [(f"src_{h(b'src', xy.tobytes())}", ("src", scale, wind, K, seed, k)) for k, xy in enumerate(sxy)]


def _save_atomic(path, write):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        write(fh)
    os.replace(tmp, path)


def run_field(task):
    """Solve one field (truth, or unit response of source k) unless cached; returns its timing record."""
    (name, (kind, scale, wind, K, seed, k)), out = task
    f = Path(out) / "fields" / f"{name}.npy"
    if f.exists():
        return None
    w0, c0 = time.time(), time.process_time()
    cfg = make_cfg(scale, wind)
    if kind == "truth":
        field = B.generate_scenario(cfg, seed, K).truth.field
    else:
        _, _, sxy, winds, _, _ = B.scenario_drivers(cfg, seed, K)
        field = X.source_field(cfg, sxy[k], winds, len(winds))
    tm = dict(name=name, kind=kind, wind=wind, K=K, seed=seed, k=k, wall_s=time.time() - w0, cpu_s=time.process_time() - c0,
              peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, host=platform.node(), pid=os.getpid())
    _save_atomic(f.with_suffix(".json"), lambda fh: fh.write(json.dumps(tm).encode()))  # json first: .npy implies .json
    _save_atomic(f, lambda fh: np.save(fh, field))
    return tm


# ---------------------------------------------------------------- stage 2: designs + evaluation per scenario
def load_scenario(scale, wind, K, seed, out, fields=None):
    """(scenario, F, G3, stage-1 timings) from the stage-1 caches in `fields` (default out/fields; FileNotFoundError if any
    is missing). Shared with E08."""
    fl = scenario_fields(scale, wind, K, seed)
    d = Path(fields or Path(out) / "fields")
    missing = [n for n, _ in fl if not (d / f"{n}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"{wind}:{K}:{seed} lacks stage-1 fields {missing}")
    st = [json.loads((d / f"{n}.json").read_text()) for n, _ in fl]
    w0 = time.time()
    sc = B.generate_scenario(make_cfg(scale, wind), seed, K, truth_field=np.load(d / f"{fl[0][0]}.npy"))
    F = np.empty((K,) + sc.truth.field.shape)  # filled in place: no second K-field copy (0.4 GB at 64^2, K = 8)
    for i, (n, _) in enumerate(fl[1:]):
        F[i] = np.load(d / f"{n}.npy")
    G3 = X.site_responses(sc, F)
    tm = dict(truth_wall_s=st[0]["wall_s"], truth_cpu_s=st[0]["cpu_s"], resp_wall_s=sum(x["wall_s"] for x in st[1:]),
              resp_cpu_s=sum(x["cpu_s"] for x in st[1:]), stage1_peak_rss_mb=max(x["peak_rss_mb"] for x in st),
              load_wall_s=time.time() - w0)
    return sc, F, G3, tm


def run_scenario(job, fields=None, exclude_m=0.0):
    """Stage 2 for one scenario from the stage-1 caches. exclude_m > 0: candidates within exclude_m of any source centre
    are removed before every design (post-hoc gate-07 control)."""
    scale, wind, K, seed, out = job
    fl = scenario_fields(scale, wind, K, seed)
    f = Path(out) / "scen" / f"{wind}_K{K}_s{seed}_{fl[0][0]}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    w0, c0 = time.time(), time.process_time()
    sc, F, G3, tm = load_scenario(scale, wind, K, seed, out, fields)
    res = evaluate(scale, wind, K, seed, sc, F, G3, tm, w0, c0, exclude_m)
    f.parent.mkdir(exist_ok=True)
    _save_atomic(f, lambda fh: fh.write(pickle.dumps(res)))
    return res


def evaluate(scale, wind, K, seed, sc, F, G3, tm, w0, c0, exclude_m=0.0):
    """Designs and held-out evaluation of one scenario given its truth `sc` and unit responses F, G3. exclude_m > 0 drops
    the candidates within exclude_m of any source (post-hoc gate-07 control); sigma, noise draws, seeds are unchanged."""
    s = SCALES[scale]
    _, strategies, estimators = scale_lists(scale)
    T, test_h, issue = len(sc.split_hours("train")), sc.split_hours("test"), X.issue_times(sc)
    a_ref, c_bg, a_true = sc.cfg.sources.reference_rate_ug_s, sc.cfg.sources.background_ug_m3, sc.truth.strengths_ug_s
    tf, n_h = sc.truth.field, len(sc.truth.winds)
    w = X.pop_weight(tuple(sc.cfg.model.grid), tuple(sc.cfg.model.domain_m))
    seedp = sc.cfg.seeds.streams(seed)["placement"]
    lin = float(np.abs(c_bg + np.einsum("k,khxy->hxy", a_true, F) - tf).max())  # superposition check of F vs truth
    key = dict(scale=scale, wind=wind, K=K, seed=seed)
    rows, swaps = [], []
    # design-free baselines
    zero = X.score(np.zeros((len(issue), K)), F, tf, issue, a_true, a_ref, c_bg, w)
    clim = tf[sc.split_hours("train")].mean(0)
    fc = clim[None] - tf[issue + 1]
    rows.append({**key, "arm": "-", "estimator": "-", "k": 0, "strategy": "background_only", "status": "ok", **zero})
    rows.append({**key, "arm": "-", "estimator": "-", "k": 0, "strategy": "climatology", "status": "ok",
                 "forecast_rmse": float(np.sqrt(np.mean(fc ** 2))), "popw_rmse": float(np.sqrt((fc ** 2 * w).sum((1, 2)).mean()))})
    # designs (per sigma mode: scalar for white/ar1, per-site for persite)
    w2, c2 = time.time(), time.process_time()
    designs = {}
    cand = X.exclude_near_sources(sc.site_xy, sc.admissible_sites(), sc.truth.source_xy, exclude_m)
    tm["n_cand"] = len(cand)
    for ps in (False, True):
        sigma = X.site_sigma(sc, ps)
        p = X.problem(X.design_input(sc, G3, sigma), cand=cand)
        for k in s["ks"]:
            for name, net in [("random", j) for j in range(X.N_RANDOM)] + [(n, 0) for n in strategies]:
                t = time.time()
                try:
                    S, info = X.make_design(name, p, k, T, seedp, net)
                    d = dict(status="ok", sites=" ".join(map(str, S)), **info, **design_descriptors(p, S, G3, sigma, test_h, a_ref))
                except ValueError as e:
                    S, d = None, dict(status=str(e))
                designs[(ps, k, name, net)] = (S, dict(design_s=time.time() - t, **d))
        if not ps:  # exhaustive check where spacing binds: 5x5 block of the real lattice (design note §3)
            ii, jj = np.divmod(np.arange(sc.n_candidate_sites), sc.lattice_n)
            adm = cand
            lo, hi = (sc.lattice_n // 2 - 3, sc.lattice_n // 2 + 1)
            blk = adm[(ii[adm] >= lo) & (ii[adm] <= hi) & (jj[adm] >= lo) & (jj[adm] <= hi)]
            pb = X.problem(X.design_input(sc, G3, sigma), cand=blk)
            for k in (2, 3):
                for cn in ("dopt", "fieldvar"):
                    crit = X.SWAP[cn]
                    try:
                        v, bS, nf = P.brute_force(pb, k, crit)
                        g = P.greedy(pb, k, crit)
                        _, vs, nsw = P.swap_refine(pb, g, crit)
                        swaps.append({**key, "k": k, "criterion": cn, "n_block_cand": len(blk), "n_feasible": nf, "exhaustive": v,
                                      "greedy": crit(pb, g), "greedy_swap": vs, "n_swaps": nsw})
                    except ValueError as e:
                        swaps.append({**key, "k": k, "criterion": cn, "n_block_cand": len(blk), "status": str(e)})
    tm["design_wall_s"], tm["design_cpu_s"] = time.time() - w2, time.process_time() - c2
    # evaluation: same R noise draws per arm for every design
    w3, c3 = time.time(), time.process_time()
    for arm, (kind, ps) in X.ARMS.items():
        sigma = X.site_sigma(sc, ps)
        Y = [X.readings(sc, sigma, kind, r, n_h) for r in range(s["R"])]
        for (q, k, name, net), (S, d) in designs.items():
            if q != ps:
                continue
            base = {**key, "arm": arm, "k": k, "strategy": name, "net": net, "sigma_med": float(np.median(sigma)), **d}
            if S is None:
                rows += [{**base, "estimator": e} for e in estimators]
                continue
            A = a_ref * G3[S] / sigma[S][:, None, None]
            ev = X.window_info(A, issue)
            wi = dict(window_prank_mean=float((ev >= 1).sum(1).mean()), frac_window_rank_lt_K=float(((ev >= 1).sum(1) < K).mean()))
            for e, nn in estimators.items():
                try:
                    m = [X.score(X.estimate_series(A, (y[S] - c_bg) / sigma[S][:, None], issue, nonneg=nn), F, tf, issue, a_true,
                                 a_ref, c_bg, w) for y in Y]
                    rows.append({**base, "estimator": e, **wi, **{c: float(np.mean([x[c] for x in m])) for c in m[0]}})
                except Exception as ex:  # estimator failure: recorded, scored as worst
                    rows.append({**base, "estimator": e, "status": f"estimator: {ex}"})
    tm["eval_wall_s"], tm["eval_cpu_s"] = time.time() - w3, time.process_time() - c3
    tm.update(key, total_wall_s=time.time() - w0, total_cpu_s=time.process_time() - c0,
              peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, n_hours=n_h, n_issue=len(issue),
              linearity_max_abs=lin, a_true=" ".join(f"{x:.4g}" for x in a_true), pid=os.getpid())
    return rows, swaps, tm


# ---------------------------------------------------------------- analysis (reads only the per-row table)
ENDPOINTS = ["forecast_rmse", "analysis_rmse", "popw_rmse", "source_rmse"]
SELECTORS = ENDPOINTS + ["D_train", "A_train", "fieldvar_train", "D_test"]  # D/A/fieldvar: larger is better


def scenario_table(df, arm="white", est="map"):
    """Per scenario x k x strategy (random = mean over its 10 networks); undefined -> NaN status flag kept."""
    d = df[(df.arm == arm) & (df.estimator == est)].copy()
    d["ok"] = d.status == "ok"
    g = d.groupby(["wind", "K", "seed", "k", "strategy"])
    agg = g[ENDPOINTS + ["D_train", "A_train", "fieldvar_train", "D_test", "prank_test", "window_prank_mean"]].mean()
    agg["defined"] = g.ok.all()
    return agg.reset_index()


def summarise(df):
    out, reg = [], []
    for arm in X.ARMS:
        for est in [e for e in ESTIMATORS if (df.estimator == e).any()]:
            st = scenario_table(df, arm, est)
            for (K, k), c in st.groupby(["K", "k"]):
                piv = {m: c.pivot_table(index=["wind", "seed"], columns="strategy", values=m, dropna=False) for m in ENDPOINTS}
                dfn = c.pivot_table(index=["wind", "seed"], columns="strategy", values="defined")
                ps = []
                for strat in [x for x in X.STRATEGIES if x in piv[ENDPOINTS[0]]]:  # sweep runs a subset
                    for m in ENDPOINTS:
                        rnd, v = piv[m]["random"], piv[m][strat]
                        rho = ((rnd - v) / rnd).where(dfn[strat].astype(bool), -np.inf).to_numpy()
                        pos, neg, tie, p = X.sign_test(rho)
                        lo, hi = X.boot_median_ci(rho)
                        out.append(dict(arm=arm, estimator=est, K=K, k=k, strategy=strat, metric=m, n=len(rho),
                                        n_undefined=int((~dfn[strat].astype(bool)).sum()), median_rel_reduction=float(np.median(rho)),
                                        ci_lo=lo, ci_hi=hi, wins=pos, losses=neg, ties=tie, sign_p=p))
                        if m == "forecast_rmse":
                            ps.append(len(out) - 1)
                adj = X.holm([out[i]["sign_p"] for i in ps])
                for i, a in zip(ps, adj):
                    out[i]["holm_p"] = float(a)
            # cross-objective regret: selector metric picks the best strategy per scenario, regret on each endpoint
            for (w_, K, seed, k), c in st.groupby(["wind", "K", "seed", "k"]):
                c = c.set_index("strategy")
                ok = c.defined.astype(bool)
                for sel in SELECTORS:
                    vals = c[sel].where(ok)
                    if sel in ("D_train", "A_train", "fieldvar_train", "D_test"):
                        vals = vals.drop("random")  # design criteria: random has no single network value
                        b = vals.idxmax()
                    else:
                        b = vals.idxmin()
                    for m in ENDPOINTS:
                        best = c[m].where(ok).min()
                        reg.append(dict(arm=arm, estimator=est, wind=w_, K=K, seed=seed, k=k, selector=sel, chosen=b, metric=m,
                                        regret=float((c.loc[b, m] - best) / best)))
    return pd.DataFrame(out), pd.DataFrame(reg)


def report(scale, out_dir):
    pre, src = ("local", "e07-local-pilot.csv") if scale == "pilot" else ("sweep", "e07-sweep-rows.csv")
    df = pd.read_csv(tab / src)
    summ, reg = summarise(df)
    summ.to_csv(tab / f"e07-{pre}-summary.csv", index=False)
    reg.to_csv(tab / f"e07-{pre}-regret.csv", index=False)
    plot(summ, reg, pre, n=df.groupby("K").seed.nunique().iloc[0] * df.wind.nunique())
    if scale == "sweep":
        confirmatory(df).to_csv(tab / "e07-sweep-confirmatory.csv", index=False)


def confirmatory(df):
    """Protocol test (design note §7 + deviation 6): dopt vs random, forecast RMSE, white + MAP, per (wind, K, k>=8) cell."""
    st = scenario_table(df, "white", "map")
    out = []
    for (w_, K, k), c in st[st.k >= 8].groupby(["wind", "K", "k"]):
        piv = c.pivot_table(index="seed", columns="strategy", values="forecast_rmse")
        dfn = c.pivot_table(index="seed", columns="strategy", values="defined")
        rho = ((piv["random"] - piv["dopt"]) / piv["random"]).where(dfn["dopt"].astype(bool), -np.inf).to_numpy()
        lo, hi = X.boot_median_ci(rho)
        pos, neg, tie, p = X.sign_test(rho)
        # gate-07 sensitivity (added after the sweep, reported beside the committed rule, never replacing it):
        # distribution-free order-statistic CI for the median (>= 95 % binomial coverage) and percentile bootstrap of the mean
        n_ = len(rho)
        dlo, dhi = X.df_median_ci(rho)
        g = np.random.default_rng(4000).choice(rho, (10000, n_)).mean(1)
        out.append(dict(wind=w_, K=K, k=k, n=n_, median=float(np.median(rho)), ci_lo=lo, ci_hi=hi, wins=pos, losses=neg,
                        sign_p=p, verdict="above" if lo > .1 else "below" if hi < .1 else "straddles",
                        df_ci_lo=dlo, df_ci_hi=dhi, df_verdict="above" if dlo > .1 else "below" if dhi < .1 else "straddles",
                        mean=float(rho.mean()), mean_ci_lo=float(np.percentile(g, 2.5)), mean_ci_hi=float(np.percentile(g, 97.5))))
    t = pd.DataFrame(out)
    f = lambda v: (t.verdict == v).mean()
    t.attrs["overall"] = "supported" if f("above") >= .7 else "contradicted" if f("below") >= .7 else "unresolved"
    print(t.to_string(index=False), "\nprotocol hypothesis:", t.attrs["overall"])
    return t


def plot(summ, reg, pre="local", n=9):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = reg[(reg.arm == "white") & (reg.estimator == "map")]
    M = r.groupby(["selector", "metric"]).regret.median().unstack().reindex(index=SELECTORS, columns=ENDPOINTS)
    s = summ[(summ.arm == "white") & (summ.estimator == "map") & (summ.metric == "forecast_rmse")]
    Ks = sorted(s.K.unique())
    fg, ax = plt.subplots(1, 1 + len(Ks), figsize=(5.2 + 4.6 * len(Ks), 4.4))
    im = ax[0].imshow(M.to_numpy(), cmap="viridis", vmin=0)
    ax[0].set_xticks(range(len(ENDPOINTS)), [e.replace("_rmse", "") for e in ENDPOINTS], rotation=30)
    ax[0].set_yticks(range(len(SELECTORS)), ["best on " + x.replace("_rmse", "") for x in SELECTORS])
    for (i, j), v in np.ndenumerate(M.to_numpy()):
        ax[0].text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v < np.nanmax(M.to_numpy()) / 2 else "k", fontsize=8)
    ax[0].set_title("median relative regret on held-out metric\n(white noise, MAP; all k, K)", fontsize=9)
    fg.colorbar(im, ax=ax[0], shrink=.8)
    for a, K in zip(ax[1:], Ks):
        for st in X.STRATEGIES:
            q = s[(s.K == K) & (s.strategy == st)].sort_values("k")
            y = q.median_rel_reduction.replace(-np.inf, np.nan)
            a.plot(q.k, y, "o-", label=st, ms=4)
        a.axhline(0, color="k", lw=.8)
        a.axhline(.1, color="k", lw=.6, ls=":")
        a.set_xscale("log", base=2)
        a.set_xlabel("sensors k")
        a.set_title(f"K = {K}: forecast RMSE reduction vs random\n(median over {n} scenarios; dotted = 10 %)", fontsize=9)
    ax[1].set_ylabel("(RMSE_random - RMSE_design) / RMSE_random")
    ax[-1].legend(fontsize=7, loc="best")
    fg.tight_layout()
    fg.savefig(fig_dir / f"e07-{pre}-crossobjective.png", dpi=130)
    plt.close(fg)


# ---------------------------------------------------------------- preflight (64^2 sweep settings, engineering only)
def _pf_solve(job):
    """One timed solve with the sweep model on scenario (mixed, seed 0, K): unit source `which`, or 'joint' (true strengths)."""
    scheme, K, which, hours, spinup = job
    cfg = make_cfg("sweep", "mixed")
    cfg.model.advection_scheme, cfg.model.spinup_h = scheme, spinup
    m = cfg.model
    _, a, sxy, winds, _, _ = B.scenario_drivers(cfg, 0, K)
    w0, c0 = time.time(), time.process_time()
    if which == "joint":  # the truth's source, built as in benchmarks.generate_scenario
        S = sum(ak * B.source_basis(tuple(m.grid), tuple(m.domain_m), c, cfg.sources.footprint_sigma_m)
                for ak, c in zip(a, sxy)) / m.mixing_depth_m
        f = anomaly_fields(m, winds, S, hours)
    else:
        f = X.source_field(cfg, sxy[which % K], winds, hours)
    tm = dict(scheme=scheme, which=which, hours=hours, spinup_h=spinup, start=w0, wall_s=time.time() - w0,
              cpu_s=time.process_time() - c0, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, pid=os.getpid())
    return tm, f, a


def preflight(out, ns, hours):
    """(a) s per simulated hour per worker with N concurrent 64^2 solves; (b) superposition error, vanleer and upwind."""
    pf = out / "preflight"
    pf.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in ns:  # one task per worker; all N run concurrently (each lasts minutes, longer than pool start-up)
        with ProcessPoolExecutor(n) as ex:
            res = list(ex.map(_pf_solve, [("vanleer", 8, i, hours, 0) for i in range(n)]))
        rows += [dict(n_workers=n, s_per_sim_h=tm["wall_s"] / hours, **tm) for tm, _, _ in res]
        d = pd.DataFrame(rows)
        d.to_csv(pf / "throughput.csv", index=False)
        q = d[d.n_workers == n]
        print(f"N={n}: s/sim-h median {q.s_per_sim_h.median():.2f} max {q.s_per_sim_h.max():.2f}; distinct pids "
              f"{q.pid.nunique()}; rss max {q.peak_rss_mb.max():.0f} MB", flush=True)
    K, sup = 4, {}
    jobs = [(sch, K, w, 48, 48) for sch in ("vanleer", "upwind") for w in ["joint", *range(K)]]
    with ProcessPoolExecutor(min(len(jobs), max(ns))) as ex:
        res = list(ex.map(_pf_solve, jobs))
    bg = make_cfg("sweep", "mixed").sources.background_ug_m3
    for sch in ("vanleer", "upwind"):
        r = [x for j, x in zip(jobs, res) if j[0] == sch]
        joint, a = bg + r[0][1], r[0][2]
        rec = bg + np.einsum("k,khxy->hxy", a, np.stack([x[1] for x in r[1:]]))
        e, anom = rec - joint, joint - bg
        sup[sch] = dict(max_abs_err=float(np.abs(e).max()), max_abs_anomaly=float(np.abs(anom).max()),
                        rel_max_err=float(np.abs(e).max() / np.abs(anom).max()),
                        rel_rms_err=float(np.sqrt((e ** 2).mean() / (anom ** 2).mean())), min_joint_anomaly=float(anom.min()),
                        wall_s=[x[0]["wall_s"] for x in r])
        print(sch, {k: v for k, v in sup[sch].items() if k != "wall_s"}, flush=True)
    meta = dict(scenario="sweep model, wind mixed, seed 0, K=4, dt 30 s, spin-up 48 h + 48 h; both schemes at dt 30 s",
                strengths=[float(x) for x in a], **provenance())
    (pf / "superposition.json").write_text(json.dumps(dict(meta=meta, **sup), indent=1, default=str))


def provenance():
    git = lambda *c: subprocess.run(["git", *c], cwd=root, capture_output=True, text=True).stdout.strip()
    return dict(command="OMP_NUM_THREADS=1 uv run python scripts/e07_cross_objective.py " + " ".join(sys.argv[1:]),
                git_rev=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain", "--", "src", "scripts")),
                host=platform.node(), python=platform.python_version(), numpy=np.__version__,
                slurm_job=os.environ.get("SLURM_JOB_ID"), design_note="docs/research/cross-objective-design.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--throughput-n", default="1,16,32,64", help="preflight: concurrent worker counts")
    ap.add_argument("--preflight-hours", type=int, default=24)
    ap.add_argument("--stage", choices=("1", "2", "all"), default="all")
    ap.add_argument("--shard", default="0/1", help="i/n: this job takes every n-th task starting at i")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count()))
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma list wind:K:seed (smoke)")
    ap.add_argument("--fields", default=None, help="field cache dir (default OUT/fields); e.g. an earlier run's, read-only")
    ap.add_argument("--exclude-radius", type=float, default=0.0, help="m; drop candidates this close to any source")
    a = ap.parse_args()
    scale = "sweep" if a.sweep else "pilot"
    out = Path(a.out or root / f".cache/e07/{scale}")
    if a.report:
        return report(scale, out)
    if a.fields and a.stage != "2":
        ap.error("--fields reads an existing (read-only) cache: use it with --stage 2")
    fdir = Path(a.fields or out / "fields")
    out.mkdir(parents=True, exist_ok=True)
    if a.fields is None:
        fdir.mkdir(exist_ok=True)
    if a.preflight:
        return preflight(out, [int(x) for x in a.throughput_n.split(",")], a.preflight_hours)
    si, sn = map(int, a.shard.split("/"))
    if not 0 <= si < sn:
        ap.error("--shard i/n needs 0 <= i < n")
    s = SCALES[scale]
    jobs = [(scale, w, K, sd, str(out)) for K in sorted(s["Ks"], reverse=True) for w in scale_lists(scale)[0]
            for sd in s["seeds"]]  # longest first
    if a.only:
        sel = {tuple(x.split(":")) for x in a.only.split(",")}
        jobs = [j for j in jobs if (j[1], str(j[2]), str(j[3])) in sel]
    t0 = time.time()
    if a.stage in ("1", "all"):
        tasks = list({n: sp for j in jobs for n, sp in scenario_fields(*j[:4])}.items())  # dedup identical solves
        tasks = tasks[si::sn] if a.stage == "1" else tasks  # every solve spans one calendar (same cost): round-robin balances
        todo = [t for t in tasks if not (fdir / f"{t[0]}.npy").exists()]
        print(f"stage 1: {len(tasks)} solves in shard {a.shard}, {len(todo)} to run on {a.workers} workers", flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(run_field, (t, str(out))) for t in todo]
            for i, f in enumerate(as_completed(futs)):
                if (tm := f.result()) is None:  # finished meanwhile by another job sharing --out
                    continue
                print(f"[{i + 1}/{len(todo)}] {tm['kind']} {tm['wind']}:{tm['K']}:{tm['seed']} k={tm['k']} "
                      f"wall={tm['wall_s']:.0f}s rss={tm['peak_rss_mb']:.0f}MB", flush=True)
        if a.stage == "1":
            return
    if a.stage == "2":
        jobs = jobs[si::sn]
    rows, swaps, comp = [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_scenario, j, a.fields, a.exclude_radius): j for j in jobs}
        for f in as_completed(futs):
            r, sw, tm = f.result()
            rows += r; swaps += sw; comp.append(tm)
            print(f"done {futs[f][1:4]} wall={tm['total_wall_s']:.0f}s rss={tm['peak_rss_mb']:.0f}MB", flush=True)
    if a.only or sn > 1:
        return
    meta = dict(**provenance(), scale=s, fields=str(fdir), exclude_radius_m=a.exclude_radius, winds={w: WINDS[w] for w in scale_lists(scale)[0]}, workers=a.workers, total_wall_s=time.time() - t0)
    if scale != "pilot":  # whole-sweep tables next to the caches; the lead curates what enters reports/
        pd.DataFrame(rows).to_csv(out / "e07-sweep-rows.csv", index=False)
        pd.DataFrame(swaps).to_csv(out / "e07-sweep-swapcheck.csv", index=False)
        pd.DataFrame(comp).to_csv(out / "e07-sweep-compute.csv", index=False)
        (out / "e07-sweep-meta.json").write_text(json.dumps(meta, indent=1, default=str))
        return
    pd.DataFrame(rows).to_csv(tab / "e07-local-pilot.csv", index=False)
    pd.DataFrame(swaps).to_csv(tab / "e07-local-swapcheck.csv", index=False)
    pd.DataFrame(comp).to_csv(tab / "e07-local-compute.csv", index=False)
    (tab / "e07-local-meta.json").write_text(json.dumps(meta, indent=1, default=str))
    report(scale, out)


if __name__ == "__main__":
    main()

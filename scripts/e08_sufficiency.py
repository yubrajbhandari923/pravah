"""E08 sensor sufficiency (design: docs/research/sufficiency-design.md). Reuses the E07 two-stage driver and its field caches.

Pilot (32^2, local):  OMP_NUM_THREADS=1 uv run python scripts/e08_sufficiency.py --pilot --workers 6
Sweep (64^2, HPC):    ... --sweep --out DIR --workers 64     (stage 1 solves, then stage 2 per scenario; rerun to resume)
Variants (stage 2 only, reuse DIR/fields; outputs get a tag, reference files are never touched):
                      --exclude-radius 1200  --estimator map_full   (tags excl1200 / mapfull; scen/ cache in scen-<tag>/)
Report only:          uv run python scripts/e08_sufficiency.py --pilot --report   (or --sweep --report --out DIR)

Stage 1 is E07's `run_field` (content-hashed caches in DIR/fields; E08's sweep scale has the same model, calendar and winds
as E07's sweep, so E07's `mixed` fields can be symlinked into DIR/fields and are not re-solved). Stage 2 writes
DIR/scen/<K>_<seed>_<truth hash>.pkl per scenario.
"""
import argparse, importlib.util, json, multiprocessing as mp, os, pickle, resource, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from pravah.experiments import cross_objective as X  # noqa: E402
from pravah.experiments import sufficiency as S  # noqa: E402

_spec = importlib.util.spec_from_file_location("e07_driver", root / "scripts" / "e07_cross_objective.py")
E7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E7)

KS = (2, 4, 8, 16, 32)
E7.SCALES.update({  # registered at import, so worker processes (which re-import this module) see them too
    "e08pilot": dict(grid=(32, 32), dt=300, scheme="upwind", spinup=12, split=((1, 4), (5, 7), (8, 13)), Ks=(2, 4, 8), ks=KS,
                     seeds=(0, 1, 2, 3), R=5, winds=("mixed",)),
    "e08sweep": dict(grid=(64, 64), dt=30, scheme="vanleer", spinup=48, split=None, Ks=(2, 4, 8), ks=KS,  # = E07 sweep model (A2)
                     seeds=tuple(range(20)), R=5, winds=("mixed",)),
})
WIND = "mixed"
E07_OVERLAP = lambda K, seed: K in (2, 4) and seed <= 9  # sweep scenarios identical to E07's (not fresh for H5)
tab, fig_dir = root / "reports/quarto/tables", root / "reports/quarto/figures"


def run_field(task):
    return E7.run_field(task)


def variant_tag(excl, full):
    return "-".join(([f"excl{int(excl)}"] if excl else []) + (["mapfull"] if full else []))


def run_scenario(job):
    scale, K, seed, out, excl, full = job
    fl = E7.scenario_fields(scale, WIND, K, seed)
    tag = variant_tag(excl, full)
    f = Path(out) / (f"scen-{tag}" if tag else "scen") / f"K{K}_s{seed}_{fl[0][0]}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    w0, c0 = time.time(), time.process_time()
    sc, F, G3, tm = E7.load_scenario(scale, WIND, K, seed, out)
    s = E7.SCALES[scale]
    rows, drows, t2 = S.evaluate(sc, F, G3, s["ks"], s["R"], exclude_m=excl, full_window=full)
    key = dict(scale=scale, K=K, seed=seed)
    tm["n_cand"] = len(X.exclude_near_sources(sc.site_xy, sc.admissible_sites(), sc.truth.source_xy, excl)) if excl else len(sc.admissible_sites())
    lin = float(np.abs(sc.cfg.sources.background_ug_m3 + np.einsum("k,khxy->hxy", sc.truth.strengths_ug_s, F) - sc.truth.field).max())
    tm.update(key, **t2, total_wall_s=time.time() - w0, total_cpu_s=time.process_time() - c0, linearity_max_abs=lin,
              peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, n_hours=len(sc.truth.winds),
              a_true=" ".join(f"{x:.4g}" for x in sc.truth.strengths_ug_s), pid=os.getpid())
    res = ([{**key, **r} for r in rows], [{**key, **r} for r in drows], tm)
    f.parent.mkdir(exist_ok=True)
    E7._save_atomic(f, lambda fh: fh.write(pickle.dumps(res)))
    return res


# ---------------------------------------------------------------- analysis (reads only the row table)
def with_nrmse(df):
    """Adds nRMSE (overall and per regime) = RMSE / background-only RMSE of the same scenario; undefined -> inf."""
    bg = df[df.strategy == "background_only"].set_index(["K", "seed"])
    d = df[df.strategy != "background_only"].copy()
    regs = [c[len("fc_rmse_"):] for c in df.columns if c.startswith("fc_rmse_")]
    for c, b in [("forecast_rmse", "forecast_rmse")] + [(f"fc_rmse_{g}", f"fc_rmse_{g}") for g in regs]:
        den = bg[b].reindex(pd.MultiIndex.from_frame(d[["K", "seed"]])).to_numpy()
        v = d[c].where(d.status == "ok", np.inf).to_numpy()
        d["n" + c] = v / den
    return d, regs


def curves(d):
    out = []
    for (K, st, arm, sch, k), g in d.groupby(["K", "strategy", "arm", "schedule", "k"]):
        lo, hi, cov = S.df_ci(g.nforecast_rmse)
        out.append(dict(K=K, strategy=st, arm=arm, schedule=sch, k=k, n=len(g), n_undefined=int((g.status != "ok").sum()),
                        median_nrmse=float(np.median(g.nforecast_rmse)), ci_lo=lo, ci_hi=hi, ci_coverage=cov,
                        median_rmse=float(g.forecast_rmse.median()), median_source_rmse=float(g.source_rmse.median()),
                        median_frac_prank_lt_K=float(g.frac_prank_lt_K.median()), median_tti=float(g.tti_med.median())
                        if "tti_med" in g else np.nan))
    return pd.DataFrame(out)


def kstars(d):
    rows = []
    for (K, seed, st, arm, sch), g in d.groupby(["K", "seed", "strategy", "arm", "schedule"]):
        rows += [dict(K=K, seed=seed, strategy=st, arm=arm, schedule=sch, tau=t, kstar=S.breakpoint(g.k, g.nforecast_rmse, t))
                 for t in S.TAUS]
    return pd.DataFrame(rows)


def breakpoints(ksd, cur):
    out = []
    for (K, st, arm, sch, t), g in ksd.groupby(["K", "strategy", "arm", "schedule", "tau"]):
        lo, hi, cov = S.df_ci(g.kstar)
        c = cur[(cur.K == K) & (cur.strategy == st) & (cur.arm == arm) & (cur.schedule == sch)].sort_values("k")
        conf = [k for k, h in zip(c.k, c.ci_hi) if h <= t]
        r = dict(K=K, strategy=st, arm=arm, schedule=sch, tau=t, n=len(g), median_kstar=float(np.median(g.kstar)), ci_lo=lo,
                 ci_hi=hi, ci_coverage=cov, k_ci_sufficient=min(conf) if conf else S.K_NONE)
        for k in KS:
            s_ = int((g.kstar <= k).sum())
            wl, wh = S.wilson(s_, len(g))
            r.update({f"frac_suff_k{k}": s_ / len(g), f"frac_suff_k{k}_lo": wl, f"frac_suff_k{k}_hi": wh})
        out.append(r)
    return pd.DataFrame(out)


def _paired(name, x, unit, note=""):
    x = np.asarray(x, float)
    pos, neg, tie, p = X.sign_test(x)
    lo, hi, cov = S.df_ci(x)
    return dict(test=name, n=len(x), unit=unit, median=float(np.median(x)), ci_lo=lo, ci_hi=hi, ci_coverage=cov, wins=pos,
                losses=neg, ties=tie, sign_p=p, note=note)


def contrasts(d, ksd, sweep):
    """H1-H5 primary tests (design note §9; Holm family at 64^2) plus reported variants (labelled secondary)."""
    arm, sch = S.REF
    ref = d[(d.arm == arm) & (d.schedule == sch)]
    val = lambda q, c="forecast_rmse": q.set_index(["K", "seed"])[c].where(q.set_index(["K", "seed"]).status == "ok", np.inf)
    out = []
    # H1: k*(0.2) K=8 vs K=2, paired by seed (log2 scale)
    for st in S.STRATEGIES:
        for t in S.TAUS:
            q = ksd[(ksd.strategy == st) & (ksd.arm == arm) & (ksd.schedule == sch) & (ksd.tau == t)].pivot(index="seed", columns="K", values="kstar")
            if {2, 8} <= set(q.columns):
                out.append({**_paired(f"H1 {st} tau={t}", np.log2(q[8]) - np.log2(q[2]), "seed", "log2 k*(K=8) - log2 k*(K=2) > 0"),
                            "primary": st == "dopt" and t == 0.2})
    # H2: noise ratio random vs source_near, k=8, scenario
    for k in KS:
        r = {st: np.log(val(d[(d.arm == "w5") & (d.schedule == sch) & (d.k == k) & (d.strategy == st)])
                        / val(d[(d.arm == "w1") & (d.schedule == sch) & (d.k == k) & (d.strategy == st)])) for st in ("random", "source_near", "coverage", "dopt")}
        out.append({**_paired(f"H2 k={k}", r["random"] - r["source_near"], "scenario", "log(w5/w1) random - source_near > 0"), "primary": k == 8})
    # H3: span gain (S4 -> S6) in steady_southerly minus other regimes, dopt k=8
    def pooled_other(q):
        regs = [c[len("fc_rmse_"):] for c in q.columns if c.startswith("fc_rmse_") and c != "fc_rmse_steady_southerly"]
        num = sum(q[f"n_issue_{g}"] * q[f"fc_rmse_{g}"] ** 2 for g in regs)
        return np.sqrt(num / sum(q[f"n_issue_{g}"] for g in regs))
    for st in S.STRATEGIES:
        for k in KS:
            a = d[(d.arm == arm) & (d.k == k) & (d.strategy == st) & (d.schedule == "S4")].set_index(["K", "seed"])
            b = d[(d.arm == arm) & (d.k == k) & (d.strategy == st) & (d.schedule == "S6")].set_index(["K", "seed"])
            if (a.status != "ok").any() or (b.status != "ok").any():
                continue
            gs = np.log(a.fc_rmse_steady_southerly) - np.log(b.fc_rmse_steady_southerly)
            go = np.log(pooled_other(a)) - np.log(pooled_other(b))
            out.append({**_paired(f"H3 {st} k={k}", gs - go, "scenario", "span gain (S4->S6) southerly - other > 0"),
                        "primary": st == "dopt" and k == 8})
            out.append({**_paired(f"H3b {st} k={k} total span gain", np.log(a.forecast_rmse) - np.log(b.forecast_rmse), "scenario",
                                  "log RMSE S4 - S6 (all regimes)"), "primary": False})
    # H4: matched readings, (k, n=24) vs (4k, n=6)
    for st in S.STRATEGIES:
        for W, s24, s6 in ((168, "S6", "S5"), (24, "S4", "S3")):
            for k in (2, 4, 8):
                a = val(d[(d.arm == arm) & (d.schedule == s24) & (d.k == k) & (d.strategy == st)])
                b = val(d[(d.arm == arm) & (d.schedule == s6) & (d.k == 4 * k) & (d.strategy == st)])
                out.append({**_paired(f"H4 {st} W={W} k={k}->{4 * k}", np.log(a) - np.log(b), "scenario",
                                      "log RMSE(k, n=24) - log RMSE(4k, n=6) > 0"), "primary": st == "random" and W == 168 and k == 2})
    # H5: source_near vs dopt on fresh scenarios
    for k in KS:
        q = ref[ref.k == k]
        x = np.log(val(q[q.strategy == "dopt"])) - np.log(val(q[q.strategy == "source_near"]))
        fresh = np.array([not (sweep and E07_OVERLAP(K, s)) for K, s in x.index])
        out.append({**_paired(f"H5 k={k} fresh", x[fresh], "scenario", "log RMSE dopt - source_near > 0"), "primary": k == 8})
        if sweep:
            out.append({**_paired(f"H5 k={k} E07-overlap", x[~fresh], "scenario", "not new evidence"), "primary": False})
    t = pd.DataFrame(out)
    pm = t.primary.to_numpy()
    t["holm_p"] = np.nan
    t.loc[pm, "holm_p"] = X.holm(t.loc[pm, "sign_p"].to_numpy())
    t["verdict"] = np.where(~pm, "", np.where((t.holm_p < .05) & (t.wins > t.losses), "supported",
                                              np.where((t.holm_p < .05) & (t.wins < t.losses), "contradicted", "unresolved")))
    if not sweep:
        t.loc[pm, "verdict"] = "pilot (exploratory, no verdict)"
    return t


def protocol_cells(d):
    """Design note §10: the protocol primary rule (dopt vs random, >= 10 %, k >= 8) per K at the reference configuration."""
    arm, sch = S.REF
    q = d[(d.arm == arm) & (d.schedule == sch) & (d.k >= 8)]
    out = []
    for (K, k), g in q.groupby(["K", "k"]):
        p = g.pivot(index="seed", columns="strategy", values="forecast_rmse")
        ok = g.pivot(index="seed", columns="strategy", values="status") == "ok"
        rho = ((p.random - p.dopt) / p.random).where(ok.dopt, -np.inf).where(ok.random, np.nan).to_numpy()
        lo, hi, cov = S.df_ci(rho)
        blo, bhi = X.boot_median_ci(rho[np.isfinite(rho) | (rho == -np.inf)])
        out.append(dict(K=K, k=k, n=len(rho), median=float(np.median(rho)), df_ci_lo=lo, df_ci_hi=hi, df_coverage=cov,
                        df_verdict="above" if lo > .1 else "below" if hi < .1 else "straddles", boot_ci_lo=blo, boot_ci_hi=bhi,
                        boot_verdict="above" if blo > .1 else "below" if bhi < .1 else "straddles"))
    return pd.DataFrame(out)


def report(pre, rows_csv, sweep):
    df = pd.read_csv(rows_csv)
    d, _ = with_nrmse(df)
    cur = curves(d)
    ksd = kstars(d)
    bp = breakpoints(ksd, cur)
    con = contrasts(d, ksd, sweep)
    pc = protocol_cells(d)
    cur.to_csv(tab / f"e08-{pre}-curves.csv", index=False)
    bp.to_csv(tab / f"e08-{pre}-breakpoints.csv", index=False)
    con.to_csv(tab / f"e08-{pre}-contrasts.csv", index=False)
    pc.to_csv(tab / f"e08-{pre}-protocolcells.csv", index=False)
    plot(cur, bp, pre, n=int(df.groupby("K").seed.nunique().min()))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(con[con.primary | con.test.str.startswith("H5")][["test", "n", "median", "ci_lo", "ci_hi", "wins", "losses", "ties", "sign_p", "verdict"]].to_string(index=False))
        print(pc.to_string(index=False))


def plot(cur, bp, pre, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ks = sorted(cur.K.unique())
    col = dict(random="#888888", coverage="#1b9e77", source_near="#d95f02", dopt="#7570b3")
    fg, ax = plt.subplots(3, len(Ks), figsize=(4.4 * len(Ks), 11), sharey="row", squeeze=False)
    sel = lambda K, st, arm, sch: cur[(cur.K == K) & (cur.strategy == st) & (cur.arm == arm) & (cur.schedule == sch)].sort_values("k")
    for j, K in enumerate(Ks):
        a = ax[0, j]
        for st in S.STRATEGIES:
            q = sel(K, st, *S.REF)
            a.plot(q.k, q.median_nrmse.replace(np.inf, np.nan), "o-", color=col[st], label=st, ms=4)
            a.fill_between(q.k, q.ci_lo.clip(upper=10), q.ci_hi.clip(upper=10), color=col[st], alpha=.15)
        a.set_title(f"K = {K}: reference (w1, 24 h hourly)", fontsize=9)
        a = ax[1, j]
        for st in ("random", "source_near"):
            for arm, ls in (("w0.2", ":"), ("w1", "-"), ("w5", "--"), ("ar1", "-.")):
                q = sel(K, st, arm, "S4")
                a.plot(q.k, q.median_nrmse.replace(np.inf, np.nan), ls, color=col[st], label=f"{st} {arm}", ms=3)
        a.set_title(f"K = {K}: noise arms (24 h hourly)", fontsize=9)
        a = ax[2, j]
        for sch, c_ in zip(S.SCHEDULES, plt.cm.viridis(np.linspace(0, .9, len(S.SCHEDULES)))):
            q = sel(K, "random", "w1", sch)
            W, nn = S.SCHEDULES[sch]
            a.plot(q.k, q.median_nrmse.replace(np.inf, np.nan), "o-", color=c_, ms=3, label=f"{sch} W={W} n={nn}")
        a.set_title(f"K = {K}: sampling schedules (random networks, w1)", fontsize=9)
    for a in ax.ravel():
        a.set_xscale("log", base=2)
        a.set_yscale("log")
        a.set_xticks(KS, [str(k) for k in KS])
        a.axhline(.2, color="k", lw=.6, ls=":")
        a.set_xlabel("sensors k")
    for i in range(3):
        ax[i, 0].set_ylabel(f"median nRMSE (forecast / background-only)\n{n} scenarios per K; dotted = tau 0.2")
        ax[i, -1].legend(fontsize=6, loc="best")
    fg.suptitle(f"E08 {pre}: held-out forecast nRMSE vs sensors (bands: distribution-free CI of the median, top row)", fontsize=10)
    fg.tight_layout()
    fg.savefig(fig_dir / f"e08-{pre}-sufficiency.png", dpi=120)
    plt.close(fg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count()))
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude-radius", type=float, default=0, help="m; drop candidate sites within it of any true source (stage-2 variant)")
    ap.add_argument("--estimator", choices=("map_window", "map_full"), default="map_window",
                    help="map_full: all-past-data MAP instead of the schedule's W-hour window (stage-2 variant)")
    ap.add_argument("--only", default=None, help="comma list K:seed (smoke)")
    a = ap.parse_args()
    scale, pre = ("e08sweep", "sweep") if a.sweep else ("e08pilot", "local")
    out = Path(a.out or root / f".cache/e08/{pre}")
    excl, full = a.exclude_radius, a.estimator == "map_full"
    tag = variant_tag(excl, full)
    pre_t = pre + (f"-{tag}" if tag else "")  # output prefix; the reference run (no tag) keeps its names
    if a.report:
        return report(pre_t, tab / f"e08-{pre_t}-rows.csv" if pre == "local" else out / f"e08-{pre_t}-rows.csv", pre == "sweep")
    (out / "fields").mkdir(parents=True, exist_ok=True)
    s = E7.SCALES[scale]
    jobs = [(scale, K, sd, str(out), excl, full) for K in sorted(s["Ks"], reverse=True) for sd in s["seeds"]]  # longest first
    if a.only:
        sel = {tuple(map(int, x.split(":"))) for x in a.only.split(",")}
        jobs = [j for j in jobs if (j[1], j[2]) in sel]
    t0 = time.time()
    tasks = list({n: sp for j in jobs for n, sp in E7.scenario_fields(scale, WIND, j[1], j[2])}.items())
    todo = [t for t in tasks if not (out / "fields" / f"{t[0]}.npy").exists()]
    print(f"stage 1: {len(tasks)} solves, {len(todo)} to run on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as ex:  # fork: forkserver workers lost their semaphores on the cluster (jobs 205035-8)
        for i, f in enumerate(as_completed([ex.submit(run_field, (t, str(out))) for t in todo])):
            if (tm := f.result()) is not None:
                print(f"[{i + 1}/{len(todo)}] {tm['kind']} K={tm['K']} seed={tm['seed']} k={tm['k']} wall={tm['wall_s']:.0f}s", flush=True)
    rows, drows, comp = [], [], []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as ex:  # fork: forkserver workers lost their semaphores on the cluster (jobs 205035-8)
        futs = {ex.submit(run_scenario, j): j for j in jobs}
        for f in as_completed(futs):
            r, dr, tm = f.result()
            rows += r; drows += dr; comp.append(tm)
            print(f"done K={tm['K']} seed={tm['seed']} design={tm['design_wall_s']:.0f}s eval={tm['eval_wall_s']:.0f}s "
                  f"rss={tm['peak_rss_mb']:.0f}MB", flush=True)
    if a.only:
        return
    meta = {**E7.provenance(), "scale": s, "wind": WIND, "workers": a.workers, "exclude_radius_m": excl, "estimator": a.estimator, "total_wall_s": time.time() - t0,
            "design_note": "docs/research/sufficiency-design.md"}
    meta["command"] = "OMP_NUM_THREADS=1 uv run python scripts/e08_sufficiency.py " + " ".join(sys.argv[1:])
    dest = tab if pre == "local" else out  # sweep tables stay next to the caches; the lead curates what enters reports/
    pd.DataFrame(rows).to_csv(dest / f"e08-{pre_t}-rows.csv", index=False)
    pd.DataFrame(drows).to_csv(dest / f"e08-{pre_t}-designs.csv", index=False)
    pd.DataFrame(comp).to_csv(dest / f"e08-{pre_t}-compute.csv", index=False)
    (dest / f"e08-{pre_t}-meta.json").write_text(json.dumps(meta, indent=1, default=str))
    if pre == "local":
        report(pre_t, tab / f"e08-{pre_t}-rows.csv", False)


if __name__ == "__main__":
    main()

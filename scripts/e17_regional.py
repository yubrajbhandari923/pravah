"""E17 regional case: synthetic tracer under ERA5 winds, not validation (design: docs/research/regional-case-design.md).

Pilot (32², local):   OMP_NUM_THREADS=1 uv run python scripts/e17_regional.py --pilot --workers 6
Sweep (64², one job): OMP_NUM_THREADS=1 uv run python scripts/e17_regional.py --sweep --out DIR --workers 64 [--dry-run]
Wind rose only:       uv run python scripts/e17_regional.py --rose
Report from rows:     uv run python scripts/e17_regional.py --pilot --report

Truth and every unit response use the box-mean ERA5 10 m wind (the solver takes spatially uniform winds only).
Stage 1 caches each solve in OUT/fields/<name>.npy (name hashes model, winds and source xy); stage 2 evaluates one
scenario per task: designs from the oracle training responses, the E07 map24 and the all-past-data MAP (map_grow),
held-out forecast RMSE (t+1 h) as the primary endpoint, plus E12-style wrong-model-wind specs on the 'coolA' case.
"""
import argparse, functools, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "src")]
from pravah import benchmarks as B  # noqa: E402
from pravah import uncertainty as U  # noqa: E402
from pravah.data import region as R  # noqa: E402
from pravah.experiment_config import ExperimentConfig, Split  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402

RAW = Path(os.environ.get("PRAVAH_E17_RAW", root / ".cache/e17/raw"))  # set it in a worktree or on the cluster
WORLDPOP = RAW / "worldpop/npl_ppp_2020_1km_Aggregated_UNadj.tif"
LABEL = "synthetic concentration under ERA5 winds (not validation)"
POINT = (28.75, 80.25)  # ERA5 grid point nearest Belauri: the 'one station' model wind of the 'point' spec
# case -> [(UTC start date, days)] concatenated; split = (train, val, test) days. Sweep cases mirror E07 (30/12/18 days).
CASES = {
    "pilot": {"coolA": ([("2021-11-01", 22)], (10, 2, 10)), "coolB": ([("2022-01-01", 22)], (10, 2, 10)),
              "monsoon": ([("2021-11-01", 12), ("2021-07-01", 10)], (10, 2, 10))},
    "sweep": {"coolA": ([("2021-11-01", 60)], (30, 12, 18)), "coolB": ([("2022-01-01", 59)], (30, 12, 17)),
              "monsoon": ([("2021-11-01", 42), ("2021-07-01", 18)], (30, 12, 18))},
}
SCALES = {"pilot": dict(grid=(32, 32), dt=300, scheme="upwind", Ks=(2, 4), ks=(2, 4, 8), seeds=(0, 1, 2), R=3),
          "sweep": dict(grid=(64, 64), dt=240, scheme="vanleer", Ks=(2, 4), ks=(4, 8, 16), seeds=tuple(range(10)), R=5)}
SPECS = (U.ORACLE, U.WindSpec("point"), U.WindSpec("dir", 30))  # wrong-wind specs run on SPEC_CASE only
SPEC_CASE = "coolA"
STRATS = ("dopt", "source_near", "coverage")
N_RAND = 5
ESTS = {"map24": X.WINDOW_H, "map_grow": None}
tab, fig_dir = root / "reports/quarto/tables", root / "reports/quarto/figures"


def make_cfg(scale, case):
    """Regional model: every benchmark length x10 (domain 100 km, footprint 4 km, source spacing 15 km, sensor spacing
    12.5 km) and K x10 (500 m2/s), which keeps the cell Peclet number u dx / K of E07 at equal wind speed."""
    s, (_, sp) = SCALES[scale], CASES[scale][case]
    c = ExperimentConfig.default("EXP-E17")
    c.model.domain_m, c.model.grid, c.model.dt_s, c.model.advection_scheme = (1e5, 1e5), s["grid"], s["dt"], s["scheme"]
    c.model.diffusivity_m2_s, c.model.spinup_h = 500.0, 48
    c.sources.footprint_sigma_m, c.sources.min_separation_m = 4000.0, 15000.0
    c.design.min_sensor_separation_m = 12500.0
    c.split = Split((1, sp[0]), (sp[0] + 1, sp[0] + sp[1]), (sp[0] + sp[1] + 1, sum(sp)))
    c.validate()
    return c


_ERA5 = {}


def era5():
    if "e" not in _ERA5:
        _ERA5["e"] = R.read_era5(sorted(RAW.glob("era5_sl_belauri_*.nc")))
    return _ERA5["e"]


def case_times(scale, case):
    return np.concatenate([R.hourly_times(d, n) for d, n in CASES[scale][case][0]])


def _ro(*a):
    for x in a:
        x.flags.writeable = False  # cached arrays are shared between callers
    return a if len(a) > 1 else a[0]


@functools.cache
def case_winds(scale, case, spec=U.ORACLE):
    """Hourly (T, 2) model winds: box mean of the bilinear ERA5 field (truth/oracle) or a wrong model wind."""
    e, t = era5(), case_times(scale, case)
    w = R.model_winds(e, R.BELAURI, SCALES[scale]["grid"], t)[0]
    if spec.family == "point":
        i, j = np.searchsorted(e["time"], t), (np.abs(e["lat"] - POINT[0]).argmin(), np.abs(e["lon"] - POINT[1]).argmin())
        return _ro(np.stack([e["u10"][i, j[0], j[1]], e["v10"][i, j[0], j[1]]], 1))
    return _ro(U.perturb_winds(w, spec, 0))


@functools.cache
def masks(grid):
    c, la, lo = R.read_worldpop(WORLDPOP)
    pop, vf = R.aggregate_counts(c, la, lo, R.BELAURI, grid)
    pl, _ = R.aggregate_counts(c, la, lo, R.BELAURI, (12, 12))
    return _ro(pop, vf, R.source_mask(pl))


def drivers(scale, case, K, seed):
    cfg = make_cfg(scale, case)
    cells, a = B.draw_sources(cfg, seed, K, allowed=masks(cfg.model.grid)[2])
    L, (na, nb) = cfg.model.domain_m, cfg.sources.lattice
    sxy = np.array([[(p + .5) * L[0] / na, (q + .5) * L[1] / nb] for p, q in cells])
    hrs = {s: (getattr(cfg.split, s)[1] - getattr(cfg.split, s)[0] + 1) * 24 for s in B.SPLITS}
    reg = {s: [f"era5:{case}"] for s in B.SPLITS}
    return cfg, (cells, a, sxy, case_winds(scale, case), reg, hrs)


def specs_for(case):
    return SPECS if case == SPEC_CASE else SPECS[:1]


def field_tasks(scale, case, K, seed):
    """[(name, (scale, case, K, seed, kind, k, spec name))]: truth + K unit responses per spec."""
    cfg, (_, a, sxy, w, _, _) = drivers(scale, case, K, seed)
    out = []
    base = json.dumps({"model": asdict(cfg.model), "sources": asdict(cfg.sources)}, sort_keys=True).encode()
    h = lambda *b: hashlib.sha256(base + b"|".join(b)).hexdigest()[:20]
    out.append((f"truth_{h(w.tobytes(), sxy.tobytes(), a.tobytes())}", (scale, case, K, seed, "truth", -1, "oracle")))
    for sp in specs_for(case):
        wm = case_winds(scale, case, sp)
        out += [(f"src_{h(wm.tobytes(), xy.tobytes())}", (scale, case, K, seed, "src", k, sp.name)) for k, xy in enumerate(sxy)]
    return out


def _spec(name):
    return next(s for s in SPECS if s.name == name)


def run_field(task):
    (name, (scale, case, K, seed, kind, k, spn)), out = task
    f = Path(out) / "fields" / f"{name}.npy"
    if f.exists():
        return None
    w0 = time.time()
    cfg, drv = drivers(scale, case, K, seed)
    if kind == "truth":
        field = B.generate_scenario(cfg, seed, K, drivers=drv).truth.field
    else:
        field = X.source_field(cfg, drv[2][k], case_winds(scale, case, _spec(spn)), len(drv[3]))
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, field)
    os.replace(tmp, f)
    return dict(name=name, kind=kind, wall_s=time.time() - w0)


# ---------------------------------------------------------------- stage 2
def evaluate(job):
    scale, case, K, seed, out = job
    s = SCALES[scale]
    cfg, drv = drivers(scale, case, K, seed)
    ft = field_tasks(scale, case, K, seed)
    ld = lambda n: np.load(Path(out) / "fields" / f"{n}.npy")
    sc = B.generate_scenario(cfg, seed, K, truth_field=ld(ft[0][0]), drivers=drv)
    Fm = {sp.name: np.stack([ld(n) for n, t in ft[1:] if t[6] == sp.name]) for sp in specs_for(case)}
    pop, vf, _ = masks(cfg.model.grid)
    wpop = pop / pop.sum()
    a_ref, cbg, a_true = cfg.sources.reference_rate_ug_s, cfg.sources.background_ug_m3, sc.truth.strengths_ug_s
    tf, T, issue = sc.truth.field, len(sc.truth.winds), X.issue_times(sc)
    T_train, seedp = len(sc.split_hours("train")), cfg.seeds.streams(seed)["placement"]
    cand = sc.admissible_sites()
    cand = cand[R.candidate_mask(vf, sc.site_ij[cand])]  # deployable: inside Nepal (WorldPop raster)
    sigma = X.site_sigma(sc, False)
    Y = [X.readings(sc, sigma, "white", r, T) for r in range(s["R"])]
    G3 = {n: X.site_responses(sc, F) for n, F in Fm.items()}
    p = X.problem(X.design_input(sc, G3["oracle"], sigma), cand=cand)
    key = dict(scale=scale, case=case, K=K, seed=seed)
    lin = float(np.abs(cbg + np.einsum("k,khxy->hxy", a_true, Fm["oracle"]) - tf).max() / (tf - cbg).max())
    rows = []
    zero = X.score(np.zeros((len(issue), K)), Fm["oracle"], tf, issue, a_true, a_ref, cbg, wpop)
    pers = tf[issue] - tf[issue + 1]
    rows.append({**key, "spec": "-", "estimator": "-", "k": 0, "strategy": "background_only", **zero})
    rows.append({**key, "spec": "-", "estimator": "-", "k": 0, "strategy": "persistence",
                 "forecast_rmse": float(np.sqrt(np.mean(pers ** 2))), "popw_rmse": float(np.sqrt((pers ** 2 * wpop).sum((1, 2)).mean()))})
    designs = {}
    for k in s["ks"]:
        for name, net in [("random", j) for j in range(N_RAND)] + [(n, 0) for n in STRATS]:
            designs[(k, name, net)] = X.make_design(name, p, k, T_train, seedp, net)[0]
    test_h = sc.split_hours("test")
    for spn, F in Fm.items():
        derr, sratio = U.wind_error(sc.truth.winds[test_h], case_winds(scale, case, _spec(spn))[test_h])
        for (k, name, net), S in designs.items():
            A = a_ref * G3[spn][S] / sigma[S][:, None, None]
            for est, win in ESTS.items():
                m = [X.score(X.estimate_series(A, (y[S] - cbg) / sigma[S][:, None], issue, window=win), F, tf, issue,
                             a_true, a_ref, cbg, wpop) for y in Y]
                rows.append({**key, "spec": spn, "dir_err_deg": derr, "speed_ratio": sratio, "estimator": est, "k": k,
                             "strategy": name, "net": net, "sites": " ".join(map(str, S)),
                             **{c: float(np.mean([x[c] for x in m])) for c in m[0]}})
    tm = {**key, "n_cand": len(cand), "n_hours": T, "n_issue": len(issue), "superposition_rel_max": lin,
          "a_true": " ".join(f"{x:.4g}" for x in a_true / a_ref), "source_xy_km": " ".join(f"{x/1e3:.1f}" for x in sc.truth.source_xy.ravel())}
    return rows, tm


def run(scale, out, workers, dry=False):
    jobs = [(scale, c, K, sd) for c in CASES[scale] for K in SCALES[scale]["Ks"] for sd in SCALES[scale]["seeds"]]
    tasks = {n: t for j in jobs for n, t in field_tasks(*j)}
    todo = [((n, t), out) for n, t in tasks.items() if not (Path(out) / "fields" / f"{n}.npy").exists()]
    print(f"{len(jobs)} scenarios, {len(tasks)} solves, {len(todo)} to do", flush=True)
    if dry:
        sys.exit(0)
    w0 = time.time()
    ctx = get_context("spawn")
    timing = []
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        for i, r in enumerate(ex.map(run_field, todo)):
            timing.append(r)
            print(f"solve {i + 1}/{len(todo)} {r['kind']} {r['wall_s']:.0f}s", flush=True)
        res = list(ex.map(evaluate, [j + (out,) for j in jobs]))
    rows = pd.DataFrame([r for rr, _ in res for r in rr])
    meta = pd.DataFrame([tm for _, tm in res])
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root).stdout.strip()
    prov = dict(scale=scale, git=rev, workers=workers, wall_s=time.time() - w0, solves=len(todo),
                solve_wall_s=[t["wall_s"] for t in timing if t], label=LABEL)
    return rows, meta, prov


# ---------------------------------------------------------------- analysis
def summarise(rows):
    """Paired relative forecast-RMSE reductions per (case, spec, estimator, k): strategy vs random (mean of the N_RAND
    networks) and dopt vs source_near; cluster = (K, seed)."""
    d = rows[rows.spec != "-"].groupby(["case", "spec", "K", "seed", "estimator", "k", "strategy"], as_index=False)[
        ["forecast_rmse", "popw_rmse", "source_rmse"]].mean()
    out = []
    for (case, spec, est, k), g in d.groupby(["case", "spec", "estimator", "k"]):
        pv = g.pivot_table(index=["K", "seed"], columns="strategy", values="forecast_rmse")
        pairs = [(s, "random") for s in STRATS] + [("dopt", "source_near")]
        for a, b in pairs:
            rho = ((pv[b] - pv[a]) / pv[b]).to_numpy()
            pos, neg, _, pval = X.sign_test(rho)
            out.append(dict(case=case, spec=spec, estimator=est, k=k, comparison=f"{a}_vs_{b}", n=len(rho),
                            median_rel_reduction=float(np.median(rho)), min=float(rho.min()), max=float(rho.max()),
                            wins=pos, losses=neg, sign_p=pval))
    return pd.DataFrame(out)


def wrong_wind(rows):
    """E12-style D = forecast RMSE(spec) / RMSE(oracle), per design x estimator x k on SPEC_CASE (median over clusters)."""
    d = rows[(rows.case == SPEC_CASE) & (rows.spec != "-")].groupby(
        ["spec", "K", "seed", "estimator", "k", "strategy"], as_index=False)[["forecast_rmse", "dir_err_deg", "speed_ratio"]].mean()
    o = d[d.spec == "oracle"].set_index(["K", "seed", "estimator", "k", "strategy"]).forecast_rmse
    d["D"] = d.forecast_rmse.to_numpy() / o.loc[list(zip(d.K, d.seed, d.estimator, d.k, d.strategy))].to_numpy()
    return d.groupby(["spec", "estimator", "k", "strategy"], as_index=False).agg(
        n=("D", "size"), D_median=("D", "median"), D_min=("D", "min"), D_max=("D", "max"),
        dir_err_deg=("dir_err_deg", "median"), speed_ratio=("speed_ratio", "median"))


def rose_table(scale="pilot"):
    """Box-mean wind statistics per calendar month and per pilot/sweep split segment (UTC)."""
    e = era5()
    w = R.model_winds(e, R.BELAURI, SCALES[scale]["grid"])[0]
    s, th = U.to_polar(w)
    mon = e["time"].astype("datetime64[M]")
    pt = [(e["lat"][i], e["lon"][j]) for i in range(len(e["lat"])) for j in range(len(e["lon"]))]
    rows = []
    for m in np.unique(mon):
        i = mon == m
        th_m = np.degrees(np.arctan2(-w[i, 0].mean(), -w[i, 1].mean())) % 360
        rose = R.wind_rose(w[i])
        dom = rose.sum(1).argmax()
        rows.append(dict(month=str(m), hours=int(i.sum()), mean_speed=float(s[i].mean()), p95_speed=float(np.percentile(s[i], 95)),
                         max_speed=float(s[i].max()), frac_calm_lt1=float((s[i] < 1).mean()),
                         vector_mean_from_deg=float(th_m), vector_mean_speed=float(np.hypot(*w[i].mean(0))),
                         steadiness=float(np.hypot(*w[i].mean(0)) / s[i].mean()), modal_from_sector_deg=float(dom * 22.5),
                         modal_sector_frac=float(rose.sum(1)[dom]),
                         blh_median_m=float(np.median(e["blh"][i])) if "blh" in e else np.nan,
                         gridpoint_speed_max=float(np.hypot(e["u10"][i], e["v10"][i]).max())))
    return pd.DataFrame(rows), w, mon


def plot_rose(w, mon, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = {"Cool season Nov 2021-Feb 2022": mon != np.datetime64("2021-07"), "Monsoon Jul 2021": mon == np.datetime64("2021-07")}
    fig, axs = plt.subplots(1, 2, subplot_kw=dict(projection="polar"), figsize=(10, 5.2))
    bins = ["<1", "1-2", "2-3", "3-5", ">5"]
    cols = plt.cm.viridis(np.linspace(0, .9, len(bins)))
    for ax, (t, i) in zip(axs, groups.items()):
        r = R.wind_rose(w[i]) * 100
        th = np.radians(np.arange(16) * 22.5)
        bot = np.zeros(16)
        for b in range(r.shape[1]):
            ax.bar(th, r[:, b], width=np.radians(20), bottom=bot, color=cols[b], label=f"{bins[b]} m/s", edgecolor="white", lw=.4)
            bot += r[:, b]
        ax.set_theta_zero_location("N"), ax.set_theta_direction(-1)
        ax.set_title(f"{t}\n(n = {int(i.sum())} h; % of hours, direction FROM)", fontsize=10)
    axs[1].legend(loc="lower left", bbox_to_anchor=(1.0, 0), fontsize=8, title="box-mean speed")
    fig.suptitle("ERA5 10 m box-mean wind, Belauri 100 km box (UTC hourly). Forcing for: " + LABEL, fontsize=10)
    fig.text(.5, .01, "Contains modified Copernicus Climate Change Service information [2021-2022]", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .03, 1, .95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_results(summ, ww, path, what="local pilot (32²"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    o = summ[summ.spec == "oracle"]
    cases, ests = list(CASES["pilot"]), list(ESTS)
    comps = [f"{s}_vs_random" for s in STRATS] + ["dopt_vs_source_near"]
    fig, axs = plt.subplots(len(ests), len(cases) + 1, figsize=(15, 7))
    mk = dict(zip(comps, "osv^"))
    for r, est in enumerate(ests):
        for c, case in enumerate(cases):
            ax = axs[r, c]
            for cp in comps:
                g = o[(o.case == case) & (o.estimator == est) & (o.comparison == cp)]
                ax.plot(g.k, g.median_rel_reduction, marker=mk[cp], label=cp)
                ax.fill_between(g.k, g["min"], g["max"], alpha=.08)
            ax.axhline(0, color="k", lw=.6)
            ax.set_title(f"{case} | {est}", fontsize=9)
            ax.set_xlabel("sensors k")
        lo, hi = zip(*[axs[r, c].get_ylim() for c in range(len(cases))])
        for c in range(len(cases)):
            axs[r, c].set_ylim(min(lo), max(hi))
        axs[r, 0].set_ylabel("median rel. forecast-RMSE reduction\n(> 0: first design better)")
        ax = axs[r, -1]
        g = ww[(ww.estimator == est) & (ww.spec != "oracle")]
        for (sp, st), gg in g.groupby(["spec", "strategy"]):
            if st in ("dopt", "source_near", "random"):
                ax.plot(gg.k, np.log(gg.D_median), marker="o" if sp.startswith("point") else "x", label=f"{sp}:{st}")
        ax.axhline(0, color="k", lw=.6)
        ax.set_title(f"wrong model wind ({SPEC_CASE}) | {est}\nlog D = log RMSE(spec)/RMSE(oracle)", fontsize=9)
        ax.set_xlabel("sensors k")
        ax.legend(fontsize=6)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"E17 {what}, n = {int(summ.n.max())} clusters per cell, held-out forecast t+1 h): {LABEL}", fontsize=10)
    fig.text(.5, .005, "Bands: min-max over clusters. Contains modified Copernicus Climate Change Service information [2021-2022]",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .02, 1, .96))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--sweep", action="store_true")
    g.add_argument("--rose", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default=str(RAW.parent / "pilot"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true", help="only count the solves still to do")
    a = ap.parse_args()
    if a.rose or a.pilot:
        t, w, mon = rose_table()
        t.to_csv(tab / "e17-local-winds.csv", index=False)
        plot_rose(w, mon, fig_dir / "e17-local-windrose.png")
        print(t.to_string(index=False))
        if a.rose:
            return
    scale = "pilot" if a.pilot else "sweep"
    pre = "local" if a.pilot else "sweep"
    dest = tab if a.pilot else Path(a.out)
    if not a.report:
        rows, meta, prov = run(scale, a.out, a.workers, a.dry_run)
        rows.to_csv(dest / f"e17-{pre}-rows.csv", index=False)
        meta.to_csv(dest / f"e17-{pre}-scenarios.csv", index=False)
        (dest / f"e17-{pre}-provenance.json").write_text(json.dumps(prov, indent=1))
    rows = pd.read_csv(dest / f"e17-{pre}-rows.csv")
    summ, ww = summarise(rows), wrong_wind(rows)
    summ.to_csv(dest / f"e17-{pre}-summary.csv", index=False)
    ww.to_csv(dest / f"e17-{pre}-wrongwind.csv", index=False)
    bl = rows[rows.spec == "-"].groupby(["case", "strategy"]).forecast_rmse.median()
    print(bl.to_string())
    print(summ[(summ.spec == "oracle")].to_string(index=False))
    print(ww.to_string(index=False))
    if not a.pilot:  # sweep outputs live on the cluster; copy the small summaries into the report
        summ.to_csv(tab / f"e17-{pre}-summary.csv", index=False)
        ww.to_csv(tab / f"e17-{pre}-wrongwind.csv", index=False)
    plot_results(summ, ww, fig_dir / f"e17-{pre}-results.png", "local pilot (32²" if a.pilot else "sweep (64², VanLeer")


if __name__ == "__main__":
    main()

"""E16 generalization / sensitivity cells (design: docs/research/generalization-ablations.md). Reuses the E07 two-stage
driver and its content-hashed field caches; each sensitivity cell is registered as one E07 scale with config overrides.

Pilot (32^2, local):  OMP_NUM_THREADS=1 uv run python scripts/e16_generalization.py --pilot --workers 6
Sweep (64^2, HPC):    ... --sweep --out DIR --workers 64   (stage 1 solves, then stage 2 per scenario; rerun to resume)
Reuse check:          ... --sweep --out DIR --dry-run   (solves still missing per cell; nominal must be 0 at 64^2)
Report only:          uv run python scripts/e16_generalization.py --pilot --report   (or --sweep --report --out DIR)

`bg_hi` changes only the background, which enters the truth as an additive constant (the anomaly is solved with c_bg = 0)
and not the unit responses, so it reuses the nominal fields with the truth shifted (no solves). At 64^2 every cell uses
VanLeer dt 30 s (amendment A2), so `diff_lo` and the interaction coincide with nominal / loss_hi and are not run there;
the 32^2-vs-64^2 resolution contrast is 32^2 `diff_lo` (VanLeer) against 64^2 nominal (VanLeer).
"""
import argparse, importlib.util, json, multiprocessing as mp, os, pickle, resource, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from pravah import benchmarks as B  # noqa: E402
from pravah.experiments import generalization as G  # noqa: E402

_spec = importlib.util.spec_from_file_location("e07_driver", root / "scripts" / "e07_cross_objective.py")
E7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E7)

WIND = "mixed"
BASE = {
    "local": dict(grid=(32, 32), dt=300, scheme="upwind", spinup=12, split=((1, 4), (5, 5), (6, 11)), Ks=(2, 4), ks=G.KS,
                  seeds=tuple(range(5)), R=5, winds=(WIND,)),  # = E07 pilot world, mixed winds
    "sweep": dict(grid=(64, 64), dt=30, scheme="vanleer", spinup=48, split=None, Ks=(2, 4), ks=G.KS,  # = E07/E08 sweep model
                  seeds=tuple(range(20)), R=5, winds=(WIND,)),
}
CELLS = {"local": list(G.CELLS), "sweep": ["nominal", "diff_hi", "loss_hi", "src_wide", "src_sparse", "bg_hi"]}
SHARE = {"bg_hi": "nominal"}  # cell -> cell whose fields it reuses (truth shifted by the background difference)
scale_name = lambda pre, cell: f"e16{pre}_{cell}"
for _pre, _cells in CELLS.items():  # registered at import, so worker processes see them too
    for _c in _cells:
        G.validate_cell(G.CELLS[_c])
        E7.SCALES[scale_name(_pre, _c)] = {**BASE[_pre], "overrides": G.CELLS[_c]}
for _c, _s in SHARE.items():  # sharing is exact only if the cells differ in the background alone
    assert set(G.CELLS[_c]) ^ set(G.CELLS[_s]) <= {"sources.background_ug_m3"}
tab, fig_dir = root / "reports/quarto/tables", root / "reports/quarto/figures"


def cfg_of(pre, cell):
    return E7.make_cfg(scale_name(pre, cell), WIND)


def run_field(task):  # module-level wrapper: E7's functions are not picklable by reference
    return E7.run_field(task)


def run_scenario(job):
    pre, cell, K, seed, out = job
    src = SHARE.get(cell, cell)
    fl = E7.scenario_fields(scale_name(pre, src), WIND, K, seed)
    f = Path(out) / "scen" / f"{cell}_K{K}_s{seed}_{fl[0][0]}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    w0, c0 = time.time(), time.process_time()
    sc, F, G3, tm = E7.load_scenario(scale_name(pre, src), WIND, K, seed, out)
    cfg = cfg_of(pre, cell)
    if src != cell:  # exact: truth = c_bg + anomaly, anomaly and unit responses independent of c_bg
        sc = B.generate_scenario(cfg, seed, K, truth_field=sc.truth.field + cfg.sources.background_ug_m3 - sc.cfg.sources.background_ug_m3)
    s = E7.SCALES[scale_name(pre, cell)]
    t1 = time.time()
    rows = G.evaluate(sc, F, G3, s["ks"], s["R"])
    lin = float(np.abs(cfg.sources.background_ug_m3 + np.einsum("k,khxy->hxy", sc.truth.strengths_ug_s, F) - sc.truth.field).max())
    key = dict(cell=cell, K=K, seed=seed)
    tm.update(key, eval_wall_s=time.time() - t1, total_wall_s=time.time() - w0, total_cpu_s=time.process_time() - c0,
              linearity_max_abs=lin, n_hours=len(sc.truth.winds), fields_from=src, source_xy=" ".join(f"{x:.0f}" for x in sc.truth.source_xy.ravel()),
              a_true=" ".join(f"{x:.4g}" for x in sc.truth.strengths_ug_s), peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    res = ([{**key, **r} for r in rows], tm)
    f.parent.mkdir(exist_ok=True)
    E7._save_atomic(f, lambda fh: fh.write(pickle.dumps(res)))
    return res


def cell_table(pre):
    out = []
    for c in CELLS[pre]:
        cfg = cfg_of(pre, c)
        out.append(dict(cell=c, overrides=json.dumps(G.CELLS[c]), fields_from=SHARE.get(c, c),
                        scheme=cfg.model.advection_scheme, dt_s=cfg.model.dt_s, K_phys_m2_s=cfg.model.diffusivity_m2_s,
                        removal_per_s=cfg.model.removal_per_s, footprint_sigma_m=cfg.sources.footprint_sigma_m,
                        min_separation_m=cfg.sources.min_separation_m, background_ug_m3=cfg.sources.background_ug_m3, **G.dimensionless(cfg)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- report (reads only the row table)
def report(pre, rows_csv, dest):
    df = pd.read_csv(rows_csv)
    d = G.with_nrmse(df)
    rho = G.contrast_rows(d)
    vt = G.verdict_table(rho)
    kt, ks_ = G.kendall_table(d)
    kk, kv = G.kstar_table(d)
    rb = G.robustness(vt, kv)
    sec = pd.concat([G.verdict_table(G.contrast_rows(d, k)).assign(k=k) for k in G.KS if k != G.K_REF and k > 2], ignore_index=True)
    for name, t in (("verdicts", vt), ("verdicts-k", sec), ("kendall", ks_), ("kendall-scen", kt), ("kstar", kv), ("kstar-scen", kk),
                    ("robustness", rb), ("contrasts-scen", rho)):
        t.to_csv(dest / f"e16-{pre}-{name}.csv", index=False)
    plot(vt, ks_, kv, pre)
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_rows", 200):
        print(vt[["conclusion", "cell", "n", "median", "ci_lo", "ci_hi", "wins", "losses", "verdict", "delta_vs_nominal"]].to_string(index=False))
        print(ks_.to_string(index=False))
        print(kv.to_string(index=False))
        print(rb.to_string(index=False))


def plot(vt, ks_, kv, pre):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = [c for c in G.CELLS if c in set(vt.cell)]
    concl = list(G.CONTRASTS) + list(G.MEMORY)
    M = vt.pivot(index="conclusion", columns="cell", values="median").reindex(index=concl, columns=cells)
    V = vt.set_index(["conclusion", "cell"])
    fg, ax = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw=dict(width_ratios=[2.4, 1.2, 1.3]))
    lim = np.nanmax(np.abs(M.replace(-np.inf, np.nan).to_numpy()))
    im = ax[0].imshow(M.replace(-np.inf, np.nan).to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    mark = {"holds": "", "weakened": " ~", "flipped (indicative)": " ×?", "reversed": " ×"}
    for i, c in enumerate(concl):
        for j, cell in enumerate(cells):
            r = V.loc[(c, cell)]
            ax[0].text(j, i, f"{r['median']:+.2f}{mark[r.verdict]}\n[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]\n{r.wins}/{r.n}", ha="center",
                       va="center", fontsize=7)
    ax[0].set_xticks(range(len(cells)), cells, rotation=25, fontsize=8)
    ax[0].set_yticks(range(len(concl)), concl, fontsize=8)
    ax[0].set_title("median paired relative RMSE gain (k = 8, held-out 1 h forecast) [95 % DF CI] wins/n\n"
                    "> 0 = conclusion's direction; ~ weakened, ×? flipped (CI straddles), × reversed", fontsize=9)
    fg.colorbar(im, ax=ax[0], shrink=.8)
    for est, off, col in (("map24", -.2, "#d95f02"), ("map_grow", .2, "#7570b3")):
        q = ks_[ks_.estimator == est].set_index("cell").reindex(cells)
        ax[1].barh(np.arange(len(cells)) + off, q.frac_same_top, .4, color=col, label=f"{est} (median τ in text)")
        for y, (v, t) in enumerate(zip(q.frac_same_top, q.median_tau)):
            ax[1].text(v + .02, y + off, f"τ={t:.2f}", va="center", fontsize=7)
    ax[1].set_yticks(range(len(cells)), cells, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlim(0, 1.3)
    ax[1].set_xlabel("fraction of scenarios whose best strategy (k = 8) equals nominal's")
    ax[1].legend(fontsize=7, loc="lower right")
    ax[1].set_title("ranking stability vs nominal (4 strategies)", fontsize=9)
    sts = list(G.STRATEGIES)
    for j, K in enumerate(sorted({int(c.split("_K")[1]) for c in kv.columns if c.startswith("med_kstar_")})):
        for i, st in enumerate(sts):
            y = kv.set_index("cell").reindex(cells)[f"med_kstar_{st}_K{K}"]
            ax[2].scatter(y, np.arange(len(cells)) + (i - 1.5) * .12 + j * .5 - .25, s=18, marker="os"[j],
                          color=["#888888", "#1b9e77", "#d95f02", "#7570b3"][i], label=f"{st} K={K}")
    ax[2].set_xscale("log", base=2)
    ax[2].set_xticks([2, 4, 8, 16, 64], ["2", "4", "8", "16", ">16"])
    ax[2].set_yticks(range(len(cells)), [f"{c} ({v})" for c, v in zip(cells, kv.set_index("cell").reindex(cells).verdict)], fontsize=8)
    ax[2].invert_yaxis()
    ax[2].set_xlabel("median k*(0.2) (map24)")
    ax[2].legend(fontsize=6, ncol=2, loc="lower right")
    ax[2].set_title("C5: sensors needed (o K=2, □ K=4)", fontsize=9)
    n = int(vt.n.max())
    fg.suptitle(f"E16 {pre} (32², exploratory): do E07/E08/E09 conclusions survive one-at-a-time physics/source changes? "
                f"n = {n} scenarios per cell", fontsize=10)
    fg.tight_layout()
    fg.savefig(fig_dir / f"e16-{pre}-regime-ranks.png", dpi=120)
    plt.close(fg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count()))
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma list cell:K:seed (smoke)")
    ap.add_argument("--n-seeds", type=int, default=None, help="first N seeds only (staged 64^2 runs; later runs resume)")
    ap.add_argument("--cells", default=None, help="comma list: run/report only these cells")
    ap.add_argument("--dry-run", action="store_true", help="print the solves still to run per cell (cache-reuse check) and exit")
    a = ap.parse_args()
    if a.pilot == a.sweep:
        ap.error("give exactly one of --pilot / --sweep")
    pre = "sweep" if a.sweep else "local"
    out = Path(a.out or root / f".cache/e16/{pre}")
    dest = tab if pre == "local" else out  # sweep tables stay next to the caches; the lead curates what enters reports/
    if a.report:
        return report(pre, dest / f"e16-{pre}-rows.csv", dest)
    (out / "fields").mkdir(parents=True, exist_ok=True)
    jobs = [(pre, c, K, sd, str(out)) for c in (a.cells.split(",") if a.cells else CELLS[pre]) for K in sorted(BASE[pre]["Ks"], reverse=True) for sd in BASE[pre]["seeds"][:a.n_seeds]]
    if a.only:
        sel = {tuple(x.split(":")) for x in a.only.split(",")}
        jobs = [j for j in jobs if (j[1], str(j[2]), str(j[3])) in sel]
    t0 = time.time()
    tasks = {n: sp for j in jobs for n, sp in E7.scenario_fields(scale_name(pre, SHARE.get(j[1], j[1])), WIND, j[2], j[3])}
    dt = lambda t: E7.SCALES[t[1][1]]["overrides"].get("model.dt_s", BASE[pre]["dt"])  # smallest dt (slowest) first
    todo = sorted([t for t in tasks.items() if not (out / "fields" / f"{t[0]}.npy").exists()], key=dt)
    print(f"stage 1: {len(tasks)} solves, {len(todo)} to run on {a.workers} workers", flush=True)
    if a.dry_run:
        return print(pd.Series([t[1][1] for t in todo], dtype=str).value_counts().to_dict())
    ctx = mp.get_context("fork")  # fork: forkserver workers lost their semaphores on the cluster (E08 jobs 205035-8)
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex:
        for i, f in enumerate(as_completed([ex.submit(run_field, (t, str(out))) for t in todo])):
            if (tm := f.result()) is not None:
                print(f"[{i + 1}/{len(todo)}] {tm['kind']} K={tm['K']} seed={tm['seed']} k={tm['k']} wall={tm['wall_s']:.0f}s", flush=True)
    rows, comp = [], []
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_scenario, j): j for j in jobs}
        for f in as_completed(futs):
            r, tm = f.result()
            rows += r; comp.append(tm)
            print(f"done {tm['cell']} K={tm['K']} seed={tm['seed']} eval={tm['eval_wall_s']:.0f}s", flush=True)
    if a.only:
        return
    stage1 = [json.loads((out / "fields" / f"{n}.json").read_text()) for n in tasks]
    meta = {**E7.provenance(), "base": BASE[pre], "cells": {c: G.CELLS[c] for c in sorted({j[1] for j in jobs})}, "share": SHARE, "wind": WIND,
            "workers": a.workers, "total_wall_s": time.time() - t0, "design_note": "docs/research/generalization-ablations.md",
            "stage1_solves": len(tasks), "stage1_cpu_s": sum(x["cpu_s"] for x in stage1)}
    meta["command"] = "OMP_NUM_THREADS=1 uv run python scripts/e16_generalization.py " + " ".join(sys.argv[1:])
    pd.DataFrame(rows).to_csv(dest / f"e16-{pre}-rows.csv", index=False)
    pd.DataFrame(comp).to_csv(dest / f"e16-{pre}-compute.csv", index=False)
    pd.DataFrame(stage1).to_csv(dest / f"e16-{pre}-solves.csv", index=False)
    cell_table(pre).to_csv(dest / f"e16-{pre}-cells.csv", index=False)
    (dest / f"e16-{pre}-meta.json").write_text(json.dumps(meta, indent=1, default=str))
    report(pre, dest / f"e16-{pre}-rows.csv", dest)


if __name__ == "__main__":
    main()

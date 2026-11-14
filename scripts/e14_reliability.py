"""E14 reliability diagnostics (design: docs/research/reliability-design.md). Reuses the E12 pilot pipeline unmodified.

Fields first (E12 driver, private cache; ~2.1 CPU-h at 32²):
  OMP_NUM_THREADS=1 uv run python scripts/e12_wind_mismatch.py --pilot --workers 4 --out .cache/e14/e12pilot \
      --only 4:0,4:1,4:2,2:0,2:1,2:2
Pilot:   OMP_NUM_THREADS=1 uv run python scripts/e14_reliability.py --fields .cache/e14/e12pilot --workers 4
Report:  uv run python scripts/e14_reliability.py --report
"""
import argparse, json, os, platform, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
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
from pravah.experiments import cross_objective as X  # noqa: E402
from pravah.experiments import reliability as RL  # noqa: E402

SCALE, KS_SRC, SEEDS, KS, N_RAND, R = "pilot", (2, 4), (0, 1, 2), (2, 4, 8), 5, 3
ESTS = {"map24": 24, "map_full": None}
EXTRA = ("noise_x2", "noise_ar1", "omit1")       # simple non-wind mismatches, oracle wind (design §2)
N_VAL_BLOCKS = 2
LABELS = ("worse_than_persistence", "coverage_failure")
tab, fig_dir = root / "reports/quarto/tables", root / "reports/quarto/figures"


def run_cluster(job):
    K, seed, fields = job
    specs = E12.SCALES[SCALE]["specs"]
    names = E12.field_names(SCALE, K, seed, specs)
    dirs = [Path(fields) / "fields"]
    sc = B.generate_scenario(E7.make_cfg(SCALE, E12.WIND), seed, K, truth_field=np.load(E12._find(names["truth"][0][0], dirs)))
    Fm = {sp.name: np.stack([np.load(E12._find(n, dirs)) for n, _ in names[sp.name]]) for sp in specs}
    a_ref, cbg = sc.cfg.sources.reference_rate_ug_s, sc.cfg.sources.background_ug_m3
    T, N = len(sc.truth.winds), int(np.prod(sc.cfg.model.grid))
    tf = sc.truth.field.reshape(T, N) - cbg                       # truth: labels only
    a_true = sc.truth.strengths_ug_s / a_ref
    sigma = X.site_sigma(sc, False)
    site_truth = np.asarray(sc.truth.site_truth)[:, :T]
    Yw = np.stack([X.readings(sc, sigma, "white", r, T) for r in range(R)], 2)
    Ya = np.stack([X.readings(sc, sigma, "ar1", r, T) for r in range(R)], 2)
    T_train, seedp = len(sc.split_hours("train")), sc.cfg.seeds.streams(seed)["placement"]
    G3 = {n: X.site_responses(sc, F) for n, F in Fm.items()}
    p0 = X.problem(X.design_input(sc, G3["oracle"], sigma))
    designs = {(k, name, net): X.make_design(name, p0, k, T_train, seedp, net)[0]
               for k in KS for name, net in [("random", j) for j in range(N_RAND)] + [("dopt", 0), ("source_near", 0)]}
    issue = X.issue_times(sc)
    blocks = [issue[i:i + RL.W_H] for i in range(0, len(issue) - RL.W_H + 1, RL.W_H)]
    sd_act = 0.01 * float(np.median(sigma))
    arms = [(sp.name, sp.family, sp.magnitude, sp.name, Yw, slice(None)) for sp in specs]
    arms += [("noise_x2", "noise", 2.0, "oracle", 2 * Yw - site_truth[..., None], slice(None)),   # 2x the same draws
             ("noise_ar1", "noise", 0.8, "oracle", Ya, slice(None)),
             ("omit1", "basis", 1.0, "oracle", Yw, slice(0, K - 1))]
    rows = []
    for arm, fam, mag, resp, Y, ks in arms:
        Fr = a_ref * Fm[resp].reshape(K, T, N)[ks]
        G = G3[resp][:, :, ks]
        for (k, name, net), S in designs.items():
            A = a_ref * G[S] / sigma[S][:, None, None]
            z = (Y[S] - cbg) / sigma[S][:, None, None]
            for est, win in ESTS.items():
                for b, bl in enumerate(blocks):
                    d = RL.block_diagnostics(A, z, int(bl[0]), win, Fr[:, bl[0] + 1])
                    mu, sd, u, usd = RL.forecast(A, z, bl, win, Fr)
                    lab = RL.block_labels(mu, sd, tf, bl, sd_act, u, usd, a_true[ks])
                    for r in range(R):
                        rows.append(dict(K=K, seed=seed, cluster=f"{K}:{seed}", arm=arm, family=fam, magnitude=mag, k=k,
                                         design=name, net=net, estimator=est, draw=r, block=b,
                                         split="validation" if b < N_VAL_BLOCKS else "evaluation",
                                         **{n: float(v[r]) for n, v in d.items()},
                                         **{n: (bool(v[r]) if v.dtype == bool else float(v[r])) for n, v in lab.items()}))
    return rows


# ---------------------------------------------------------------- analysis (reads only the unit table)
def summarise(df):
    ev, va = df[df.split == "evaluation"], df[df.split == "validation"]
    # deviation 1 (post hoc): within-configuration AUC; strata = what a deployer knows (k, design class, estimator)
    st = ev.k.astype(str) + "|" + ev.design + "|" + ev.estimator
    auc, fam, thr = [], [], []
    for lab in LABELS:
        for dg in RL.DIAGS:
            auc.append(dict(label=lab, diagnostic=dg, scope="pooled", base_rate=float(ev[lab].mean()),
                            **RL.cluster_auc(ev[dg], ev[lab], ev.cluster)))
            auc.append(dict(label=lab, diagnostic=dg, scope="within_config", base_rate=float(ev[lab].mean()),
                            **RL.cluster_auc(ev[dg], ev[lab], ev.cluster, n_boot=500, strata=st)))
            for f in [f for f in df.arm.unique() if f != "oracle"]:
                g = ev[ev.arm.isin(["oracle", f])]
                fam.append(dict(label=lab, diagnostic=dg, arm=f, base_rate=float(g[lab].mean()),
                                **RL.cluster_auc(g[dg], g[lab], g.cluster, n_boot=300)))
                w = RL.cluster_auc(g[dg], g[lab], g.cluster, n_boot=300, strata=st[g.index])
                fam[-1].update(auc_within=w["auc"], within_ci_lo=w["ci_lo"], within_ci_hi=w["ci_hi"])
            t = RL.youden_threshold(va[dg], va[lab])
            flag = ev[dg] >= t
            y = ev[lab].astype(bool)
            ratio = ev.rmse_model / ev.rmse_persist
            thr.append(dict(label=lab, diagnostic=dg, threshold_val=t, tpr=float(flag[y].mean()), fpr=float(flag[~y].mean()),
                            precision=float(y[flag].mean()) if flag.any() else np.nan, frac_flagged=float(flag.mean()),
                            risk_all=float(ratio.median()), risk_kept=float(ratio[~flag].median()) if (~flag).any() else np.nan,
                            fail_rate_all=float(y.mean()), fail_rate_kept=float(y[~flag].mean()) if (~flag).any() else np.nan))
    pit = [f"pit{j}" for j in range(10)]
    cal = ev.groupby(["arm", "estimator"]).agg(
        n=("cov90", "size"), cov50=("cov50", "median"), cov90=("cov90", "median"), cov95=("cov95", "median"),
        crps=("crps", "median"), cov95_a=("cov95_a", "median"), fail_cov=("coverage_failure", "mean"),
        fail_persist=("worse_than_persistence", "mean"), skill_ratio=("rmse_model", "median")).reset_index()
    cal["skill_ratio"] = ev.assign(q=ev.rmse_model / ev.rmse_persist).groupby(["arm", "estimator"]).q.median().values
    pt = ev.groupby("arm")[pit].sum()
    pt = pt.div(pt.sum(1), axis=0).reset_index()
    # overconfident false negatives: failing units that no diagnostic flags at its validation threshold
    th = {(r["label"], r["diagnostic"]): r["threshold_val"] for r in thr}
    lab = "worse_than_persistence"
    anyflag = np.zeros(len(ev), bool)
    for dg in RL.DIAGS:
        anyflag |= (ev[dg] >= th[(lab, dg)]).to_numpy()
    miss = ev[ev[lab].to_numpy() & ~anyflag].assign(ratio=lambda x: x.rmse_model / x.rmse_persist)
    miss = miss.sort_values("ratio", ascending=False).head(30)
    return pd.DataFrame(auc), pd.DataFrame(fam), pd.DataFrame(thr), cal, pt, miss, int(anyflag.sum()), int(ev[lab].sum())


def report():
    df = pd.read_csv(tab / "e14-local-units.csv.gz")
    auc, fam, thr, cal, pit, miss, n_any, n_fail = summarise(df)
    for n, d in dict(auc=auc, auc_family=fam, thresholds=thr, calibration=cal, pit=pit, misses=miss).items():
        d.to_csv(tab / f"e14-local-{n}.csv", index=False, float_format="%.4g")
    print(auc.to_string(index=False, float_format="%.3f"))
    print(f"units flagged by >=1 diagnostic: {n_any}; failing units (L1): {n_fail}; unflagged failures: {len(miss)} shown")
    plot(df, auc, fam, cal, pit)


def plot(df, auc, fam, cal, pit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = df[df.split == "evaluation"]
    fg, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for j, lab in enumerate(LABELS):
        y = ev[lab].astype(bool).to_numpy()
        for i, dg in enumerate(RL.DIAGS):
            s = ev[dg].to_numpy()
            ok = np.isfinite(s)
            ts = np.quantile(s[ok], np.linspace(0, 1, 101))
            tpr = [(s[ok & y] >= t).mean() for t in ts]
            fpr = [(s[ok & ~y] >= t).mean() for t in ts]
            a = auc[(auc.label == lab) & (auc.diagnostic == dg) & (auc.scope == "within_config")].iloc[0]
            ax[j].plot(fpr, tpr, color=f"C{i}", lw=1.2, ls="--" if dg in ("sensor_fit", "spread", "nis") else "-",
                       label=f"{dg} {a.auc:.2f} [{a.ci_lo:.2f},{a.ci_hi:.2f}]")
        ax[j].plot([0, 1], [0, 1], "k:", lw=.7)
        ax[j].set(xlabel="false-alarm rate", ylabel="hit rate", title=f"{lab} (base rate {y.mean():.2f})")
        ax[j].legend(fontsize=6.5, title="pooled ROC curves; legend: within-config AUC [cluster boot. 95 %]", title_fontsize=6.5)
    pv = fam[fam.label == "worse_than_persistence"].pivot(index="diagnostic", columns="arm", values="auc_within").loc[list(RL.DIAGS)]
    im = ax[2].imshow(pv.to_numpy(), cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax[2].set_xticks(range(pv.shape[1]), pv.columns, rotation=60, fontsize=7)
    ax[2].set_yticks(range(pv.shape[0]), pv.index, fontsize=7)
    for (i, j), v in np.ndenumerate(pv.to_numpy()):
        ax[2].text(j, i, "–" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6)
    ax[2].set_title("within-config AUC by arm (oracle + arm), worse-than-persistence", fontsize=9)
    fg.colorbar(im, ax=ax[2], shrink=.8)
    fg.suptitle("E14 32² pilot: truth-free early warnings on held-out blocks (6 clusters; exploratory)", fontsize=10)
    fg.tight_layout()
    fg.savefig(fig_dir / "e14-local-roc.png", dpi=130)
    plt.close(fg)

    arms = list(cal[cal.estimator == "map_full"].sort_values("cov90", ascending=False).arm)
    fg, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for est, mk in (("map24", "s"), ("map_full", "o")):
        c = cal[cal.estimator == est].set_index("arm").loc[arms]
        x = np.arange(len(arms)) + (0.15 if est == "map_full" else -0.15)
        for lv, col in ((50, "C0"), (90, "C1"), (95, "C3")):
            ax[0].plot(x, c[f"cov{lv}"], mk, color=col, ms=4, label=f"{lv} % ({est})")
            ax[0].axhline(lv / 100, color=col, lw=.6, ls=":")
    ax[0].set_xticks(range(len(arms)), arms, rotation=60, fontsize=7)
    ax[0].set(ylabel="median empirical coverage (field, 1 h)", title="nominal field-forecast intervals")
    ax[0].legend(fontsize=6, ncol=2)
    for a_ in ("oracle", "spd0.8", "dir30", "lag6", "noise_ar1", "omit1"):
        if a_ in set(pit.arm):
            ax[1].step(np.arange(10) / 10 + .05, pit.set_index("arm").loc[a_].to_numpy(), where="mid", label=a_)
    ax[1].axhline(.1, color="k", lw=.6, ls=":")
    ax[1].set(xlabel="PIT", ylabel="fraction", title="PIT of field forecasts (active cells, both estimators)")
    ax[1].legend(fontsize=7)
    fg.tight_layout()
    fg.savefig(fig_dir / "e14-local-calibration.png", dpi=130)
    plt.close(fg)


def provenance():
    git = lambda *c: subprocess.run(["git", *c], cwd=root, capture_output=True, text=True).stdout.strip()
    return dict(command="OMP_NUM_THREADS=1 uv run python scripts/e14_reliability.py " + " ".join(sys.argv[1:]),
                git_rev=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain", "--", "src", "scripts")),
                host=platform.node(), python=platform.python_version(), numpy=np.__version__,
                slurm_job=os.environ.get("SLURM_JOB_ID"), design_note="docs/research/reliability-design.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default=str(root / ".cache/e14/e12pilot"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report()
    t0 = time.time()
    jobs = [(K, s, a.fields) for K in KS_SRC for s in SEEDS]
    with ProcessPoolExecutor(a.workers, mp_context=get_context("fork")) as ex:
        rows = [r for rs in ex.map(run_cluster, jobs) for r in rs]
    pd.DataFrame(rows).to_csv(tab / "e14-local-units.csv.gz", index=False, float_format="%.6g")
    meta = dict(**provenance(), Ks=KS_SRC, seeds=SEEDS, ks=KS, n_random=N_RAND, R=R, estimators=ESTS, extra_arms=EXTRA,
                n_val_blocks=N_VAL_BLOCKS, block_h=RL.W_H, n_units=len(rows), wall_s=time.time() - t0, workers=a.workers)
    (tab / "e14-local-meta.json").write_text(json.dumps(meta, indent=1, default=str))
    report()


if __name__ == "__main__":
    main()

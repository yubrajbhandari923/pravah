"""E06 step 3: illustrative placement designs at smoke scale (32^2, 12 h spin-up, local).

Run: `uv run python scripts/e06_placement.py` (~30 min). Scenarios: winds {steady_westerly, veering} x seeds {0, 1} x
K {2, 4} random sources (benchmark draw). Designs (pravah.placement) at k = 2/4/8 on two candidate sets:
  full  = admissible lattice sites (exclusion mask: outer ring, within 1 footprint sigma of a source); min separation binds;
  small = admissible sites on every 3rd lattice node (<= 25 candidates; spacing 1875 m > min separation, so it never binds),
          where exhaustive search gives each greedy objective's optimality gap.
Design inputs are TRAINING-split only: G3 over the HOURS = 96 training hours (known source locations, the truth's own
solver and winds), POD snapshots = noise-free truth over the train hours, sigma = preregistered scenario sigma,
a_ref = SourceConfig().reference_rate_ug_s. The validation split (never test) is used only to SCORE field reconstruction.
Outputs: reports/quarto/tables/e06-criteria.csv (one row per scenario x candidate set x design x k), e06-*.md fragments,
e06-meta.json and reports/quarto/figures/e06-designs.png.
"""
import json, subprocess, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src")); sys.path.insert(0, str(root / "scripts"))
import e02_examples as E  # noqa: E402
from pravah import benchmarks as B  # noqa: E402
from pravah import placement as P  # noqa: E402
from pravah.experiment_config import SourceConfig  # noqa: E402
from pravah.identifiability import practical_rank  # noqa: E402
from pravah.inverse import rank_info  # noqa: E402
from pravah.sensitivity import scenario_response_matrix  # noqa: E402

# HOURS = all 96 training hours: in the first 24 h the veering and steady streams are identical (same seed, same first block)
WINDS, SEEDS, KS, BUDGETS, HOURS = ("steady_westerly", "veering"), (0, 1), (2, 4), (2, 4, 8), 96
OWN = {"dopt": ("dopt", True), "aopt": ("aopt", True), "diversity": ("coherence", False), "podqr": ("podqr_value", True),
       "podqr_num": ("podqr_value", True)}
tab, fig_dir = root / "reports/quarto/tables", root / "reports/quarto/figures"


def field_recon_error(p, S, Y):
    """Relative error of the r-mode POD reconstruction (r = min(k, r_eff)) of validation snapshots Y (n_sites, T) from the
    sites S (noise-free readings; r = 0 predicts the training mean, error 1)."""
    X = p.snapshots
    mu = X.mean(1, keepdims=True)
    if min(len(S), p.r_eff) == 0:
        return 1.0
    Psi = p.pod(min(len(S), p.r_eff))
    c = np.linalg.lstsq(Psi[S], (Y - mu)[S], rcond=None)[0]
    return float(np.linalg.norm(mu + Psi @ c - Y) / np.linalg.norm(Y - mu))


def evaluate(p, S, sigma, Y):
    G = p.G3[S].reshape(-1, p.G3.shape[2])
    d = np.hypot(*(p.xy[S][:, None] - p.xy[S][None]).transpose(2, 0, 1)) + np.eye(len(S)) * 1e18
    return dict(dopt=P.dopt(p, S), aopt=P.aopt(p, S), eopt=P.eopt(p, S), coherence=P.coherence(p, S),
                podqr_value=P.podqr_value(p, S), practical_rank=practical_rank(G, sigma, p.a_ref), float_rank=rank_info(G)["rank"],
                n_above_noise=int((p.snr[S] >= 1).sum()), min_spacing_m=float(d.min()) if len(S) > 1 else np.nan,
                mean_dist_to_source_m=float(np.hypot(*(p.xy[S][:, None] - p.source_xy[None]).transpose(2, 0, 1)).min(1).mean()),
                field_rel_err_val=field_recon_error(p, S, Y))


def main():
    rows, maps, t_all = [], {}, time.time()
    for wind in WINDS:
        for seed in SEEDS:
            for K in KS:
                t0 = time.time()
                sc = B.generate_scenario(E.make_cfg([wind]), seed, K)
                G3 = scenario_response_matrix(sc, HOURS).reshape(sc.n_candidate_sites, HOURS, K)
                t_sim = time.time() - t0
                sigma, Y = float(sc.noise_sigma[0]), np.asarray(sc.truth.site_truth)[:, sc.split_hours("validation")]
                ii, jj = np.divmod(np.arange(sc.n_candidate_sites), sc.lattice_n)
                adm = sc.admissible_sites()
                small = adm[(ii[adm] % 3 == 1) & (jj[adm] % 3 == 1)]
                rseed = sc.cfg.seeds.streams(seed)["placement"]
                for cs, cand in (("full", adm), ("small", small)):
                    p = P.problem_from_scenario(sc, G3, cand)
                    for k in BUDGETS:
                        bf = {}
                        if cs == "small":
                            for name, (crit, mx) in OWN.items():
                                ids = p.cand[p.snr[p.cand] >= 1] if name == "diversity" else None  # diversity's goal is among above-noise sites
                                if (name == "podqr" and k > p.r_eff) or (name == "podqr_num" and k > p.r_num):
                                    continue  # design undefined
                                t1 = time.time()
                                v, bS, nf = (P.brute_force(p, k, getattr(P, crit), mx, ids) if ids is None or len(ids) >= k else (np.nan, None, 0))
                                bf[name] = (v, nf, time.time() - t1)
                        for name in P.STRATEGIES:
                            t1 = time.time()
                            base = dict(wind=wind, seed=seed, K=K, cand_set=cs, n_cand=len(cand), k=k, design=name, r_eff=p.r_eff, r_num=p.r_num,
                                        sigma=sigma, a_ref=p.a_ref, sim_runtime_s=t_sim)
                            try:
                                S = P.design(name, p, k, seed=rseed)
                            except ValueError as e:  # recorded, not hidden (podqr with k > r_eff)
                                rows.append({**base, "status": str(e)})
                                continue
                            rt = time.time() - t1
                            r = dict(**base, status="ok", sites=" ".join(map(str, S)), runtime_s=rt, **evaluate(p, S, sigma, Y))
                            if name in bf:
                                crit, mx = OWN[name]
                                v, nf, bt = bf[name]
                                ok = name != "diversity" or (p.snr[S] >= 1).all()
                                r.update(own_criterion=crit, brute_value=v if ok else np.nan, brute_n_feasible=nf, brute_runtime_s=bt,
                                         opt_gap=((v - r[crit]) if mx else (r[crit] - v)) if ok else np.nan)
                            rows.append(r)
                            if cs == "full" and seed == 0 and K == 4:
                                maps[(wind, name, k)] = S
                    if cs == "full" and seed == 0 and K == 4:
                        maps[(wind, "_snr")] = p.snr.reshape(sc.lattice_n, sc.lattice_n)
                        maps[(wind, "_src")] = sc.truth.source_xy
                        maps[(wind, "_xy")] = sc.site_xy
                print(f"done {wind} seed={seed} K={K} sim={t_sim:.0f}s total={time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(tab / "e06-criteria.csv", index=False)
    plot(maps)
    meta = dict(command="uv run python scripts/e06_placement.py", git_rev=subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                capture_output=True, text=True).stdout.strip(), winds=WINDS, seeds=SEEDS, K=KS, budgets=BUDGETS, G_hours=HOURS,
                a_ref=SourceConfig().reference_rate_ug_s, total_runtime_s=time.time() - t_all, grid=[32, 32], spinup_h=12)
    (tab / "e06-meta.json").write_text(json.dumps(meta, indent=1))
    report()


def md(df, digits=3):
    f = lambda v: f"{v:.{digits}g}" if isinstance(v, float) else str(v)
    return ("| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * df.shape[1] + "\n"
            + "\n".join("| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)) + "\n")


def report():
    """Markdown fragments for the Quarto chapter, computed from e06-criteria.csv and e06-meta.json only."""
    df = pd.read_csv(tab / "e06-criteria.csv")
    meta = json.loads((tab / "e06-meta.json").read_text())
    undefined = df[df.status != "ok"].groupby(["cand_set", "design", "k"]).size().rename("undefined_cases").reset_index()
    modes = df.groupby(["wind", "seed", "K", "cand_set"])[["r_eff", "r_num"]].first().reset_index()
    df = df[df.status == "ok"]
    full, small = df[df.cand_set == "full"], df[df.cand_set == "small"].dropna(subset=["own_criterion"])
    # 1. greedy vs exhaustive on the small set (own objective)
    # relative gap is not meaningful for coherence (bounded in [0, 1], optimum often ~0): absolute only there
    g = small.assign(rel_gap=(small.opt_gap / small.brute_value.abs()).where(small.own_criterion != "coherence"))
    gap = g.groupby(["design", "own_criterion", "k"]).agg(cases=("design", "size"), evaluable=("opt_gap", "count"),
                                                          exact=("opt_gap", lambda x: int((x <= 1e-9).sum())),
                                                          median_gap=("opt_gap", "median"), max_gap=("opt_gap", "max"),
                                                          max_rel_gap=("rel_gap", "max"), max_brute_s=("brute_runtime_s", "max"),
                                                          max_feasible_subsets=("brute_n_feasible", "max")).reset_index()
    # 2. does each design attain the best value of its own objective among all eight designs (full set)?
    best = []
    for d, (crit, mx) in OWN.items():
        n = hit = 0
        for _, grp in full.groupby(["wind", "seed", "K", "k"]):
            if d not in grp.design.values:
                continue  # design undefined in this case
            top = grp[crit].max() if mx else grp[crit].min()
            n += 1
            hit += int(abs(grp.loc[grp.design == d, crit].iloc[0] - top) <= 1e-9)
        best.append(dict(design=d, criterion=crit, larger_is_better=mx, cases=n, attains_best=hit))
    best = pd.DataFrame(best)
    # 3. cross-criterion medians over scenarios (full set)
    cols = ["dopt", "aopt", "eopt", "coherence", "podqr_value", "field_rel_err_val", "mean_dist_to_source_m", "min_spacing_m"]
    cross = full.groupby(["k", "design"], sort=False)[cols].median().reset_index()
    # 4. practical rank at the preregistered sigma: mean [min-max] over scenarios
    pr = full.groupby(["K", "k", "design"], sort=False).practical_rank.agg(["min", "mean", "max"]).reset_index()
    prw = pr.assign(v=pr.apply(lambda r: f"{r['mean']:.2g} [{r['min']:.0f}-{r['max']:.0f}]", axis=1)).pivot(
        index=["K", "k"], columns="design", values="v").reset_index().reindex(columns=["K", "k", *P.STRATEGIES]).fillna("undefined")
    rt = full.groupby("design", sort=False).runtime_s.agg(["median", "max"]).reset_index()
    for name, t in (("gap", gap), ("best", best), ("cross", cross), ("prank", prw), ("runtime", rt), ("undefined", undefined),
                    ("modes", modes)):
        (tab / f"e06-{name}.md").write_text(md(t))
    ex, ev = int(gap.exact.sum()), int(gap.evaluable.sum())
    scen = df.groupby(["wind", "seed", "K"])
    txt = (f"{len(scen)} scenarios, {len(df)} design rows, total wall time {meta['total_runtime_s'] / 60:.1f} min "
           f"(simulation {scen.sim_runtime_s.first().sum() / 60:.1f} min). Preregistered sigma across scenarios: "
           f"{sorted(df.sigma.round(4).unique().tolist())} ug m^-3; a_ref = {meta['a_ref']:g} ug s^-1. "
           f"Above-noise POD modes of the training snapshots r_eff = {sorted(modes.r_eff.unique().tolist())} (numerical rank "
           f"r_num = {sorted(modes.r_num.unique().tolist())}), so noise-aware POD/QR was defined in {len(df[df.design == 'podqr'])} "
           f"of {len(df[df.design == 'podqr']) + int(undefined[undefined.design == 'podqr'].undefined_cases.sum())} design cases. "
           f"On the small candidate set the greedy/QR designs reached the exhaustive optimum of their own objective in {ex} of {ev} "
           f"evaluable cases; the largest absolute gap was {gap.max_gap.max():.3g} and the largest relative gap {gap.max_rel_gap.max():.3g}. "
           f"Practical rank at the preregistered sigma (full set) ranged {int(full[full.K == 4].practical_rank.min())}-"
           f"{int(full[full.K == 4].practical_rank.max())} of 4 for K = 4 and {int(full[full.K == 2].practical_rank.min())}-"
           f"{int(full[full.K == 2].practical_rank.max())} of 2 for K = 2, over all designs and budgets.\n")
    (tab / "e06-results.md").write_text(txt)
    print(txt)


def plot(maps):
    names = list(P.STRATEGIES)
    fg, ax = plt.subplots(len(WINDS), len(names), figsize=(2.6 * len(names), 2.9 * len(WINDS)), squeeze=False)
    for r, w in enumerate(WINDS):
        xy, snr, src = maps[(w, "_xy")], maps[(w, "_snr")], maps[(w, "_src")]
        ext = (0, 10000, 0, 10000)
        for c, n in enumerate(names):
            a = ax[r, c]
            im = a.imshow(np.log10(np.maximum(snr, 1e-3)).T, origin="lower", extent=ext, cmap="Greys", vmin=-3, vmax=1.5)
            a.plot(*src.T, "r*", ms=10)
            for kk, sty in ((8, dict(mfc="none", mec="C0", ms=9, mew=1.4)), (4, dict(color="C1", ms=5))):
                if (w, n, kk) in maps:  # podqr may be undefined at this k
                    a.plot(*xy[maps[(w, n, kk)]].T, "o", **sty)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(n, fontsize=10)
            if c == 0:
                a.set_ylabel(w, fontsize=10)
    fg.colorbar(im, ax=ax, shrink=0.8, label="log10 single-site SNR (largest sv of a_ref G_i / sigma)")
    fg.suptitle("Designs, seed 0, K = 4, full candidate set. orange dots: k = 4; blue rings: k = 8 (none = design undefined); red stars: sources",
                fontsize=10)
    fg.savefig(fig_dir / "e06-designs.png", dpi=130, bbox_inches="tight"); plt.close(fg)


if __name__ == "__main__":
    report() if "--report" in sys.argv else main()

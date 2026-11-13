"""E04 (EXP-005, T-025): footprints for two winds, response-matrix heatmap, singular spectrum, confidence ellipses for an
informative vs a redundant design, practical rank at the preregistered noise, and a near-null source counterexample.

Run: `uv run python scripts/e04_footprints.py`. Smoke scale (32^2, 24 h window), perfect model, known
source locations (the E02 'changing4' layout), two training wind regimes. G comes from the truth's own solver. Designs are
chosen from G only (greedy Bayesian D-opt / farthest-point coverage / clustered), never from truth strengths; truth
strengths enter ONLY the counterexample (the 'true vector'). Practical rank: modes whose reference-sized change (A_REF) moves
the data by >= 1 sigma (pravah.identifiability). The informative design is chosen at an extra, NON-preregistered noise level
(0.1 x rms of the reference signal) because at the preregistered sigma nothing is above noise. Outputs: figure, csv, md fragments.
"""
import csv, json, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src")); sys.path.insert(0, str(root / "scripts"))
from pravah.experiment_config import SourceConfig  # noqa: E402
from pravah import benchmarks as B  # noqa: E402
from pravah.identifiability import greedy_design, null_counterexample, practical_rank  # noqa: E402
from pravah.information import confidence_ellipse, fisher, rank_coherence  # noqa: E402
from pravah.observations import bilinear_sample  # noqa: E402
from pravah.sensitivity import _anomaly_fields  # noqa: E402
import e02_examples as E  # noqa: E402

HOURS, A_REF, N, R_INFO, TOL, REP = 24, SourceConfig().reference_rate_ug_s, 4, 0.1, 1e-8, 2
WINDS = ("steady_westerly", "veering")
rd = Path(os.environ["RESEARCH_RUN_DIR"]) if os.environ.get("RESEARCH_RUN_DIR") else None
tab, fig_dir = root / "reports/quarto/tables", root / "reports/quarto/figures"
metrics = []


def log(name, v):
    metrics.append({"name": name, "value": float(v), "step": 0})


def build(wind):
    rep, cells, a, _, _, _ = E.EXAMPLES["changing4"]
    with E.fixed_sources(cells, a):
        sc = B.generate_scenario(E.make_cfg([wind]), REP, len(cells))
    m, dom = sc.cfg.model, tuple(sc.cfg.model.domain_m)
    F = np.array([_anomaly_fields(sc, B.source_basis(tuple(m.grid), dom, xy, sc.cfg.sources.footprint_sigma_m)
                                  / m.mixing_depth_m, HOURS) for xy in sc.truth.source_xy])  # (K, hour, i, j)
    G3 = np.array([np.stack([bilinear_sample(c, sc.site_xy, dom) for c in f], 1) for f in F]).transpose(1, 2, 0)  # (site, hour, K)
    return sc, F, G3


def main():
    sc, F, G3 = {}, {}, {}
    for w in WINDS:
        sc[w], F[w], G3[w] = build(w)
    s0 = sc[WINDS[0]]
    a_true, K = s0.truth.strengths_ug_s, len(s0.truth.strengths_ug_s)
    assert all(np.allclose(sc[w].truth.strengths_ug_s, a_true) and np.allclose(sc[w].site_xy, s0.site_xy) for w in WINDS)
    sig_pre = float(s0.view("train", E._design(s0, N, False), "tuning").noise_sigma[0])  # preregistered, via the leak-checked view
    adm = np.asarray(s0.admissible_sites())
    Gw0 = G3[WINDS[0]]
    sig_info = R_INFO * float(np.sqrt(np.mean((Gw0[adm] @ np.full(K, A_REF)) ** 2)))  # from G only
    noise = {"prereg": sig_pre, "info_scale": sig_info}
    xy = s0.site_xy
    sep = s0.cfg.design.min_sensor_separation_m
    # designs (chosen under the first wind, same sites evaluated under both winds)
    informative = greedy_design(Gw0[adm], xy[adm], N, sig_info, A_REF, sep)
    informative = [int(adm[i]) for i in informative]
    c0 = xy[informative[0]]  # redundant: the N admissible sites nearest the first informative site (min separation respected)
    redundant = []
    for i in adm[np.argsort(np.hypot(*(xy[adm] - c0).T))]:
        if all(np.hypot(*(xy[i] - xy[j])) >= sep for j in redundant):
            redundant.append(int(i))
        if len(redundant) == N:
            break
    cover = [int(adm[np.argmax(np.hypot(*(xy[adm] - xy[adm].mean(0)).T))])]  # coverage: farthest-point sampling
    while len(cover) < N:
        d = np.min([np.hypot(*(xy[adm] - xy[j]).T) for j in cover], 0)
        cover.append(int(adm[np.argmax(d)]))
    designs = {"informative": informative, "redundant": redundant, "coverage": cover}
    for n, d in designs.items():
        log(f"{n}_min_pairwise_sep_m", min(np.hypot(*(xy[i] - xy[j])) for i in d for j in d if i < j))

    rows, ell = [], {}
    for w in WINDS:
        for dn, ids in designs.items():
            Gs = G3[w][ids].reshape(-1, K)
            for nn, sg in noise.items():
                M = fisher(Gs, sg, np.eye(K) * A_REF ** 2)  # posterior information with prior sd A_REF (no unregularised CRB: singular)
                rc = rank_coherence(Gs / sg)
                ce = null_counterexample(Gs, sg, a_true)
                ax = confidence_ellipse(M, [0, 1])["semi_axes"]
                prior_ax = confidence_ellipse(np.eye(K) / A_REF ** 2, [0, 1])["semi_axes"]
                ell[(w, dn, nn)] = confidence_ellipse(M, [0, 1])
                rows.append(dict(wind=w, design=dn, sites=" ".join(map(str, ids)), noise=nn, sigma_ug_m3=sg, n_obs=len(Gs),
                                 float_rank=rc["rank"], practical_rank=practical_rank(Gs, sg, A_REF),
                                 s_max_scaled=float(rc["singular_values"][0] * A_REF), s_min_scaled=float(rc["singular_values"][-1] * A_REF),
                                 coherence=rc["coherence"], posterior_logdet=float(np.linalg.slogdet(M)[1]),
                                 ellipse_area_over_prior=float(np.prod(ax) / np.prod(prior_ax)),
                                 cx_rel_source_diff=ce["rel_diff"], cx_data_diff_sigma=ce["data_diff_sigma"],
                                 cx_similar_fit_diff_source=bool(ce["data_diff_sigma"] < 1 and ce["rel_diff"] > 0.1)))
                for k in ("float_rank", "practical_rank", "coherence", "cx_data_diff_sigma", "cx_rel_source_diff"):
                    log(f"{w}.{dn}.{nn}.{k}", rows[-1][k])
    with open(tab / "e04-identifiability.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, rows[0].keys()); wr.writeheader(); wr.writerows(rows)

    # ---- figure
    fg, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    Lx, Ly = s0.cfg.model.domain_m
    for j, w in enumerate(WINDS):
        a_ = ax[0, j]
        fp = F[w][:, -1].sum(0)  # all-source unit-response footprint at the last window hour
        im = a_.imshow(fp.T, origin="lower", extent=(0, Lx, 0, Ly), norm=LogNorm(vmin=fp.max() * 1e-4, vmax=fp.max()), cmap="viridis")
        a_.plot(*np.array(s0.truth.source_xy).T, "r*", ms=13, label="source")
        a_.plot(*xy[informative].T, "wo", mec="k", label="informative sensors")
        a_.plot(*xy[redundant].T, "ms", mec="k", label="redundant sensors")
        a_.set_title(f"footprint, wind {w} (sum of unit responses, h {HOURS})", fontsize=9); a_.set_xlabel("x [m]"); a_.set_ylabel("y [m]")
        fg.colorbar(im, ax=a_, label="ug m$^{-3}$ per ug s$^{-1}$ (log)")
    ax[0, 0].legend(fontsize=7, loc="upper right")
    a_ = ax[0, 2]
    Gh = G3[WINDS[0]][informative].reshape(-1, K)
    im = a_.imshow(np.log10(np.abs(Gh) + 1e-18), aspect="auto", cmap="magma", vmin=np.log10(np.abs(Gh).max()) - 8)
    a_.set_xlabel("source k"); a_.set_ylabel(f"(sensor, hour) row, {N} sensors x {HOURS} h"); a_.set_xticks(range(K))
    a_.set_title(f"response matrix |G|, informative design, wind {WINDS[0]}", fontsize=9); fg.colorbar(im, ax=a_, label="log10 |G|")
    a_ = ax[1, 0]
    mk = {"informative": "o", "redundant": "s", "coverage": "^"}
    for (w, ls) in zip(WINDS, ("-", ":")):
        for dn, ids in designs.items():
            s = np.linalg.svd(G3[w][ids].reshape(-1, K), compute_uv=False)
            for nn, c in (("prereg", "C0"), ("info_scale", "C1")):
                a_.semilogy(range(1, K + 1), np.maximum(s * A_REF / noise[nn], 1e-30), ls, color=c, marker=mk[dn], ms=5, alpha=.8)
    a_.axhline(1, color="k", lw=1); a_.set_ylim(1e-12, None)
    a_.set_xticks(range(1, K + 1)); a_.set_xlabel("mode index"); a_.set_ylabel("s_i * A_ref / sigma  (1 = one noise sigma)")
    a_.set_title("singular spectrum. colour: C0 prereg sigma, C1 info-scale sigma;\nmarker: design; solid/dotted: wind 1/2", fontsize=8)
    a_ = ax[1, 1]
    for dn, c in (("informative", "C2"), ("redundant", "C3")):
        for nn, ls in (("info_scale", "-"), ("prereg", "--")):
            e = ell[(WINDS[0], dn, nn)]
            a_.add_patch(Ellipse((0, 0), 2 * e["semi_axes"][0], 2 * e["semi_axes"][1], angle=np.degrees(e["angle"]), fill=False, ec=c, ls=ls, lw=1.6,
                                 label=f"{dn}, {nn}"))
    lim = 1.15 * A_REF * np.sqrt(5.99)
    a_.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="source 0 error [ug/s]", ylabel="source 1 error [ug/s]")
    a_.legend(fontsize=7); a_.set_title("95% local ellipses (posterior, prior sd = A_ref). dashed at prereg sigma\nlie on the prior circle: no information", fontsize=8)
    a_ = ax[1, 2]
    xs, wd = np.arange(len(designs)), 0.2
    for i, (nn, w) in enumerate([(n, w) for n in noise for w in WINDS]):
        v = [next(r["cx_data_diff_sigma"] for r in rows if r["design"] == d and r["noise"] == nn and r["wind"] == w) for d in designs]
        a_.bar(xs + (i - 1.5) * wd, np.maximum(v, 1e-12), wd, label=f"{nn}, {w}")
    a_.set_yscale("log"); a_.axhline(1, color="k", lw=1); a_.set_xticks(xs); a_.set_xticklabels(designs)
    a_.set_ylabel("rms data change / sigma"); a_.legend(fontsize=7)
    a_.set_title("counterexample: data change for a source vector moved along the weakest mode\n(below 1 = fits equally well)", fontsize=8)
    fg.tight_layout(); fg.savefig(fig_dir / "e04-footprints.png", dpi=130); plt.close(fg)

    # ---- generated markdown (all numbers computed above)
    g = lambda x: f"{x:.3g}"
    cols = ["wind", "design", "noise", "sigma_ug_m3", "float_rank", "practical_rank", "s_min_scaled", "coherence", "ellipse_area_over_prior",
            "cx_rel_source_diff", "cx_data_diff_sigma", "cx_similar_fit_diff_source"]
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join(
        "| " + " | ".join(g(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |" for r in rows) + "\n"
    n_sim = sum(r["cx_similar_fit_diff_source"] for r in rows)
    pre = [r for r in rows if r["noise"] == "prereg"]
    inf = [r for r in rows if r["noise"] == "info_scale"]
    txt = (f"At the preregistered noise sigma = {g(sig_pre)} ug m^-3 the practical rank is {sorted({r['practical_rank'] for r in pre})} "
           f"(values across the 6 wind/design cases) of K = {K} sources, although the floating-point rank (tol {TOL:g}) is "
           f"{sorted({r['float_rank'] for r in pre})}. At the non-preregistered information scale sigma = {g(sig_info)} ug m^-3 "
           f"the practical rank is {sorted({r['practical_rank'] for r in inf})}. "
           f"In {n_sim} of {len(rows)} wind/design/noise cases a source vector moved by a median {g(np.median([r['cx_rel_source_diff'] for r in rows]))} "
           f"of ||a|| along the weakest mode changes the data by less than one noise sigma (rms); the smallest and largest such data changes are "
           f"{g(min(r['cx_data_diff_sigma'] for r in rows))} and {g(max(r['cx_data_diff_sigma'] for r in rows))} sigma.\n")
    (tab / "e04-identifiability.md").write_text(md); (tab / "e04-results.md").write_text(txt)
    meta = dict(command="python3 scripts/e04_footprints.py", seed_replicate=REP, winds=list(WINDS), noise=noise, A_REF=A_REF, hours=HOURS,
                truth_ids=[sc[w].truth_id for w in WINDS], sites={k: v for k, v in designs.items()})
    (tab / "e04-meta.json").write_text(json.dumps(meta, indent=1, default=float))
    if rd:
        with open(rd / "metrics.jsonl", "a") as f:
            f.writelines(json.dumps(m) + "\n" for m in metrics)
        with open(rd / "artifacts.jsonl", "a") as f:
            for p in (fig_dir / "e04-footprints.png", tab / "e04-identifiability.csv", tab / "e04-meta.json"):
                f.write(json.dumps({"path": str(p)}) + "\n")
    print(txt)


main()

"""E03 static inverse study (T-015): K in 2/4/8 sources x n in 2/4/8 sensors x 3 replicate seeds x noise levels, known wind.

Run: `uv run python scripts/e03_inversion.py` (`--smoke`: K=2, one seed, no outputs registered).
Smoke scale (32^2 grid, 12 h spin-up, 24 h window of train hours 0-23), perfect model, known source locations and wind.
G comes from the truth's own solver (sensitivity._anomaly_fields); the estimator sees only G, readings, sigma and the
network. Truth strengths/fields are used ONLY to score. Noise levels sigma0 = r * rms(noise-free anomaly over all admissible
sites), r in R_LEVELS, per-site sigma_i = sigma0 * site_sigma(...) (heteroscedastic so weighting is not a no-op); plus
the preregistered homoscedastic sigma (0.05 x median truth) read through Scenario.view. Ridge strength is fixed a priori.
Everything in the tables is computed here; nothing is typed in.
"""
import csv, json, os, sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src")); sys.path.insert(0, str(root / "scripts"))
from pravah.experiment_config import SourceConfig  # noqa: E402
from pravah import benchmarks as B  # noqa: E402
from pravah.experiment_config import ExperimentConfig, Split  # noqa: E402
from pravah.inverse import rank_info, solve_nnls  # noqa: E402
from pravah.inversion_study import crb_fraction, failure_mode, site_sigma, source_scores  # noqa: E402
from pravah.observations import bilinear_sample  # noqa: E402
from pravah.sensitivity import _anomaly_fields, scenario_response_matrix  # noqa: E402
import e02_examples as E  # noqa: E402  (only its sensor-design helper)

SMOKE = "--smoke" in sys.argv
DRY = "--dry-nonid" in sys.argv  # local debug: seed 2 only, K=2, plus the non-identifiable block; writes nothing
KS, NS, REPS = ((2,), (2,), (2,)) if DRY else ((2,), (2, 4), (0,)) if SMOKE else ((2, 4, 8), (2, 4, 8), (0, 1, 2))
R_LEVELS = (0.01, 0.1, 1.0)           # sigma0 / rms(noise-free anomaly at sensors); fixed before any run
HOURS, A_REF, RIDGE = 24, SourceConfig().reference_rate_ug_s, 1e-3  # window length, reference rate (protocol), a-priori relative ridge
ESTIMATORS = (("unweighted", 0.0), ("weighted", 0.0), ("weighted_ridge", RIDGE))
tab = root / "reports" / "quarto" / "tables"
OUT = Path(os.environ["RESEARCH_RUN_DIR"]) if os.environ.get("RESEARCH_RUN_DIR") else None
metrics = []


def log(name, value):
    metrics.append((name, float(value)))


def make_cfg():
    c = ExperimentConfig.default("EXP-004")
    c.model.grid, c.model.spinup_h = (32, 32), 12
    c.design.n_sources = (2, 4, 8)
    c.split = Split(train=(1, 2), validation=(3, 3), test=(4, 4))
    c.validate()
    return c


def fields_and_G(sc):
    """Unit-response anomaly fields F[k, hour, i, j] and G[(site, hour), k] (same sampling as scenario_response_matrix)."""
    m, dom = sc.cfg.model, tuple(sc.cfg.model.domain_m)
    F = np.array([_anomaly_fields(sc, B.source_basis(tuple(m.grid), dom, xy, sc.cfg.sources.footprint_sigma_m)
                                  / m.mixing_depth_m, HOURS) for xy in sc.truth.source_xy])
    G = np.array([np.stack([bilinear_sample(c, sc.site_xy, dom) for c in f], 1).ravel() for f in F]).T
    return F, G


def rows_of(G, sites, n_sites):  # (site, hour) row order
    return (np.asarray(sites)[:, None] * HOURS + np.arange(HOURS)).ravel()


def main():
    cfg = make_cfg()
    bg = cfg.sources.background_ug_m3
    table, est_rows, valid = [], [], []
    for K in KS:
        for rep in REPS:
            sc = B.generate_scenario(cfg, rep, K)
            F, Gall = fields_and_G(sc)
            a = sc.truth.strengths_ug_s
            tag = f"K{K}_s{rep}"
            truth_anom = sc.truth.field[:HOURS] - bg
            sup = float(np.linalg.norm(np.tensordot(a, F, 1) - truth_anom) / np.linalg.norm(truth_anom))
            log(f"{tag}.field_superposition_rel_err", sup)
            valid.append((tag, "G,F reproduce hidden truth anomaly on the whole grid (rel err)", sup))
            if K == KS[0] and rep == REPS[0]:
                d = float(np.abs(Gall - scenario_response_matrix(sc, HOURS)).max())
                log("G_vs_scenario_response_matrix_max_abs_diff", d)
                valid.append((tag, "G equals sensitivity.scenario_response_matrix (max abs diff)", d))
            for n in NS:
                ids = E._design(sc, n, False)
                if len(ids) < n:
                    raise RuntimeError(f"could not place {n} sensors")
                ridx = rows_of(Gall, ids, n)
                Gs = Gall[ridx] * A_REF  # columns in ug m^-3 per reference source
                y0 = sc.truth.site_truth[ids, :HOURS].ravel() - bg  # noise-free anomaly (scoring side)
                view = sc.view("train", ids, "tuning")
                # reference signal = rms anomaly over ALL admissible sites (scenario-level; the chosen sensors may see exactly zero)
                ref = float(np.sqrt(np.mean((sc.truth.site_truth[sc.admissible_sites()][:, :HOURS] - bg) ** 2)))
                if not ref > 0:
                    raise RuntimeError("no admissible site sees any plume; noise reference undefined")
                levels = [(f"r={r:g}", r * ref, np.repeat(site_sigma(r * ref, n, 7000 + rep), HOURS), None) for r in R_LEVELS]
                levels.append(("prereg", float(view.noise_sigma[0]), np.repeat(view.noise_sigma, HOURS), view.readings[:, :HOURS].ravel() - bg))
                for li, (lname, s0, sig, ydata) in enumerate(levels):
                    if ydata is None:
                        rng = np.random.default_rng(np.random.SeedSequence([cfg.seeds.streams(rep)["noise"], K, n, li]))
                        ydata = y0 + sig * rng.standard_normal(len(y0))
                    snr = float(np.sqrt(np.mean(y0**2)) / s0)
                    rk = rank_info(Gs / sig[:, None]); rka = rank_info(np.column_stack([Gs, np.ones(len(sig)) * A_REF]) / sig[:, None])
                    crb = crb_fraction(Gs, sig, a / A_REF)
                    for est, ridge in ESTIMATORS:
                        for bgm in ("known", "estimated"):
                            y = ydata + (bg if bgm == "estimated" else 0.0)
                            nu = np.ones((len(y), 1)) / s0 if bgm == "estimated" else None
                            r = solve_nnls(Gs / s0, y / s0, noise_cov=None if est == "unweighted" else sig / s0, ridge=ridge, nuisance=nu)
                            ah = r["coef"] * A_REF
                            sc_ = source_scores(ah, a, A_REF)
                            d_hat = float(r["nuisance_coef"][0]) if bgm == "estimated" else 0.0
                            fit = float(np.sqrt(np.mean(((y - Gs @ (ah / A_REF) - d_hat) / sig) ** 2)))
                            ferr = np.tensordot(ah - a, F, 1)
                            row = dict(truth_id=sc.truth_id, K=K, n_sensors=n, seed=rep, noise_level=lname, sigma0=s0,
                                       snr_rms=snr, estimator=est, ridge=ridge, background=bgm, n_obs=len(y),
                                       rank=rk["rank"], cond=rk["cond"], rank_with_background=rka["rank"],
                                       cond_with_background=rka["cond"], crb_fraction=crb,
                                       source_relative_rmse=sc_["source_relative_rmse"], source_rmse_ref=sc_["source_rmse_ref"],
                                       parameter_bias=sc_["parameter_bias"], measurement_fit=fit,
                                       field_rmse=float(np.sqrt(np.mean(ferr**2))),
                                       field_rmse_zero_estimate=float(np.sqrt(np.mean(truth_anom**2))),
                                       background_error=d_hat - bg if bgm == "estimated" else "",
                                       n_zero_coef=int((ah == 0).sum()), failure_mode=failure_mode(rk["rank"], K, ah))
                            table.append(row)
                            est_rows += [dict(truth_id=sc.truth_id, K=K, n_sensors=n, seed=rep, noise_level=lname, estimator=est,
                                              background=bgm, source=k, true_ug_s=float(a[k]), est_ug_s=float(ah[k])) for k in range(K)]
            print(tag, "done", len(table), flush=True)

    # intentionally non-identifiable case: K=4, 2 sensors, ONE snapshot hour (rows = 2 < K), noiseless, seed 0
    nonid = []
    if DRY or not SMOKE:
        sc = B.generate_scenario(cfg, 0, 4)
        F, Gall = fields_and_G(sc)
        a, ids = sc.truth.strengths_ug_s, E._design(sc, 2, False)
        h = HOURS - 1
        Gs = Gall[(np.asarray(ids) * HOURS + h)]
        rk = rank_info(Gs)
        v = np.linalg.svd(Gs)[2][-1]  # null-space direction (rank <= 2 < 4)
        up = min((a[k] / -v[k] for k in range(4) if v[k] < 0), default=np.inf)   # largest +t keeping a >= 0
        dn = min((a[k] / v[k] for k in range(4) if v[k] > 0), default=np.inf)    # largest -t keeping a >= 0
        t = 0.5 * up if up >= dn else -0.5 * dn
        alt = a + t * v
        scale = np.linalg.norm(Gs @ a)
        ahat = solve_nnls(Gs * A_REF, Gs @ a)["coef"] * A_REF
        nonid = dict(rank=rk["rank"], K=4, n_rows=Gs.shape[0], alt_min_coef=float(alt.min()),
                     alt_vs_true_rel=float(np.linalg.norm(alt - a) / np.linalg.norm(a)),
                     alt_data_misfit_rel=float(np.linalg.norm(Gs @ (alt - a)) / scale),
                     alt_field_diff_rel=float(np.linalg.norm(np.tensordot(alt - a, F[:, h], 1)) / np.linalg.norm(np.tensordot(a, F[:, h], 1))),
                     nnls_noiseless_source_rel=source_scores(ahat, a, A_REF)["source_relative_rmse"],
                     nnls_noiseless_misfit_rel=float(np.linalg.norm(Gs @ (ahat - a)) / scale))
        for k, v_ in nonid.items():
            log(f"nonidentifiable.{k}", v_)

    # headline metrics = seed mean of the weighted / background-known estimator per cell
    for K in KS:
        for n in NS:
            for lv in [f"r={r:g}" for r in R_LEVELS] + ["prereg"]:
                sel = [r for r in table if (r["K"], r["n_sensors"], r["noise_level"], r["estimator"], r["background"]) == (K, n, lv, "weighted", "known")]
                for m in ("source_relative_rmse", "field_rmse", "measurement_fit", "parameter_bias"):
                    log(f"cell.K{K}.n{n}.{lv}.{m}", np.mean([r[m] for r in sel]))
                log(f"cell.K{K}.n{n}.{lv}.estimated_condition", np.mean([min(r["cond"], 1e300) for r in sel]))
    log("runs_rows", len(table))

    if not SMOKE and not DRY:
        for name, data in (("e03-inversion.csv", table), ("e03-estimates.csv", est_rows)):
            with open(tab / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
        with open(tab / "e03-validity.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["scenario", "check", "value"]); w.writerows(valid)
        (tab / "e03-nonidentifiable.json").write_text(json.dumps(nonid, indent=1))
    if OUT and not SMOKE and not DRY:
        with open(OUT / "metrics.jsonl", "a") as f:
            for k, v in metrics:
                f.write(json.dumps({"name": k, "value": v, "step": 0}) + "\n")
        with open(OUT / "artifacts.jsonl", "a") as f:
            for p in ("e03-inversion.csv", "e03-estimates.csv", "e03-validity.csv", "e03-nonidentifiable.json"):
                f.write(json.dumps({"path": str((tab / p).resolve())}) + "\n")
    print(f"{len(table)} rows, {len(metrics)} metrics")
    for r in table[:6]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()

"""Gate-07 field-level checks on cached 64² fields of job 204894 (copied read-only from the cluster).

uv run python docs/research/gates/gate-07/check_fields.py FIELD_DIR wind:K:seed [wind:K:seed ...]

Per scenario:
  (a) re-implements the MAP estimator / forecast scorer from scratch (numpy only) for the committed dopt and source_near
      k = 8 networks (sites read from e07-sweep-rows.csv) and compares with the committed white-arm forecast_rmse;
  (b) model-error floor: forecast RMSE with the TRUE strengths (c_bg + sum a F vs the joint VanLeer truth);
  (c) superposition residual (truth - c_bg - sum a F) at the chosen sites over the test calendar, in noise-sd units;
  (d) re-scores both designs on a LINEAR truth (truth := c_bg + sum a F, readings = linear truth + the same noise draws):
      if source_near's advantage were an artefact of VanLeer non-superposition it would shrink here;
  (e) error decomposition by window rank: share of dopt's squared forecast error from issue times whose 24 h window has
      practical rank < K;
  (f) leakage probe: dopt / source_near designs are recomputed with every non-training hour of G3 and all truth replaced
      by NaN (design must be identical).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root / "src")); sys.path.insert(0, str(root / "scripts"))
sys.argv, args = sys.argv[:1], sys.argv[1:]
import e07_cross_objective as E  # noqa: E402  (driver only used for config + cache names + design construction)
from pravah import benchmarks as B  # noqa: E402
from pravah.experiments import cross_objective as X  # noqa: E402
from pravah.observations import noise_field  # noqa: E402

fdir = Path(args[0])
rows = pd.read_csv(root / "reports/quarto/tables/e07-sweep-rows.csv")


def my_map(G3, S, sigma, y, issue, a_ref, c_bg, K, W=24):
    """Independent MAP: for each issue t, solve (I + A^T A) u = A^T z over hours t-W+1..t (explicit loop)."""
    A = a_ref * G3[S] / sigma
    z = (y[S] - c_bg) / sigma
    u = np.empty((len(issue), K))
    rk = np.empty(len(issue), int)
    for n, t in enumerate(issue):
        Aw = A[:, t - W + 1:t + 1].reshape(-1, K)
        zw = z[:, t - W + 1:t + 1].ravel()
        N = np.eye(K) + Aw.T @ Aw
        u[n] = np.linalg.solve(N, Aw.T @ zw)
        rk[n] = (np.linalg.eigvalsh(Aw.T @ Aw) >= 1).sum()
    return u * a_ref, rk


def fc_err(a_hat, F, truth, issue, c_bg):
    return c_bg + np.einsum("tk,ktxy->txy", a_hat, F[:, issue + 1]) - truth[issue + 1]


for spec in args[1:]:
    wind, K, seed = spec.split(":"); K, seed = int(K), int(seed)
    names = [n for n, _ in E.scenario_fields("sweep", wind, K, seed)]
    cfg = E.make_cfg("sweep", wind)
    tf = np.load(fdir / f"{names[0]}.npy")
    F = np.stack([np.load(fdir / f"{n}.npy") for n in names[1:]])
    sc = B.generate_scenario(cfg, seed, K, truth_field=tf)
    G3 = X.site_responses(sc, F)
    a, a_ref, c_bg = sc.truth.strengths_ug_s, cfg.sources.reference_rate_ug_s, cfg.sources.background_ug_m3
    issue, test_h = X.issue_times(sc), sc.split_hours("test")
    sigma = float(X.site_sigma(sc, False)[0])
    lin = c_bg + np.einsum("k,khxy->hxy", a, F)
    print(f"\n=== {spec}: a/a_ref = {np.round(a / a_ref, 3)}, sigma = {sigma:.4f}, n_issue = {len(issue)}, "
          f"peak test anomaly = {(tf[test_h] - c_bg).max():.2f}")
    res = tf - lin
    print(f"(b) floor: forecast RMSE with TRUE strengths = {np.sqrt(np.mean((lin[issue + 1] - tf[issue + 1]) ** 2)):.5f}; "
          f"max|truth - linear| all hours {np.abs(res).max():.4f} (hour {np.unravel_index(np.abs(res).argmax(), res.shape)[0]}), "
          f"test hours {np.abs(res[test_h]).max():.4f}")
    sub = rows[(rows.wind == wind) & (rows.K == K) & (rows.seed == seed) & (rows.arm == "white") & (rows.k == 8)]
    Y = [X.readings(sc, np.full(sc.n_candidate_sites, sigma), "white", r, len(tf)) for r in range(5)]
    # independent noise reconstruction check
    e0 = noise_field(np.random.SeedSequence([cfg.seeds.streams(seed)["noise"], 0]), sc.n_candidate_sites, len(tf),
                     np.full(sc.n_candidate_sites, sigma), "white")
    assert np.allclose(Y[0] - sc.truth.site_truth, e0)
    from pravah.observations import bilinear_sample
    lin_sites = np.stack([bilinear_sample(f, sc.site_xy, tuple(cfg.model.domain_m)) for f in lin], 1)
    for strat in ["dopt", "source_near"]:
        r0 = sub[sub.strategy == strat].iloc[0]
        S = np.array(r0.sites.split(), int)
        out, outL, share = [], [], []
        for y in Y:
            ah, rk = my_map(G3, S, sigma, y, issue, a_ref, c_bg, K)
            e = fc_err(ah, F, tf, issue, c_bg)
            out.append(np.sqrt(np.mean(e ** 2)))
            se = (e ** 2).mean((1, 2))
            share.append(se[rk < K].sum() / se.sum())
            yL = lin_sites + (y - sc.truth.site_truth)  # same noise draw on the linear truth
            ahL, _ = my_map(G3, S, sigma, yL, issue, a_ref, c_bg, K)
            outL.append(np.sqrt(np.mean(fc_err(ahL, F, lin, issue, c_bg) ** 2)))
        resid = (np.asarray(sc.truth.site_truth)[S][:, test_h] - lin_sites[S][:, test_h]) / sigma
        print(f"{strat:12s} sites {r0.sites}: committed {r0.forecast_rmse:.5f} | (a) re-implemented {np.mean(out):.5f} | "
              f"(d) linear truth {np.mean(outL):.5f} | (c) site residual rms {np.sqrt(np.mean(resid ** 2)):.3f} sd, "
              f"max {np.abs(resid).max():.3f} sd | (e) windows rank<K {np.mean(rk < K):.3f}, their share of SE {np.mean(share):.3f}")
    # (f) leakage probe
    tr = sc.split_hours("train")
    G3p = np.full_like(G3, np.nan); G3p[:, tr] = G3[:, tr]
    di = X.design_input(sc, G3p, X.site_sigma(sc, False))
    p = X.problem(di)
    for strat in ["dopt", "source_near"]:
        S2, _ = X.make_design(strat, p, 8, len(tr), cfg.seeds.streams(seed)["placement"])
        same = " ".join(map(str, S2)) == sub[sub.strategy == strat].iloc[0].sites
        print(f"(f) {strat}: design rebuilt from training-only G3 (rest NaN) identical to committed: {same}")

    # (g) is dopt's window rank deficit specific to the held-out winds? same statistic on training-period windows
    print("(g) fraction of 24 h windows with practical rank < K, train hours vs test issue times; distance (m) of sites to each source")
    tr_issue = np.arange(23, len(tr))
    for strat in ["dopt", "source_near", "random"]:
        rr = sub[sub.strategy == strat]
        ftr, fte = [], []
        for _, r0 in rr.iterrows():
            S = np.array(r0.sites.split(), int)
            A = a_ref * G3[S] / sigma
            ftr.append(((X.window_info(A, tr_issue) >= 1).sum(1) < K).mean())
            fte.append(((X.window_info(A, issue) >= 1).sum(1) < K).mean())
        S = np.array(rr.iloc[0].sites.split(), int)
        dd = np.linalg.norm(sc.site_xy[S][:, None] - sc.truth.source_xy[None], axis=2)
        print(f"    {strat:12s} train {np.mean(ftr):.3f}  test {np.mean(fte):.3f}  (n nets {len(rr)}); nearest site per source "
              f"{np.round(dd.min(0)).astype(int).tolist()}")

    # (h) control: only candidates OUTSIDE the 3-sigma emission disc (> 1200 m from every source); source_near and dopt
    # rebuilt on that candidate set, scored with the same estimator and noise. Tests whether source_near's win needs
    # sensors sitting inside the emitting area.
    dmin = np.linalg.norm(sc.site_xy[:, None] - sc.truth.source_xy[None], axis=2).min(1)
    far = p.cand[dmin[p.cand] > 3 * cfg.sources.footprint_sigma_m]
    pf = X.problem(di, cand=far)
    for k in (8, 16):
        msg = []
        for strat in ["dopt", "source_near"]:
            S, _ = X.make_design(strat, pf, k, len(tr), cfg.seeds.streams(seed)["placement"])
            v = [np.sqrt(np.mean(fc_err(my_map(G3, S, sigma, y, issue, a_ref, c_bg, K)[0], F, tf, issue, c_bg) ** 2)) for y in Y]
            msg.append(f"{strat} {np.mean(v):.4f}")
        print(f"(h) k={k}, candidates > 1200 m from every source ({len(far)} of {len(p.cand)}): " + ", ".join(msg))

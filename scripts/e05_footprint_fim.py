"""E04 (EXP-005, T-023): source-sensor footprints G, Fisher M = G^T R^-1 G, D/A/E, background projection, rank vs conditioning.

Run: `uv run python scripts/e05_footprint_fim.py`. Reuses the E02 smoke pipeline and the E03
`scenario_response_matrix` unchanged (32^2 grid, known wind, perfect model, anomaly form). Truth strengths are used only
to size the CRB relative to a reference rate; no estimator sees them. Metrics -> $RESEARCH_RUN_DIR/metrics.jsonl.
"""
import json, os, sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src")); sys.path.insert(0, str(root / "scripts"))
from pravah.experiment_config import SourceConfig  # noqa: E402
from pravah.inverse import _whiten, rank_info  # noqa: E402
from pravah.sensitivity import scenario_response_matrix  # noqa: E402
import e02_examples as E  # noqa: E402

OUT = Path(os.environ["RESEARCH_RUN_DIR"]) / "metrics.jsonl" if "RESEARCH_RUN_DIR" in os.environ else None
HOURS, A_REF, TOL = 24, SourceConfig().reference_rate_ug_s, 1e-8


def log(name, value):
    print(f"{name} = {float(value):.6g}")
    if OUT:
        with open(OUT, "a") as f:
            f.write(json.dumps({"name": name, "value": float(value), "step": 0}) + "\n")


def pick_sites(sc, n):  # same deterministic spread rule as E03
    pick = []
    for i in sc.admissible_sites()[::7]:
        if all(np.hypot(*(sc.site_xy[i] - sc.site_xy[j])) >= sc.cfg.design.min_sensor_separation_m for j in pick):
            pick.append(int(i))
    return pick[:n]


def fim(G, R=None):
    W = _whiten(R, G.shape[0])
    return (W @ G).T @ (W @ G)


def criteria(M, tag):
    w = np.linalg.eigvalsh(M)
    log(f"{tag}_lambda_min_over_max", w[0] / w[-1])
    if w[0] > TOL ** 2 * w[-1]:  # D/A/E only defined (finite) for a nonsingular M
        log(f"{tag}_logdet", np.linalg.slogdet(M)[1]); log(f"{tag}_trace_inv", np.trace(np.linalg.inv(M)))
        log(f"{tag}_crb_max_rel_std", np.sqrt(np.diag(np.linalg.inv(M))).max() / A_REF)
    else:
        log(f"{tag}_singular_M", 1)


def main():
    for name, K in (("separated2", 2), ("changing4", 4)):
        sc = E.build_examples([name])[name]
        G = scenario_response_matrix(sc, HOURS)
        ids = pick_sites(sc, 8)
        sig = float(sc.view("train", ids, "tuning").noise_sigma[0])
        G3 = G.reshape(sc.n_candidate_sites, HOURS, K)
        Gs = G3[ids]  # (sites, hours, K)
        t = f"{name}"
        log(f"{t}_sigma_ug_m3", sig)
        # 1. identities: M = V S^2 V^T / sigma^2, and time additivity M = sum_h M_h
        Gw = Gs.reshape(-1, K) / sig
        M = Gw.T @ Gw
        s = np.linalg.svd(Gw, compute_uv=False)
        log(f"{t}_eig_vs_sv2_rel_err", np.abs(np.sort(np.linalg.eigvalsh(M)) - np.sort(s ** 2)).max() / s[0] ** 2)
        Mh = sum(fim(Gs[:, h, :], sig ** 2 * np.eye(len(ids))) for h in range(HOURS))
        log(f"{t}_time_additivity_rel_err", np.abs(Mh - M).max() / np.abs(M).max())
        # 2. snapshot vs window vs sensors: rank (tol 1e-8) and conditioning
        for n in (1, 2, 4, 8):
            for lab, Gx in (("snap", Gs[:n, -1, :]), ("win24", Gs[:n].reshape(-1, K))):
                r = rank_info(Gx, TOL)
                log(f"{t}_n{n}_{lab}_rank", r["rank"])
                log(f"{t}_n{n}_{lab}_cond", r["cond"])
        criteria(M, f"{t}_n8_win24")
        # 3. correlated noise: AR(1) in time per sensor (rho 0.8), same marginal sigma; whitened vs direct inverse
        idx = np.arange(HOURS)
        Rt = sig ** 2 * 0.8 ** np.abs(idx[:, None] - idx[None, :])
        R = np.kron(np.eye(len(ids)), Rt)  # (site, hour) row order
        Mc = fim(Gs.reshape(-1, K), R)
        Md = Gs.reshape(-1, K).T @ np.linalg.solve(R, Gs.reshape(-1, K))
        log(f"{t}_corr_whitened_vs_direct_rel_err", np.abs(Mc - Md).max() / np.abs(Md).max())
        log(f"{t}_corr_over_white_logdet_diff", (np.linalg.slogdet(Mc)[1] - np.linalg.slogdet(M)[1]) if np.linalg.matrix_rank(M) == K else np.nan)
        # 4. background nuisance b (uniform, 1 column): Schur complement vs projection
        Gaug = np.hstack([Gs.reshape(-1, K), np.ones((len(ids) * HOURS, 1))]) / sig
        Ma = Gaug.T @ Gaug
        Schur = Ma[:K, :K] - Ma[:K, K:] @ np.linalg.solve(Ma[K:, K:], Ma[K:, :K])
        n1 = np.ones((len(ids) * HOURS, 1)) / sig
        P = np.eye(len(n1)) - n1 @ n1.T / (n1.T @ n1)
        Mproj = Gw.T @ P @ Gw
        log(f"{t}_schur_vs_projection_rel_err", np.abs(Schur - Mproj).max() / np.abs(Mproj).max())
        for lab, X in (("known_bg", M), ("nuisance_bg", Schur)):
            w = np.linalg.eigvalsh(X)
            log(f"{t}_{lab}_fim_rank", int((w > TOL ** 2 * w[-1]).sum()))  # eigenvalues are squared singular values: tol is on lambda
        w0, w1 = np.linalg.eigvalsh(M)[0], np.linalg.eigvalsh(Schur)[0]
        log(f"{t}_lambda_min_ratio_nuisance_over_known", w1 / w0)
        # 5. prior precision: M + P_0 is nonsingular even when M is not; equals ridge-Tikhonov normal matrix
        M1 = Gs[:1, -1, :].T @ Gs[:1, -1, :] / sig ** 2  # one sensor, one hour: rank <= 1
        P0 = np.eye(K) / A_REF ** 2
        log(f"{t}_snapshot1_M_rank", np.linalg.matrix_rank(M1, tol=TOL ** 2 * np.linalg.eigvalsh(M1)[-1]))
        log(f"{t}_snapshot1_posterior_cond", np.linalg.cond(M1 + P0))


main()

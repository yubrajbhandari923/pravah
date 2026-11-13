"""E03 derivation checks (T-013): does y = c_bg + G a hold for the real solver, and how do NNLS / weighted NNLS behave?

Run: `uv run python scripts/e03_derivation_checks.py`. Reuses the E02 smoke pipeline unchanged
(32^2 grid, 12 h spin-up). G is built column by column from the SAME solver, wind stream and spin-up replay that made the
truth, so every check compares the operator with the hidden truth only to measure the operator. Truth is never given to
an estimator: estimators see `Scenario.view` and the known source locations. Metrics go to $RESEARCH_RUN_DIR/metrics.jsonl.
"""
import json, os, sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src")); sys.path.insert(0, str(root / "scripts"))
from pravah import benchmarks as B  # noqa: E402
from pravah.forward import simulate  # noqa: E402
from pravah.observations import bilinear_sample  # noqa: E402
import e02_examples as E  # noqa: E402

OUT = Path(os.environ["RESEARCH_RUN_DIR"]) / "metrics.jsonl" if "RESEARCH_RUN_DIR" in os.environ else None
METRICS = {}


def log(name, value):
    METRICS[name] = float(value)
    print(f"{name} = {float(value):.6g}")
    if OUT:
        with open(OUT, "a") as f:
            f.write(json.dumps({"name": name, "value": float(value), "step": 0}) + "\n")


def anomaly_fields(sc, S, hours, spin_h=None):
    """Anomaly c - c_bg after each hour h in [0, hours), same loop and spin-up as benchmarks.generate_scenario."""
    m, cfg = sc.cfg.model, sc.cfg
    grid, dom = tuple(m.grid), tuple(m.domain_m)
    w = sc.truth.winds
    spin_h = m.spinup_h if spin_h is None else spin_h
    spin = np.resize(w[:max(m.spinup_h, 1)], (spin_h, 2))
    d, out = np.zeros(grid), []
    for h, u in enumerate(np.concatenate([spin, w[:hours]])):
        d = simulate(nx=grid[0], ny=grid[1], Lx=dom[0], Ly=dom[1], dt=m.dt_s, n_steps=3600 // m.dt_s, u=tuple(u),
                     K=m.diffusivity_m2_s, lam=m.removal_per_s, c_bg=0.0, c0=d, bc="study", source=S)
        if h >= spin_h:
            out.append(d)
    return np.array(out)


def response_matrix(sc, hours):
    """G[(site, hour), k]: reading anomaly per unit a_k [ug s^-1]; includes spin-up start and transport lag."""
    m, dom = sc.cfg.model, tuple(sc.cfg.model.domain_m)
    cols = []
    for xy0 in sc.truth.source_xy:
        S = B.source_basis(tuple(m.grid), dom, xy0, sc.cfg.sources.footprint_sigma_m) / m.mixing_depth_m
        f = anomaly_fields(sc, S, hours)
        cols.append(np.stack([bilinear_sample(c, sc.site_xy, dom) for c in f], 1).ravel())  # (site, hour) order
    return np.array(cols).T


def main():
    HOURS = 24
    # ---------------- 1. linearity, superposition, affine background (solver level)
    sc = E.build_examples(["separated2"])["separated2"]
    a = sc.truth.strengths_ug_s
    G = response_matrix(sc, HOURS)
    y_truth = (sc.truth.site_truth[:, :HOURS] - sc.cfg.sources.background_ug_m3).ravel()
    log("superposition_rel_err", np.linalg.norm(G @ a - y_truth) / np.linalg.norm(y_truth))
    Gs = response_matrix(sc, HOURS)  # determinism: identical second build
    log("G_rebuild_max_abs_diff", np.abs(G - Gs).max())
    m = sc.cfg.model
    S1 = B.source_basis(tuple(m.grid), tuple(m.domain_m), sc.truth.source_xy[0], sc.cfg.sources.footprint_sigma_m) / m.mixing_depth_m
    f1, f3 = anomaly_fields(sc, S1, 6), anomaly_fields(sc, 3 * S1, 6)
    log("homogeneity_rel_err_x3", np.abs(f3 - 3 * f1).max() / np.abs(3 * f1).max())
    # background: raw c_bg = 5 run with a source amplified x1e4 (see gate-02 solver-tolerance note) vs 5 + anomaly
    u0, g, dom = tuple(sc.truth.winds[0]), tuple(m.grid), tuple(m.domain_m)
    Sbig = 1e4 * S1
    raw = simulate(nx=g[0], ny=g[1], Lx=dom[0], Ly=dom[1], dt=m.dt_s, n_steps=12, u=u0, K=m.diffusivity_m2_s,
                   lam=m.removal_per_s, c_bg=5.0, c0=np.full(g, 5.0), bc="study", source=Sbig)
    ano = simulate(nx=g[0], ny=g[1], Lx=dom[0], Ly=dom[1], dt=m.dt_s, n_steps=12, u=u0, K=m.diffusivity_m2_s,
                   lam=m.removal_per_s, c_bg=0.0, c0=np.zeros(g), bc="study", source=Sbig)
    log("affine_background_rel_err", np.abs(raw - (5.0 + ano)).max() / np.abs(ano).max())
    # transport lag: release switched on at t = 0 from e = 0 (no spin-up); arrival hour at each admissible site
    f0 = anomaly_fields(sc, S1, 12, spin_h=0)
    ok = sc.admissible_sites()
    r = np.stack([bilinear_sample(c, sc.site_xy[ok], dom) for c in f0], 1)
    arr = np.array([int(np.argmax(x > 1e-3 * r.max())) if (x > 1e-3 * r.max()).any() else 99 for x in r])
    log("lag_arrival_hours_min", arr.min()); log("lag_arrival_hours_max_reached", arr[arr < 99].max())
    log("lag_sites_not_reached_in_12h", int((arr == 99).sum()))
    log("lag_sites_reached_after_hour0", int(((arr > 0) & (arr < 99)).sum()))

    # ---------------- 2. identifiable case: noiseless and noisy NNLS, benchmark scale and amplified
    pick = []  # greedy, deterministic: first admissible sites >= min separation apart, stride 7 for spread
    for i in sc.admissible_sites()[::7]:
        if all(np.hypot(*(sc.site_xy[i] - sc.site_xy[j])) >= sc.cfg.design.min_sensor_separation_m for j in pick):
            pick.append(int(i))
    view = sc.view("train", pick[:8], "tuning")
    ids = list(view.site_ids)
    Gk = G.reshape(sc.n_candidate_sites, HOURS, -1)[ids].reshape(len(ids) * HOURS, -1)
    sv = np.linalg.svd(Gk, compute_uv=False)
    log("identifiable_rank", int((sv > 1e-8 * sv[0]).sum())); log("identifiable_cond", sv[0] / sv[-1])
    yk = (view.readings[:, :HOURS] - sc.cfg.sources.background_ug_m3).ravel()  # background known
    sig = float(view.noise_sigma[0])
    log("noise_sigma_ug_m3", sig)
    est0, _ = nnls(Gk, Gk @ a)
    log("noiseless_nnls_rel_err", np.linalg.norm(est0 - a) / np.linalg.norm(a))
    est, _ = nnls(Gk / sig, yk / sig)  # weighted NNLS, R = sigma^2 I
    log("benchmark_scale_snr_peak", np.abs(Gk @ a).max() / sig)
    log("benchmark_scale_nnls_rel_err", np.linalg.norm(est - a) / np.linalg.norm(a))
    for amp in (1e2, 1e3, 1e4):  # amplified emissions: NOT the preregistered scale, labelled as such
        rng = np.random.default_rng(0)
        yy = amp * (Gk @ a) + sig * rng.standard_normal(len(yk))
        e, _ = nnls(Gk / sig, yy / sig)
        log(f"amp{int(amp)}_snr_peak", np.abs(amp * (Gk @ a)).max() / sig)
        log(f"amp{int(amp)}_nnls_rel_err", np.linalg.norm(e / amp - a) / np.linalg.norm(a))
        lam = 1e-3 * np.linalg.norm(Gk / sig, 2) ** 2  # ridge scale fixed a priori, relative to ||W G||^2
        Gr = np.vstack([Gk / sig, np.sqrt(lam) * np.eye(len(a))])
        er, _ = nnls(Gr, np.concatenate([yy / sig, np.zeros(len(a))]))
        log(f"amp{int(amp)}_ridgennls_rel_err", np.linalg.norm(er / amp - a) / np.linalg.norm(a))

    # ---------------- 3. underdetermined: K=4 sources, snapshot at one hour, 2 sensors => rank <= 2
    s4 = E.build_examples(["changing4"])["changing4"]
    G4 = response_matrix(s4, HOURS)
    G4 = G4.reshape(s4.n_candidate_sites, HOURS, -1)
    adm = s4.admissible_sites()
    two = adm[[5, int(np.argmax(np.linalg.norm(s4.site_xy[adm] - s4.site_xy[adm[5]], axis=1)))]]
    Gsnap = G4[two, HOURS - 1, :]
    sv = np.linalg.svd(Gsnap, compute_uv=False)
    log("snapshot_K4_k2_rank", int((sv > 1e-8 * sv[0]).sum()))
    a4 = s4.truth.strengths_ug_s
    null = np.linalg.svd(Gsnap)[2][-1]
    t = 0.99 * np.min(a4 / np.maximum(np.abs(null), 1e-300))
    alt = a4 + t * null
    log("nonunique_alt_min_coefficient", alt.min())
    log("nonunique_alt_vs_true_distance_rel", np.linalg.norm(alt - a4) / np.linalg.norm(a4))
    log("nonunique_alt_data_misfit_rel", np.linalg.norm(Gsnap @ (alt - a4)) / np.linalg.norm(Gsnap @ a4))
    Gall = G4[two].reshape(2 * HOURS, -1)  # 24 h window on the same 2 sensors
    sv = np.linalg.svd(Gall, compute_uv=False)
    log("window24_K4_k2_rank", int((sv > 1e-8 * sv[0]).sum())); log("window24_K4_k2_cond", sv[0] / sv[-1])


if __name__ == "__main__":
    main()

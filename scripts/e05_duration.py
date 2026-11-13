"""E05 (EXP-006, T-029): sensors N x window W x readings m x wind x source count K, 32^2 smoke scale, perfect model.

Run: `uv run python scripts/e05_duration.py A|B`.
A = constant amplitude (p = K unknowns). B = diurnal basis {1, sin, cos} (p = 3K unknowns; K=2 only, cost); true source is constant,
so B is deliberately over-parameterised. Truth strengths set only the data y = G theta + noise and the error score; sites are random
and nested (placement seed stream), never optimised. Calendar: one 7-day train segment so every window uses training winds only.
Winds: constant = constant_westerly (exactly 3 m/s from 270 deg, null control), drifting = steady_westerly (per-48 h-block direction
offset ~N(0, 10 deg)), changing = veering (270 -> 315 deg in the first 12 h of each block). drifting and changing share speed factors.
Noise: preregistered sigma (scale 1) and a 0.1x sensitivity (not preregistered). a_ref from the config (amendment A1, 1 g/s).
20 seeds, scenarios run in parallel worker processes (set OMP_NUM_THREADS=1). Estimator: unconstrained MAP/ridge LS with prior sd a_ref
(not tuned). Output: reports/quarto/tables/e05-duration-<branch>.csv ; metrics -> $RESEARCH_RUN_DIR/metrics.jsonl.
"""
import csv, json, os, sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src")); sys.path.insert(0, str(root / "scripts"))
import e02_examples as E  # noqa: E402
from pravah.experiment_config import SourceConfig  # noqa: E402
from pravah import benchmarks as B  # noqa: E402
from pravah.experiment_config import Split  # noqa: E402
from pravah.forward import simulate  # noqa: E402
from pravah.identifiability import practical_rank  # noqa: E402
from pravah.information import fisher  # noqa: E402
from pravah.inverse import rank_info  # noqa: E402
from pravah.observations import bilinear_sample  # noqa: E402
from pravah.sensitivity import scenario_response_matrix  # noqa: E402
from pravah.temporal_basis import kernel_to_response, periodic_basis, reading_hours, stack_window  # noqa: E402

BRANCH = sys.argv[1] if len(sys.argv) > 1 else "A"
HOURS, ISSUE, A_REF, NDRAW = 168, 167, SourceConfig().reference_rate_ug_s, 5
WINDS = {"constant": ["constant_westerly"], "drifting": ["steady_westerly"], "changing": ["veering"]}  # constant = exact null control (gate-05)
KS = (2, 4) if BRANCH == "A" else (2,)
SEEDS, NS, WS, MS, NOISE = tuple(range(20)), (2, 4, 8), (1, 6, 24, 168), (1, 2, 4, 8, 16), (1.0, 0.1)
J = 1 if BRANCH == "A" else 3
OUT = Path(os.environ["RESEARCH_RUN_DIR"]) / "metrics.jsonl" if "RESEARCH_RUN_DIR" in os.environ else None
ORACLES = []


def log(name, value):
    print(f"{name} = {float(value):.6g}", flush=True)
    if name.startswith("O"):  # oracle checks: kept beside the table so e05_report.py needs no run logs
        ORACLES.append((name, float(value)))
    if OUT:
        with open(OUT, "a") as f:
            f.write(json.dumps({"name": name, "value": float(value), "step": 0}) + "\n")


def scenario(wind, seed, K):
    c = E.make_cfg(WINDS[wind]); c.split = Split(train=(1, 7), validation=(8, 8), test=(9, 10)); c.validate()
    return B.generate_scenario(c, seed, K)


def response_B(sc):
    """G3[site, hour, k*3+j] (ug m^-3 per ug s^-1) with source k driven by b_j(t); spin-up uses b_j at t<0 (same periodic source)."""
    m, dom, grid = sc.cfg.model, tuple(sc.cfg.model.domain_m), tuple(sc.cfg.model.grid)
    w = sc.truth.winds
    spin = np.resize(w[:max(m.spinup_h, 1)], (m.spinup_h, 2))
    t = np.arange(-m.spinup_h, HOURS)
    bas = periodic_basis("diurnal", t)
    cols = []
    for xy0 in sc.truth.source_xy:
        S = B.source_basis(grid, dom, xy0, sc.cfg.sources.footprint_sigma_m) / m.mixing_depth_m
        for j in range(3):
            d, f = np.zeros(grid), []
            for h, u in enumerate(np.concatenate([spin, w[:HOURS]])):
                d = simulate(nx=grid[0], ny=grid[1], Lx=dom[0], Ly=dom[1], dt=m.dt_s, n_steps=3600 // m.dt_s, u=tuple(u),
                             K=m.diffusivity_m2_s, lam=m.removal_per_s, c_bg=0.0, c0=d, bc="study", scheme=m.advection_scheme, source=S * (bas[h, j] + (j > 0)))  # solver needs source >= 0: drive 1+b_j, subtract column 0 below
                if h >= m.spinup_h:
                    f.append(bilinear_sample(d, sc.site_xy, dom))
            f = np.stack(f, 1)  # (sites, hours)
            cols.append(f - cols[-j] if j > 0 else f)  # linearity: response(b_j) = response(1+b_j) - response(1)
    return np.stack(cols, 2)


def nested_sites(sc, seed, n=8):
    rng = np.random.default_rng(sc.cfg.seeds.streams(seed)["placement"])
    pick = []
    for i in rng.permutation(sc.admissible_sites()):
        if all(np.hypot(*(sc.site_xy[i] - sc.site_xy[j])) >= sc.cfg.design.min_sensor_separation_m for j in pick):
            pick.append(int(i))
        if len(pick) == n:
            return pick
    raise ValueError("could not place nested sites")


def ridge_ls(G, y, sigma):
    """MAP estimate with Gaussian prior sd a_ref on each coefficient: ridge (sigma/a_ref)^2 in data units (gate-05: the earlier
    ridge relative to ||G/sigma||^2 vanished in no-signal cells and errors blew up to 1e23)."""
    Gw, p = G / sigma, G.shape[1]
    lam = 1.0 / A_REF ** 2
    return np.linalg.lstsq(np.vstack([Gw, np.sqrt(lam) * np.eye(p)]), np.concatenate([y / sigma, np.zeros(p)]), rcond=None)[0]


def run_scenario(job):
    """One (wind, seed, K) scenario -> (rows, oracle checks). Runs in a worker process."""
    wind, seed, K = job
    rows, orc = [], []
    log = lambda n, v: orc.append((n, float(v)))
    sc = scenario(wind, seed, K)
    if BRANCH == "A":
        G3 = scenario_response_matrix(sc, HOURS).reshape(sc.n_candidate_sites, HOURS, K)
    else:
        G3 = response_B(sc)
        if wind == "constant" and seed == 0:  # oracle O1: the b_0 = 1 columns are Branch A's columns
            GA = scenario_response_matrix(sc, HOURS).reshape(sc.n_candidate_sites, HOURS, K)
            log("O1_basis_b0_vs_branchA_rel_err", np.abs(G3[:, :, ::3] - GA).max() / np.abs(GA).max())
    p = G3.shape[2]
    theta = np.zeros(p); theta[::J] = sc.truth.strengths_ug_s
    sites, sig0 = nested_sites(sc, seed), float(sc.noise_sigma[0])
    rng = np.random.default_rng([4000, seed, K])  # explicit: identical draws for every wind (paired)
    eps = rng.standard_normal((NDRAW, sc.n_candidate_sites, HOURS))  # paired: same draws for every cell and branch-A/B alike
    if seed == 0 and K == KS[0]:  # oracle P2: duplicated snapshot rows add no rank and double the information
        Gs = G3[sites[:4], ISSUE]
        M1, M2 = fisher(Gs, sig0), fisher(np.vstack([Gs, Gs]), sig0)
        log(f"O4_{wind}_dup_rank_change", np.linalg.matrix_rank(np.vstack([Gs, Gs]), 1e-8 * np.linalg.norm(Gs, 2)) - np.linalg.matrix_rank(Gs, 1e-8 * np.linalg.norm(Gs, 2)))
        log(f"O4_{wind}_dup_info_ratio_err", abs(np.trace(M2) / np.trace(M1) - 2))
    for N in NS:
        for W in WS:
            for m_ in MS:
                try:
                    hrs = reading_hours(ISSUE, W, m_)
                except ValueError:
                    continue  # infeasible: m distinct hourly readings do not fit in W (never padded)
                Gc = stack_window(G3, sites[:N], hrs, ISSUE)
                if not np.isfinite(Gc).all():
                    raise ValueError("non-finite G")
                ri = rank_info(Gc)
                sv = ri["singular_values"]
                base = dict(branch=BRANCH, wind=wind, seed=seed, K=K, p=p, N=N, m=m_, W=W, B=N * m_, rows=Gc.shape[0],
                            rank=ri["rank"], rank_over_p=ri["rank"] / p, sigma_prereg=sig0,
                            prank_aref1=practical_rank(Gc, sig0, A_REF), prank_aref10=practical_rank(Gc, sig0, 10 * A_REF),
                            prank_aref100=practical_rank(Gc, sig0, 100 * A_REF), s_max=sv[0], s_min=sv[-1] if len(sv) == p else 0.0)
                for ns in NOISE:
                    sig = sig0 * ns
                    e_src, e_const, e_fld = [], [], []
                    for r in range(NDRAW):
                        y = (Gc @ theta).reshape(N, m_) + sig * eps[r][np.ix_(sites[:N], hrs)]
                        th = ridge_ls(Gc, y.ravel(), sig)
                        d = th - theta
                        e_src.append(np.sqrt(np.mean(d ** 2))); e_const.append(np.sqrt(np.mean(d[::J] ** 2)))
                        e_fld.append(np.sqrt(np.mean((G3[:, ISSUE, :] @ d) ** 2)))
                    rows.append({**base, "noise_scale": ns, "prank_at_noise": practical_rank(Gc, sig, A_REF),
                                 "src_rmse": np.mean(e_src), "src_rmse_const": np.mean(e_const), "field_rmse_issue": np.mean(e_fld),
                                 "true_rms": float(np.sqrt(np.mean(theta[::J] ** 2)))})
    print("done", wind, seed, K, flush=True)

    return rows, orc


if __name__ == "__main__":  # guard: workers re-import this module
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(w, s, k) for w in WINDS for s in SEEDS for k in KS]
    rows = []
    with ProcessPoolExecutor(max_workers=min(len(jobs), max(1, (os.cpu_count() or 2) - 4))) as ex:
        for r, orc in ex.map(run_scenario, jobs):  # ordered, so the table is deterministic
            rows += r
            for n, v in orc:
                log(n, v)

    path = root / "reports/quarto/tables" / f"e05-duration-{BRANCH}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(path.with_name(f"e05-oracles-{BRANCH}.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("check", "value"), *ORACLES])
    log("n_rows", len(rows))
    for wind in WINDS:
        sel = [r for r in rows if r["wind"] == wind and r["noise_scale"] == 1.0]
        log(f"mean_rank_over_p_{wind}", np.mean([r["rank_over_p"] for r in sel]))
        log(f"mean_prank_aref1_prereg_{wind}", np.mean([r["prank_aref1"] for r in sel]))
    if OUT:
        with open(OUT.parent / "artifacts.jsonl", "a") as f:
            f.write(json.dumps({"path": str(path)}) + "\n")

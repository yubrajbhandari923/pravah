"""Gate-00 B3: upwind vs FiPy VanLeer advection — accuracy, effective diffusivity, dt vs dx error, cost.

Run: OMP_NUM_THREADS=1 uv run python docs/research/gates/b3/b3_scheme_study.py
Writes CSVs next to this script. References are the closed-form oracles of tests/test_forward_oracles.py.
No protocol value is changed: K = 50 m^2/s, protocol dt = 300 s, 10 km domain, winds 3 and 5 m/s.
"""
import csv, math, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

here = Path(__file__).resolve().parent
root = here.parents[3]
sys.path[:0] = [str(root / "src"), str(root / "tests")]
import test_forward_oracles as o  # noqa: E402
import pravah.forward as fw  # noqa: E402

L = o.L
K, SIG0, T = 50.0, 1000.0, 3600.0  # puff: physical K, resolved initial width, one simulated hour
WINDS = {"x3": (3.0, 0.0), "x5": (5.0, 0.0), "diag3": (3 / math.sqrt(2), 3 / math.sqrt(2))}
GRIDS = (16, 32, 64, 128)
DTS = (300.0, 150.0, 60.0, 30.0, 15.0)


def courant(u, n, dt):
    return max(abs(u[0]), abs(u[1])) * dt / (L / n)  # per-axis Courant (what the split vanleer guard checks)


def puff(n, u, t):
    x, _ = o.centres(n)
    return o.ref_advected_periodic(x, x, t, u=u, K=K, sig0=SIG0)


def fit_cov(c, n):
    """Least-squares fit of a unit-mass periodic Gaussian with full covariance (3x3 image sum); returns Sigma (2x2)."""
    x, dx = o.centres(n)
    c = c / (c.sum() * dx * dx)
    X, Y = np.meshgrid(x, x, indexing="ij")
    i, j = np.unravel_index(c.argmax(), c.shape)

    def model(p):
        a, b, d, cx, cy = p  # Cholesky factor [[a, 0], [b, d]] keeps Sigma positive definite
        S = np.array([[a * a, a * b], [a * b, b * b + d * d]])
        P = np.linalg.inv(S)
        ex = (X - cx + L / 2) % L - L / 2
        ey = (Y - cy + L / 2) % L - L / 2
        m = 0.0
        for k in (-1, 0, 1):
            for l in (-1, 0, 1):
                u, v = ex + k * L, ey + l * L
                m = m + np.exp(-0.5 * (P[0, 0] * u * u + 2 * P[0, 1] * u * v + P[1, 1] * v * v))
        return m / (2 * np.pi * np.sqrt(np.linalg.det(S))), S

    r = lambda p: ((model(p)[0] - c) / c.max()).ravel()  # O(1) residuals so the solver's tolerances bite
    p = least_squares(r, [SIG0, 0.0, SIG0, x[i], x[j]], x_scale=[1e3, 1e3, 1e3, 1e3, 1e3]).x
    return model(p)[1]


def puff_job(args):
    scheme, wname, n, dt, guard = args
    fw.VANLEER_MAX_COURANT = guard
    u = WINDS[wname]
    c0, ref = puff(n, u, 0.0), puff(n, u, T)
    t0 = time.perf_counter()
    try:
        c = fw.simulate(nx=n, ny=n, Lx=L, Ly=L, dt=dt, n_steps=round(T / dt), u=u, K=K, lam=0.0, c_bg=0.0,
                        c0=c0, bc="periodic", scheme=scheme)
    except ValueError as e:
        return dict(scheme=scheme, wind=wname, n=n, dt=dt, courant=courant(u, n, dt), status=f"rejected: {e}"), None
    wall = time.perf_counter() - t0
    row = dict(scheme=scheme, wind=wname, n=n, dt=dt, courant=courant(u, n, dt), status="ok" if np.isfinite(c).all() else "nonfinite",
               wall_s_per_sim_hour=wall * 3600 / T, rel_l2=float(np.linalg.norm(c - ref) / np.linalg.norm(ref)),
               peak_ratio=float(c.max() / ref.max()), min_over_max=float(c.min() / c.max()),
               mass_rel_err=float(abs(c.sum() - c0.sum()) / c0.sum()))
    if not np.isfinite(c).all() or row["peak_ratio"] > 2:
        return row, None  # unstable run: no meaningful width
    # K_eff along / across the wind; baseline is the fitted covariance of the discrete initial field, not SIG0^2
    dS = (fit_cov(c, n) - fit_cov(c0, n)) / (2 * T)
    e = np.array(u) / np.hypot(*u)
    p = np.array([-e[1], e[0]])
    row.update(K_eff_along=float(e @ dS @ e), K_eff_cross=float(p @ dS @ p))
    h = L / n
    # modified-equation estimates. upwind (implicit): K + |u_i| h/2 per axis + |u|^2 dt/2 along the wind.
    # vanleer unsplit (FiPy, explicit + per-face (h - u_n dt) correction): off-diagonal -u_x u_y dt/2 survives,
    # i.e. K - u_x u_y dt e_x e_y along a direction e (results in vanleer_unsplit/). Split vanleer: leading term K.
    if scheme == "upwind":
        row["K_theory_along"] = K + h / 2 * (abs(u[0]) * e[0] ** 2 + abs(u[1]) * e[1] ** 2) + (u[0] ** 2 + u[1] ** 2) * dt / 2
        row["K_theory_cross"] = K + h / 2 * (abs(u[0]) * p[0] ** 2 + abs(u[1]) * p[1] ** 2)
    else:
        row["K_theory_along"] = row["K_theory_cross"] = K
    return row, (c if n == 64 else None)


def oracle_job(args):
    scheme, name, bc, Ko, u, To, sig0, n = args
    x, _ = o.centres(n)
    f = (lambda t: o.ref_diffusion_noflux(x, x, t, K=Ko, sig0=sig0)) if bc == "noflux" else \
        (lambda t: o.ref_advected_periodic(x, x, t, u=u, K=Ko, sig0=sig0))
    c0, ref, dt = f(0.0), f(To), 60.0 * 32 / n
    t0 = time.perf_counter()
    c = fw.simulate(nx=n, ny=n, Lx=L, Ly=L, dt=dt, n_steps=round(To / dt), u=u, K=Ko, lam=0.0, c_bg=0.0, c0=c0, bc=bc,
                    scheme=scheme)
    return dict(scheme=scheme, case=name, n=n, dt=dt, courant=courant(u, n, dt), wall_s=time.perf_counter() - t0,
                rel_l2=float(np.linalg.norm(c - ref) / np.linalg.norm(ref)), min_over_max=float(c.min() / c.max()),
                peak_ratio=float(c.max() / ref.max()), mass_rel_err=float(abs(c.sum() - c0.sum()) / c0.sum()))


def write(name, rows):
    out = here / (sys.argv[1] if len(sys.argv) > 1 else "results")  # tag = which forward.py variant was run
    out.mkdir(exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / name, "w", newline="") as f:
        w = csv.DictWriter(f, cols, restval="")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    CASES = [("diffusion", "noflux", 200.0, (0.0, 0.0), 6 * 3600.0, o.SIG0),
             ("advect_K20", "periodic", 20.0, (2.0, 1.0), 360.0, 2500.0),
             ("advect_K200", "periodic", 200.0, (2.0, 1.0), 360.0, 2500.0)]
    puff_jobs = [(s, w, n, dt, 1.0) for s in ("upwind", "vanleer") for w in WINDS for n in GRIDS for dt in DTS]
    puff_jobs += [("vanleer", "x3", 64, dt, math.inf) for dt in (300.0, 60.0)]  # stability bound switched off on purpose
    with Pool(int(os.environ.get("NPROC", 12))) as pool:
        orc = pool.map(oracle_job, [(s, *c, n) for s in ("upwind", "vanleer") for c in CASES for n in GRIDS])
        out = pool.map(puff_job, puff_jobs)
    for s in ("upwind", "vanleer"):
        for c in CASES:
            e = {r["n"]: r["rel_l2"] for r in orc if r["scheme"] == s and r["case"] == c[0]}
            for a, b in zip(GRIDS, GRIDS[1:]):
                next(r for r in orc if r["scheme"] == s and r["case"] == c[0] and r["n"] == b)["order_from_prev"] = math.log2(e[a] / e[b])
    write("oracle_convergence.csv", orc)
    rows = [r for r, _ in out]
    # dt-halving gate (protocol §3: change of endpoint field <= 1% rel L2) at 64^2, per scheme/wind
    fields = {(r["scheme"], r["wind"], r["dt"]): c for (r, c), j in zip(out, puff_jobs) if c is not None and j[4] == 1.0}
    halving = []
    for (s, w, dt), c in fields.items():
        if (s, w, dt / 2) in fields:
            c2 = fields[(s, w, dt / 2)]
            halving.append(dict(scheme=s, wind=w, n=64, dt=dt, dt_half=dt / 2, rel_change=float(np.linalg.norm(c - c2) / np.linalg.norm(c2))))
    write("puff_k_eff.csv", rows)
    write("dt_halving_64.csv", halving)
    for r in orc + rows + halving:
        print(r)

"""E01 local verification: convergence, dt refinement, CFL/Peclet sensitivity, mass budget, snapshots.

Run via `uv run python scripts/e01_verification.py`. References are the independent closed-form
oracles in tests/test_forward_oracles.py (no inverse-generated truth). Nothing is tuned: settings are the preregistered
oracle settings of doc 5.3; every number lands in the CSV, and metrics go to $RESEARCH_RUN_DIR/metrics.jsonl.
"""
import csv, json, math, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "src"), str(root / "tests")]
import test_forward_oracles as o  # noqa: E402
from pravah.forward import simulate  # noqa: E402

L, rows, metrics = o.L, [], []


def l2(c, ref):
    return float(np.linalg.norm(c - ref) / np.linalg.norm(ref))


def refs(bc, K, u, sig0, n, t):
    x, _ = o.centres(n)
    f = (lambda tt: o.ref_diffusion_noflux(x, x, tt, K=K, sig0=sig0)) if bc == "noflux" else \
        (lambda tt: o.ref_advected_periodic(x, x, tt, u=u, K=K, sig0=sig0))
    return f(0.0), f(t)


def run(bc, K, u, sig0, n, T, dt):
    c0, ref = refs(bc, K, u, sig0, n, T)
    c = simulate(nx=n, ny=n, Lx=L, Ly=L, dt=dt, n_steps=round(T / dt), u=u, K=K, lam=0.0, c_bg=0.0, c0=c0, bc=bc)
    dx = L / n
    return dict(rel_l2=l2(c, ref), min_over_max=float(c.min() / c.max()), peak_ratio=float(c.max() / ref.max()),
                courant=float(max(abs(u[0]), abs(u[1])) * dt / dx), peclet=float(max(abs(u[0]), abs(u[1])) * dx / K if K else math.inf),
                mass_rel_err=float(abs(c.sum() - c0.sum()) / c0.sum())), c, ref


def add(group, case, **kw):
    rows.append(dict(group=group, case=case, **kw))
    for k, v in kw.items():
        if isinstance(v, float) and math.isfinite(v):
            metrics.append((f"{group}.{case}.{k}", v))


CASES = [("diffusion", "noflux", 200.0, (0.0, 0.0), 6 * 3600.0, o.SIG0),
         ("advect_K20", "periodic", 20.0, (2.0, 1.0), 360.0, 2500.0),
         ("advect_K200", "periodic", 200.0, (2.0, 1.0), 360.0, 2500.0)]

# 1. grid convergence at constant Courant number (dt = 60 s * 32/n), as in the oracle tests
conv, snaps = {}, {}
for name, bc, K, u, T, sig0 in CASES:
    errs = []
    for n in (16, 32, 64):
        d, c, ref = run(bc, K, u, sig0, n, T, 60.0 * 32 / n)
        errs.append(d["rel_l2"])
        snaps[(name, n)] = (c, ref)
        add("grid", f"{name}_n{n}", n=float(n), dt=60.0 * 32 / n, **d)
    conv[name] = errs
    for a, b, na in ((0, 1, "16_32"), (1, 2, "32_64")):
        add("order", f"{name}_{na}", observed_order=math.log2(errs[a] / errs[b]))

# 2. time-step refinement at fixed 32^2 grid (spatial error fixed, temporal error varies)
for name, bc, K, u, T, sig0 in CASES[:2]:
    for dt in (120.0, 60.0, 30.0, 15.0):
        d, _, _ = run(bc, K, u, sig0, 32, T, dt)
        add("dt", f"{name}_dt{dt:g}", dt=dt, **d)

# 3. advection-dominated sensitivity: 32^2, K=20, narrow-ish pulse, growing wind (Courant and cell Peclet both rise)
for U in (1.0, 2.0, 4.0, 8.0):
    d, _, _ = run("periodic", 20.0, (U, 0.0), 1500.0, 32, 360.0, 60.0)
    add("cfl_peclet", f"U{U:g}", U=U, **d)

# 4. mass budget: oracle iv (noflux, prescribed theta), discrete-exact recurrence, 32^2
n, theta, h, cbg, T, lam = 32, 10.0, 500.0, 5e-6, 6 * 3600.0, 1 / (24 * 3600.0)
x, dx = o.centres(n)
b = np.exp(-((x[:, None] - 5000) ** 2 + (x[None, :] - 5000) ** 2) / (2 * 400.0**2))
b /= b.sum() * dx * dx
dt, ns, mass_t, mass_err, M, c = 60.0, 360, [], [], h * cbg * L * L, np.full((n, n), cbg)
for k in range(ns):
    c = simulate(nx=n, ny=n, Lx=L, Ly=L, dt=dt, n_steps=1, u=(1.0, 0.5), K=50.0, lam=lam, c_bg=cbg, c0=c,
                 bc="noflux", source=theta * b / h)
    M = (M + dt * (theta + lam * h * cbg * L * L)) / (1 + lam * dt)
    mass_t.append((k + 1) * dt / 3600)
    mass_err.append(abs(h * c.sum() * dx * dx - M) / M)
add("mass", "noflux_theta10", max_rel_mass_err=float(max(mass_err)), final_mass_g=float(h * c.sum() * dx * dx))

# CSV (all rows, union of columns)
cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("group", "case"), k))
(root / "reports/quarto/tables").mkdir(parents=True, exist_ok=True)
with open(root / "reports/quarto/tables/e01-verification.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols, restval="")
    w.writeheader()
    w.writerows(rows)

# Figure
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for name, e in conv.items():
    ax[0, 0].loglog([L / 16, L / 32, L / 64], e, "o-", label=name)
ax[0, 0].loglog([L / 16, L / 64], [conv["diffusion"][0], conv["diffusion"][0] / 4], "k--", label="slope 1")
ax[0, 0].set(xlabel="dx (m)", ylabel="relative L2 error", title="Grid convergence (constant Courant)")
ax[0, 0].legend()
ax[0, 1].semilogy(mass_t, np.maximum(mass_err, 1e-18))
ax[0, 1].set(xlabel="time (h)", ylabel="|M - M_exact| / M_exact", title="Mass budget, noflux + forcing (32^2)")
xs = np.arange(64)
for n_ in (16, 32, 64):
    c_, ref_ = snaps[("advect_K20", n_)]
    ax[1, 0].plot((np.arange(n_) + .5) * L / n_, c_[:, n_ // 2], label=f"n={n_}")
ax[1, 0].plot((np.arange(64) + .5) * L / 64, snaps[("advect_K20", 64)][1][:, 32], "k--", label="reference")
ax[1, 0].set(xlabel="x (m), row j = n/2", ylabel="c (arb.)", title="Translating pulse, K=20, t=360 s")
ax[1, 0].legend()
U = [r["U"] for r in rows if r["group"] == "cfl_peclet"]
ax[1, 1].plot(U, [r["rel_l2"] for r in rows if r["group"] == "cfl_peclet"], "o-")
ax[1, 1].set(xlabel="wind U (m/s) at 32^2, dt=60 s", ylabel="relative L2 error", title="Advection-dominated sensitivity")
fig.tight_layout()
(root / "reports/quarto/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(root / "reports/quarto/figures/e01-convergence.png", dpi=130)

# Metrics / artifacts under the Research OS run contract
rd = os.environ.get("RESEARCH_RUN_DIR")
if rd:
    with open(Path(rd) / "metrics.jsonl", "a") as f:
        for k, v in metrics:
            f.write(json.dumps({"name": k, "value": v, "step": 0}) + "\n")
    with open(Path(rd) / "artifacts.jsonl", "a") as f:
        for p in ("tables/e01-verification.csv", "figures/e01-convergence.png"):
            f.write(json.dumps({"path": str(root / "reports/quarto" / p)}) + "\n")
for r in rows:
    print(r)

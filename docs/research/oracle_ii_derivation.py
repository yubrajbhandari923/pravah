"""A-priori regime derivation for convergence oracle (ii), doc section 5.3. No solver is imported.

K_num (per axis i, doc 5.2): |u_i| dx/2 * (1 + nu_i), nu_i = |u_i| dt/dx, dt = 60 s * 32/n.
Numerical variance adds 2 K_num_i T to the physical s = sig0^2 + 2 K T, so rho_i = 2 K_num_i T / s.
Relative L2 error between Gaussians of variance s and s(1+rho), exact in closed form (2-D, product over axes):
e^2 = B + 1 - 2 C with B = prod (1+rho_i)^-1/2, C = prod (1+rho_i/2)^-1/2.
"""
import json
import math
import os

L, U, DT0, N0 = 10_000.0, (2.0, 1.0), 60.0, 32
SIG0, T = 2500.0, 360.0  # chosen from the criterion rho_x(16) <= 0.10, not from solver output
RHO_MAX = 0.10


def case(K, n, sig0=SIG0, T=T):
    dx, dt = L / n, DT0 * N0 / n
    s = sig0**2 + 2 * K * T
    rho = [2 * (abs(u) * dx / 2) * (1 + abs(u) * dt / dx) * T / s for u in U]
    B = math.prod((1 + r) ** -0.5 for r in rho)
    C = math.prod((1 + r / 2) ** -0.5 for r in rho)
    return rho, math.sqrt(B + 1 - 2 * C), round(T / dt), T / dt


out = {}
for K in (20.0, 200.0):
    errs = []
    for n in (16, 32, 64):
        rho, e, steps, exact = case(K, n)
        assert abs(steps - exact) < 1e-9, "T must be a whole number of steps on every grid"
        errs.append(e)
        out[f"K{K:g}_n{n}"] = dict(rho_x=rho[0], rho_y=rho[1], e_pred=e, steps=steps)
        print(f"K={K:g} n={n}: rho_x={rho[0]:.4f} rho_y={rho[1]:.4f} predicted rel L2={e:.4f} steps={steps}")
    order = math.log2(errs[1] / errs[2])
    out[f"K{K:g}_order"] = order
    print(f"K={K:g}: predicted order log2(e32/e64) = {order:.3f}")
    assert case(K, 16)[0][0] <= RHO_MAX and order >= 0.8 + 0.1  # criterion declared before running anything

if os.environ.get("RESEARCH_RUN_DIR"):
    import research
    for k, v in out.items():
        if isinstance(v, dict):
            for m, x in v.items():
                research.log_metric(f"{k}_{m}", float(x))
        else:
            research.log_metric(f"{k}", float(v))

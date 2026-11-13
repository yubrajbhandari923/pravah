"""Source-response (sensitivity) operator G for the linear passive-tracer forward model. See tests/test_source_inversion.py."""
import numpy as np

from . import benchmarks as B
from .forward import simulate
from .observations import bilinear_sample


def build_response(forward, K):
    """G[:, k] = forward(e_k). Exact only if `forward` is one fixed linear map (check with superposition_error)."""
    cols = []
    for k in range(K):
        e = np.zeros(K)
        e[k] = 1.0
        c = np.asarray(forward(e), float).ravel()
        if not np.isfinite(c).all():
            raise ValueError("forward returned non-finite readings")
        cols.append(c)
    return np.column_stack(cols)


def superposition_error(forward, q1, q2, alpha, beta):
    """Relative ||H(a q1 + b q2) - a H(q1) - b H(q2)||; ~0 only for fixed linear physics."""
    lhs = np.asarray(forward(alpha * np.asarray(q1, float) + beta * np.asarray(q2, float)), float)
    rhs = alpha * np.asarray(forward(q1), float) + beta * np.asarray(forward(q2), float)
    return float(np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs))


def anomaly_fields(model, winds, S, hours):
    """Anomaly c - c_bg after each hour in [0, hours) for source field S [ug m^-3 s^-1] under hourly `winds`, after
    `model.spinup_h` spin-up hours replaying the first winds; same loop as benchmarks.generate_scenario."""
    grid, dom, m = tuple(model.grid), tuple(model.domain_m), model
    spin = np.resize(winds[:max(m.spinup_h, 1)], (m.spinup_h, 2))
    d, out = np.zeros(grid), []
    for h, u in enumerate(np.concatenate([spin, winds[:hours]])):
        d = simulate(nx=grid[0], ny=grid[1], Lx=dom[0], Ly=dom[1], dt=m.dt_s, n_steps=3600 // m.dt_s, u=tuple(u),
                     K=m.diffusivity_m2_s, lam=m.removal_per_s, c_bg=0.0, c0=d, bc="study", scheme=m.advection_scheme, source=S)
        if h >= m.spinup_h:
            out.append(d)
    return np.array(out)


def _anomaly_fields(sc, S, hours):
    """anomaly_fields with the scenario's own model and winds."""
    return anomaly_fields(sc.cfg.model, sc.truth.winds, S, hours)


def scenario_response_matrix(sc, hours):
    """G[(site, hour), k] in ug m^-3 per ug s^-1, from the truth's own solver, winds and spin-up. Anomaly only:
    the (known or nuisance) background is NOT in G. Uses sc.truth.source_xy (known locations), never strengths."""
    m, dom = sc.cfg.model, tuple(sc.cfg.model.domain_m)
    cols = []
    for xy0 in sc.truth.source_xy:
        S = B.source_basis(tuple(m.grid), dom, xy0, sc.cfg.sources.footprint_sigma_m) / m.mixing_depth_m
        f = _anomaly_fields(sc, S, hours)
        cols.append(np.stack([bilinear_sample(c, sc.site_xy, dom) for c in f], 1).ravel())
    return np.array(cols).T

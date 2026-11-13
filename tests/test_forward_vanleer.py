"""Contract tests for scheme="vanleer" in pravah.forward (gate-00 B3; docs/research/gates/b3-advection-scheme.md)."""
import math

import numpy as np
import pytest

from pravah.forward import simulate
from test_forward_oracles import L, centres, ref_advected_periodic

N = 32
KW = dict(nx=N, ny=N, Lx=L, Ly=L, dt=60.0, n_steps=60, K=50.0)


def gauss(n=N, x0=5000.0, y0=5000.0, s=600.0):
    x, dx = centres(n)
    b = np.exp(-((x[:, None] - x0) ** 2 + (x[None, :] - y0) ** 2) / (2 * s**2))
    return b / (b.sum() * dx * dx)


def test_default_is_upwind_bit_identical():
    kw = dict(KW, u=(2.0, 1.0), lam=0.0, c_bg=0.0, c0=gauss(), bc="study")
    assert np.array_equal(simulate(**kw), simulate(scheme="upwind", **kw))


def test_noflux_mass_with_source_and_nonnegative():
    _, dx = centres(N)
    h, theta = 500.0, 10.0
    c = simulate(u=(1.0, 0.5), lam=0.0, c_bg=0.0, c0=np.zeros((N, N)), bc="noflux", source=theta * gauss() / h,
                 scheme="vanleer", **KW)
    assert h * c.sum() * dx * dx == pytest.approx(theta * KW["dt"] * KW["n_steps"], rel=1e-9)
    assert c.min() >= 0


def test_study_uniform_background_is_steady():
    for u in ((3.0, 0.0), (0.0, -2.0), (2.0, 1.0)):
        c = simulate(u=u, lam=1 / 86400, c_bg=5.0, c0=np.full((N, N), 5.0), bc="study", scheme="vanleer", **KW)
        assert np.allclose(c, 5.0, rtol=1e-8, atol=0)


def test_courant_bound_enforced():
    with pytest.raises(ValueError, match="explicit"):
        simulate(u=(3.0, 0.0), lam=0.0, c_bg=0.0, c0=gauss(), bc="periodic", scheme="vanleer", **dict(KW, dt=300.0))


def test_converges_to_analytic_and_beats_upwind():
    # e01 oracle (ii), K = 20, constant Courant 0.58; floor 0.8 is the same a-priori floor used for upwind
    errs, up = [], None
    for n in (16, 32, 64):
        x, _ = centres(n)
        dt = 60.0 * 32 / n
        kw = dict(nx=n, ny=n, Lx=L, Ly=L, dt=dt, n_steps=round(360.0 / dt), u=(2.0, 1.0), K=20.0, lam=0.0, c_bg=0.0,
                  c0=ref_advected_periodic(x, x, 0.0, K=20.0, sig0=2500.0), bc="periodic")
        ref = ref_advected_periodic(x, x, 360.0, K=20.0, sig0=2500.0)
        c = simulate(scheme="vanleer", **kw)
        errs.append(np.linalg.norm(c - ref) / np.linalg.norm(ref))
        if n == 32:
            up = np.linalg.norm(simulate(**kw) - ref) / np.linalg.norm(ref)
    assert errs[0] > errs[1] > errs[2] and math.log2(errs[1] / errs[2]) >= 0.8
    assert errs[1] < up


def test_oblique_wind_stays_positive_and_beats_upwind():
    # FiPy's unsplit VanLeer undershot here (min/max -0.07: anti-diffusion -u_x u_y dt/2 > K along the wind)
    u = (3 / math.sqrt(2), 3 / math.sqrt(2))
    x, _ = centres(N)
    kw = dict(KW, u=u, lam=0.0, c_bg=0.0, c0=ref_advected_periodic(x, x, 0.0, u=u, K=50.0, sig0=1000.0), bc="periodic")
    ref = ref_advected_periodic(x, x, 3600.0, u=u, K=50.0, sig0=1000.0)
    c = simulate(scheme="vanleer", **kw)
    assert c.min() >= 0
    assert np.linalg.norm(c - ref) < np.linalg.norm(simulate(**kw) - ref)


def test_limiter_is_nonlinear_but_homogeneous():
    # Documents the contract change: no superposition under vanleer (NNLS Jacobians must not assume it).
    a = gauss(s=500.0)
    b = np.roll(a, 10, 0)
    f = lambda c0: simulate(u=(2.0, 1.0), lam=0.0, c_bg=0.0, c0=c0, bc="periodic", scheme="vanleer", **KW)
    assert np.linalg.norm(f(a + b) - f(a) - f(b)) > 1e-6 * np.linalg.norm(f(a + b))
    assert np.allclose(f(3 * a), 3 * f(a), rtol=1e-10, atol=0)

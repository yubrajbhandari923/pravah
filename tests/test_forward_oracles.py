"""Independent oracles for the Phase-1 forward model (T-005; see docs/research/pde-and-discretization.md).

Part A checks the exact references themselves (numpy only, no solver).
Part B checks a solver `pravah.forward` against them; it skips until that module exists.
No tolerance is fitted to solver output: each comes from the method (doc section 5).
"""
import importlib.util
import math
import os

import numpy as np
import pytest

L = 10_000.0  # m, domain side
SIG0 = 1500.0  # m, initial Gaussian width
X0 = (4000.0, 5000.0)  # m, pulse centre


def _g(d, s2):
    return np.exp(-d * d / (2 * s2)) / math.sqrt(2 * math.pi * s2)


def ref_diffusion_noflux(x, y, t, K=200.0, sig0=SIG0, x0=X0, nimg=3):
    """Unit-mass Gaussian in a closed box: method of images (reflections), per-axis product."""
    s2 = sig0**2 + 2 * K * t
    f = lambda z, z0: sum(_g(z - 2 * k * L - z0, s2) + _g(z - 2 * k * L + z0, s2) for k in range(-nimg, nimg + 1))
    return f(x, x0[0])[:, None] * f(y, x0[1])[None, :]


def ref_advected_periodic(x, y, t, u=(2.0, 1.0), K=20.0, sig0=SIG0, x0=X0, nimg=3):
    """Unit-mass Gaussian translated by u t on a periodic box (image sum)."""
    s2 = sig0**2 + 2 * K * t
    f = lambda z, z0, uz: sum(_g(z - (z0 + uz * t) - k * L, s2) for k in range(-nimg, nimg + 1))
    return f(x, x0[0], u[0])[:, None] * f(y, x0[1], u[1])[None, :]


def centres(n):
    dx = L / n
    return (np.arange(n) + 0.5) * dx, dx


def _residual(ref, n_pts, t, dt_frac=1e-3, **kw):
    """Max PDE residual |c_t + u.grad c - K lap c| by central differences on an interior grid."""
    K = kw.get("K", 200.0)
    u = kw.get("u", (0.0, 0.0))
    x = np.linspace(2000, 8000, n_pts)
    h = x[1] - x[0]
    dt = t * dt_frac
    c = lambda tt, xx, yy: ref(xx, yy, tt, **kw)
    c0 = c(t, x, x)
    ct = (c(t + dt, x, x) - c(t - dt, x, x)) / (2 * dt)
    cx = (np.roll(c0, -1, 0) - np.roll(c0, 1, 0)) / (2 * h)
    cy = (np.roll(c0, -1, 1) - np.roll(c0, 1, 1)) / (2 * h)
    lap = (np.roll(c0, -1, 0) + np.roll(c0, 1, 0) + np.roll(c0, -1, 1) + np.roll(c0, 1, 1) - 4 * c0) / h**2
    r = (ct + u[0] * cx + u[1] * cy - K * lap)[1:-1, 1:-1]  # c_t + u.grad c = K lap c
    return np.abs(r).max() / np.abs(ct).max()


# ---------------- Part A: references are right before any solver is trusted ----------------

@pytest.mark.parametrize("ref,kw,t", [
    (ref_diffusion_noflux, dict(K=200.0), 6 * 3600.0),
    (ref_advected_periodic, dict(K=20.0, u=(2.0, 1.0)), 3 * 3600.0),
    (ref_advected_periodic, dict(K=200.0, u=(2.0, 1.0)), 3 * 3600.0),
])
def test_reference_satisfies_pde_and_residual_is_second_order(ref, kw, t):
    r1, r2 = _residual(ref, 61, t, **kw), _residual(ref, 121, t, **kw)
    # central differences are O(h^2): halving h cuts the residual by ~4. One-sided (>= 3.25, i.e. order >= 1.7):
    # a ratio above 4 is a legitimate higher-order term, only a lower ratio signals a wrong reference.
    # (First draft used a two-sided band; the Gaussian K=20 case gave 4.78, so the upper bound was dropped, a
    # decision recorded in doc section 4, made on reference output, not solver output.)
    assert r2 < r1 and r1 / r2 >= 3.25, (r1, r2)


def test_reference_mass_is_one():
    for n in (64, 128):
        x, dx = centres(n)
        assert abs(ref_diffusion_noflux(x, x, 6 * 3600.0).sum() * dx * dx - 1) < 1e-3
        assert abs(ref_advected_periodic(x, x, 3 * 3600.0).sum() * dx * dx - 1) < 1e-3


def test_noflux_reference_has_zero_wall_gradient():
    h = 1.0
    for yy in (3000.0, 5000.0):
        c = lambda xx: ref_diffusion_noflux(np.array([xx]), np.array([yy]), 6 * 3600.0)[0, 0]
        assert abs(c(h) - c(-h)) < 1e-12 * c(0.0) * 1e3  # symmetric about x=0 -> dc/dx = 0
        assert abs(c(L + h) - c(L - h)) < 1e-12 * c(L) * 1e3


def test_negative_control_wrong_reference_is_rejected():
    # A reference with the wrong K or wrong wind must fail the residual test when judged with the true K / u.
    bad_K = _residual(lambda x, y, t, **k: ref_diffusion_noflux(x, y, t, K=100.0), 61, 6 * 3600.0, K=200.0)
    bad_u = _residual(lambda x, y, t, **k: ref_advected_periodic(x, y, t, u=(1.0, 1.0)), 61, 3 * 3600.0, K=20.0, u=(2.0, 1.0))
    good = _residual(ref_diffusion_noflux, 61, 6 * 3600.0, K=200.0)
    assert bad_K > 1e3 * good and bad_u > 1e3 * good, (bad_K, bad_u, good)


def test_discrete_exact_decay_recurrence_matches_closed_form():
    lam, dt, a0, n = 1 / (6 * 3600.0), 300.0, 2.0, 72
    a = a0
    for _ in range(n):
        a = a / (1 + lam * dt)  # backward Euler on da/dt = -lam a
    assert a == pytest.approx(a0 * (1 + lam * dt) ** (-n), rel=1e-12)
    # first-order: halving dt halves the error against exp(-lam t) (leading term lam^2 dt t a0 e^{-lam t}/2)
    t = n * dt
    e = lambda d: abs(a0 * (1 + lam * d) ** (-round(t / d)) - a0 * math.exp(-lam * t))
    assert 1.9 < e(300.0) / e(150.0) < 2.1


def test_mass_recurrence_matches_prescribed_integral():
    theta, h, A, cbg, dt = 10.0, 500.0, L * L, 5e-6, 300.0
    # lam = 0: M(t) = M0 + theta t exactly (units: g s^-1 * s = g)
    M = h * cbg * A
    for _ in range(100):
        M = M + dt * theta
    assert M == pytest.approx(h * cbg * A + theta * 100 * dt, rel=1e-12)
    # lam > 0: recurrence converges to the steady column mass h cbg A + theta/lam
    lam = 1 / (24 * 3600.0)
    M = h * cbg * A
    for _ in range(20000):
        M = (M + dt * (theta + lam * h * cbg * A)) / (1 + lam * dt)
    assert M == pytest.approx(h * cbg * A + theta / lam, rel=1e-9)


# ---------------- Part B: solver oracles (skip until pravah.forward exists) ----------------

try:
    HAVE = importlib.util.find_spec("pravah.forward") is not None
except ModuleNotFoundError:  # parent package not importable (no src/ on path)
    HAVE = False
needs_solver = pytest.mark.skipif(not HAVE, reason="pravah.forward not implemented yet (T-005 delivers oracles only)")
SCHEME = os.environ.get("PRAVAH_SCHEME", "upwind")
FORMAL_ORDER = {"upwind": 1.0}  # derived in doc 5.2; unknown schemes are skipped, not given a guessed tolerance


def _sim(n, **kw):
    from pravah.forward import simulate
    dt = 60.0 * 32 / n  # constant Courant number across grids (doc 5.2)
    T = kw.pop("T")
    return simulate(nx=n, ny=n, Lx=L, Ly=L, dt=dt, n_steps=round(T / dt), scheme=SCHEME, **kw)


def _l2(n, c, ref):
    return np.linalg.norm(c - ref) / np.linalg.norm(ref)


@needs_solver
@pytest.mark.parametrize("name,bc,K,u,T,sig0", [
    ("diffusion", "noflux", 200.0, (0.0, 0.0), 6 * 3600.0, SIG0),
    # advection regime fixed a priori in doc 5.3 (option b): 2 K_num T <= 0.1 (sig0^2 + 2KT) on 16^2
    ("advect_K20", "periodic", 20.0, (2.0, 1.0), 360.0, 2500.0),
    ("advect_K200", "periodic", 200.0, (2.0, 1.0), 360.0, 2500.0),
])
def test_convergence_order(name, bc, K, u, T, sig0):
    if SCHEME not in FORMAL_ORDER:
        pytest.skip(f"no derived order for scheme {SCHEME}")
    p, errs = FORMAL_ORDER[SCHEME], []
    for n in (16, 32, 64):
        x, _ = centres(n)
        ref0 = (ref_diffusion_noflux if bc == "noflux" else ref_advected_periodic)
        c0 = ref0(x, x, 0.0, K=K, sig0=sig0) if bc == "noflux" else ref0(x, x, 0.0, u=u, K=K, sig0=sig0)
        c = _sim(n, T=T, u=u, K=K, lam=0.0, c_bg=0.0, c0=c0, bc=bc)
        ref = ref0(x, x, T, K=K, sig0=sig0) if bc == "noflux" else ref0(x, x, T, u=u, K=K, sig0=sig0)
        errs.append(_l2(n, c, ref))
        assert c.min() >= -1e-8 * c.max() or SCHEME != "upwind"  # M-matrix positivity (doc 5.5)
    print(name, "rel L2 errors 16/32/64:", errs)
    assert errs[0] > errs[1] > errs[2]
    assert math.log2(errs[1] / errs[2]) >= p - 0.2  # 0.2 = a-priori pre-asymptotic allowance (doc 5.3)


@needs_solver
def test_uniform_background_is_exact_steady_state_under_study_bc():
    for u in ((3.0, 0.0), (0.0, -2.0), (2.0, 1.0)):
        c = _sim(32, T=6 * 3600.0, u=u, K=50.0, lam=1 / (24 * 3600.0), c_bg=5.0, c0=np.full((32, 32), 5.0), bc="study")
        assert np.allclose(c, 5.0, rtol=1e-8, atol=0)


@needs_solver
def test_decay_discrete_exact():
    n, lam, a0 = 32, 1 / (6 * 3600.0), 2.0
    T = 6 * 3600.0
    dt = 60.0 * 32 / n
    c = _sim(n, T=T, u=(2.0, 1.0), K=50.0, lam=lam, c_bg=5.0, c0=np.full((n, n), 5.0 + a0), bc="periodic")
    assert np.allclose(c - 5.0, a0 * (1 + lam * dt) ** (-round(T / dt)), rtol=1e-8, atol=0)


@needs_solver
@pytest.mark.parametrize("lam", [0.0, 1 / (24 * 3600.0)])
def test_prescribed_mass_forcing_noflux(lam):
    n, theta, h, cbg, T = 32, 10.0, 500.0, 5e-6, 6 * 3600.0
    x, dx = centres(n)
    b = np.exp(-((x[:, None] - 5000) ** 2 + (x[None, :] - 5000) ** 2) / (2 * 400.0**2))
    b /= b.sum() * dx * dx  # grid-normalised: sum b dA = 1 on every grid, units m^-2
    dt = 60.0 * 32 / n
    c = _sim(n, T=T, u=(1.0, 0.5), K=50.0, lam=lam, c_bg=cbg, c0=np.full((n, n), cbg), bc="noflux", source=theta * b / h)
    M = h * cbg * L * L
    for _ in range(round(T / dt)):
        M = (M + dt * (theta + lam * h * cbg * L * L)) / (1 + lam * dt)
    assert h * c.sum() * dx * dx == pytest.approx(M, rel=1e-8)


@needs_solver
def test_observation_operator_is_cell_centre_value():
    from pravah.forward import observe
    c = np.random.default_rng(0).random((8, 8))
    assert np.array_equal(observe(c, [(1, 2), (7, 0)]), c[[1, 7], [2, 0]])

"""Scientific checks of the FiPy forward model. Nothing is mocked; tolerances are justified inline."""
from __future__ import annotations

import dataclasses as dc
import math
from pathlib import Path

import numpy as np
import pytest

from pravah.config import (Boundary, Domain, GaussianSource, Initial, Scenario, SolverOptions, Times, Transport,
                           WindSegment, load_config, scenario_from_dict)
from pravah.forward import _boundary_coefficients, simulate, step_times  # pyright: ignore[reportPrivateUsage]
from pravah.sources import cell_centres, emission_profile, quadrature_ratio, source_fields

ROOT = Path(__file__).resolve().parents[1]
LAM = 1.0 / 21600.0


def small(nx: int = 25, mode: str = "open", wind: tuple[WindSegment, ...] = (WindSegment(0.0, 1.0, 0.3),),
          lam: float = LAM, K: float = 100.0, bg: float = 0.0, initial: Initial | None = None,
          T: float = 3600.0, dt: float = 60.0, out: float = 300.0,
          sources: tuple[GaussianSource, ...] | None = None) -> Scenario:
    kw = {} if sources is None else {"sources": sources}
    return Scenario(domain=Domain(10000.0, 10000.0, nx, nx), transport=Transport(wind, K, lam, 500.0),
                    boundary=Boundary(mode, bg), initial=initial or Initial(), times=Times(T, dt, out), **kw)


def reference() -> tuple[Scenario, list[float]]:
    cfg = load_config(ROOT / "configs" / "reference.json")
    return scenario_from_dict(cfg["scenario"]), cfg["q"]


# ---- sources -------------------------------------------------------------------------------------------------

def test_source_integrals_resolution_invariant():
    for n in (25, 50, 100):
        sc = small(n)
        f = source_fields(sc)
        assert f.shape == (3, n, n)
        # normalised sum is exact to rounding (1e-12 >> accumulated 1e-16 round-off over <= 1e4 cells)
        np.testing.assert_allclose(f.sum(axis=(1, 2)) * sc.domain.dx * sc.domain.dy, 1.0, atol=1e-12)


def test_quadrature_ratio_converges():
    r = {n: np.abs(quadrature_ratio(small(n)) - 1).max() for n in (10, 25, 50, 100)}
    # sigma = 500 m: midpoint sum of a smooth Gaussian is spectrally accurate once dx << sigma (measured ~1e-7 at 25
    # cells for the worst source, which sits near a wall and is truncated), but visibly off at dx = 1000 m = 2 sigma.
    assert r[10] > 1e-2
    assert r[25] < 1e-6 and r[50] < 1e-6 and r[100] < 1e-6


def test_emission_profile_piecewise_constant():
    s = GaussianSource("a", 1.0, 1.0, 1.0, ((0.0, 1.0), (100.0, 0.0), (200.0, 2.5)))
    assert [emission_profile(s, t) for t in (0.0, 99.9, 100.0, 199.0, 200.0, 1e4)] == [1.0, 1.0, 0.0, 0.0, 2.5, 2.5]


# ---- simple invariants ---------------------------------------------------------------------------------------

def test_zero_stays_zero():
    r = simulate(small(), [0.0, 0.0, 0.0])
    assert np.all(r.concentration.values == 0.0)
    assert r.diagnostics["max_abs_budget_residual"] == 0.0


def test_result_metadata():
    sc = small()
    r = simulate(sc, [1e5, 0, 0])
    c = r.concentration
    assert c.name == "concentration" and c.dims == ("time", "y", "x") and c.attrs["units"] == "microgram/m^3"
    np.testing.assert_allclose(c.time.values, sc.output_times)
    np.testing.assert_allclose(c.x.values, (np.arange(25) + 0.5) * 400.0)
    assert r.status == "ok" and r.budget is not None and r.budget.attrs["units"] == "microgram"
    assert r.budget.sizes["time"] == len(sc.output_times) - 1


@pytest.mark.parametrize("mode,wind", [("closed", (WindSegment(0.0, 0.0, 0.0),)),
                                        ("periodic", (WindSegment(0.0, 1.0, 0.3),))])
def test_uniform_field_stays_uniform(mode: str, wind: tuple[WindSegment, ...]):
    # lam = 0, no sources: a uniform field is an exact discrete steady state (all differences vanish).
    sc = small(mode=mode, wind=wind, lam=0.0, initial=Initial("uniform", 3.0), T=1200.0)
    r = simulate(sc, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(r.concentration.values, 3.0, rtol=0, atol=1e-12)


def test_open_uniform_matches_background():
    # IC = c_b everywhere, lam = 0: inflow faces see c_b - c_P = 0, so the state stays uniform and in = out.
    sc = small(lam=0.0, bg=2.0, initial=Initial("uniform", 2.0), T=1200.0)
    r = simulate(sc, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(r.concentration.values, 2.0, rtol=0, atol=1e-12)
    assert r.budget is not None
    np.testing.assert_allclose(r.budget["inflow"].values, r.budget["outflow"].values, rtol=1e-10)
    assert (r.budget["outflow"].values > 0).all()


# ---- removal-only decay --------------------------------------------------------------------------------------

def _decay(dt: float, lam: float = 1e-3, T: float = 1000.0) -> tuple[float, int]:
    sc = small(10, mode="closed", wind=(WindSegment(0.0, 0.0, 0.0),), lam=lam, initial=Initial("uniform", 1.0),
               T=T, dt=dt, out=500.0)
    r = simulate(sc, [0.0, 0.0, 0.0])
    c = r.concentration.values[-1]
    assert np.ptp(c) < 1e-12  # stays uniform
    return float(c.mean()), r.diagnostics["n_steps"]


def test_removal_only_matches_backward_euler_formula():
    c, n = _decay(100.0)
    assert n == 10
    # Backward Euler on dc/dt = -lam c gives (1+lam dt)^-n exactly; LU solve error is ~1e-15.
    assert abs(c - (1 + 1e-3 * 100.0) ** -n) < 1e-10


def test_removal_only_first_order_to_exponential():
    errs = [abs(_decay(dt)[0] - math.exp(-1.0)) for dt in (100.0, 50.0, 25.0, 12.5)]
    ratios = [a / b for a, b in zip(errs, errs[1:])]
    # first order: error ratio -> 2 as dt halves (measured 1.96, 1.98, 1.99 at these dt)
    assert all(1.9 < r < 2.1 for r in ratios), ratios


# ---- periodic advection-diffusion-decay vs analytical Gaussian ----------------------------------------------

def _periodic_error(nx: int, dt: float, T: float = 1800.0) -> float:
    """Relative L2 error against the exact translating, spreading, decaying Gaussian (sum over periodic images)."""
    L, s0, K, (u, v) = 10000.0, 1000.0, 100.0, (1.0, 0.3)
    sc = small(nx, mode="periodic", wind=(WindSegment(0.0, u, v),), initial=Initial("gaussian", 1.0, 5000.0, 5000.0, s0),
               T=T, dt=dt, out=T)
    c = simulate(sc, [0.0, 0.0, 0.0]).concentration.values[-1]
    x, y = cell_centres(sc)
    X, Y = np.meshgrid(x, y)
    s2 = s0**2 + 2 * K * T
    exact = sum(np.exp(-((X - 5000 - u * T - i * L) ** 2 + (Y - 5000 - v * T - j * L) ** 2) / (2 * s2))
                for i in range(-2, 3) for j in range(-2, 3))
    exact = exact * s0**2 / s2 * math.exp(-LAM * T)
    return float(np.linalg.norm(c - exact) / np.linalg.norm(exact))


def test_periodic_gaussian_spatial_convergence():
    # Fixed small dt = 15 s keeps the (first-order) time error well below the spatial error at dx = 500, 250, 125 m.
    # Measured: 0.155, 0.0607, 0.0238 (ratios 2.56 and 2.55: pre-asymptotic; the exponential scheme's numerical
    # diffusion K*((Pe/2)coth(Pe/2) - 1) ~ K Pe^2/12 shrinks as cell Pe = 5, 2.5, 1.25, so the order tends to 2).
    e = [_periodic_error(n, 15.0) for n in (20, 40, 80)]
    assert e[0] > e[1] > e[2]
    assert e[0] / e[1] > 2.0 and e[1] / e[2] > 2.0
    assert e[2] < 0.03


def test_periodic_gaussian_time_convergence():
    # Backward Euler: halving dt halves the time-error part. Differences between successive dt are in ratio ~2.
    e = [_periodic_error(40, dt) for dt in (60.0, 30.0, 15.0, 7.5)]
    d = [a - b for a, b in zip(e, e[1:])]
    assert all(x > 0 for x in d)
    assert all(1.7 < a / b < 2.3 for a, b in zip(d, d[1:])), d


# ---- conservation and budget ---------------------------------------------------------------------------------

def _mass(r, sc: Scenario) -> np.ndarray:
    return sc.transport.H * sc.domain.dx * sc.domain.dy * r.concentration.values.sum(axis=(1, 2))


def test_closed_no_source_conserves_mass():
    sc = small(mode="closed", wind=(WindSegment(0.0, 0.0, 0.0),), lam=0.0,
               initial=Initial("gaussian", 5.0, 3000.0, 4000.0, 800.0), T=7200.0)
    r = simulate(sc, [0.0, 0.0, 0.0])
    m = _mass(r, sc)
    assert np.abs(m / m[0] - 1).max() < 1e-10  # LU solve at round-off, 24 steps x 1e-15
    assert r.concentration.values[-1].std() < r.concentration.values[0].std()  # but it has diffused


def test_closed_source_only_mass_equals_integrated_emission():
    q = np.array([1e5, 2e5, 5e4])
    srcs = tuple(dc.replace(s, profile=p) for s, p in zip(small().sources,
                 [((0.0, 1.0),), ((0.0, 1.0), (1000.0, 0.0), (2000.0, 2.0)), ((0.0, 0.5),)]))
    sc = small(mode="closed", wind=(WindSegment(0.0, 0.0, 0.0),), lam=0.0, T=3000.0, sources=srcs)
    r = simulate(sc, q)
    expected = q[0] * 3000.0 + q[1] * (1000.0 + 2.0 * 1000.0) + q[2] * 0.5 * 3000.0
    assert abs(_mass(r, sc)[-1] - expected) / expected < 1e-10
    # profile breakpoints (1000, 2000) are step boundaries even though they are not output times
    t = step_times(sc)
    assert any(abs(t - 1000.0) < 1e-9) and any(abs(t - 2000.0) < 1e-9)


def test_periodic_conserves_mass_with_decay_budget():
    sc = small(mode="periodic", initial=Initial("gaussian", 5.0, 3000.0, 4000.0, 800.0), T=3600.0)
    r = simulate(sc, [1e5, 0.0, 0.0])
    assert r.diagnostics["max_rel_budget_residual"] < 1e-10
    assert r.budget is not None and np.all(r.budget["inflow"].values == 0) and np.all(r.budget["outflow"].values == 0)


def test_open_budget_closes_with_background_and_ic():
    sc = small(bg=1.0, initial=Initial("gaussian", 4.0, 6000.0, 5000.0, 1500.0), T=7200.0)
    r = simulate(sc, [1e5, 1.5e5, 8e4])
    b = r.budget
    assert b is not None
    assert r.diagnostics["max_rel_budget_residual"] < 1e-9
    assert (b["outflow"].values > 0).all() and (b["inflow"].values > 0).all()
    # totals telescope: sum of mass changes = M(T) - M(0)
    m = _mass(r, sc)
    np.testing.assert_allclose(b["mass_change"].sum(), m[-1] - m[0], rtol=1e-10)


def test_open_net_export_without_background():
    # c_b = 0: the only inflow-face term is the (negative) diffusive draw-down; the plume and IC are exported.
    sc = small(initial=Initial("gaussian", 4.0, 6000.0, 5000.0, 1500.0), T=7200.0)
    r = simulate(sc, [1e5, 1.5e5, 8e4])
    b = r.budget
    assert b is not None
    assert (b["outflow"].values > 0).all() and (b["inflow"].values <= 0).all()
    assert r.diagnostics["max_rel_budget_residual"] < 1e-9
    # the whole emission + IC is either removed, exported or still inside
    m = _mass(r, sc)
    total_in = float(b["emission"].sum()) + m[0]
    np.testing.assert_allclose(m[-1] + float(b["removal"].sum()) + float((b["outflow"] - b["inflow"]).sum()), total_in,
                               rtol=1e-9)


def test_steady_throughflow_balance():
    # Long run to (near) steady state with wind along x: out = in + emission - removal in the last interval.
    sc = small(wind=(WindSegment(0.0, 1.0, 0.0),), bg=1.0, T=150_000.0, dt=500.0, out=15_000.0)
    r = simulate(sc, [1e5, 1.5e5, 8e4])
    b = r.budget
    assert b is not None
    last = b.isel(time=-1)
    assert abs(float(last["mass_change"])) < 1e-5 * float(last["emission"])  # steady to 1e-5 (e^-20 relaxation)
    balance = float(last["inflow"] + last["emission"] - last["removal"])
    assert abs(float(last["outflow"]) - balance) < 1e-5 * float(last["outflow"])
    assert float(last["removal"]) > 0 and float(last["outflow"]) > 0


# ---- reference settings, nonnegativity, schemes --------------------------------------------------------------

def test_reference_nonnegative_and_fast():
    sc, q = reference()
    r = simulate(sc, q)
    c = r.concentration.values
    assert c.min() >= -1e-10 * c.max() and c.max() > 0  # M-matrix scheme: no clipping needed
    assert r.diagnostics["n_steps"] == 240 and r.diagnostics["max_rel_budget_residual"] < 1e-9
    assert r.runtime_s < 15.0


def test_exponential_vs_upwind_differ_and_are_nonnegative():
    sc = small(T=3600.0)
    q = [1e5, 1.5e5, 8e4]
    a = simulate(sc, q, SolverOptions("exponential")).concentration.values
    b = simulate(sc, q, SolverOptions("upwind")).concentration.values
    assert a.min() >= -1e-10 * a.max() and b.min() >= -1e-10 * b.max()
    # upwind adds ~u*dx/2 of numerical diffusion; exponential adds less here, so fields differ at the % level
    assert np.abs(a - b).max() > 1e-2 * a.max()


# ---- piecewise-constant wind ---------------------------------------------------------------------------------

def test_wind_reversal_flips_classification_and_budget():
    wind = (WindSegment(0.0, 1.0, 0.0), WindSegment(2400.0, -1.0, 0.0))
    sc = small(wind=wind, lam=0.0, initial=Initial("gaussian", 5.0, 8500.0, 5000.0, 600.0), T=4800.0)
    a_out0, a_in0, _ = _boundary_coefficients(sc, wind[0])
    a_out1, a_in1, _ = _boundary_coefficients(sc, wind[1])
    assert a_out0[:, -1].min() > 0 and a_out0[:, :-1].max() == 0 and a_in0[:, 0].min() > 0
    assert a_out1[:, 0].min() > 0 and a_out1[:, 1:].max() == 0 and a_in1[:, -1].min() > 0
    assert 2400.0 in step_times(sc)
    r = simulate(sc, [0.0, 0.0, 0.0])
    b = r.budget
    assert b is not None
    out_before = float(b["outflow"].sel(time=slice(0, 2400)).sum())
    out_after = float(b["outflow"].sel(time=slice(2700, 4800)).sum())
    # plume near the east wall leaves through it under +u; after reversal the east wall is an inflow face and the
    # west wall (far from the plume) exports almost nothing. Measured ratio is ~1e-4 or less.
    assert out_before > 0 and out_after < 1e-2 * out_before
    assert r.diagnostics["max_rel_budget_residual"] < 1e-9


def test_step_plan_splits_at_discontinuity():
    wind = (WindSegment(0.0, 1.0, 0.0), WindSegment(1234.0, -1.0, 0.0))
    sc = small(wind=wind, T=3600.0, dt=60.0, out=300.0)
    t = step_times(sc)
    assert np.isclose(t, 1234.0).any() and t[0] == 0.0 and t[-1] == 3600.0
    assert np.all(np.diff(t) <= 60.0 + 1e-9) and np.all(np.diff(t) > 0)
    assert all(np.isclose(t, o).any() for o in sc.output_times)


# ---- input validation ----------------------------------------------------------------------------------------

def test_closed_with_wind_raises():
    with pytest.raises(ValueError):
        small(mode="closed")  # default wind is nonzero


@pytest.mark.parametrize("q", [[1.0, 2.0], [1.0, -1.0, 1.0], [1.0, float("nan"), 1.0], [1.0, float("inf"), 1.0],
                               [[1.0, 1.0, 1.0]]])
def test_invalid_q_raises(q):
    with pytest.raises(ValueError):
        simulate(small(), q)


def test_exponential_requires_positive_K():
    with pytest.raises(ValueError):
        simulate(small(K=0.0), [0.0, 0.0, 0.0])

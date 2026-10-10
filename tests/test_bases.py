"""Inflow-concentration bases and stream-function wind in the FiPy forward model. Real FiPy solves on small grids.

Tolerances: roundoff-level identities use 1e-12 (relative to the field maximum; LU solve tolerance 1e-10 default is
for the linear solver but the systems here are identical or differ only by rounding of the RHS); linearity /
superposition 1e-10 (two LU solves of the same matrix with different right-hand sides, solver tolerance 1e-10 in
relative residual); budget residuals 1e-9 (they close to ~1e-13 in practice)."""
from __future__ import annotations

import dataclasses as dc
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pravah.config import (Boundary, Domain, GaussianSource, InflowBasis, Initial, Scenario, SolverOptions, Times,
                           Transport, WindSegment, load_config, scenario_from_dict)
from pravah.forward import (FIPY, _psi_face_velocity, fp, psi_face_fluxes,  # pyright: ignore[reportPrivateUsage]
                            simulate, step_times)

ROOT = Path(__file__).resolve().parents[1]
L = 10000.0
N = 24
SRC = (GaussianSource("s", 6000.0, 5000.0, 600.0),)
EXP = SolverOptions(convection="exponential")
UPW = SolverOptions(convection="upwind")


def scen(wind: tuple[WindSegment, ...] = (WindSegment(0.0, 1.0, 0.0),), bases: tuple[InflowBasis, ...] = (),
         bg: float = 0.0, K: float = 100.0, lam: float = 1.0 / 21600.0, T: float = 3600.0,
         initial: Initial | None = None, sources: tuple[GaussianSource, ...] = SRC) -> Scenario:
    return Scenario(domain=Domain(L, L, N, N), transport=Transport(wind, K, lam, 500.0),
                    boundary=Boundary("open", bg, bases), initial=initial or Initial(),
                    times=Times(T, 60.0, 300.0), sources=sources)


def corner_xy(sc: Scenario) -> tuple[np.ndarray, np.ndarray]:
    d = sc.domain
    return (np.arange(d.nx + 1) * d.dx)[None, :], (np.arange(d.ny + 1) * d.dy)[:, None]


def uniform_psi(sc: Scenario, u: float, v: float) -> np.ndarray:
    x, y = corner_xy(sc)
    return u * y - v * x + 0.0 * x


def psi_wind(psi: np.ndarray, u: float = 1.0, v: float = 0.0, t0: float = 0.0) -> WindSegment:
    return WindSegment.from_streamfunction(t0, psi, u, v)


def maxabs(a: np.ndarray) -> float:
    return float(np.abs(a).max())


def budget(res) -> xr.Dataset:
    assert res.budget is not None
    return res.budget


def values(res) -> np.ndarray:
    return np.asarray(res.concentration.values)


# ---- (a) a basis acts only on inflow faces ---------------------------------------------------------------------

def test_basis_acts_only_on_inflow_faces():
    base = simulate(scen(), [1e5])
    mk = lambda side: scen(bases=(InflowBasis("b", (side,)),))  # noqa: E731
    west = values(simulate(mk("west"), [1e5, 0.05])) - values(base)
    assert west.min() >= -1e-15 and west[-1].max() > 1e-3
    # plume enters from the west: the increment decays away from the west edge along the centreline
    prof = west[-1, N // 2, :]
    assert prof[0] > prof[N // 2] > prof[-1] >= 0
    # outflow face (east) and tangential faces (south/north, v = 0) have zero inflow coefficient: exactly no change
    for side in ("east", "north", "south"):
        assert np.array_equal(values(simulate(mk(side), [1e5, 0.05])), values(base)), side


# ---- (b) wind reversal flips which basis faces act --------------------------------------------------------------

def test_wind_reversal_flips_inflow_faces():
    T = 3600.0
    wind = (WindSegment(0.0, 1.0, 0.0), WindSegment(T / 2, -1.0, 0.0))
    west = simulate(scen(wind, (InflowBasis("w", ("west",)),), T=T), [0.0, 0.05])
    east = simulate(scen(wind, (InflowBasis("e", ("east",)),), T=T), [0.0, 0.05])
    both = simulate(scen(wind, (InflowBasis("w", ("west",)), InflowBasis("e", ("east",))), T=T),
                    [0.0, 0.05, 0.05])
    tt = budget(west)["t_start"].values
    first, second = tt < T / 2, tt >= T / 2
    wi, ei = budget(west)["inflow"].values, budget(east)["inflow"].values
    assert (wi[first] > 0).all()
    # after reversal the west faces are outflow: no basis source; only the (non-positive) a_in*c back-diffusion at
    # the new inflow side remains
    assert (wi[second] <= 0).all()
    # east basis: before reversal east faces are outflow (a_in = s_in = 0) -> exactly nothing enters or exists
    assert (ei[first] == 0).all() and not values(east)[tt.searchsorted(0):int(T / 2 / 300) + 1].any()
    assert (ei[second] > 0).all()
    assert values(east)[-1].max() > 1e-3
    for r in (west, east, both):
        assert r.diagnostics["max_rel_budget_residual"] <= 1e-9
    # linearity of the two bases in the reversal case
    np.testing.assert_allclose(values(both), values(west) + values(east), rtol=0, atol=1e-10 * maxabs(values(both)))


# ---- (c) uniform all-sides basis == scalar background -----------------------------------------------------------

@pytest.mark.parametrize("opts", [EXP, UPW])
def test_uniform_basis_equals_scalar_background(opts: SolverOptions):
    a = 0.04
    wind = (WindSegment(0.0, 1.0, 0.4), WindSegment(1800.0, -0.5, 0.3))
    ref = values(simulate(scen(wind, bg=a), [1e5], opts))
    got = values(simulate(scen(wind, (InflowBasis("all"),)), [1e5, a], opts))
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12 * maxabs(ref))


# ---- (d) superposition, (e) linearity -----------------------------------------------------------------------------

def test_superposition_and_linearity():
    sc = scen((WindSegment(0.0, 1.0, 0.3),), (InflowBasis("w", ("west", "south")),))
    em, ba, both = (values(simulate(sc, th)) for th in ([1e5, 0.0], [0.0, 0.05], [1e5, 0.05]))
    np.testing.assert_allclose(em + ba, both, rtol=0, atol=1e-10 * maxabs(both))
    double = values(simulate(sc, [0.0, 0.10]))
    np.testing.assert_allclose(double, 2 * ba, rtol=0, atol=1e-10 * maxabs(double))
    assert maxabs(ba) > 1e-3


# ---- (f) time profile: breakpoints and onset ----------------------------------------------------------------------

def test_ramp_basis_onset():
    onset = 1700.0  # not a multiple of dt = 60 nor of the output interval
    sc = scen(bases=(InflowBasis("r", ("west",), ((0.0, 0.0), (onset, 1.0))),))
    assert np.isclose(step_times(sc), onset, atol=1e-9, rtol=0).any()
    res = simulate(sc, [0.0, 0.05])
    tt = res.concentration["time"].values
    c = values(res)
    assert not c[tt <= 1500.0].any()  # exactly zero before onset (zero IC, no sources)
    assert c[tt >= 1800.0].max() > 0 and c[-1].max() > 1e-3
    # without the breakpoint the first step would straddle the onset: it is in the plan, so a step ends at onset
    assert not c[tt == 1500.0].any()


# ---- stream-function wind -----------------------------------------------------------------------------------------

def test_psi_face_mapping_is_one_to_one_and_divergence_free():
    sc = scen()
    rng = np.random.default_rng(0)
    psi = rng.normal(size=(N + 1, N + 1)) * 1000.0
    grid = fp.Grid2D(dx=sc.domain.dx, dy=sc.domain.dy, nx=N, ny=N)
    vel = _psi_face_velocity(sc, grid, psi)
    assert vel.shape == (2, grid.numberOfFaces)
    # independent of our own indexing: net flux of the assigned face field through each cell, from FiPy's own
    # cell-face connectivity, normals and areas (outward sign from the cell's side of each face)
    def net_flux(v: np.ndarray) -> np.ndarray:
        ids = np.asarray(grid.cellFaceIDs)  # (4, ncells)
        n = np.asarray(grid.faceNormals)
        area = np.asarray(grid._faceAreas)  # pyright: ignore[reportPrivateUsage]
        flux = (v * n).sum(axis=0) * area  # along FiPy's face normal
        owner = np.asarray(grid.faceCellIDs)[0]  # normal points out of the first adjacent cell
        sign = np.where(owner[ids] == np.arange(ids.shape[1])[None, :], 1.0, -1.0)
        return (flux[ids] * sign).sum(axis=0)

    scale = np.abs(vel).max() * sc.domain.dx
    assert np.abs(net_flux(vel)).max() < 1e-12 * scale
    # a wrong assignment (scrambled faces) would not be divergence-free
    bad = vel.copy()
    ix = np.where(np.abs(np.asarray(grid.faceNormals)[0]) > 0.5)[0]
    bad[:, ix] = vel[:, ix[::-1]]
    assert np.abs(net_flux(bad)).max() > 1e-3 * scale


def test_psi_fluxes_sign_convention():
    sc = scen()
    uf, vf = psi_face_fluxes(sc, uniform_psi(sc, 2.0, -3.0))
    np.testing.assert_allclose(uf, 2.0, rtol=1e-13)
    np.testing.assert_allclose(vf, -3.0, rtol=1e-13)


@pytest.mark.parametrize("opts", [EXP, UPW])
@pytest.mark.parametrize("uv", [(1.0, 0.3), (-0.7, 0.5)])
def test_uniform_psi_equals_legacy_path(opts: SolverOptions, uv: tuple[float, float]):
    u, v = uv
    legacy_sc = scen((WindSegment(0.0, u, v),), bg=0.02)
    psi_sc = scen((psi_wind(uniform_psi(legacy_sc, u, v), u, v),), bg=0.02)
    a, b = simulate(legacy_sc, [1e5], opts), simulate(psi_sc, [1e5], opts)
    # same operator up to rounding of u_f = (psi diff)/dy; 1e-12 relative to the maximum
    np.testing.assert_allclose(values(b), values(a), rtol=0, atol=1e-12 * maxabs(values(a)))
    for k in ("inflow", "outflow", "removal"):
        np.testing.assert_allclose(budget(b)[k].values, budget(a)[k].values, rtol=1e-11, atol=0)
    assert b.diagnostics["wind_field"] is True and b.diagnostics["n_basis"] == 0
    assert b.diagnostics["max_courant"] > 0


def channel_psi(sc: Scenario, u: float = 1.0) -> np.ndarray:
    x, y = corner_xy(sc)
    s = 1500.0
    return u * y + 1000.0 * np.exp(-((x - L / 2) ** 2 + (y - L / 2) ** 2) / (2 * s**2))


def rotation_psi(sc: Scenario, omega: float = 1e-4) -> np.ndarray:
    x, y = corner_xy(sc)
    return 0.5 * omega * ((x - L / 2) ** 2 + (y - L / 2) ** 2)  # u = omega (y - yc), v = -omega (x - xc); nonzero at walls


@pytest.mark.parametrize("opts", [EXP, UPW])
@pytest.mark.parametrize("make", [channel_psi, rotation_psi])
def test_divergence_free_psi_preserves_constant_field(make, opts: SolverOptions):
    c0 = 0.07
    probe = scen()
    sc = scen((psi_wind(make(probe), 1.0, 0.0),), bg=c0, lam=0.0, initial=Initial("uniform", c0))
    res = simulate(sc, [0.0], opts)
    # conservative face fluxes: sum of interior + boundary face fluxes is exactly zero per cell -> c stays c0 to
    # solver tolerance (LU 1e-10 relative); a non-divergence-free discretisation would drift at O(div * t * c0)
    assert maxabs(values(res) - c0) <= 1e-10
    assert res.diagnostics["max_rel_budget_residual"] <= 1e-9


def test_budget_closes_with_bases_and_psi():
    sc = scen((psi_wind(rotation_psi(scen())), ), (InflowBasis("a", ("west", "south")),
                                                   InflowBasis("b", ("east", "north"), ((0.0, 0.5), (1800.0, 1.0)))),
              bg=0.01, initial=Initial("gaussian", 0.2, 5000.0, 5000.0, 800.0))
    for opts in (EXP, UPW):
        res = simulate(sc, [1e5, 0.03, 0.05], opts)
        assert res.diagnostics["max_rel_budget_residual"] <= 1e-9
        assert res.diagnostics["n_basis"] == 2 and res.diagnostics["wind_field"] is True
        assert budget(res)["inflow"].values.sum() > 0 and budget(res)["outflow"].values.sum() > 0


def test_psi_with_wind_reversal_and_basis_budget():
    p1 = uniform_psi(scen(), 1.0, 0.0)
    wind = (psi_wind(channel_psi(scen()), 1.0, 0.0), psi_wind(-p1, -1.0, 0.0, 1800.0))
    sc = scen(wind, (InflowBasis("w", ("west",)), InflowBasis("e", ("east",))))
    res = simulate(sc, [1e5, 0.05, 0.05])
    assert res.diagnostics["max_rel_budget_residual"] <= 1e-9


def test_periodic_with_psi_raises():
    sc = scen((psi_wind(uniform_psi(scen(), 1.0, 0.0)),))
    sc = dc.replace(sc, boundary=Boundary("periodic"))
    with pytest.raises(ValueError, match="stream-function"):
        simulate(sc, [1e5])


# ---- parameter vector / capabilities -------------------------------------------------------------------------------

def test_invalid_theta_length_and_values():
    sc = scen(bases=(InflowBasis("w", ("west",)),))
    for bad in ([1e5], [1e5, 0.1, 0.1], np.ones((1, 2)), [1e5, -0.1], [1e5, float("nan")]):
        with pytest.raises(ValueError):
            simulate(sc, bad)
    with pytest.raises(ValueError):
        simulate(scen(), [1e5, 0.1])  # basis amplitude without a basis


def test_capabilities_and_legacy_diagnostics():
    caps = FIPY.info.capabilities
    assert caps["boundary_bases"] is True and caps["wind_field"] is True
    res = simulate(scen(), [1e5])
    assert "n_basis" not in res.diagnostics and "wind_field" not in res.diagnostics  # legacy diagnostics unchanged


# ---- (k) legacy bit-identity vs the frozen reference ---------------------------------------------------------------

def test_legacy_reference_bit_identical():
    frozen = ROOT / "results" / "reference-v0" / "concentration.nc"
    if not frozen.exists():
        pytest.skip("results/reference-v0/concentration.nc not present")
    cfg = load_config(ROOT / "configs" / "reference.json")
    res = simulate(scenario_from_dict(cfg["scenario"]), cfg["q"])
    with xr.open_dataarray(frozen) as ref:
        assert np.array_equal(values(res), ref.values)


def test_tangential_faces_ignore_roundoff_wind() -> None:
    """|u.n| < 1e-12 is exactly tangential: a south basis under wind angle 180 deg (v = 1.2e-16 from sin(pi)) acts
    identically to v = 0, and v = +1e-16 equals v = -1e-16 (no inflow through the south side in either case)."""
    import math
    south = (InflowBasis("s", ("south",), ((0.0, 1.0),)),)
    v_sin = math.sin(math.pi)  # 1.2246e-16
    assert 0 < v_sin < 1e-12

    def run(v: float) -> np.ndarray:
        sc = scen(wind=(WindSegment(0.0, -1.0, v),), bases=south, T=1800.0)
        return np.asarray(simulate(sc, np.array([1e5, 0.05])).concentration.values)

    f0, fs, fp_, fm = run(0.0), run(v_sin), run(1e-16), run(-1e-16)
    assert np.array_equal(f0, fs) and np.array_equal(fp_, fm) and np.array_equal(f0, fp_)
    # the south basis delivered nothing: the field equals the source-only field
    only = scen(wind=(WindSegment(0.0, -1.0, 0.0),), T=1800.0)
    assert np.array_equal(f0, np.asarray(simulate(only, np.array([1e5])).concentration.values))
    # a real (non-tangential) southerly inflow does act
    assert not np.array_equal(f0, run(0.01))

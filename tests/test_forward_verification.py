"""Contract tests for pravah.forward: shape, units, boundary flux, nonnegative source, observation (T-006)."""
import numpy as np
import pytest

from pravah.forward import observe, simulate

L = 1e4
KW = dict(nx=16, ny=8, Lx=L, Ly=L / 2, dt=120.0, n_steps=20, K=50.0, lam=0.0, c_bg=0.0)


def blob(nx=16, ny=8):
    c = np.zeros((nx, ny))
    c[7:9, 3:5] = 1e-6
    return c


def mass(c, nx=16, ny=8):
    return c.sum() * (L / nx) * (L / 2 / ny)


def test_shape_and_determinism():
    a = simulate(u=(1.0, 0.5), c0=blob(), bc="study", **KW)
    assert a.shape == (16, 8)
    assert np.array_equal(a, simulate(u=(1.0, 0.5), c0=blob(), bc="study", **KW))


def test_noflux_wall_flux_is_zero_closed_box_conserves_mass():
    c = simulate(u=(3.0, -2.0), c0=blob(), bc="noflux", **KW)
    assert mass(c) == pytest.approx(mass(blob()), rel=1e-9)


def test_study_outflow_leaves_inflow_is_background():
    big = dict(KW, n_steps=400)
    c = simulate(u=(3.0, 0.0), c0=blob(), bc="study", **big)
    assert mass(c) < 1e-3 * mass(blob())  # pulse exits through the downwind wall
    z = simulate(u=(3.0, 0.0), c0=np.zeros((16, 8)), bc="study", **dict(big, c_bg=2e-6))
    assert np.allclose(z, 2e-6, rtol=1e-9)  # inflow brings c_bg, nothing else


def test_units_source_mass_rate():
    # theta g/s over A=L*L/2 with b normalised to unit integral and h=500: mass grows by theta*t (lam=0, closed box)
    h, theta = 500.0, 4.0
    b = np.full((16, 8), 1.0)
    b /= mass(b)
    c = simulate(u=(0.0, 0.0), c0=np.zeros((16, 8)), bc="noflux", source=theta * b / h, **KW)
    assert h * mass(c) == pytest.approx(theta * KW["dt"] * KW["n_steps"], rel=1e-9)


def test_source_nonnegative_and_inputs_validated():
    for bad in (dict(source=-np.ones((16, 8))), dict(bc="dirichlet"), dict(scheme="central"),
                dict(u=(1.0, 2.0, 3.0)), dict(c0=np.zeros((8, 16)))):
        kw = dict(u=(1.0, 0.0), c0=blob(), bc="study", **KW) | bad
        with pytest.raises(ValueError):
            simulate(**kw)


def test_nonneg_solution_for_nonneg_data():
    c = simulate(u=(2.0, 1.0), c0=blob(), bc="study", source=1e-9, **KW)
    assert c.min() >= -1e-14


def test_observation_exact_and_bounds():
    c = np.arange(12.0).reshape(3, 4)
    assert np.array_equal(observe(c, [(2, 3), (0, 1)]), [11.0, 1.0])
    with pytest.raises(IndexError):
        observe(c, [(3, 0)])

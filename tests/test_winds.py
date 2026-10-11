"""EXP-018: prescribed winds (divergence, closure, reversal, series round trip) and the returned-mass definition."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pravah import forward as F
from pravah import winds as W
from pravah.config import Boundary, Domain, GaussianSource, Scenario, Times, Transport

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prescribed_winds", ROOT / "scripts/prescribed_winds.py")
pw = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(pw)  # type: ignore[union-attr]

D = Domain(10_000.0, 10_000.0, 20, 20)
C = (5000.0, 5000.0)
BUILDERS = {
    "steady": W.steady_uniform(D, 2.0, 0.5),
    "reversal": W.delayed_reversal(D, 2.0, 0.5, 1200.0),
    "ridge": W.ridge_deflection(D, 2.0, 0.0, C, 1500.0, 3000.0),
    "ridge_reversal": W.ridge_deflection(D, 2.0, 0.0, C, 1500.0, 3000.0, 1200.0),
    "eddy": W.recirculation_eddy(D, 2.0, 0.0, C, 1500.0, 8000.0),
}


def _scn(wind):
    return Scenario(domain=D, sources=(GaussianSource("s", 3000.0, 4000.0, 800.0),), transport=Transport(wind, 100.0, 1 / 21600, 500.0),
                    boundary=Boundary("open"), times=Times(2400.0, 60.0, 300.0))


@pytest.mark.parametrize("name", list(BUILDERS))
def test_psi_divergence_free(name):
    for w in BUILDERS[name]:
        if w.psi is not None:
            assert np.abs(W.face_divergence(D, w.psi)).max() < 1e-12


@pytest.mark.parametrize("name", list(BUILDERS))
def test_mass_closure_each_builder(name):
    scn = _scn(BUILDERS[name])
    res = F.simulate(scn, [1e5])
    t = pw.mass_terms(res, scn)
    assert t["closure_rel"] < 1e-8 and t["step_residual_rel"] < 1e-8 and t["min_conc"] > -1e-12 * t["max_conc"]


def test_reversal_reverses_flux_sign():
    for name in ("reversal", "ridge_reversal"):
        scn = _scn(BUILDERS[name])
        a, b = (F._side_normal_velocities(scn, w) for w in scn.transport.wind)  # pyright: ignore[reportPrivateUsage]
        assert all(np.allclose(a[s], -b[s], atol=1e-12) for s in a)
        assert (a["east"] > 0).all() and (b["east"] < 0).all() and (a["west"] < 0).all()


def test_series_round_trip():
    t = np.array([0.0, 600.0, 1200.0])
    psi = np.array([W._corner_psi(D, 2.0, 0.0, C, 1500.0, k * 2000.0) for k in (0, 1, 2)])  # type: ignore[attr-defined]
    seg = W.segments_from_series(D, t, psi=psi)
    assert [s.t_start for s in seg] == [0.0, 600.0, 1200.0] and np.array_equal(seg[1].psi, psi[1])
    # u, v at cell centres (average of the face fluxes) -> lsqr psi: gradients agree with the original away from the edges
    uf, vf = (np.array(x) for x in zip(*[F.psi_face_fluxes(_scn(seg), p) for p in psi]))
    uc, vc = 0.5 * (uf[:, :, 1:] + uf[:, :, :-1]), 0.5 * (vf[:, 1:] + vf[:, :-1])
    back = W.segments_from_series(D, t, u=uc, v=vc)
    for p, s in zip(psi, back):
        e = np.abs(np.diff(s.psi - p, axis=0)[2:-2, 2:-2]).max() / D.dy  # type: ignore[operator]
        assert e < 0.05 * np.abs(uc).max()


def test_returned_mass_two_humps():
    t = np.arange(0.0, 1000.0, 10.0)
    hump = lambda c: np.exp(-((t - c) / 40.0) ** 2)  # noqa: E731
    one = pw.returned_mass(t, hump(200), np.zeros_like(t), 0.0, 1.0, 1.0, 1.0)
    assert one[0] < 1e-12  # one pass, no return (lam = 0: net inward exchange is just d m)
    two = pw.returned_mass(t, hump(200) + 0.5 * hump(600), np.zeros_like(t), 0.0, 1.0, 1.0, 1.0)
    assert two[0] == pytest.approx(0.5, rel=0.02) and 200 < two[1] < 260  # return mass = height of the second hump


def _smoke_cfg():
    cfg = json.loads((ROOT / "configs/prescribed_winds.json").read_text())
    cfg.update(cfg["profiles"]["smoke"])
    return cfg


def test_steady_has_no_return_and_reversal_has():
    cfg = _smoke_cfg()
    steady = pw.run_case(cfg, "steady", 2.0, None)
    rev = pw.run_case(cfg, "reversal", 2.0, 12600.0)
    assert steady["returned_mass_fraction"] < 1e-3
    assert rev["returned_mass_fraction"] > 0.01
    assert max(steady["ext_closure_rel"], rev["ext_closure_rel"], rev["loc_closure_rel"]) < 1e-8


def test_linearity_total_is_sum():
    scn = _scn(BUILDERS["ridge_reversal"])
    scn2 = Scenario(domain=D, sources=(GaussianSource("a", 3000.0, 4000.0, 800.0), GaussianSource("b", 7000.0, 5000.0, 800.0)),
                    transport=scn.transport, boundary=scn.boundary, times=scn.times)
    tot = F.simulate(scn2, [1e5, 2e5]).concentration.values
    parts = F.simulate(scn2, [1e5, 0.0]).concentration.values + F.simulate(scn2, [0.0, 2e5]).concentration.values
    assert np.abs(tot - parts).max() < 1e-9 * tot.max()

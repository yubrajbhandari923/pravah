"""EXP-021: deflected_channel (prescribed wind, not terrain physics): divergence, turning angle, net offset, mass closure."""
import numpy as np
import pytest

from pravah import forward as F
from pravah import winds as W
from pravah.config import Boundary, Domain, GaussianSource, Scenario, Times, Transport

D = Domain(40_000.0, 20_000.0, 80, 40)
U, X0, WID, OFF = 2.0, 20_000.0, 3000.0, 3000.0


def _faces(seg):
    scn = Scenario(domain=D, sources=(GaussianSource("s", 5000.0, 10000.0, 800.0),), transport=Transport((seg,), 100.0, 0.0, 500.0),
                   boundary=Boundary("open"), times=Times(600.0, 60.0, 300.0))
    return scn, F.psi_face_fluxes(scn, seg.psi)


@pytest.mark.parametrize("kind", ["step", "bump"])
def test_divergence_free_and_turning_angle(kind):
    seg = W.deflected_channel(D, U, X0, WID, OFF, kind)[0]
    assert np.abs(W.face_divergence(D, seg.psi)).max() < 1e-12
    _, (uf, vf) = _faces(seg)
    assert np.allclose(uf, U)  # u = dpsi/dy = U everywhere
    peak = np.degrees(np.arctan(np.abs(vf / U).max()))
    want = np.degrees(np.arctan(OFF / (2 * WID) if kind == "step" else OFF / (WID * np.sqrt(np.e))))
    assert peak == pytest.approx(want, rel=0.03)  # grid-sampled slope of h


def test_net_offset_step_vs_bump():
    for kind, net in (("step", OFF), ("bump", 0.0)):
        psi = W.deflected_channel(D, U, X0, WID, OFF, kind)[0].psi
        # the streamline entering at y_in leaves at y_in + net: psi(y_out, east) == psi(y_in, west)
        y = np.arange(D.ny + 1) * D.dy
        y_out = np.interp(psi[20, 0], psi[:, -1], y)
        assert y_out - y[20] == pytest.approx(net, abs=2.0)
    with pytest.raises(ValueError):
        W.deflected_channel(D, U, X0, WID, OFF, "ramp")


@pytest.mark.parametrize("kind", ["step", "bump"])
def test_mass_closure(kind):
    seg = W.deflected_channel(D, U, X0, WID, OFF, kind)
    scn = Scenario(domain=D, sources=(GaussianSource("s", 5000.0, 10000.0, 800.0),), transport=Transport(seg, 100.0, 1 / 21600, 500.0),
                   boundary=Boundary("open"), times=Times(3600.0, 60.0, 600.0))
    r = F.simulate(scn, [1e5])
    b = r.budget
    rem = 500.0 * D.dx * D.dy * r.concentration.values[-1].sum()
    assert abs(rem - (b.emission.sum() - b.removal.sum() - (b.outflow - b.inflow).sum())) / b.emission.sum() < 1e-8
    assert r.diagnostics["min_concentration"] > -1e-12 * r.concentration.values.max()

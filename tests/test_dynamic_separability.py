"""DYN-06 separability: unit tests on synthetic operators (no solver); noise seeds 9000-9999 only."""
import numpy as np
import pytest

from pravah.studies.toy import dynamic_separability as sep
from pravah.studies.toy.dynamic_study import LAYOUTS, N_INV, NP, WINDS

SCEN = ("S1", "S2", "S4", "S5", "S6")


def _rand_op(seed, sigma=1.0):
    rng = np.random.default_rng(seed)
    Ax = rng.uniform(0, 5, (72, NP * N_INV))
    return sep._op(Ax / np.repeat(sep.REF, N_INV), sigma, np.repeat(np.arange(N_INV), 12))


def test_confounded_fixture_d_after_rescale_zero():
    op = sep.fixture_confounded()
    xa, xb, _ = sep.confounded_pair()
    r = sep.analytic(op, op, xa, xb)
    assert r["d_after_rescale"] < 1e-9
    assert r["d_raw"] > 1.0  # amplitudes differ, so only the rescale removes the difference
    assert r["cos_min"] > 1 - 1e-9


def test_rescale_nonneg_and_minimises():
    rng = np.random.default_rng(9001)
    u, v = rng.normal(size=20), rng.normal(size=20)
    for v_ in (v, -v):
        d, c = sep.rescale_distance(u, v_, 2.0)
        assert c >= 0
        grid = np.linspace(0, 5, 2001)
        assert d <= min(np.linalg.norm(u - g * v_) for g in grid) / 2.0 + 1e-9


def test_cross_wind_distance_symmetric():
    oa, ob = _rand_op(9002), _rand_op(9003, sigma=2.0)
    xa, xb = sep._base(), sep._base()
    xb[1, :3] = 0
    f, g = sep.analytic(oa, ob, xa, xb), sep.analytic(ob, oa, xb, xa)
    assert f["d_raw"] == pytest.approx(g["d_raw"])
    assert f["d_raw"] > 0


@pytest.mark.parametrize("name", SCEN)
def test_builders_nonneg_shape(name):
    sc = sep.build(name, _rand_op(9004))
    assert sc["valid"]
    for x in (sc["xa"], sc["xb"]):
        assert x.shape == (NP, N_INV) and (x >= 0).all()
    assert not np.allclose(sc["xa"], sc["xb"])


def test_s3_builder_and_dead_window_invalid():
    op = _rand_op(9005)
    sc = sep.build("S3", op, _rand_op(9006))
    assert sc["xb"][1, :3].sum() == 0 and sc["groups"] is None
    op.live[3:] = False
    assert not sep.build("S1", op)["valid"]


def test_attribution_deterministic_and_seed_range():
    op = sep.fixture_confounded()
    xa, _, g = sep.confounded_pair()
    r1, r2 = sep.attribute(op, xa, g, reps=6), sep.attribute(op, xa, g, reps=6)
    assert r1 == r2
    with pytest.raises(ValueError):
        sep.attribute(op, xa, g, reps=1001)


def test_confounded_attribution_is_chance():
    op = sep.fixture_confounded()
    xa, _, g = sep.confounded_pair()
    r = sep.attribute(op, xa, g, reps=20)
    assert r["truth"] == "s0" and r["conf_rate"] == pytest.approx(0.5, abs=0.2)


def test_no_variation_reports_no_change():
    r = sep.fixture_no_variation(_rand_op(9007), reps=20)
    assert r["truth"] == "none" and r["frac_none"] >= 0.8


def test_s7_layout_four_distinct_cells():
    for w in WINDS:
        cells = sep.l4_ref_cells(w)
        assert len(set(cells)) == 4 and cells[:3] == LAYOUTS["L4"][:3]
        ix, iy = cells[3]
        assert 0 <= ix < 24 and 0 <= iy < 18 and (ix in (1, 22) or iy in (1, 16))

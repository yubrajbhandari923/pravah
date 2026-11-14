"""E12 contract (docs/research/wind-mismatch-design.md §2, §11): truth untouched; inverse uses the model wind only."""
import numpy as np
import pytest

from pravah import benchmarks as B
from pravah import uncertainty as U
from pravah.experiment_config import ExperimentConfig, Split, Winds
from pravah.experiments import cross_objective as X


@pytest.fixture(scope="module")
def tiny():
    c = ExperimentConfig.default("EXP-E12-test")
    c.model.grid, c.model.spinup_h = (16, 16), 6
    c.design.n_sources = (2,)
    c.split = Split(train=(1, 1), validation=(2, 2), test=(3, 3))
    c.winds = Winds(train=["veering"], heldout=c.winds.heldout)
    return c, B.generate_scenario(c, 0, 2)


def test_perturbations_physical_and_input_untouched():
    w = B.make_winds(["veering", "steady_southerly"], 96, 5)[0]
    w0 = w.copy()
    s, th = U.to_polar(w)
    assert np.allclose(U.from_polar(s, th), w)
    for spec in U.PILOT_SPECS:
        m = U.perturb_winds(w, spec, 3)
        assert m.shape == w.shape and m is not w
        d, r = U.wind_error(w, m)
        if spec.family == "dir":
            assert d == pytest.approx(spec.magnitude) and r == pytest.approx(1)
        if spec.family == "spd":
            assert d == pytest.approx(0, abs=1e-9) and r == pytest.approx(spec.magnitude)
        if spec.family == "lag":
            L = int(spec.magnitude)
            assert np.array_equal(m[L:], w[:-L]) and np.array_equal(m[:L], np.tile(w[0], (L, 1)))
        if spec.oracle:
            assert np.array_equal(m, w)
    np.testing.assert_array_equal(w, w0)
    # AR error: seed-consistent across magnitudes (30 deg path = 2 x the 15 deg path), different across replicates
    a15 = U.to_polar(U.perturb_winds(w, U.WindSpec("ar", 15), 3))[1] - th
    a30 = U.to_polar(U.perturb_winds(w, U.WindSpec("ar", 30), 3))[1] - th
    wrap = lambda x: (x + 180) % 360 - 180
    assert np.allclose(wrap(a30), wrap(2 * a15), atol=1e-8)
    assert not np.allclose(U.ar_path(96, 3), U.ar_path(96, 4))
    e = U.ar_path(20000, 0)
    assert abs(e.std() - 1) < .1 and abs(np.corrcoef(e[:-1], e[1:])[0, 1] - np.exp(-1 / 12)) < .03


def test_truth_unchanged_and_oracle_equals_e07(tiny):
    c, sc = tiny
    tf, tw = sc.truth.field.copy(), sc.truth.winds.copy()
    F0, G0 = X.responses(sc)
    Fo, Go, wo = U.model_responses(sc, U.ORACLE)
    np.testing.assert_array_equal(Fo, F0)
    np.testing.assert_array_equal(Go, G0)
    Fd, _, wd = U.model_responses(sc, U.WindSpec("dir", 30))
    assert np.abs(Fd - F0).max() > 1e-3 * np.abs(F0).max()  # the wrong wind really changes the inverse model
    np.testing.assert_array_equal(sc.truth.field, tf)
    np.testing.assert_array_equal(sc.truth.winds, tw)
    np.testing.assert_array_equal(B.generate_scenario(c, 0, 2).truth.field, tf)


@pytest.mark.parametrize("spec", [s for s in U.PILOT_SPECS if not s.oracle], ids=lambda s: s.name)
def test_inverse_never_solves_with_true_wind(tiny, monkeypatch, spec):
    _, sc = tiny
    seen = []
    monkeypatch.setattr(X, "source_field", lambda cfg, xy, winds, n: seen.append(np.array(winds)) or
                        np.zeros((n,) + tuple(cfg.model.grid)))
    U.model_responses(sc, spec)
    assert len(seen) == sc.n_sources
    for w in seen:
        assert not np.array_equal(w, sc.truth.winds)
        np.testing.assert_array_equal(w, U.perturb_winds(sc.truth.winds, spec, sc.replicate))

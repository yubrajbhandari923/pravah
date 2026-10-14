import numpy as np

from pravah.studies.toy import dynamic_robustness as rb
from pravah.studies.toy import dynamic_study as ds


def test_perturbation_changes_truth_but_nominal_is_identical():
    h = np.tile(np.array([[1e5], [1e5], [1e5], [0.0], [0.0]]), (1, ds.N_FINE))
    a = ds.truth_field("steady_a", h, np.random.default_rng(1))
    b = ds.truth_field("steady_a", h, np.random.default_rng(1), {})
    assert np.array_equal(a, b)
    rot = ds.truth_field("steady_a", h, np.random.default_rng(1), {"rot_deg": 20.0})
    ex = ds.truth_field("steady_a", h, np.random.default_rng(1), {"extra_frac": 0.3})
    assert np.abs(rot - a).max() > 1e-4 * a.max()
    assert (ex - a).min() >= -1e-12 and (ex - a).max() > 0  # an extra source can only add pollutant


def test_robust_episodes_use_disjoint_seeds_and_pair_across_conditions():
    e1, e2 = rb.episodes("nominal"), rb.episodes("rot10")
    assert [e["seed"] for e in e1] == [e["seed"] for e in e2]
    assert min(e["seed"] for e in e1) == 8200 and max(e["seed"] for e in e1) < 8300
    assert all(e["perturb"] is None for e in e1) and all(e["perturb"] == {"rot_deg": 10.0} for e in e2)

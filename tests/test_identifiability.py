"""Contract for pravah.identifiability (E04 / T-025). Written before the implementation.

practical_rank(G, sigma, a_ref): number of singular values of G/sigma whose reference-scale signal s*a_ref is >= 1 (one
noise sigma). Differs from float rank (rank_info tol 1e-8): a mathematically full-rank G can have practical rank 0.
null_counterexample(G, sigma, a, rel): a2 = a + t*v with v the weakest right singular vector of G/sigma, t capped so a2 >= 0
and ||a2-a|| <= rel*||a||. Returns a2, rel_diff=||a2-a||/||a||, data_diff_sigma=rms(G(a2-a))/sigma, s_min.
greedy_design(G3, xy, n, sigma, a_ref, min_sep): greedy max logdet(I + a_ref^2 Gw^T Gw) over sites (G3: sites,hours,K).
"""
import numpy as np
import pytest

from pravah.identifiability import greedy_design, null_counterexample, practical_rank


def test_practical_rank_vs_float_rank():
    G = np.diag([1.0, 1e-3, 1e-6])
    assert practical_rank(G, sigma=1.0, a_ref=1.0) == 1
    assert practical_rank(G, sigma=1.0, a_ref=1e3) == 2
    assert practical_rank(G, sigma=1e-9, a_ref=1.0) == 3
    assert practical_rank(G, sigma=1e6, a_ref=1.0) == 0  # full float rank, nothing above noise


def test_practical_rank_boundary_inclusive_and_validation():
    assert practical_rank(np.diag([1.0, 0.5]), sigma=1.0, a_ref=1.0) == 1
    with pytest.raises(ValueError):
        practical_rank(np.eye(2), sigma=0.0, a_ref=1.0)
    with pytest.raises(ValueError):
        practical_rank(np.ones(3), sigma=1.0, a_ref=1.0)


def test_counterexample_exact_null():
    G = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # null direction (1,-1,0)/sqrt2
    a = np.array([2.0, 2.0, 1.0])
    r = null_counterexample(G, sigma=0.5, a=a, rel=0.5)
    assert np.all(r["a2"] >= 0) and r["rel_diff"] > 0.1
    assert r["data_diff_sigma"] < 1e-12 and r["s_min"] < 1e-12
    assert np.isclose(np.linalg.norm(r["a2"] - a) / np.linalg.norm(a), r["rel_diff"])


def test_counterexample_identifiable_is_detected():
    a = np.array([1.0, 1.0])
    r = null_counterexample(np.eye(2), sigma=0.1, a=a, rel=0.5)
    assert np.isclose(r["data_diff_sigma"], np.linalg.norm(r["a2"] - a) / np.sqrt(2) / 0.1)  # rms over 2 rows
    assert r["data_diff_sigma"] > 1  # distinguishable at this noise


def test_counterexample_nonneg_cap():
    G = np.array([[1.0, 1.0]])
    r = null_counterexample(G, sigma=1.0, a=np.array([1.0, 0.0]), rel=5.0)  # cap by nonnegativity, not by rel
    assert np.all(r["a2"] >= -1e-15) and r["rel_diff"] <= 5.0


def test_greedy_prefers_new_direction_over_duplicate():
    G3 = np.zeros((3, 1, 2))
    G3[0, 0], G3[1, 0], G3[2, 0] = [2, 0], [1.9, 0], [0, 1]
    xy = np.array([[0, 0], [5000, 0], [10000, 0]], float)
    assert sorted(greedy_design(G3, xy, 2, sigma=0.1, a_ref=1.0, min_sep=1000.0)) == [0, 2]


def test_greedy_respects_min_sep_and_n():
    G3 = np.random.default_rng(0).normal(size=(4, 2, 2))
    xy = np.array([[0, 0], [10, 0], [5000, 0], [9000, 0]], float)
    ids = greedy_design(G3, xy, 3, sigma=1.0, a_ref=1.0, min_sep=1000.0)
    assert len(ids) == 3 and not ({0, 1} <= set(ids))
    with pytest.raises(ValueError):
        greedy_design(G3, xy, 4, sigma=1.0, a_ref=1.0, min_sep=1000.0)  # cannot place 4 with the separation rule

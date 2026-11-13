"""Contract tests for src/pravah/inversion_study.py (T-015). Written before the code.

Contract (units: a in ug s^-1, readings in ug m^-3, a_ref = 100 ug s^-1 per protocol 10.2)
site_sigma(sigma0, n_sites, seed) -> (n_sites,) positive per-site noise std, sigma0 * factor, factor log-uniform in
    [1/3, 3]; deterministic in (sigma0, n_sites, seed); independent of any truth. ValueError for sigma0 <= 0, n_sites < 1.
source_scores(a_hat, a, a_ref=100.) -> dict(source_rmse_ref = sqrt(mean((a_hat-a)^2))/a_ref  [protocol 10.2 item 1],
    source_relative_rmse = ||a_hat-a||_2/||a||_2, parameter_bias = mean(a_hat-a)/mean(a)). ValueError on shape mismatch,
    non-finite input, or a == 0 everywhere.
crb_fraction(G, sigma, a) -> fraction of k with Cramer-Rao sd_k <= 0.5 a_k, sd = sqrt(diag((G^T diag(sigma^-2) G)^-1))
    (protocol 9). Rank deficient at relative tol 1e-8 -> 0.0 (information matrix singular; no finite CRB). Local,
    unconstrained approximation. ValueError on shape mismatch / sigma <= 0.
failure_mode(rank, K, a_hat) -> "non_identifiable" if rank < K (checked first); else "all_zero_estimate" if no
    coefficient is > 0 (protocol 10.2: undefined estimates are failures); else "estimated". Descriptive only: it is
    not a pass/fail threshold on error.
"""
import numpy as np
import pytest

from pravah.inversion_study import crb_fraction, failure_mode, site_sigma, source_scores


def test_site_sigma_range_determinism_and_errors():
    s = site_sigma(0.2, 500, 7)
    assert s.shape == (500,) and (s >= 0.2 / 3 - 1e-12).all() and (s <= 0.2 * 3 + 1e-12).all()
    assert np.array_equal(s, site_sigma(0.2, 500, 7)) and not np.array_equal(s, site_sigma(0.2, 500, 8))
    assert s.std() > 0.05 * 0.2  # genuinely heteroscedastic
    assert np.allclose(site_sigma(2.0, 500, 7), 10 * s)  # pure scale
    for bad in [(0.0, 3, 1), (-1.0, 3, 1), (1.0, 0, 1)]:
        with pytest.raises(ValueError):
            site_sigma(*bad)


def test_source_scores_hand_computed():
    r = source_scores(np.array([110., 0.]), np.array([100., 50.]))
    # errors (10, -50): rms = sqrt((100+2500)/2); relative = sqrt(2600)/sqrt(12500); bias = -20/75
    assert r["source_rmse_ref"] == pytest.approx(np.sqrt(1300) / 100)
    assert r["source_relative_rmse"] == pytest.approx(np.sqrt(2600 / 12500))
    assert r["parameter_bias"] == pytest.approx(-40 / 150)
    exact = source_scores(np.array([1., 2.]), np.array([1., 2.]))
    assert exact["source_relative_rmse"] == 0 and exact["parameter_bias"] == 0


def test_source_scores_errors():
    for bad in [(np.ones(2), np.ones(3)), (np.array([np.nan, 1.]), np.ones(2)), (np.ones(2), np.zeros(2))]:
        with pytest.raises(ValueError):
            source_scores(*bad)


def test_crb_fraction_hand_computed():
    # G = I (2x2), sigma = 1 -> sd = (1, 1). a = (10, 1): sd <= 0.5 a only for k=0.
    assert crb_fraction(np.eye(2), np.ones(2), np.array([10., 1.])) == 0.5
    assert crb_fraction(np.eye(2), np.ones(2), np.array([10., 10.])) == 1.0
    assert crb_fraction(np.eye(2), 10 * np.ones(2), np.array([10., 10.])) == 0.0  # noisier -> sd 10 > 5
    # scalar sigma accepted via broadcast; per-site sigma: G=[[1],[1]], sigma=(1,1) -> sd = 1/sqrt(2)
    assert crb_fraction(np.ones((2, 1)), np.ones(2), np.array([1.4])) == 0.0  # sd 0.7071 > 0.5*1.4 = 0.7
    assert crb_fraction(np.ones((2, 1)), np.ones(2), np.array([1.5])) == 1.0  # 0.7071 <= 0.75
    assert crb_fraction(np.ones((2, 1)), np.array([2.]) * np.ones(2), np.array([2.9])) == 1.0  # sd 1.414 <= 1.45


def test_crb_fraction_rank_deficient_and_errors():
    G = np.array([[1., 1.], [2., 2.], [3., 3.]])
    assert crb_fraction(G, np.ones(3), np.array([1e6, 1e6])) == 0.0
    with pytest.raises(ValueError):
        crb_fraction(np.eye(2), np.array([1., 0.]), np.ones(2))
    with pytest.raises(ValueError):
        crb_fraction(np.eye(2), np.ones(3), np.ones(2))


def test_failure_mode():
    assert failure_mode(1, 2, np.array([1., 1.])) == "non_identifiable"
    assert failure_mode(1, 2, np.zeros(2)) == "non_identifiable"  # rank checked first
    assert failure_mode(2, 2, np.zeros(2)) == "all_zero_estimate"
    assert failure_mode(2, 2, np.array([0., 3.])) == "estimated"

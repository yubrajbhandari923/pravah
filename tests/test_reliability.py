import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pravah.experiments import reliability as RL  # noqa: E402


def _toy(T=120, k=4, K=2, R=3, N=50, seed=0, ar=0.0):
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.5, 3, (k, T, K))
    a = np.array([1.0, 2.0])[:K]
    Fr = rng.uniform(0, 5, (K, T, N))
    e = rng.standard_normal((k, T, R))
    for t in range(1, T):
        e[:, t] = ar * e[:, t - 1] + np.sqrt(1 - ar ** 2) * e[:, t]
    z = np.einsum("stk,k->st", A, a)[..., None] + e
    tf = np.einsum("k,ktn->tn", a, Fr)
    return A, z, Fr, tf, a


def test_map_series_matches_e09():
    import e09_filter as E9
    A, z, *_ = _toy()
    hrs = np.arange(30, 100)
    for w in (24, None):
        u, C = RL.map_series(A, z, hrs, w)
        u9, C9 = E9.map_series(A, z, hrs, np.zeros(2), 1.0, w)
        assert np.allclose(u, u9) and np.allclose(C, C9)


def test_diagnostics_truth_free():
    """The diagnostic path never touches the truth: NaN-ing every truth array changes nothing."""
    A, z, Fr, tf, a = _toy()
    d0 = RL.block_diagnostics(A, z, 60, 24, Fr[:, 61])
    tf[:], a[:] = np.nan, np.nan
    d1 = RL.block_diagnostics(A, z, 60, 24, Fr[:, 61])
    assert set(d0) == set(RL.DIAGS)
    for n in RL.DIAGS:
        assert np.isfinite(d1[n]).all() and np.array_equal(d0[n], d1[n])


def test_diagnostics_under_correct_model():
    A, z, Fr, *_ = _toy(R=200, seed=1)
    d = RL.block_diagnostics(A, z, 80, None, Fr[:, 81])
    assert abs(d["nis"].mean() - 1) < 0.05            # NIS/k ~ 1 under the model
    assert abs(d["innov_ac1"].mean()) < 0.05           # white innovations
    assert abs(d["loso"].mean() - 1) < 0.1
    d_ar = RL.block_diagnostics(*_toy(R=200, seed=1, ar=0.8)[:2], 80, None, Fr[:, 81])
    assert d_ar["innov_ac1"].mean() > 0.5              # autocorrelated error is detected by lag-1


def test_gaussian_coverage_and_pit():
    """Known calibrated Gaussian forecasts: coverage ~ nominal, PIT ~ uniform; overconfident ones under-cover."""
    rng = np.random.default_rng(3)
    n, N, R = 50, 400, 2
    sd = rng.uniform(0.5, 2, (n, N))
    mu = rng.standard_normal((n, N, R))
    tf = np.zeros((n + 1, N))
    issue = np.arange(n)
    x = mu[..., 0] + sd * rng.standard_normal((n, N))
    tf[1:] = x
    mu[..., 1] = mu[..., 0]
    out = RL.block_labels(mu, sd, tf, issue, 0.0)
    for lv in RL.LEVELS:
        assert abs(out[f"cov{int(lv * 100)}"][0] - lv) < 0.02
    pit = np.array([out[f"pit{j}"][0] for j in range(10)]) / (n * N)
    assert np.abs(pit - 0.1).max() < 0.01
    bad = RL.block_labels(mu, sd / 4, tf, issue, 0.0)
    assert bad["cov90"][0] < 0.5 and bad["coverage_failure"][0]
    assert bad["crps"][0] > out["crps"][0]


def test_degenerate_zero_spread_has_no_active_cells():
    mu = np.zeros((5, 10, 1))
    out = RL.block_labels(mu, np.zeros((5, 10)), np.ones((6, 10)), np.arange(5), 1e-3)
    assert out["n_active"][0] == 0 and np.isnan(out["cov90"][0])


def test_auc_and_threshold():
    rng = np.random.default_rng(0)
    s, y = rng.standard_normal(300), rng.random(300) < 0.4
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]])
    assert RL.auc(s, y) == pytest.approx(brute)
    assert RL.auc(y.astype(float), y) == 1.0 and np.isnan(RL.auc(s, np.zeros(300, bool)))
    t = RL.youden_threshold(s + 3 * y, y)
    assert np.isfinite(t)
    r = RL.cluster_auc(s + y, y, np.arange(300) % 6, n_boot=200)
    assert r["ci_lo"] <= r["auc"] <= r["ci_hi"] and r["n_clusters"] == 6


def test_stratified_auc_removes_between_stratum_signal():
    rng = np.random.default_rng(1)
    g = np.repeat([0, 1], 500)
    y = rng.random(1000) < np.where(g, 0.8, 0.2)
    s = g + 0.01 * rng.standard_normal(1000)          # knows the stratum only
    assert RL.auc(s, y) > 0.7 and abs(RL.stratified_auc(s, y, g) - 0.5) < 0.08

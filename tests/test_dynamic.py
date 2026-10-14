import numpy as np
import pytest
from scipy.optimize import minimize

from pravah import dynamic as dyn
from pravah.config import Boundary, Domain, GaussianSource, InflowBasis, Scenario, Times, Transport, WindSegment
from pravah.forward import simulate
from pravah.inverse import compute_responses


def _scn(n_src=2, bases=()):
    srcs = tuple(GaussianSource(f"s{i}", 2000.0 + 3000 * i, 3000.0, 800.0) for i in range(n_src))
    return Scenario(domain=Domain(lx=8000.0, ly=6000.0, nx=16, ny=12), sources=srcs,
                    transport=Transport(wind=(WindSegment(0.0, 1.0, 0.3),)),
                    boundary=Boundary(mode="open", bases=bases), times=Times(duration=3600.0, dt=120.0, output_interval=600.0))


def test_expanded_responses_match_direct_history():
    """Superposition: sum of bin responses equals one direct run with the same piecewise profile."""
    scn = _scn(bases=(InflowBasis("west", sides=("west",)),))
    nb = 3
    r = compute_responses(dyn.expand(scn, nb))
    hist = np.array([[0.0, 5.0, 2.0], [3.0, 3.0, 0.0], [0.0, 0.0, 0.4]])
    direct = simulate(dyn.truth_scenario(scn, hist, dyn.bin_edges(3600.0, nb)), np.ones(3)).concentration.values
    np.testing.assert_allclose(r.field(hist.reshape(-1)).values - r.base, direct, rtol=1e-6, atol=1e-9)


def test_tv_nnls_matches_smoothed_reference_and_nonneg():
    rng = np.random.default_rng(0)
    A = rng.random((30, 6))
    y = A @ np.array([0, 2, 2, 2, 0.5, 0.5]) + 0.01 * rng.standard_normal(30)
    sig = np.full(30, 0.01)
    D = dyn._diff(1, 6)
    a, b = 1e-4, 0.5
    th, _ = dyn.tv_nnls(A, y, sig, np.zeros(6), a, b, D)

    def f(t):
        return 0.5 * np.sum(((A @ t - y) / sig) ** 2) + a * t @ t + b * np.sum(np.sqrt((D @ t) ** 2 + 1e-12))

    ref = minimize(f, np.ones(6), bounds=[(0, None)] * 6, method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12})
    assert th.min() >= 0
    assert f(th) <= f(ref.x) + 1e-3 * abs(f(ref.x)) + 1e-6


def test_noiseless_exact_recovery_and_zero_source():
    rng = np.random.default_rng(1)
    A = rng.random((40, 8))
    truth = np.array([0, 0, 3.0, 3.0, 3.0, 0, 0, 1.0])
    fit = dyn.fit_history(A, A @ truth, np.full(40, 1e-3), 1, 8, "M1", alpha=1e-12)
    np.testing.assert_allclose(fit.theta[0], truth, atol=1e-3)
    zero = dyn.fit_history(A, np.zeros(40), np.full(40, 1.0), 1, 8, "M2", alpha=1e-3, beta=1.0)
    assert np.allclose(zero.theta, 0, atol=1e-8)


def test_m2_beats_m1_on_piecewise_constant_with_noise():
    rng = np.random.default_rng(2)
    A = rng.random((24, 10))
    truth = np.repeat([0.0, 2.0], 5)
    err = {"M1": [], "M2": []}
    for _ in range(20):
        y = A @ truth + 0.3 * rng.standard_normal(24)
        for m in err:
            th = dyn.fit_history(A, y, np.full(24, 0.3), 1, 10, m, alpha=1e-3, beta=3.0).theta[0]
            err[m].append(np.sqrt(np.mean((th - truth) ** 2)))
    assert np.mean(err["M2"]) < np.mean(err["M1"])


def test_m0_is_constant_and_rejects_unknown_model():
    A = np.random.default_rng(3).random((10, 4))
    assert np.ptp(dyn.fit_history(A, A.sum(1), np.ones(10), 1, 4, "M0").theta) < 1e-9
    with pytest.raises(ValueError):
        dyn.fit_history(A, A.sum(1), np.ones(10), 1, 4, "M9")


def test_converges_to_exact_solution_at_production_conditioning():
    """Review of F-037: with rho=1 ADMM stopped at the cap; here the whitened Gram eigenvalues reach ~1e6+."""
    from scipy.optimize import lsq_linear
    rng = np.random.default_rng(5)
    Z = rng.standard_normal((144, 6)) @ rng.standard_normal((6, 30))  # strongly correlated columns
    A = np.abs(Z) * rng.uniform(1e-3, 1e-1, 30) * (rng.random(30) > 0.15)  # some dead columns
    truth = rng.uniform(0, 2, 30)
    sig = np.full(144, 3e-4)
    y = A @ truth + sig * rng.standard_normal(144)
    for alpha in (1e-6, 1e-2, 1.0):
        th, it = dyn.tv_nnls(A, y, sig, np.full(30, 0.5), alpha, 0.0, dyn._diff(5, 6))
        Aw = np.vstack([A / sig[:, None], np.sqrt(2 * alpha) * np.eye(30)])
        bw = np.concatenate([y / sig, np.sqrt(2 * alpha) * np.full(30, 0.5)])
        ref = lsq_linear(Aw, bw, bounds=(0, np.inf), tol=1e-14, max_iter=5000)
        f = lambda x: 0.5 * np.sum((Aw @ x - bw) ** 2)  # noqa: E731
        assert f(th) <= f(ref.x) * (1 + 1e-6) + 1e-9, (alpha, it, f(th), f(ref.x))


def test_not_converged_raises_and_scaling_is_layout_independent():
    rng = np.random.default_rng(6)
    A = rng.random((20, 6))
    with pytest.raises(dyn.NotConverged):
        dyn.tv_nnls(A, A.sum(1), np.full(20, 1e-3), np.zeros(6), 1e-6, 1.0, dyn._diff(1, 6), max_iter=3)
    # fixed reference amplitudes: duplicating rows (a different layout) must not change the prior meaning of alpha
    ref = np.array([2.0])
    f1 = dyn.fit_history(A, A.sum(1), np.full(20, 0.1), 1, 6, "M1", alpha=1e3, ref=ref).theta
    f2 = dyn.fit_history(np.vstack([A, A]), np.concatenate([A.sum(1)] * 2), np.full(40, 0.1), 1, 6, "M1", alpha=1e3, ref=ref).theta
    assert np.linalg.norm(f2) >= np.linalg.norm(f1) - 1e-9  # more data pulls estimates away from the (zero) prior, never toward it


def test_sensitivity_flags_dead_columns():
    A = np.random.default_rng(7).random((10, 6))
    A[:, 3:] = 0.0
    s = dyn.sensitivity(A, np.ones(10), 2, 3)
    assert (s[1] == 0).all() and (s[0] > 0).all()

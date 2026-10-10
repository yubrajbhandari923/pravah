import numpy as np
import pytest
import xarray as xr
from scipy.integrate import solve_ivp

from pravah.indoor import (IndoorBox, fit_indoor_emission, indoor_linear_parts, indoor_only_ambiguity,
                           indoor_response, misattribution_demo, outdoor_at_building, paired_observations)

X = (np.arange(10) + 0.5) * 100.0
Y = (np.arange(8) + 0.5) * 100.0
TIMES = np.linspace(0.0, 3600.0, 37)


def make_field(fn):
    T, YY, XX = np.meshgrid(TIMES, Y, X, indexing="ij")
    return xr.DataArray(fn(T, YY, XX), dims=("time", "y", "x"), coords={"time": TIMES, "y": Y, "x": X},
                        attrs={"units": "microgram/m^3"})


def box(P=0.6, a=1e-3, k=2e-4, V=200.0, x=420.0, y=330.0):
    return IndoorBox("b", x, y, P, a, k, V)


def const_field(c):
    return make_field(lambda T, Y_, X_: c + 0 * T)


def test_constant_outdoor_transient():
    b, c = box(), 7.0
    got = indoor_response(b, TIMES, np.full_like(TIMES, c))
    # default q profile is 1 microgram/s
    ref = (0.6 * 1e-3 * c + 1.0 / 200.0) / 1.2e-3 * (1 - np.exp(-1.2e-3 * TIMES))
    assert np.allclose(got, ref, rtol=1e-12, atol=0)


def test_q_zero_constant_outdoor_exact():
    b, c = box(), 7.0
    got = indoor_response(b, TIMES, np.full_like(TIMES, c), ((0.0, 0.0),))
    ref = b.P * b.a * c / b.lam * (1 - np.exp(-b.lam * TIMES))
    assert np.allclose(got, ref, rtol=1e-12, atol=0)


def test_steady_state():
    b = box(a=1e-2, k=1e-2)
    t = np.array([0.0, 1e6])
    got = indoor_response(b, t, np.array([5.0, 5.0]), ((0.0, 3.0),))[-1]
    assert got == pytest.approx(b.P * b.a * 5.0 / b.lam + 3.0 / (b.V * b.lam), rel=1e-12)


def test_emission_only():
    b = box()
    got = indoor_response(b, TIMES, np.zeros_like(TIMES), ((0.0, 2.0),))
    ref = 2.0 / (b.V * b.lam) * (1 - np.exp(-b.lam * TIMES))
    assert np.allclose(got, ref, rtol=1e-12, atol=0)


def test_lam_zero_accumulation():
    b = box(a=0.0, k=0.0)
    got = indoor_response(b, TIMES, np.full_like(TIMES, 9.0), ((0.0, 4.0),))
    assert np.allclose(got, 4.0 * TIMES / b.V, rtol=1e-14, atol=0)


def test_lam_zero_with_ramp_is_zero_outdoor_effect():
    # a = 0 means no outdoor coupling at all, even with k_dep = 0
    b = box(a=0.0, k=0.0)
    got = indoor_response(b, TIMES, TIMES * 0.01, ((0.0, 0.0),), c0=1.5)
    assert np.allclose(got, 1.5, rtol=0, atol=0)


def test_small_lam_continuous_with_series_branch():
    # a tiny but nonzero lam must match the lam=0 limit smoothly
    b0 = box(a=0.0, k=0.0)
    b1 = box(a=0.0, k=1e-12)
    r0 = indoor_response(b0, TIMES, TIMES * 0, ((0.0, 4.0),))
    r1 = indoor_response(b1, TIMES, TIMES * 0, ((0.0, 4.0),))
    assert np.allclose(r0, r1, rtol=1e-8)


def test_a_positive_no_deposition_steady_state():
    b = box(a=5e-3, k=0.0)
    t = np.array([0.0, 1e6])
    got = indoor_response(b, t, np.array([4.0, 4.0]), ((0.0, 2.0),))[-1]
    assert got == pytest.approx(b.P * 4.0 + 2.0 / (b.V * b.a), rel=1e-12)


def test_P_zero_decouples():
    b = box(P=0.0)
    r1 = indoor_response(b, TIMES, np.full_like(TIMES, 100.0), ((0.0, 1.0),))
    r2 = indoor_response(b, TIMES, np.zeros_like(TIMES), ((0.0, 1.0),))
    assert np.array_equal(r1, r2)


def test_linear_ramp_matches_solve_ivp():
    b = box()
    u = 2.0 + 0.004 * TIMES
    got = indoor_response(b, TIMES, u, ((0.0, 0.5),), c0=1.0)
    sol = solve_ivp(lambda t, c: b.P * b.a * (2.0 + 0.004 * t) - b.lam * c + 0.5 / b.V, (0, TIMES[-1]), [1.0],
                    t_eval=TIMES, method="DOP853", rtol=1e-12, atol=1e-14)
    # DOP853 at rtol 1e-12 is accurate to ~1e-11 relative on this smooth problem
    assert np.allclose(got, sol.y[0], rtol=1e-10, atol=1e-13)


def test_piecewise_q_steps_and_ramp():
    b = box()
    prof = ((0.0, 1.0), (1000.0, 0.0), (2200.0, 3.0))
    u = 1.0 + 0.001 * TIMES
    got = indoor_response(b, TIMES, u, prof)

    def q(t):
        return 1.0 if t < 1000 else (0.0 if t < 2200 else 3.0)
    c = 0.0
    out = {0.0: 0.0}
    for lo, hi in [(0, 1000), (1000, 2200), (2200, 3600)]:
        s = solve_ivp(lambda t, cc, qq=q(lo + 1): b.P * b.a * (1 + 0.001 * t) - b.lam * cc + qq / b.V,
                      (lo, hi), [c], t_eval=[hi], method="DOP853", rtol=1e-12, atol=1e-14)
        c = s.y[0, -1]
        out[float(hi)] = c
    for tq in (1000.0, 2200.0, 3600.0):
        i = int(np.argmin(abs(TIMES - tq)))
        assert got[i] == pytest.approx(out[tq], rel=1e-9)


def test_superposition():
    b = box()
    rng = np.random.default_rng(1)
    u1, u2 = rng.uniform(0, 5, TIMES.size), rng.uniform(0, 5, TIMES.size)
    z = ((0.0, 0.0),)
    r12 = indoor_response(b, TIMES, 2 * u1 + 3 * u2, z)
    assert np.allclose(r12, 2 * indoor_response(b, TIMES, u1, z) + 3 * indoor_response(b, TIMES, u2, z),
                       rtol=1e-12, atol=1e-13)
    zero = np.zeros_like(TIMES)
    q1 = indoor_response(b, TIMES, zero, ((0.0, 1.0),))
    q5 = indoor_response(b, TIMES, zero, ((0.0, 5.0),))
    assert np.allclose(q5, 5 * q1, rtol=1e-12, atol=0)


def test_linear_parts_match_direct():
    b = box()
    f = make_field(lambda T, Y_, X_: 3.0 + 0.001 * T + 0.002 * X_)
    drv, g = indoor_linear_parts(b, f)
    tt, co = outdoor_at_building(f, b)
    full = indoor_response(b, tt, co, ((0.0, 2.5),))
    assert np.allclose(drv + 2.5 * g, full, rtol=1e-12, atol=1e-14)
    assert co[0] == pytest.approx(3.0 + 0.002 * 420.0, rel=1e-12)  # bilinear exact for affine field


def synth(q_true, sigma, seed, b=None):
    b = b or box()
    f = make_field(lambda T, Y_, X_: 4.0 + 2.0 * np.sin(T / 900.0) ** 2 + 0 * X_)
    return paired_observations(b, f, TIMES, q_true, sigma, sigma, seed)


def test_fit_noise_free():
    d = synth(3.7, 0.0, 0)
    fit = fit_indoor_emission(box(), d["y_in"], 0.1, d["driven"], d["g_in"])
    assert fit.q_hat == pytest.approx(3.7, rel=1e-9)
    assert fit.identifiable and fit.singular_value > 1


def test_fit_noise_statistics():
    q, sig = 3.0, 0.05
    z = []
    for seed in range(200):
        d = synth(q, sig, seed)
        fit = fit_indoor_emission(box(), d["y_in"], sig, d["driven"], d["g_in"])
        z.append((fit.q_hat - q) / fit.std)
    z = np.array(z)
    # std error of mean z ~ 1/sqrt(200)=0.07; 4-sigma bounds
    assert abs(z.mean()) < 0.3 and 0.8 < z.std() < 1.2


def test_fit_clips_at_zero():
    b = box()
    d = synth(0.0, 0.0, 0)
    y = d["driven"] - 0.5 * d["g_in"]  # data that wants negative q
    fit = fit_indoor_emission(b, y, 0.1, d["driven"], d["g_in"])
    assert fit.q_hat == 0.0


def test_fit_alpha_shrinks_and_unidentifiable():
    d = synth(3.0, 0.0, 0)
    f0 = fit_indoor_emission(box(), d["y_in"], 0.1, d["driven"], d["g_in"])
    f1 = fit_indoor_emission(box(), d["y_in"], 0.1, d["driven"], d["g_in"], alpha=1e4)
    assert 0 < f1.q_hat < f0.q_hat and f1.std < f0.std
    tiny = fit_indoor_emission(box(), d["y_in"], 1.0, d["driven"], 1e-12 * d["g_in"])
    assert not tiny.identifiable


def test_paired_vs_indoor_only():
    b = box()
    t = TIMES
    shape = 1.0 + 0.8 * np.sin(t / 700.0)
    const = indoor_only_ambiguity(b, t, 0.1)
    assert const.cosine == pytest.approx(1.0, abs=1e-12)
    assert const.sv_indoor_only[1] < 1e-10 * const.sv_indoor_only[0]
    assert const.sv_paired[1] > 1e-3 * const.sv_paired[0]
    var = indoor_only_ambiguity(b, t, 0.1, shape=shape)
    assert var.cosine < 1.0 - 1e-6  # time-varying forcing partially breaks the collinearity
    assert var.sv_paired[1] > var.sv_indoor_only[1]


def test_outside_hull_raises():
    f = const_field(1.0)
    with pytest.raises(ValueError):
        outdoor_at_building(f, box(x=10.0))
    with pytest.raises(ValueError):
        indoor_linear_parts(box(y=790.0), f)


@pytest.mark.parametrize("kw", [dict(P=-0.1), dict(P=1.1), dict(a=-1.0), dict(k=-1.0), dict(V=0.0), dict(V=-3.0)])
def test_invalid_parameters_raise(kw):
    with pytest.raises(ValueError):
        box(**kw)


def test_negative_emission_raises():
    with pytest.raises(ValueError):
        indoor_response(box(), TIMES, np.ones_like(TIMES), ((0.0, -1.0),))


def test_misattribution_decomposition():
    b = box()
    f = make_field(lambda T, Y_, X_: 5.0 + 0 * T)
    m = misattribution_demo(b, f, TIMES, q_in=4.0)
    assert np.allclose(m["from_outdoor"] + m["from_indoor"], m["reading"], rtol=1e-13)
    fac = b.P * b.a / b.lam
    assert np.allclose(m["outdoor_equivalent"] - m["overstatement"], m["from_outdoor"] / fac, rtol=1e-12)
    # at late time (steady state) the from_outdoor part equals fac*c_out and the naive estimate overstates by the indoor part
    late = TIMES[-1]
    assert late * b.lam > 4
    assert m["from_outdoor"][-1] == pytest.approx(fac * 5.0, rel=0.02)
    assert m["outdoor_equivalent"][-1] > m["c_out_true"][-1]
    with pytest.raises(ValueError):
        misattribution_demo(box(P=0.0), f, TIMES, 1.0)

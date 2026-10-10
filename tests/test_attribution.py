"""Tests for the parameter-vector generalisation of inverse.py and for attribution.py.  Real FiPy solves on a small grid
(30x30 cells, 12 km, T = 7200 s, dt = 120 s).  Tolerances: tag sums / base split 1e-10 relative to the field maximum
(superposition of LU solves differing by rounding), responses vs direct simulate 1e-6 (as in test_inverse), reference
amplitude changes 1e-9, noise-free recovery 1e-4."""
from __future__ import annotations

import dataclasses as dc

import numpy as np
import pytest
import xarray as xr

from pravah.attribution import (base_parts, component_errors, contributions, emission_error, external_fraction,
                                identifiability_report, origin_fractions, total_field)
from pravah.config import (Boundary, Domain, GaussianSource, InflowBasis, Initial, Scenario, StudyRegion, Times,
                           Transport, WindSegment)
from pravah.forward import simulate
from pravah.inverse import (REF_BY_KIND, InferenceOptions, ResponseCache, compute_responses, fit_sources,
                            reconstruct, reconstruct_linear)
from pravah.observations import Observations, SensorLayout, sample
from pravah.sources import check_origin_tags

L_SRC = GaussianSource("L", 6000.0, 6000.0, 600.0, group="local")
E_SRC = GaussianSource("E", 1500.0, 6000.0, 700.0, group="external")
WEST = InflowBasis("west", ("west",))
STUDY = StudyRegion(4000.0, 4000.0, 8000.0, 8000.0)
IC = Initial("gaussian", 2.0, 6000.0, 6000.0, 900.0)
OBS_T = tuple(float(t) for t in range(600, 7201, 600))


def make(sources=(L_SRC, E_SRC), bases=(WEST,), bg=0.3, initial=IC, wind=(WindSegment(0.0, 1.0, 0.0),),
         T=7200.0) -> Scenario:
    return Scenario(domain=Domain(12_000.0, 12_000.0, 30, 30), sources=sources,
                    transport=Transport(wind=wind, K=100.0, lam=1 / 21600.0, H=500.0),
                    boundary=Boundary("open", bg, bases), initial=initial,
                    times=Times(duration=T, dt=120.0, output_interval=600.0), study=STUDY)


THETA = np.array([1.5e5, 2.0e5, 0.04])
SC = make()


def rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


@pytest.fixture(scope="module")
def cache() -> ResponseCache:
    return ResponseCache()


@pytest.fixture(scope="module")
def resp(cache):
    return cache.get(SC)


@pytest.fixture(scope="module")
def parts(cache):
    return base_parts(SC, cache=cache)


# ------------------------------------------------------------------ responses / parameter bookkeeping
def test_param_bookkeeping(resp):
    assert resp.g.shape[0] == 3
    assert resp.param_names == ("L", "E", "west")
    assert resp.param_kinds == ("emission", "emission", "inflow")
    assert resp.param_groups == ("local", "external", "boundary")
    assert resp.param_units == ("microgram/s", "microgram/s", "microgram/m^3")
    assert list(resp.refs) == [REF_BY_KIND["emission"], REF_BY_KIND["emission"], REF_BY_KIND["inflow"]]


@pytest.mark.parametrize("kw", [dict(), dict(bg=0.0, initial=Initial())], ids=["ic+bg", "zero"])
def test_decomposition_matches_direct_simulation(kw):
    sc = make(**kw)
    r = ResponseCache().get(sc)
    direct = simulate(sc, THETA).concentration.values
    assert rel(r.field(THETA).values, direct) < 1e-6
    assert np.array_equal(r.base, simulate(sc, np.zeros(3)).concentration.values)
    # the inflow response is a real, positive signal
    assert r.g[2].max() > 0


def test_reference_amplitudes_do_not_change_fields(resp):
    other = compute_responses(SC, q_ref=3e4, inflow_ref=0.4)
    assert list(other.refs) == [3e4, 3e4, 0.4]
    assert rel(other.g, resp.g) < 1e-9
    assert np.array_equal(other.base, resp.base)
    assert rel(other.field(THETA).values, resp.field(THETA).values) < 1e-9


def test_cache_distinguishes_refs(cache):
    n = cache.misses
    cache.get(SC, inflow_ref=0.4)
    assert cache.misses == n + 1
    cache.get(SC, inflow_ref=0.4)
    assert cache.misses == n + 1


# ------------------------------------------------------------------ attribution
def test_base_parts_sum_to_base(resp, parts):
    assert rel(parts["initial"] + parts["background"], resp.base) < 1e-12
    assert parts["initial"].max() > 0.5 and parts["background"].max() > 0.05


def test_base_parts_cached(cache, parts):
    assert base_parts(SC, cache=cache) is parts


@pytest.mark.parametrize("kw", [dict(), dict(bg=0.0, initial=Initial()), dict(bg=0.5, initial=Initial())],
                         ids=["ic+bg", "zero", "bg-only"])
def test_tags_sum_to_total(kw):
    sc = make(**kw)
    c = ResponseCache()
    r = c.get(sc)
    contrib = contributions(r, THETA, base_parts(sc, cache=c))
    assert set(contrib) == {"local", "external_sources", "boundary", "background", "initial"}
    total = total_field(contrib).values
    assert rel(total, r.field(THETA).values) < 1e-10
    assert rel(total, simulate(sc, THETA).concentration.values) < 1e-6  # DIRECT run of the full theta
    # each tag really carries its own piece
    assert contrib["local"].values.max() > 0 and contrib["external_sources"].values.max() > 0
    assert contrib["boundary"].values.max() > 0
    assert (contrib["initial"].values.max() > 0) == ("initial" not in kw)  # IC tag only with a configured IC
    assert (contrib["background"].values.max() > 0) == (kw.get("bg", 0.3) > 0)


def test_local_plume_leaving_and_reentering_stays_local():
    wind = (WindSegment(0.0, 1.2, 0.0), WindSegment(3600.0, -1.2, 0.0))
    pulse = GaussianSource("L", 6000.0, 6000.0, 600.0, profile=((0.0, 1.0), (1200.0, 0.0)), group="local")
    sc = make(sources=(pulse, E_SRC), wind=wind, bases=(), bg=0.0, initial=Initial())
    check_origin_tags(sc)
    c = ResponseCache()
    r = c.get(sc)
    th = np.array([1.5e5, 2.0e5])
    contrib = contributions(r, th, base_parts(sc, cache=c))
    w = np.zeros((30, 30))
    w[10:20, 10:20] = 1.0  # study cells (4000..8000 m)
    mean_local = np.einsum("tyx,yx->t", contrib["local"].values, w) / w.sum()
    t = np.asarray(r.times)
    mid, end = int(np.argmin(abs(t - 3600.0))), len(t) - 1
    # the pulse (emitted until 1200 s, centre 6000 + 1.2*3600 = 10.3 km at the turn) has left and is back by the end
    assert mean_local[mid] < 0.3 * mean_local[:mid].max()
    assert mean_local[end] > 2 * mean_local[mid] and mean_local[end] > 0
    assert float(contrib["local"].values[end][10:20, 10:20].min()) > 0
    # removing the external source leaves the local tag unchanged (real separate solve)
    sc1 = make(sources=(pulse,), wind=wind, bases=(), bg=0.0, initial=Initial())
    c1 = ResponseCache()
    contrib1 = contributions(c1.get(sc1), th[:1], base_parts(sc1, cache=c1))
    assert rel(contrib1["local"].values, contrib["local"].values) < 1e-9
    assert float(np.abs(contrib1["external_sources"].values).max()) == 0.0
    F = external_fraction(contrib, sc, t_exclude=0.0)
    assert F is not None and 0.0 <= F <= 1.0


def test_origin_fractions_sum_and_range(resp, parts):
    contrib = contributions(resp, THETA, parts)
    fr = origin_fractions(contrib, SC, t_exclude=1200.0)
    assert all(v is not None and 0.0 <= v <= 1.0 for v in fr.values())
    assert sum(v for v in fr.values() if v is not None) == pytest.approx(1.0, abs=1e-12)
    assert external_fraction(contrib, SC, 1200.0) == fr["external"]


# ------------------------------------------------------------------ external_fraction on hand-built fields
def hand(local, ext, bnd, bg, ini, nt=5):
    s = make()
    times = np.arange(nt) * 600.0
    xs, ys = (np.arange(30) + 0.5) * 400.0, (np.arange(30) + 0.5) * 400.0

    def da(v):
        return xr.DataArray(np.full((nt, 30, 30), float(v)), dims=("time", "y", "x"),
                            coords={"time": times, "x": xs, "y": ys})
    return s, {"local": da(local), "external_sources": da(ext), "boundary": da(bnd), "background": da(bg),
               "initial": da(ini)}


def test_external_fraction_analytical():
    s, c = hand(1.0, 2.0, 0.5, 0.5, 1.0)
    assert external_fraction(c, s) == pytest.approx(3.0 / 5.0, abs=1e-14)
    fr = origin_fractions(c, s)
    assert fr == {k: pytest.approx(v, abs=1e-14) for k, v in {"local": 0.2, "external": 0.6, "initial": 0.2}.items()}


def test_external_fraction_undefined_for_zero_total():
    s, c = hand(0, 0, 0, 0, 0)
    assert external_fraction(c, s) is None
    assert origin_fractions(c, s) == {"local": None, "external": None, "initial": None}


def test_external_fraction_undefined_for_negligible_mean():
    s, c = hand(0, 1.0, 0, 0, 0)
    # a single-time spike: window mean is 1/4 of the peak mean -> below eps_rel = 0.5 -> undefined
    v = c["external_sources"].values.copy()
    v[:] = 0.0
    v[2] = 1.0
    c["external_sources"] = c["external_sources"].copy(data=v)
    assert external_fraction(c, s, eps_rel=0.5) is None
    assert external_fraction(c, s, eps_rel=1e-3) == pytest.approx(1.0)


def test_external_fraction_needs_study():
    s, c = hand(1, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="study"):
        external_fraction(c, dc.replace(s, study=None))
    assert external_fraction(c, dc.replace(s, study=None), weights=np.ones((30, 30))) == pytest.approx(0.6)


# ------------------------------------------------------------------ errors
def test_component_errors(resp, parts):
    c = contributions(resp, THETA, parts)
    e = component_errors(c, c, SC, t_exclude=600.0)
    assert e["rmse_total"] == 0.0 and e["rmse_local"] == 0.0 and e["rmse_external"] == 0.0
    assert e["external_fraction_error"] == 0.0
    c2 = contributions(resp, THETA * np.array([1.0, 0.5, 1.0]), parts)
    e2 = component_errors(c2, c, SC, t_exclude=600.0)
    assert e2["rmse_local"] == 0.0 and e2["rmse_external"] > 0 and e2["rmse_total"] > 0
    assert e2["external_fraction_error"] is not None and e2["external_fraction_error"] < 0


def test_emission_and_inflow_errors_not_mixed():
    t_true = np.array([1.0e5, 2.0e5, 0.04])
    only_inflow = emission_error(t_true + np.array([0, 0, 0.02]), t_true, SC)
    assert only_inflow["all_emission"] == 0.0 and only_inflow["local"] == 0.0 and only_inflow["external"] == 0.0
    assert only_inflow["inflow"]["abs_error_ug_m3"] == {"west": pytest.approx(0.02)}
    assert only_inflow["inflow"]["relative_l2"] == pytest.approx(0.5)
    only_em = emission_error(t_true + np.array([1.0e4, 0, 0]), t_true, SC)
    assert only_em["local"] == pytest.approx(0.1) and only_em["external"] == 0.0
    assert only_em["all_emission"] == pytest.approx(1.0e4 / np.hypot(1e5, 2e5))
    assert only_em["inflow"]["abs_error_ug_m3"] == {"west": 0.0}
    with pytest.raises(ValueError):
        emission_error(np.zeros(2), np.zeros(2), SC)


# ------------------------------------------------------------------ fits
def obs_from(truth: xr.DataArray, layout: SensorLayout, sigma: float = 0.01) -> Observations:
    vals = sample(truth, layout, OBS_T)
    return Observations(vals, np.full(len(layout), sigma), layout, OBS_T)


GOOD = SensorLayout(("up", "w2", "s1", "s2", "s3", "s4", "d1", "d2"),
                    (800.0, 3000.0, 5000.0, 6500.0, 7500.0, 5500.0, 9000.0, 10500.0),
                    (6000.0, 6000.0, 5200.0, 6800.0, 5900.0, 7300.0, 6200.0, 5400.0))


def test_noise_free_recovery_of_all_three_kinds(cache):
    truth = simulate(SC, THETA).concentration
    fit = fit_sources(SC, obs_from(truth, GOOD), InferenceOptions(), cache)
    assert fit.param_names == ("L", "E", "west") and fit.param_units[2] == "microgram/m^3"
    assert fit.q_hat.shape == fit.q_std.shape == fit.q_scale.shape == (3,)
    assert list(fit.q_scale) == [1e5, 1e5, 0.05]
    assert np.max(np.abs(fit.q_hat / THETA - 1)) < 1e-4
    rep = identifiability_report(fit)
    assert rep["flags"] == [] and rep["effective_rank"] == 3
    # reconstructions agree
    r = cache.get(SC)
    assert rel(reconstruct_linear(r, fit).values, truth.values) < 1e-4
    assert rel(reconstruct(SC, fit).concentration.values, truth.values) < 1e-4


def test_direct_method_matches_linear_with_inflow(cache):
    truth = simulate(SC, THETA).concentration
    o = obs_from(truth, GOOD)
    lin = fit_sources(SC, o, InferenceOptions(), cache)
    drc = fit_sources(SC, o, InferenceOptions(method="direct"), cache)
    assert np.max(np.abs(drc.q_hat / lin.q_hat - 1)) < 1e-4


def test_scale_and_prior_vectors_validated(cache):
    truth = simulate(SC, THETA).concentration
    o = obs_from(truth, GOOD)
    with pytest.raises(ValueError, match="q_scale"):
        fit_sources(SC, o, InferenceOptions(q_scale=(1e5, 1e5)), cache)
    f = fit_sources(SC, o, InferenceOptions(q_scale=(1e5, 1e5, 0.5), alpha=1e-6, prior=tuple(THETA)), cache)
    assert np.max(np.abs(f.q_hat / THETA - 1)) < 1e-3


def test_no_arrival_is_flagged_not_hidden(cache):
    # E sits downwind of every sensor (u > 0): upwind decay length K/u = 100 m, so its signal is ~exp(-25)
    far = GaussianSource("E", 10500.0, 6000.0, 500.0, group="external")
    sc = make(sources=(L_SRC, far))
    check_origin_tags(sc)
    layout = GOOD.select([0, 1, 2, 3, 4, 5])  # all at x <= 7500
    truth = simulate(sc, THETA).concentration
    o = obs_from(truth, layout)
    fit0 = fit_sources(sc, o, InferenceOptions(alpha=0.0), cache)
    rep0 = identifiability_report(fit0)
    assert fit0.column_norms[1] < 1e-6 * fit0.column_norms[0]
    assert "rank_deficient" in rep0["flags"] and "unidentifiable:E" in rep0["flags"]
    assert "weakly_constrained" in rep0["flags"]
    assert "unidentifiable:L" not in rep0["flags"]
    fit1 = fit_sources(sc, o, InferenceOptions(alpha=1e-3), cache)
    assert abs(fit1.q_hat[1]) < 1e-6 * fit1.q_scale[1]  # = prior (0): the data say nothing, the estimate is not recovered
    assert fit1.q_std[1] / fit1.q_scale[1] <= 1 / np.sqrt(1e-3) * (1 + 1e-9)
    assert "unidentifiable:E" in identifiability_report(fit1)["flags"]  # still flagged despite the prior


def test_collinear_sources_are_flagged(cache):
    a = GaussianSource("A", 5500.0, 6000.0, 600.0, group="local")
    b = GaussianSource("B", 5600.0, 6000.0, 600.0, group="local")
    sc = make(sources=(a, b), bases=())
    layout = SensorLayout(("d1", "d2", "d3"), (9000.0, 10000.0, 10800.0), (6000.0, 6100.0, 5900.0))
    truth = simulate(sc, np.array([1e5, 1e5])).concentration
    fit = fit_sources(sc, obs_from(truth, layout), InferenceOptions(), cache)
    assert fit.column_cosines[0, 1] > 0.99
    assert "confounded:A,B" in identifiability_report(fit)["flags"]


def test_weakly_constrained_is_separate_from_rank_deficient(cache):
    truth = simulate(SC, THETA).concentration
    # very large stated noise: all whitened scaled singular values fall below 1, but the problem is not singular
    fit = fit_sources(SC, obs_from(truth, GOOD, sigma=1e3), InferenceOptions(), cache)
    rep = identifiability_report(fit)
    assert fit.rank == 3 and fit.condition < 1e6 and fit.effective_rank < 3
    assert "weakly_constrained" in rep["flags"] and "rank_deficient" not in rep["flags"]
    ok = identifiability_report(fit_sources(SC, obs_from(truth, GOOD), InferenceOptions(), cache))
    assert "weakly_constrained" not in ok["flags"] and "rank_deficient" not in ok["flags"]

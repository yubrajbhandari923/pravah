"""ForwardSolver interface: contract checker, `forward=` threading and solver-aware response caching.

The alternative backend is a real, different numerical scheme (first-order upwind convection), not a mock."""
import dataclasses
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr

from pravah.config import Domain, FieldResult, GaussianSource, Scenario, SolverOptions, Times, Transport, WindSegment
from pravah.design import Candidates, DesignOptions, build_case
from pravah.forward import ForwardSolver, check_field_result, simulate
from pravah.inverse import (InferenceOptions, ResponseCache, cache_key, compute_responses, fit_sources,
                            reconstruct)
from pravah.observations import ObservationSpec, SensorLayout, observe

SRC = (GaussianSource("S1", 2500.0, 2500.0, 600.0), GaussianSource("S2", 4000.0, 6000.0, 600.0))
SC = Scenario(domain=Domain(10_000.0, 10_000.0, 12, 12), sources=SRC,
              transport=Transport(wind=(WindSegment(0.0, 1.0, 0.3),)),
              times=Times(duration=1800.0, dt=150.0, output_interval=600.0))
LAY = SensorLayout(("a", "b", "c", "d"), (3000.0, 5000.0, 4500.0, 7000.0), (3000.0, 6500.0, 2800.0, 7000.0))
Q = np.array([1.2e5, 0.8e5])


class UpwindSolver:
    """Alternative numerical scheme (upwind convection) behind the ForwardSolver interface."""
    forward_id = "upwind-v1"

    def __call__(self, scenario: Scenario, q, solver: SolverOptions = SolverOptions()) -> FieldResult:
        return simulate(scenario, q, dataclasses.replace(solver, convection="upwind"))


ALT = UpwindSolver()


def _res(mutate) -> FieldResult:
    res = simulate(SC, Q)
    return dataclasses.replace(res, concentration=mutate(res.concentration.copy(deep=True)))


def test_simulate_satisfies_protocol():
    fwd: ForwardSolver = simulate
    alt: ForwardSolver = ALT
    check_field_result(fwd(SC, Q), SC)
    check_field_result(alt(SC, Q), SC)


@pytest.mark.parametrize("mutate,msg", [
    (lambda c: c.transpose("time", "x", "y"), "dims"),
    (lambda c: c.assign_coords(time=c["time"].values + 1.0), "'time'"),
    (lambda c: c.assign_coords(x=c["x"].values + 1.0), "'x'"),
    (lambda c: c.copy(data=np.where(np.arange(c.size).reshape(c.shape) == 3, np.nan, c.values)), "non-finite"),
    (lambda c: c.assign_attrs(units=""), "units"),
])
def test_check_rejects(mutate, msg):
    with pytest.raises(ValueError, match=msg):
        check_field_result(_res(mutate), SC)


def test_check_rejects_missing_units_and_bad_status():
    res = simulate(SC, Q)
    no_units = dataclasses.replace(res, concentration=xr.DataArray(res.concentration.values,
                                   dims=("time", "y", "x"), coords=res.concentration.coords))
    with pytest.raises(ValueError, match="units"):
        check_field_result(no_units, SC)
    with pytest.raises(ValueError, match="status"):
        check_field_result(dataclasses.replace(res, status=cast(Any, 3)), SC)


def test_alt_backend_chain_recovers_with_own_responses():
    truth = ALT(SC, Q).concentration
    obs = observe(truth, LAY, ObservationSpec((600.0, 1200.0, 1800.0), 0.0, 0))
    cache = ResponseCache()
    fit = fit_sources(SC, obs, InferenceOptions(method="linear"), cache, forward=ALT)
    assert np.allclose(fit.q_hat, Q, rtol=1e-4)
    assert cache.misses == 1
    rec = reconstruct(SC, fit, forward=ALT).concentration
    assert np.allclose(rec.values, truth.values, rtol=1e-4, atol=1e-8)
    # the default solver's own responses are different, so fitting alt data with them is biased
    cache_d = ResponseCache()
    biased = fit_sources(SC, obs, InferenceOptions(method="linear"), cache_d)
    assert not np.allclose(biased.q_hat, Q, rtol=1e-4)


def test_cache_keys_and_no_mixing():
    s = SolverOptions()
    assert cache_key(SC, s, 1e5) != cache_key(SC, s, 1e5, ALT)

    class Other(UpwindSolver):
        forward_id = "upwind-v2"
    assert cache_key(SC, s, 1e5, ALT) != cache_key(SC, s, 1e5, Other())
    cache = ResponseCache()
    r_def = cache.get(SC)
    r_alt = cache.get(SC, forward=ALT)
    assert cache.misses == 2 and r_def is not r_alt
    assert not np.allclose(r_def.g, r_alt.g)
    assert cache.get(SC, forward=ALT) is r_alt and cache.get(SC) is r_def and cache.hits == 2
    assert np.array_equal(r_alt.g, compute_responses(SC, forward=ALT).g)


def test_disk_cache_not_mixed(tmp_path):
    ResponseCache(tmp_path).get(SC)
    c2 = ResponseCache(tmp_path)
    r = c2.get(SC, forward=ALT)
    assert c2.misses == 1 and c2.hits == 0
    assert np.array_equal(r.g, compute_responses(SC, forward=ALT).g)


def test_contract_violation_rejected():
    class Bad:
        def __call__(self, scenario, q, solver=SolverOptions()):
            res = simulate(scenario, q, solver)
            return dataclasses.replace(res, concentration=res.concentration.assign_coords(
                time=res.concentration["time"].values + 5.0))

    obs = observe(simulate(SC, Q).concentration, LAY, ObservationSpec((600.0, 1200.0), 0.0, 0))
    with pytest.raises(ValueError, match="coordinate 'time'"):
        fit_sources(SC, obs, InferenceOptions(), ResponseCache(), forward=Bad())
    with pytest.raises(ValueError, match="coordinate 'time'"):
        compute_responses(SC, forward=Bad())
    with pytest.raises(ValueError, match="coordinate 'time'"):
        reconstruct(SC, fit_sources(SC, obs, InferenceOptions(), ResponseCache()), forward=Bad())
    with pytest.raises(ValueError, match="coordinate 'time'"):
        fit_sources(SC, obs, InferenceOptions(method="direct"), ResponseCache(), forward=Bad())


def test_build_case_with_alt():
    cands = Candidates(("c0", "c1", "c2"), np.array([3000.0, 5000.0, 7000.0]), np.array([3000.0, 6000.0, 7000.0]))
    opt = DesignOptions(noise_std=0.05)
    cache = ResponseCache()
    case = build_case("alt", "design", SC, Q, SC, cands, opt, 1, 2, cache, forward=ALT)
    ref = build_case("def", "design", SC, Q, SC, cands, opt, 1, 2, cache)
    assert cache.misses == 2  # separate response sets
    assert not np.allclose(case.truth_readings, ref.truth_readings)  # truth came from the alt scheme
    assert case.field_error(Q) == pytest.approx(0.0, abs=1e-10)  # alt truth vs alt responses are consistent
    assert ref.field_error(Q) == pytest.approx(0.0, abs=1e-10)

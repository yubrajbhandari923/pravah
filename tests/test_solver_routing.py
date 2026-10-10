"""Solver interchangeability: every workflow goes through the ONE injected forward solver.

* RecordingForward / DistinguishableForward (tests/helpers_solver.py) are INTERFACE TESTS, NOT PHYSICS BENCHMARKS:
  they delegate to the original FiPy function (optionally adding a constant 1 ug/m^3), so they only prove routing.
* The FiPy sentinel patches ``pravah.forward.simulate`` (the FiPy implementation entry, also used by FIPY) to raise:
  any hidden FiPy call in simulate/inverse/design/experiments/cli would fail the test.
* The only real alternate numerical scheme here is FipyForward(convection='upwind').
"""
import copy
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from pravah import cli, forward as forward_mod  # noqa: E402
from pravah import experiments as ex  # noqa: E402
from pravah.config import (Domain, GaussianSource, Scenario, SolverOptions, Times, Transport,  # noqa: E402
                           WindSegment, scenario_from_dict)
from pravah.design import Candidates, DesignOptions, build_case  # noqa: E402
from pravah.forward import FIPY, BackendInfo, FipyForward, forward_info, get_forward, simulate  # noqa: E402
from pravah.inverse import (InferenceOptions, ResponseCache, cache_key, compute_responses, fit_sources,  # noqa: E402
                            reconstruct)
from pravah.observations import ObservationSpec, SensorLayout, observe  # noqa: E402
from tests import helpers_solver as H  # noqa: E402

SPEC = "tests.helpers_solver:make_recording"
CFG = {
    "scenario": {"domain": {"lx": 10000.0, "ly": 10000.0, "nx": 20, "ny": 20},
                 "sources": [{"name": "S1", "x": 2500.0, "y": 2500.0, "sigma": 600.0},
                             {"name": "S2", "x": 4000.0, "y": 6000.0, "sigma": 600.0},
                             {"name": "S3", "x": 6500.0, "y": 3500.0, "sigma": 600.0}],
                 "transport": {"wind": [{"t_start": 0.0, "u": 1.0, "v": 0.3}]},
                 "times": {"duration": 3600.0, "dt": 120.0, "output_interval": 600.0}},
    "q": [100000.0, 150000.0, 80000.0],
    "inverse": {"sensor_interval": 600.0, "noise_std": 0.005, "seed": 1, "heldout_ids": ["S4"],
                "sensors": [[2500, 4500], [5000, 5000], [7500, 4500], [4000, 7500]]},
    "design": {"candidates_n": 2, "budgets": [1, 2], "n_design": 1, "n_test": 1, "noise_reps": 1, "n_random": 2,
               "weighted": False, "heldout_n": 2, "design_seed": 11, "test_seed": 22, "design_noise_seed": 1000,
               "test_noise_seed": 5000, "layout_seed": 7, "n_validation": 1, "validation_seed": 44,
               "validation_noise_seed": 9000, "alpha": 1.0, "alpha_grid": [0.0, 1.0], "alpha_selection_file": None},
    "mismatch": {"poor_layout_sites": [[2500.0, 7500.0], [7500.0, 7500.0]]},
}
Q_REF = 1e5

SRC = (GaussianSource("S1", 2500.0, 2500.0, 600.0), GaussianSource("S2", 4000.0, 6000.0, 600.0))
SC = Scenario(domain=Domain(10_000.0, 10_000.0, 12, 12), sources=SRC,
              transport=Transport(wind=(WindSegment(0.0, 1.0, 0.3),)),
              times=Times(duration=1800.0, dt=150.0, output_interval=600.0))
LAY = SensorLayout(("a", "b", "c", "d"), (3000.0, 5000.0, 4500.0, 7000.0), (3000.0, 6500.0, 2800.0, 7000.0))
Q = np.array([1.2e5, 0.8e5])
OBS_SPEC = ObservationSpec((600.0, 1200.0, 1800.0), 0.0, 0)
CANDS = Candidates(("c0", "c1", "c2"), np.array([3000.0, 5000.0, 7000.0]), np.array([3000.0, 6000.0, 7000.0]))


def _boom(*a, **k):
    raise AssertionError("hidden FiPy call: pravah.forward.simulate was reached")


@pytest.fixture
def no_fipy(monkeypatch):
    """FiPy sentinel: any call of the FiPy entry (directly or through FIPY) raises."""
    monkeypatch.setattr(forward_mod, "simulate", _boom)


@pytest.fixture(autouse=True)
def _clear_calls():
    H.CALLS.clear()


def _fingerprints(specs):
    return {H.fingerprint(s.scenario) for s in specs}


def _is_response_q(q):
    return all(v in (0.0, Q_REF) for v in q) and sum(q) <= Q_REF


# ------------------------------------------------------------------ shared module-level routed runs
@pytest.fixture(scope="module")
def routed(tmp_path_factory):
    """benchmark -> mismatch with separate recorders for the truth and inference roles, FiPy sentinel active."""
    out = tmp_path_factory.mktemp("routed")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(forward_mod, "simulate", _boom)
        bt, bi = H.RecordingForward("bench-truth"), H.RecordingForward("bench-inf")
        bench = ex.run_benchmark(copy.deepcopy(CFG), out / "bench", truth_forward=bt, inference_forward=bi)
        at, ai = H.RecordingForward("alpha-truth"), H.RecordingForward("alpha-inf")
        alpha = ex.run_alpha_selection(copy.deepcopy(CFG), out / "alpha", truth_forward=at, inference_forward=ai)
        mt, mi = H.RecordingForward("mm-truth"), H.RecordingForward("mm-inf")
        mm = ex.run_mismatch(copy.deepcopy(CFG), bench.layouts_json(2), out / "mm", truth_forward=mt,
                             inference_forward=mi)
    return {"bench": (bench, bt, bi), "alpha": (alpha, at, ai), "mm": (mm, mt, mi), "out": out}


def test_benchmark_routes_truth_and_inference(routed):
    _, bt, bi = routed["bench"]
    specs = ex._setup(copy.deepcopy(CFG)).design_specs + ex._setup(copy.deepcopy(CFG)).test_specs
    assert len(bt.calls) == len(specs) == 2  # one truth solve per case
    assert {c[1] for c in bt.calls} == _fingerprints(specs)
    assert all(min(c[2]) > 0 for c in bt.calls)  # truth amplitudes, not unit responses
    assert len(bi.calls) == 2 * 4  # n_src + 1 response solves per case scenario
    assert {c[1] for c in bi.calls} == _fingerprints(specs)
    assert all(_is_response_q(c[2]) for c in bi.calls)


def test_alpha_selection_routes_truth_and_inference(routed):
    _, at, ai = routed["alpha"]
    assert len(at.calls) == 1 and len(ai.calls) == 4
    assert all(min(c[2]) > 0 for c in at.calls) and all(_is_response_q(c[2]) for c in ai.calls)


def test_mismatch_routes_roles_separately(routed):
    _, mt, mi = routed["mm"]
    # matched + refined + wind-perturbed + extra-source truths, all through the TRUTH solver (one test case)
    assert len(mt.calls) == 4 and len({c[1] for c in mt.calls}) == 4
    assert sorted(len(c[2]) for c in mt.calls) == [3, 3, 3, 4]  # the extra source has a 4th amplitude
    # responses (matched inference scenario only, cached across the four conditions) through the INFERENCE solver
    assert len(mi.calls) == 4 and len({c[1] for c in mi.calls}) == 1
    assert all(_is_response_q(c[2]) for c in mi.calls)
    assert {c[1] for c in mi.calls} <= {c[1] for c in mt.calls}  # the matched truth shares the inference scenario
    assert mt is not mi and not set(map(tuple, mt.calls)) & set(map(tuple, mi.calls))


def test_provenance_and_summary_record_both_backends(routed):
    for sub, kind in (("bench", "benchmark"), ("mm", "mismatch"), ("alpha", "alpha_selection")):
        prov = json.loads((routed["out"] / sub / "provenance.json").read_text())
        b = prov["extra"]["backends"]
        assert prov["extra"]["experiment"] == kind
        assert b["truth"]["name"] == {"bench": "recording-bench-truth", "mm": "recording-mm-truth",
                                      "alpha": "recording-alpha-truth"}[sub]
        assert b["inference"]["capabilities"]["linear"] is True
    for sub in ("bench", "mm"):
        s = json.loads((routed["out"] / sub / "summary.json").read_text())
        assert set(s["backend"]) == {"truth", "inference"}


# ------------------------------------------------------------------ library: direct fit, build_case, reconstruct
def test_direct_fit_build_case_and_reconstruct_route(no_fipy):
    rec = H.RecordingForward("lib")
    truth = rec(SC, Q).concentration
    assert len(rec.calls) == 1
    obs = observe(truth, LAY, OBS_SPEC)
    cache = ResponseCache()
    fit = fit_sources(SC, obs, InferenceOptions(method="direct"), cache, forward=rec)
    n_direct = len(rec.calls)
    assert n_direct > 1 + 3  # truth + 3 response solves + optimiser iterates
    assert cache.misses == 1
    reconstruct(SC, fit, forward=rec)
    assert len(rec.calls) == n_direct + 1
    rec2 = H.RecordingForward("case")
    build_case("c", "design", SC, Q, SC, CANDS, DesignOptions(noise_std=0.05), 1, 2, ResponseCache(), forward=rec2)
    assert len(rec2.calls) == 3 + 1  # responses (n_src + 1) and the truth run
    assert sum(1 for c in rec2.calls if c[2] == tuple(Q)) == 1


def test_build_case_truth_field_skips_truth_run(no_fipy):
    rec = H.RecordingForward("case")
    truth = rec(SC, Q).concentration
    n0 = len(rec.calls)
    build_case("c", "design", SC, Q, SC, CANDS, DesignOptions(noise_std=0.05), 1, 2, ResponseCache(),
               truth_field=truth, forward=rec)
    assert len(rec.calls) - n0 == 3


def test_run_forward_and_compute_responses_have_no_implicit_fipy(no_fipy):
    rec = H.RecordingForward("x")
    r = compute_responses(SC, forward=rec)
    assert r.g.shape[0] == 2 and len(rec.calls) == 3
    with pytest.raises(AssertionError, match="hidden FiPy"):
        FIPY(SC, Q)  # the sentinel really guards the FiPy entry


# ------------------------------------------------------------------ distinguishable backend (interface test, not physics)
def test_distinguishable_forward_reaches_fit_reconstruct_and_cases():
    """INTERFACE TEST, NOT A PHYSICS BENCHMARK: the +1 ug/m^3 marker shows which solver produced which field."""
    dist = H.RecordingForward("dist", offset=1.0)
    base = simulate(SC, Q).concentration
    assert np.allclose(dist(SC, Q).concentration.values - base.values, 1.0, atol=1e-12)
    ref = build_case("r", "design", SC, Q, SC, CANDS, DesignOptions(noise_std=0.05), 1, 2, ResponseCache())
    case = build_case("d", "design", SC, Q, SC, CANDS, DesignOptions(noise_std=0.05), 1, 2, ResponseCache(),
                      forward=dist)
    assert np.allclose(case.truth_readings - ref.truth_readings, 1.0, atol=1e-9)  # truth run used the marker solver
    assert not np.allclose(case.b_site, ref.b_site)  # and so did the responses
    obs = observe(base, LAY, OBS_SPEC)
    fit = fit_sources(SC, obs, InferenceOptions(), ResponseCache(), forward=dist)
    rec_d = reconstruct(SC, fit, forward=dist).concentration
    rec_f = reconstruct(SC, fit).concentration
    assert np.allclose(rec_d.values - rec_f.values, 1.0, atol=1e-12)


def test_marker_on_truth_role_changes_benchmark(routed, no_fipy):
    t1 = H.RecordingForward("t1", offset=1.0)
    res = ex.run_benchmark(copy.deepcopy(CFG), truth_forward=t1, inference_forward=H.RecordingForward("i1"))
    base = routed["bench"][0]
    assert t1.calls and not np.allclose(res.raw["rmse"].values, base.raw["rmse"].values)


# ------------------------------------------------------------------ real alternate FiPy configuration
UP = FipyForward(convection="upwind")


def test_upwind_fipy_changes_results_key_and_cache():
    d, u = FIPY(SC, Q).concentration, UP(SC, Q).concentration
    assert not np.allclose(d.values, u.values)
    assert np.array_equal(u.values, simulate(SC, Q, SolverOptions(convection="upwind")).concentration.values)
    s = SolverOptions()
    assert cache_key(SC, s, Q_REF) != cache_key(SC, s, Q_REF, UP)
    assert cache_key(SC, s, Q_REF) == cache_key(SC, s, Q_REF, simulate) == cache_key(SC, s, Q_REF, FipyForward())
    assert cache_key(SC, s, Q_REF, UP) != cache_key(SC, s, Q_REF, FipyForward(tolerance=1e-8))
    assert UP.info.settings == {"convection": "upwind", "tolerance": None}
    assert UP(SC, Q).diagnostics["backend"]["settings"]["convection"] == "upwind"


def test_upwind_never_reuses_entries_memory_and_disk(tmp_path):
    cache = ResponseCache()
    r_d, r_u = cache.get(SC), cache.get(SC, forward=UP)
    assert cache.misses == 2 and r_d is not r_u and not np.allclose(r_d.g, r_u.g)
    assert cache.get(SC, forward=UP) is r_u and cache.hits == 1
    ResponseCache(tmp_path).get(SC)
    c2 = ResponseCache(tmp_path)
    r = c2.get(SC, forward=UP)
    assert c2.misses == 1 and c2.hits == 0 and np.array_equal(r.g, r_u.g)
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_upwind_backend_in_provenance(routed, tmp_path):
    res = ex.run_benchmark(copy.deepcopy(CFG), tmp_path / "up", inference_forward=UP)
    prov = json.loads((tmp_path / "up" / "provenance.json").read_text())
    b = prov["extra"]["backends"]
    assert b["inference"]["name"] == "fipy" and b["inference"]["settings"]["convection"] == "upwind"
    assert b["truth"]["settings"] == {"convection": None, "tolerance": None}
    assert b["inference"]["version"].startswith("fipy ") and "pravah" in b["inference"]["version"]
    assert not np.allclose(res.raw["rmse"].values, routed["bench"][0].raw["rmse"].values)


# ------------------------------------------------------------------ defaults, info, refusal, get_forward
def test_default_fipy_identity_and_backend_diagnostics():
    assert forward_info(simulate) == FIPY.info and forward_info(FIPY) == FIPY.info
    assert FIPY.info.capabilities == {"linear": True, "budget": True, "boundary_bases": True,
                                       "wind_field": True}
    a, b, c = simulate(SC, Q), FIPY(SC, Q), FipyForward()(SC, Q)
    assert np.array_equal(a.concentration.values, b.concentration.values)
    assert np.array_equal(a.concentration.values, c.concentration.values)
    assert a.diagnostics["backend"] == FIPY.info.to_dict() == b.diagnostics["backend"]
    assert isinstance(FIPY.info, BackendInfo) and FIPY.forward_id != UP.forward_id


def test_derived_info_does_not_assume_linearity():
    class Plain:
        def __call__(self, scenario, q, solver=SolverOptions()):
            return simulate(scenario, q, solver)

    info = forward_info(Plain())
    assert info.name.endswith("Plain") and info.version == "unknown" and info.capabilities["linear"] is False
    assert forward_info(H.RecordingForward("z", linear=False)).capabilities["linear"] is False


def test_nonlinear_or_undeclared_backend_refused_by_response_cache():
    class Plain:
        def __call__(self, scenario, q, solver=SolverOptions()):
            return simulate(scenario, q, solver)

    for fwd in (Plain(), H.RecordingForward("nl", linear=False)):
        with pytest.raises(ValueError, match="linear"):
            ResponseCache().get(SC, forward=fwd)
        with pytest.raises(ValueError, match="linear"):
            compute_responses(SC, forward=fwd)
        obs = observe(simulate(SC, Q).concentration, LAY, OBS_SPEC)
        with pytest.raises(ValueError, match="linear"):
            fit_sources(SC, obs, InferenceOptions(), ResponseCache(), forward=fwd)
    assert H.CALLS == []  # refused before any solve


def test_get_forward():
    assert get_forward(None) is FIPY and get_forward("fipy") is FIPY and get_forward("fipy", {}) is FIPY
    assert get_forward("fipy", {"convection": "upwind"}) == UP
    f = get_forward(SPEC, {"role": "k", "offset": 2.0})
    assert isinstance(f, H.RecordingForward) and f.role == "k" and f.offset == 2.0
    for bad, msg in (("nocolon", "module:callable"), (":x", "module:callable"), ("m:", "module:callable"),
                     ("no_such_module_xyz:f", "cannot import"), ("tests.helpers_solver:nope", "no attribute"),
                     ("tests.helpers_solver:CALLS", "callable"), (SPEC + "_missing", "no attribute")):
        with pytest.raises(ValueError, match=msg):
            get_forward(bad)
    with pytest.raises(ValueError, match="failed"):
        get_forward(SPEC, {"bogus": 1})
    with pytest.raises(ValueError, match="invalid kwargs"):
        get_forward("fipy", {"bogus": 1})
    with pytest.raises(ValueError, match="non-callable"):
        get_forward("json:loads", {"s": "1"})  # the factory returns the int 1


# ------------------------------------------------------------------ config 'backends' and override precedence
def test_config_backends_and_override_precedence(no_fipy):
    cfg = copy.deepcopy(CFG)
    cfg["backends"] = {"truth": {"spec": SPEC, "kwargs": {"role": "cfg-truth"}},
                       "inference": {"spec": SPEC, "kwargs": {"role": "cfg-inf"}}}
    ex.run_alpha_selection(cfg)
    assert [r for r, _ in H.CALLS] == ["cfg-truth"] + ["cfg-inf"] * 4
    H.CALLS.clear()
    over = H.RecordingForward("override-truth")
    ex.run_alpha_selection(cfg, truth_forward=over)
    assert [r for r, _ in H.CALLS] == ["override-truth"] + ["cfg-inf"] * 4
    with pytest.raises(ValueError, match="roles"):
        ex.resolve_backends({"backends": {"oops": {}}})
    st = ex._setup(copy.deepcopy(CFG))
    assert st.truth_forward is FIPY and st.inference_forward is FIPY and st.solver == SolverOptions()


# ------------------------------------------------------------------ CLI
def _write_cfg(tmp_path, **extra):
    cfg = {**copy.deepcopy(CFG), **extra}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def _roles():
    return [r for r, _ in H.CALLS]


def test_cli_simulate_routes_through_solver(tmp_path, no_fipy):
    out = tmp_path / "sim"
    assert cli.main(["simulate", "--config", _write_cfg(tmp_path), "--out", str(out), "--solver", SPEC,
                     "--solver-kwargs", '{"role": "cli-sim"}']) == 0
    assert _roles() == ["cli-sim"]
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["extra"]["backend"]["name"] == "recording-cli-sim"
    assert (out / "concentration.nc").is_file()


def test_cli_inverse_shared_and_split_roles(tmp_path, no_fipy):
    cfg = _write_cfg(tmp_path)
    cli.main(["inverse", "--config", cfg, "--out", str(tmp_path / "a"), "--solver", SPEC])
    assert _roles() == ["rec"] * 6  # 1 truth + 4 responses + 1 reconstruction, one shared object
    H.CALLS.clear()
    cli.main(["inverse", "--config", cfg, "--out", str(tmp_path / "b"), "--truth-solver", SPEC,
              "--truth-solver-kwargs", '{"role": "T"}', "--inference-solver", SPEC,
              "--inference-solver-kwargs", '{"role": "I"}'])
    assert _roles() == ["T"] + ["I"] * 5
    prov = json.loads((tmp_path / "b" / "provenance.json").read_text())
    assert prov["extra"]["backends"]["truth"]["name"] == "recording-T"
    assert prov["extra"]["backends"]["inference"]["name"] == "recording-I"
    H.CALLS.clear()  # --direct: the truth role is still called once only, the optimiser calls the inference role
    cli.main(["inverse", "--config", cfg, "--out", str(tmp_path / "c"), "--direct", "--solver", SPEC,
              "--solver-kwargs", '{"role": "S"}', "--truth-solver-kwargs", '{"role": "T"}'])
    assert _roles().count("T") == 1 and _roles().count("S") > 6


def test_cli_inverse_default_and_upwind_inference(tmp_path):
    cfg = _write_cfg(tmp_path)
    cli.main(["inverse", "--config", cfg, "--out", str(tmp_path / "d")])
    cli.main(["inverse", "--config", cfg, "--out", str(tmp_path / "u"), "--inference-solver", "fipy",
              "--inference-solver-kwargs", '{"convection": "upwind"}'])
    d = json.loads((tmp_path / "d" / "inverse_metrics.json").read_text())["linear"]
    u = json.loads((tmp_path / "u" / "inverse_metrics.json").read_text())["linear"]
    assert d["q_hat"] != u["q_hat"]  # a different inference scheme on FiPy-generated truth gives different fits
    prov = json.loads((tmp_path / "d" / "provenance.json").read_text())
    assert prov["extra"]["backends"]["truth"]["name"] == "fipy"


def test_cli_benchmark_and_mismatch_truth_inference_split(tmp_path, no_fipy):
    split = ["--truth-solver", SPEC, "--truth-solver-kwargs", '{"role": "T"}', "--inference-solver", SPEC,
             "--inference-solver-kwargs", '{"role": "I"}']
    bench = tmp_path / "bench"
    cli.main(["benchmark", "--config", _write_cfg(tmp_path), "--out", str(bench), *split])
    assert _roles().count("T") == 2 and _roles().count("I") == 8 and set(_roles()) == {"T", "I"}
    H.CALLS.clear()
    mm_cfg = tmp_path / "mm.json"
    mm_cfg.write_text(json.dumps({**copy.deepcopy(CFG), "experiment": "mismatch",
                                  "mismatch": {**CFG["mismatch"], "layouts_from": str(bench)}}))
    cli.main(["benchmark", "--config", str(mm_cfg), "--out", str(tmp_path / "mm"), *split])
    assert _roles().count("T") == 4 and _roles().count("I") == 4
    s = json.loads((tmp_path / "mm" / "summary.json").read_text())
    assert s["backend"]["truth"]["name"] == "recording-T"
    H.CALLS.clear()
    al_cfg = tmp_path / "al.json"
    al_cfg.write_text(json.dumps({**copy.deepcopy(CFG), "experiment": "alpha_selection"}))
    cli.main(["benchmark", "--config", str(al_cfg), "--out", str(tmp_path / "al"), "--solver", SPEC,
              "--solver-kwargs", '{"role": "S"}'])
    assert _roles() == ["S"] * 5


def test_cli_solver_errors(tmp_path):
    cfg = _write_cfg(tmp_path)
    with pytest.raises(ValueError, match="module:callable"):
        cli.main(["simulate", "--config", cfg, "--out", str(tmp_path / "e1"), "--solver", "oops"])
    with pytest.raises(ValueError, match="JSON"):
        cli.main(["simulate", "--config", cfg, "--out", str(tmp_path / "e2"), "--solver-kwargs", "{bad"])
    with pytest.raises(ValueError, match="JSON object"):
        cli.main(["simulate", "--config", cfg, "--out", str(tmp_path / "e3"), "--solver-kwargs", "[1]"])

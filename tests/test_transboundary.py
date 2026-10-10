"""Tests for pravah.transboundary (synthetic external vs local study).  Real FiPy solves on the tiny grids of
configs/transboundary_smoke.json (A: 30x30 at 500 m, B: 40x40, T = 7200 s); no solver mocks."""
from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pravah import transboundary as tb
from pravah.attribution import contributions
from pravah.cli import main
from pravah.config import GaussianSource, SolverOptions, StudyRegion, load_config
from pravah.inverse import InferenceOptions, ResponseCache, fit_sources
from pravah.observations import ObservationSpec, SensorLayout, observe
from pravah.sources import check_origin_tags

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
SMOKE = load_config(CONFIGS / "transboundary_smoke.json")
FULL = load_config(CONFIGS / "transboundary.json")


def recovery_only(cfg: dict[str, Any], **recovery: Any) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    for k in ("fields", "special", "domain_size"):
        c[k]["enabled"] = False
    c["recovery"].update(recovery)
    return c


def test_named_scenarios_are_valid_on_both_domains() -> None:
    for name in FULL["scenarios"]:
        for dom, n in (("A", 80), ("B", 120)):
            scn, theta, labels = tb.build_scenario(FULL, name, dom)
            check_origin_tags(scn)
            assert scn.domain.nx == scn.domain.ny == n
            assert theta.shape == (scn.n_params,) and len(labels) == scn.n_params
            for lab, p in zip(labels, scn.param_table()):
                assert (lab["name"], lab["kind"], lab["units"], lab["group"]) == (p.name, p.kind, p.units, p.group)
                assert lab["units"] == ("microgram/s" if lab["kind"] == "emission" else "microgram/m^3")
                assert lab["true_value"] >= 0
            assert scn.boundary.background == 0.0 and scn.initial.kind == "zero"
    # same physical problem on A and B: positions relative to the study region agree
    a, _, _ = tb.build_scenario(FULL, "external_sources", "A")
    b, _, _ = tb.build_scenario(FULL, "external_sources", "B")
    assert isinstance(a.study, StudyRegion) and isinstance(b.study, StudyRegion)
    for sa, sb in zip(a.sources, b.sources):
        assert isinstance(sa, GaussianSource) and isinstance(sb, GaussianSource)
        assert (sa.x - a.study.x0, sa.y - a.study.y0) == pytest.approx((sb.x - b.study.x0, sb.y - b.study.y0))
    groups = {s.name: s.group for s in a.sources}
    assert groups == {"L1": "local", "L2": "local", "E1": "external"}


def test_scenario_composition() -> None:
    def names(n: str) -> list[str]:
        return [p.name for p in tb.build_scenario(FULL, n)[0].param_table()]
    assert names("local_only") == ["L1", "L2"]
    assert names("external_sources") == ["L1", "L2", "E1"]
    assert names("boundary_only") == ["L1", "L2", "uniform"]
    assert names("mixed") == ["L1", "L2", "E1", "uniform"]
    assert names("sector") == ["L1", "L2", "west", "south", "east", "north"]
    # documented distinct amplitudes
    assert tb.build_scenario(FULL, "mixed")[1][-1] != tb.build_scenario(FULL, "boundary_only")[1][-1]


def test_no_double_count_validator() -> None:
    tb.check_no_double_count([("E1", -4000.0, 4000.0)], [("uniform", None)])
    with pytest.raises(ValueError, match="double count"):
        tb.check_no_double_count([("E1", -4000.0, 4000.0)], [("uniform", (-7000.0, 1000.0, -1000.0, 7000.0))])
    with pytest.raises(ValueError, match="outside the simulation domain"):
        tb.check_no_double_count([("E9", -9000.0, 0.0)], [], (-5000.0, -5000.0, 15000.0, 15000.0))
    # via the builder: E1 explicit AND a basis flagged as representing E1 is rejected; without E1 it is accepted
    spec = {"sources": ["L1", "L2", "E1"], "bases": ["uniform"], "represents": FULL["domain_size"]["equivalence"]["represents"]}
    with pytest.raises(ValueError, match="double count"):
        tb.build_model(FULL, spec, "A")
    spec["sources"] = ["L1", "L2"]
    scn, _, _ = tb.build_model(FULL, spec, "A")
    assert [s.name for s in scn.sources] == ["L1", "L2"]
    # the configured equivalence models never contain E1
    for m in FULL["domain_size"]["equivalence"]["models"].values():
        assert "E1" not in m["sources"]


def test_seed_guards() -> None:
    base, th, _ = tb.build_scenario(SMOKE, "mixed")
    val = tb.draw_cases(SMOKE, base, th, 3, 202, "val")
    test = tb.draw_cases(SMOKE, base, th, 3, 303, "test")
    tb.check_ensembles(val, test, 202, 303, 20200, 30300, 2)
    with pytest.raises(ValueError, match="differ"):
        tb.check_ensembles(val, test, 202, 202, 20200, 30300, 2)
    with pytest.raises(ValueError, match="noise seeds overlap"):
        tb.check_ensembles(val, test, 202, 303, 20200, 20203, 2)  # ranges [20200, 20206) and [20203, 20209)
    with pytest.raises(ValueError, match="share a"):
        tb.check_ensembles(val, tb.draw_cases(SMOKE, base, th, 3, 202, "other"), 202, 303, 20200, 30300, 2)
    # draws are deterministic and winds respect the documented ranges
    again = tb.draw_cases(SMOKE, base, th, 3, 202, "val")
    assert all(np.array_equal(a.theta, b.theta) for a, b in zip(val, again))
    for c in val:
        speed = math.hypot(c.scenario.transport.wind[0].u, c.scenario.transport.wind[0].v)
        assert 0.6 <= speed <= 1.6
    assert len(val[1].scenario.transport.wind) == 2 and len(val[0].scenario.transport.wind) == 1
    # config-level guard: overlapping special / equivalence ranges are refused
    bad = copy.deepcopy(SMOKE)
    bad["domain_size"]["equivalence"]["noise_seed"] = bad["special"]["noise_seed"] + 1
    with pytest.raises(ValueError, match="overlap"):
        tb.run_transboundary(bad, "unused-never-created")


def test_fit_sources_cannot_see_tags() -> None:
    names = set(inspect.signature(fit_sources).parameters)
    assert names == {"scenario", "obs", "options", "cache", "forward"}
    scn, theta, _ = tb.build_scenario(SMOKE, "mixed")
    cache = ResponseCache()
    from pravah.inverse import run_forward
    lay = tb._layouts(SMOKE, scn)["interior"]
    truth = run_forward(tb.FIPY, scn, theta, SolverOptions()).concentration
    obs = observe(truth, lay, ObservationSpec(tb._obs_times(SMOKE), 0.002, 5))
    q1 = fit_sources(scn, obs, InferenceOptions(alpha=1.0), cache).q_hat
    # flip every origin tag of the declared model: the estimate is unchanged (tags are evaluation-only labels)
    srcs = [s for s in scn.sources if isinstance(s, GaussianSource)]
    flipped = dataclasses.replace(scn, sources=tuple(
        dataclasses.replace(s, group="external" if s.group == "local" else "local") for s in srcs))
    q2 = fit_sources(flipped, obs, InferenceOptions(alpha=1.0), ResponseCache()).q_hat
    assert np.array_equal(q1, q2)


def test_fraction_none_when_total_is_zero() -> None:
    scn, _, _ = tb.build_scenario(SMOKE, "mixed")
    resp = ResponseCache().get(scn)
    c = contributions(resp, np.zeros(scn.n_params), tb._zero_parts(resp))
    _, f = tb.fraction_timeseries(c, scn)
    assert all(v is None for v in f)


def test_arrival_time_helper() -> None:
    t = np.arange(0.0, 6000.0, 600.0)
    tr = np.where(t >= 1800.0, 1.0, 0.0)
    assert tb.arrival_time(tr, t, 0.05, 1e-6) == pytest.approx(1800.0 - 600.0 + 0.05 * 600.0)
    assert tb.arrival_time(np.zeros_like(t), t, 0.05, 1e-6) is None


def test_no_arrival_is_verified_by_the_model_and_not_recovered() -> None:
    scn, theta, _ = tb.build_scenario(SMOKE, "no_arrival")
    cache = ResponseCache()
    resp = cache.get(scn)
    c = contributions(resp, theta, tb._zero_parts(resp))
    ratio = tb.verify_no_arrival(c, scn, 1e-3)
    assert ratio < 1e-3
    # the same check refuses a scenario in which E1 does reach the study region
    ok, th2, _ = tb.build_scenario(SMOKE, "external_sources")
    r2 = cache.get(ok)
    with pytest.raises(ValueError, match="DOES reach"):
        tb.verify_no_arrival(contributions(r2, th2, tb._zero_parts(r2)), ok, 1e-3)
    run = tb._Run(SMOKE, Path("."), cache, SolverOptions(), None)
    lay = tb._layouts(SMOKE, scn)["interior"]
    rows, _ = tb._case_rows(run, "na", scn, theta, scn, lay, 0.02, {"selected": 1.0, "none": 0.0}, 40400, 2)
    df = pd.DataFrame(rows)
    assert df["flags"].str.contains("unidentifiable:E1").all()
    assert df["flags"].str.contains("rank_deficient").all()
    sel = df[df["alpha_role"] == "selected"]
    assert (sel["theta_hat_E1"] == 0.0).all()  # equals the prior (0): not recovered
    assert (sel["emission_rel_err_external"] == 1.0).all()  # the true E1 amplitude is missed completely


def test_confounded_total_small_decomposition_large() -> None:
    cfg = copy.deepcopy(SMOKE)
    cfg["physics"].update({"T": 14400.0, "dt": 60.0})
    cfg["scenarios"]["confounded"]["wind_speed"] = 8.0  # fast wind: the two plumes' responses become nearly collinear
    scn, theta, _ = tb.build_scenario(cfg, "confounded")
    run = tb._Run(cfg, Path("."), ResponseCache(), SolverOptions(), None)
    cl = cfg["special"]["confounded_layout"]
    lay = SensorLayout(tuple(f"C{i}" for i in range(len(cl))), tuple(p[0] + 2500.0 for p in cl), tuple(p[1] + 2500.0 for p in cl))
    rows, _ = tb._case_rows(run, "conf", scn, theta, scn, lay, 0.3, {"none": 0.0}, 40400, 4)
    df = pd.DataFrame(rows)
    assert df["flags"].str.contains("confounded:L1,E1c").all()
    # the total field error is a few % of the peak; the external-fraction and local-amplitude errors are much larger
    tot = df["rmse_total_rel_to_peak"].mean()
    assert tot < 0.1
    assert df["f_ext_error"].abs().mean() > 3 * tot
    assert df["emission_rel_err_local"].mean() > 3 * tot


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Recovery-only runs: A (via the CLI), A' (identical config, determinism), B (other test seed)."""
    root = tmp_path_factory.mktemp("tb")
    cfg = recovery_only(SMOKE)
    cfg.pop("extends", None)
    p = root / "cfg.json"
    p.write_text(json.dumps(cfg))
    assert main(["benchmark", "--config", str(p), "--out", str(root / "A")]) == 0
    tb.run_transboundary(cfg, root / "A2")
    other = recovery_only(SMOKE, test_seed=304)
    tb.run_transboundary(other, root / "B")
    return {"A": root / "A", "A2": root / "A2", "B": root / "B"}


def test_recovery_outputs_and_noise_scale(runs: dict[str, Path]) -> None:
    a = runs["A"]
    for f in ("config.json", "provenance.json", "timings.json", "recovery/alpha_selected.json", "recovery/test_raw.csv",
              "recovery/validation_raw.csv", "recovery/summary.csv", "recovery/paired.csv", "recovery/cases.csv",
              "recovery/noise_levels_test.csv", "figures/recovery_alpha_selection.png"):
        assert (a / f).is_file(), f
    for f in ("config.json", "provenance.json", "timings.json", "recovery/alpha_selected.json"):
        json.loads((a / f).read_text(), parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    prov = json.loads((a / "provenance.json").read_text())
    assert prov["seeds"]["validation_seed"] == 202 and prov["seeds"]["test_seed"] == 303
    # noise is a documented fraction of the signal scale, reported in ug/m3 and in %
    nl = pd.read_csv(a / "recovery/noise_levels_test.csv")
    assert np.allclose(nl["sigma_ug_m3"] / nl["signal_scale_ug_m3"], nl["noise_fraction"])
    assert np.allclose(nl["sigma_pct_of_scale"], 100 * nl["noise_fraction"])
    raw = pd.read_csv(a / "recovery/test_raw.csv")
    assert np.allclose(raw["sigma_ug_m3"] / raw["signal_scale_ug_m3"], raw["noise_fraction"])
    assert {"interior", "interior+upwind", "interior8"} == set(raw["layout"])
    assert set(raw.groupby("layout")["n_sensors"].first()) == {6, 8}
    for col in ("rmse_total", "rmse_local", "rmse_external", "f_ext_true", "f_ext_est", "f_ext_error",
                "emission_rel_err_local", "emission_rel_err_external", "inflow_abs_err_ug_m3_uniform", "rank",
                "effective_rank", "condition", "flags", "fit_runtime_s"):
        assert col in raw.columns
    paired = pd.read_csv(a / "recovery/paired.csv")
    assert set(paired["equal_budget"]) == {True, False}
    assert (paired["rmse_total_diff_sd"].notna() | (paired["n_pairs"] < 2)).all()
    # selected alpha is a grid point; raw rows exist for every layout x case x rep x noise x role
    sel = json.loads((a / "recovery/alpha_selected.json").read_text())
    grid = set(sel["alpha_grid"])
    assert all(v in grid for v in sel["alpha_by_noise_fraction"].values())
    n_case, n_rep, n_noise = 2, 1, 2
    assert len(raw) == 3 * n_case * n_rep * n_noise * 3


def test_alpha_selection_uses_validation_only(runs: dict[str, Path]) -> None:
    a = (runs["A"] / "recovery/alpha_selected.json").read_text()
    b = (runs["B"] / "recovery/alpha_selected.json").read_text()
    sa, sb = json.loads(a), json.loads(b)
    assert sa["seeds"]["test_seed"] == 303 and sb["seeds"]["test_seed"] == 304
    for d in (sa, sb):
        d["seeds"].pop("test_seed")
    assert sa == sb  # a different test seed does not change anything that was frozen
    ra = pd.read_csv(runs["A"] / "recovery/test_raw.csv")
    rb = pd.read_csv(runs["B"] / "recovery/test_raw.csv")
    assert not np.allclose(ra["rmse_total"], rb["rmse_total"])  # ... but the test results do differ
    assert set(ra.loc[ra["alpha_role"] == "selected", "alpha"]) <= set(sa["alpha_grid"])


def test_determinism(runs: dict[str, Path]) -> None:
    a = pd.read_csv(runs["A"] / "recovery/test_raw.csv").drop(columns=["fit_runtime_s"])
    b = pd.read_csv(runs["A2"] / "recovery/test_raw.csv").drop(columns=["fit_runtime_s"])
    pd.testing.assert_frame_equal(a, b)
    va = pd.read_csv(runs["A"] / "recovery/validation_raw.csv")
    vb = pd.read_csv(runs["A2"] / "recovery/validation_raw.csv")
    pd.testing.assert_frame_equal(va, vb)


def test_refuses_non_empty_out_dir(runs: dict[str, Path]) -> None:
    with pytest.raises(FileExistsError):
        tb.run_transboundary(recovery_only(SMOKE), runs["A"])


def test_full_smoke_run_all_studies(tmp_path: Path) -> None:
    out = tmp_path / "smoke"
    res = tb.run_transboundary(SMOKE, out)
    assert res.studies == ["fields", "recovery", "special", "domain_size"]
    for f in ("fields/scenario_summary.csv", "fields/fraction_timeseries.csv", "fields/arrival_times.csv",
              "fields/wind_direction_sweep.csv", "scenarios/param_table.csv", "special/confounded.csv",
              "special/no_arrival.csv", "special/wind_mismatch.csv", "special/prior_sensitivity.csv",
              "special/omitted_source.csv", "special/diagnostics.json", "domain_size/field_difference.csv",
              "domain_size/boundary_basis_equivalence.csv", "figures/fields_snapshots.png",
              "figures/domain_size_fields.png", "figures/boundary_basis_equivalence.png", "timings.json"):
        assert (out / f).is_file(), f
    strict = json.loads((out / "special/diagnostics.json").read_text())
    assert strict["no_arrival"]["external_to_local_study_mean_ratio"] < 1e-3
    arr = pd.read_csv(out / "fields/arrival_times.csv")
    assert arr[arr["scenario"] == "local_only"]["arrival_time_s"].isna().all()
    assert arr[(arr["scenario"] == "external_sources") & (arr["point"] == "P0")]["arrival_time_s"].notna().all()
    assert arr[arr["scenario"] == "no_arrival"]["arrival_time_s"].isna().all()
    ts = pd.read_csv(out / "fields/fraction_timeseries.csv")
    assert ts[ts["scenario"] == "local_only"]["concentration_external_fraction"].dropna().eq(0.0).all()
    assert ts[ts["scenario"] == "external_sources"]["concentration_external_fraction"].isna().any()  # t = 0 undefined
    # omitted external source: local-only model reports external fraction 0 while the truth is not 0
    om = pd.read_csv(out / "special/omitted_source.csv")
    first = om[om["case"].str.contains("by_local_only") & (om["alpha_role"] == "none")]
    assert (first["f_ext_est"] == 0.0).all() and (first["f_ext_true"] > 0.1).all()
    eq = pd.read_csv(out / "domain_size/boundary_basis_equivalence.csv")
    assert {"A_E1_explicit_true_theta", "A_local_only", "A_uniform_basis", "A_sector_bases"} == set(eq["model"])
    dsz = pd.read_csv(out / "domain_size/field_difference.csv")
    assert (dsz["rmse_total_rel_to_peak"] < 0.1).all()

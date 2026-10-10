"""End-to-end tests of experiments.py on a tiny problem (20x20 grid, T=3600 s, 3x3 candidates, real solves)."""
import copy
import dataclasses
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from pravah import experiments as ex
from pravah.config import Domain, Scenario, Times
from pravah.inverse import ResponseCache

CFG = {
    "scenario": {"domain": {"lx": 10000.0, "ly": 10000.0, "nx": 20, "ny": 20},
                 "sources": [{"name": "S1", "x": 2500.0, "y": 2500.0, "sigma": 600.0},
                             {"name": "S2", "x": 4000.0, "y": 6000.0, "sigma": 600.0},
                             {"name": "S3", "x": 6500.0, "y": 3500.0, "sigma": 600.0}],
                 "transport": {"wind": [{"t_start": 0.0, "u": 1.0, "v": 0.3}]},
                 "times": {"duration": 3600.0, "dt": 120.0, "output_interval": 600.0}},
    "q": [100000.0, 150000.0, 80000.0],
    "inverse": {"sensor_interval": 600.0, "noise_std": 0.005},
    "design": {"candidates_n": 3, "budgets": [1, 2], "n_design": 2, "n_test": 2, "noise_reps": 1, "n_random": 3,
               "weighted": True, "heldout_n": 3, "design_seed": 11, "test_seed": 22, "design_noise_seed": 1000,
               "test_noise_seed": 5000, "layout_seed": 7,
               "n_validation": 2, "validation_seed": 44, "validation_noise_seed": 9000, "alpha": 1.0,
               "alpha_grid": [0.0, 1.0, 100.0], "alpha_selection_file": None},
    "mismatch": {"refine": 2, "truth_dt_factor": 0.5, "wind_speed_error": 0.2, "wind_rotation_deg": 15.0,
                 "extra_source": {"x": 3000.0, "y": 4500.0, "sigma": 500.0, "q": 60000.0},
                 "poor_layout_sites": [[1667.0, 8333.0], [5000.0, 8333.0]]},
}


@pytest.fixture(scope="module")
def cache():
    return ResponseCache()


@pytest.fixture(scope="module")
def bench(tmp_path_factory, cache):
    out = tmp_path_factory.mktemp("bench")
    return ex.run_benchmark(copy.deepcopy(CFG), out / "run", cache=cache), out / "run"


def base_scenario() -> Scenario:
    from pravah.config import scenario_from_dict
    return scenario_from_dict(CFG["scenario"])


# ---------------------------------------------------------------- ensembles
def test_ensemble_ranges_and_determinism():
    base = base_scenario()
    a = ex.make_ensemble(base, 8, 3, CFG["q"])
    assert a == ex.make_ensemble(base, 8, 3, CFG["q"]) and a != ex.make_ensemble(base, 8, 4, CFG["q"])
    theta0 = math.atan2(0.3, 1.0)
    for i, c in enumerate(a):
        assert all(0.5 * q0 <= q <= 1.5 * q0 for q, q0 in zip(c.q, CFG["q"]))
        segs = c.scenario.transport.wind
        assert len(segs) == (2 if i % 2 else 1)
        for w in segs:
            assert 0.6 <= math.hypot(w.u, w.v) <= 1.6 + 1e-12
            dev = math.degrees(math.atan2(w.v, w.u) - theta0)
            assert abs(dev) <= 40 + 1e-9
        if len(segs) == 2:
            t = segs[1].t_start
            assert 1200.0 <= t <= 2400.0 and t % 600.0 == 0


def test_check_ensembles_guards():
    base = base_scenario()
    d, t = ex.make_ensemble(base, 2, 1, CFG["q"], "d"), ex.make_ensemble(base, 2, 2, CFG["q"], "t")
    ex.check_ensembles(d, t, 1, 2, 10, 20)
    with pytest.raises(ValueError, match="seed"):
        ex.check_ensembles(d, t, 1, 1, 10, 20)
    with pytest.raises(ValueError, match="noise"):
        ex.check_ensembles(d, t, 1, 2, 10, 11)
    with pytest.raises(ValueError, match="share"):
        ex.check_ensembles(d, ex.make_ensemble(base, 2, 1, CFG["q"], "t"), 1, 2, 10, 20)  # same draws
    cfg = copy.deepcopy(CFG)
    cfg["design"]["test_seed"] = cfg["design"]["design_seed"]
    with pytest.raises(ValueError):
        ex.run_benchmark(cfg)


def test_mismatch_helpers():
    base = base_scenario()
    fine = ex.refine_scenario(base, 2, 0.5)
    assert (fine.domain.nx, fine.domain.ny, fine.times.dt, fine.times.output_interval) == (40, 40, 60.0, 600.0)
    w = ex.perturb_wind(base, 0.2, 90.0).transport.wind[0]
    assert (w.u, w.v) == pytest.approx((-0.36, 1.2))
    from pravah.forward import simulate
    f = simulate(ex.refine_scenario(base, 2), np.array(CFG["q"])).concentration
    avg = ex.block_average(f, base)
    assert avg.shape == (7, 20, 20) and float(avg.mean()) == pytest.approx(float(f.mean()), rel=1e-12)
    assert float(avg.isel(time=-1, y=0, x=0)) == pytest.approx(float(f.isel(time=-1, y=slice(0, 2),
                                                                         x=slice(0, 2)).mean()))


# ---------------------------------------------------------------- benchmark
def test_benchmark_outputs(bench):
    res, out = bench
    for name in ("config.json", "provenance.json", "raw_metrics.csv", "summary.csv", "summary.json",
                 "paired_comparisons.csv", "design_scores.csv", "layouts.json", "timings.json"):
        assert (out / name).is_file(), name
    for name in ("config.json", "provenance.json", "summary.json", "layouts.json", "timings.json"):
        def bad(c):
            raise AssertionError(f"non-finite constant {c} in {name}")
        json.loads((out / name).read_text(), parse_constant=bad)
    assert {"error_vs_sensors_uniform.png", "error_vs_sensors_population.png", "selected_sites.png",
            "heldout_vs_regional.png"} <= set(res.figures)
    assert all((out / "figures" / f).stat().st_size > 1000 for f in res.figures)
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["seeds"]["test_seed"] == 22 and res.peak_memory_mb > 0 and res.runtime_s > 0
    assert {"cases_s", "selection_s", "evaluation_s"} <= set(res.timings)


def test_benchmark_tables(bench):
    res, _ = bench
    assert set(res.raw["method"]) == {"random", "spread", "greedy", "greedy_weighted"}
    assert set(res.raw["weighting"]) == {"uniform", "population"}
    assert {"condition", "cond_number"} & set(res.raw.columns) == {"cond_number"}
    assert len(res.raw) == 2 * (3 * 2 + 3 * 2) * 2 * 1  # weightings x layouts x cases x reps
    assert list(res.paired.columns) == ["comparison", "weighting", "budget", "mean_diff", "sd_diff", "n",
                                        "frac_a_better"]
    assert {"greedy - spread", "greedy - random_mean", "spread - random_mean"} <= set(res.paired["comparison"])
    assert {"mean_rmse", "sd_rmse", "sd_label", "design_rmse", "gap_rmse", "median_rmse"} <= set(res.summary.columns)
    assert len(res.design_scores) == 2 * (3 * 2 + 3 * 2)
    # random layouts never come from selection: evaluated rows only on test cases
    assert set(res.raw["case"]) == {"t0", "t1"}
    # greedy layouts are nested prefixes of one trace and recorded per budget
    g = res.greedy_orders["greedy"]
    assert res.layouts["greedy"][1][0] == (g[0],) and res.layouts["greedy"][2][0] == tuple(sorted(g[:2]))


def test_weighted_objective_changes_scores(bench):
    ds = bench[0].design_scores
    u = ds[ds["weighting"] == "uniform"].set_index("layout_id")["design_rmse"]
    p = ds[ds["weighting"] == "population"].set_index("layout_id")["design_rmse"]
    assert not np.allclose(u.loc[p.index].to_numpy(), p.to_numpy())


def test_seeds_and_noise_disjoint():
    st = ex._setup(CFG)
    d, t = st.d, st.test_specs
    assert d["design_seed"] != d["test_seed"]
    assert {ex._draw_key(s) for s in st.design_specs}.isdisjoint({ex._draw_key(s) for s in t})
    cache = ResponseCache()
    dcs = ex._build_cases(st, st.design_specs, "design", d["design_noise_seed"], cache, ["uniform"])["uniform"]
    tcs = ex._build_cases(st, t, "test", d["test_noise_seed"], cache, ["uniform"])["uniform"]
    assert {c.noise_seed for c in dcs}.isdisjoint({c.noise_seed for c in tcs})
    assert not np.allclose(dcs[0].noise, tcs[0].noise)
    from pravah.design import assert_disjoint
    assert_disjoint(dcs, tcs)


def test_greedy_independent_of_test_truth(bench, cache):
    cfg = copy.deepcopy(CFG)
    cfg["design"]["test_seed"], cfg["design"]["test_noise_seed"] = 99, 7000  # different test truths and noise
    other = ex.run_benchmark(cfg, cache=cache)
    for m in ("greedy", "greedy_weighted"):
        assert other.layouts[m] == bench[0].layouts[m]
    assert not other.raw["rmse"].equals(bench[0].raw["rmse"])  # but the test results do differ


def test_benchmark_deterministic(bench, cache):
    again = ex.run_benchmark(copy.deepcopy(CFG), cache=ResponseCache())
    cols = [c for c in bench[0].raw.columns if c != "runtime_s"]
    pd.testing.assert_frame_equal(again.raw[cols], bench[0].raw[cols])
    assert again.layouts == bench[0].layouts


def test_refuse_overwrite(bench, cache):
    with pytest.raises(FileExistsError):
        ex.run_benchmark(copy.deepcopy(CFG), bench[1], cache=cache)


# ---------------------------------------------------------------- mismatch
@pytest.fixture(scope="module")
def mism(bench, tmp_path_factory, cache):
    out = tmp_path_factory.mktemp("mm") / "run"
    return ex.run_mismatch(copy.deepcopy(CFG), bench[1], out, cache=cache), out


def test_layouts_round_trip(bench):
    n, lays = ex.load_layouts(bench[1])
    assert n == 3 and lays == bench[0].layouts
    assert ex.load_layouts(json.loads((bench[1] / "layouts.json").read_text()))[1] == lays


def test_mismatch_rows_and_pairing(bench, mism):
    res, out = mism
    assert set(res.raw["condition"]) == {"matched", "refined_truth", "wind_perturbed", "extra_source"}
    assert "cond_number" in res.raw.columns
    # matched baseline recomputed on the same cases / noise as the benchmark's uniform test rows
    key = ["method", "layout_id", "case", "rep", "budget"]
    a = res.raw[(res.raw["condition"] == "matched") & (res.raw["method"] != "poor")].sort_values(key)
    b = bench[0].raw[bench[0].raw["weighting"] == "uniform"].sort_values(key)
    assert np.allclose(a["rmse"].to_numpy(), b["rmse"].to_numpy(), rtol=1e-9)
    # extra-source rows know the unmodelled share; others do not
    x = res.raw[res.raw["condition"] == "extra_source"]
    assert (x["unmodelled_source_mass"] > 0).all()
    assert (res.raw[res.raw["condition"] != "extra_source"]["unmodelled_source_mass"] == 0).all()
    assert set(res.paired["condition"]) == {"refined_truth", "wind_perturbed", "extra_source"}
    assert {"mean_diff_rmse", "sd_diff_rmse", "n", "frac_worse"} <= set(res.paired.columns)
    assert {"diff_rmse", "diff_heldout_rmse", "diff_source_rel_err"} <= set(res.paired_rows.columns)
    # each mismatch genuinely changes the truth: the paired differences are not all zero
    for kind in ex.MISMATCH_KINDS:
        assert res.paired_rows[res.paired_rows["condition"] == kind]["diff_rmse"].abs().max() > 0


def test_mismatch_poor_layout_and_outputs(mism):
    res, out = mism
    d = pd.read_csv(out / "poor_layout_diagnostics.csv")
    assert {"poor", "greedy", "spread"} == set(d["method"])
    assert {"rank", "effective_rank", "cond_number", "sv_min", "q_std_S1", "q_std_rel_S3"} <= set(d.columns)
    assert set(res.poor_paired["comparison"]) == {"poor - greedy", "poor - spread"}
    for name in ("raw_metrics.csv", "summary.csv", "summary.json", "paired_mismatch.csv", "paired_mismatch_rows.csv",
                 "poor_layout_paired.csv", "layouts.json", "provenance.json", "config.json", "timings.json"):
        assert (out / name).is_file(), name
    json.loads((out / "summary.json").read_text(), parse_constant=lambda c: pytest.fail(c))
    assert {"rmse_by_condition.png", "mismatch_minus_matched.png", "poor_layout.png"} <= set(res.figures)


def test_mismatch_checks_candidate_grid(bench):
    cfg = copy.deepcopy(CFG)
    cfg["design"]["candidates_n"] = 4
    with pytest.raises(ValueError, match="candidate grid"):
        ex.run_mismatch(cfg, bench[1])


# ---------------------------------------------------------------- validation ensemble and alpha selection
@pytest.fixture(scope="module")
def sel(tmp_path_factory, cache):
    out = tmp_path_factory.mktemp("sel") / "run"
    return ex.run_alpha_selection(copy.deepcopy(CFG), out, cache=cache), out


def test_validation_ensemble_disjoint_and_guarded():
    from pravah.design import assert_disjoint
    st = ex._setup(CFG)
    keys = [{ex._draw_key(s) for s in specs} for specs in (st.design_specs, st.test_specs, st.validation_specs)]
    assert all(a.isdisjoint(b) for i, a in enumerate(keys) for b in keys[i + 1:])
    assert [s.name for s in st.validation_specs] == ["v0", "v1"]
    d = st.d
    cache = ResponseCache()
    cs = {r: ex._build_cases(st, specs, r, d[f"{r}_noise_seed"], cache, ["uniform"])["uniform"]
          for r, specs in (("design", st.design_specs), ("test", st.test_specs),
                           ("validation", st.validation_specs))}
    assert_disjoint(cs["design"], cs["test"], cs["validation"])
    assert not np.allclose(cs["validation"][0].noise, cs["test"][0].noise)
    with pytest.raises(ValueError, match="test and validation cases share"):
        assert_disjoint(cs["design"], cs["test"], [dataclasses.replace(cs["test"][0], role="validation")])
    base = base_scenario()
    dsp, tsp = ex.make_ensemble(base, 2, 1, CFG["q"], "d"), ex.make_ensemble(base, 2, 2, CFG["q"], "t")
    vsp = ex.make_ensemble(base, 2, 3, CFG["q"], "v")
    ex.check_ensembles(dsp, tsp, 1, 2, 10, 20, vsp, 3, 30)
    with pytest.raises(ValueError, match="validation_seed"):
        ex.check_ensembles(dsp, tsp, 1, 2, 10, 20, vsp, 2, 30)
    with pytest.raises(ValueError, match="noise seeds overlap"):
        ex.check_ensembles(dsp, tsp, 1, 2, 10, 20, vsp, 3, 21)
    with pytest.raises(ValueError, match="share a"):
        ex.check_ensembles(dsp, tsp, 1, 2, 10, 20, ex.make_ensemble(base, 2, 1, CFG["q"], "v"), 3, 30)
    for key, val in (("validation_seed", 22), ("validation_noise_seed", 5001), ("validation_seed", 11)):
        cfg = copy.deepcopy(CFG)
        cfg["design"][key] = val
        with pytest.raises(ValueError):
            ex.run_alpha_selection(cfg)
        with pytest.raises(ValueError):
            ex.run_benchmark(cfg)


def test_role_guards(cache):
    from pravah import design
    st = ex._setup(CFG)
    d = st.d
    vcs = ex._build_cases(st, st.validation_specs, "validation", d["validation_noise_seed"], cache, ["uniform"])["uniform"]
    tcs = ex._build_cases(st, st.test_specs, "test", d["test_noise_seed"], cache, ["uniform"])["uniform"]
    opts = ex._opts(st, False)
    k = ex._constraints(st)
    for cases in (vcs, tcs):
        with pytest.raises(ValueError, match="leakage"):
            design.greedy_trace(st.cands, 1, cases, k, opts)
        with pytest.raises(ValueError, match="leakage"):
            design.select_layout(st.cands, 1, cases, k, opts)
        with pytest.raises(ValueError, match="leakage"):
            design.score_layout((0,), cases, opts)
    with pytest.raises(ValueError, match="leakage"):
        design.evaluate_layouts({"m": [(0,)]}, vcs, opts)
    with pytest.raises(ValueError, match="validation"):
        design.evaluate_layouts({"m": [(0,)]}, vcs, opts, allow_design=True)
    with pytest.raises(ValueError, match="leakage"):
        design.evaluate_validation({"m": [(0,)]}, tcs, opts)
    assert len(design.evaluate_validation({"m": [(0,)]}, vcs, opts)) == 2
    with pytest.raises(ValueError, match="role"):
        design.build_case("x", "other", st.base, np.array(CFG["q"]), st.base, st.cands, opts, 1, 1, cache)


def test_alpha_selection_outputs(sel):
    res, out = sel
    for name in ("alpha_selected.json", "validation_raw.csv", "provenance.json", "timings.json", "config.json",
                 "layouts.json", "figures/criterion_vs_alpha.png"):
        assert (out / name).is_file(), name
    assert (out / "figures" / "criterion_vs_alpha.png").stat().st_size > 1000
    a = json.loads((out / "alpha_selected.json").read_text(), parse_constant=lambda c: pytest.fail(c))
    for key in ("alpha", "table", "alpha_grid", "validation_seed", "validation_noise_seed", "n_validation",
                "layout_seed", "q_scale", "prior", "prior_mean_uses_truth", "config_hash", "package_versions",
                "created_utc", "criterion", "tie_rule"):
        assert key in a, key
    assert a["alpha_grid"] == [0.0, 1.0, 100.0] and a["prior_mean_uses_truth"] is False and a["q_scale"] == 1e5
    assert (a["validation_seed"], a["validation_noise_seed"], a["n_validation"], a["layout_seed"]) == (44, 9000, 2, 7)
    t = pd.DataFrame(a["table"])
    assert {"criterion", "median_rmse", "p90_rmse", "n_rmse_gt_10x_no_emission", "mean_source_rel_err"} <= set(t.columns)
    assert list(t["alpha"]) == [0.0, 1.0, 100.0]
    assert a["alpha"] == t.loc[t["criterion"].idxmin(), "alpha"] == res.alpha
    # criterion is the plain mean of the raw per-row regional RMSE (all budgets, layouts, cases, reps)
    raw = pd.read_csv(out / "validation_raw.csv")
    assert set(raw["case"]) == {"v0", "v1"} and set(raw["method"]) == {"random", "spread"}
    assert len(raw) == 3 * (2 * (3 + 1)) * 2 * 1
    assert t["criterion"].to_numpy() == pytest.approx(raw.groupby("alpha")["rmse"].mean().to_numpy())
    assert (raw["rmse_no_emission"] > 0).all()
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["seeds"]["validation_seed"] == 44 and prov["extra"]["experiment"] == "alpha_selection"


def test_pick_alpha_tie_goes_to_smaller_alpha():
    t = pd.DataFrame({"alpha": [0.0, 1.0, 10.0, 100.0], "criterion": [3.0, 1.0, 1.0, float("nan")]})
    assert ex._pick_alpha(t) == 1.0
    assert ex._pick_alpha(t.iloc[::-1]) == 1.0


def test_alpha_selection_independent_of_test_ensemble(sel, cache):
    cfg = copy.deepcopy(CFG)
    cfg["design"].update({"test_seed": 99, "test_noise_seed": 8000, "n_test": 3})
    other = ex.run_alpha_selection(cfg, cache=cache)
    cols = [c for c in sel[0].raw.columns if c != "runtime_s"]
    pd.testing.assert_frame_equal(other.raw[cols], sel[0].raw[cols])
    assert other.alpha == sel[0].alpha
    pd.testing.assert_frame_equal(other.table, sel[0].table)


def test_alpha_selection_ignores_cfg_alpha_and_file(sel, cache):
    cfg = copy.deepcopy(CFG)
    cfg["design"].update({"alpha": None, "alpha_selection_file": "does/not/exist.json"})
    assert ex.run_alpha_selection(cfg, cache=cache).alpha == sel[0].alpha


def test_alpha_selection_grid_validation():
    for grid in ([], [-1.0], [float("inf")]):
        cfg = copy.deepcopy(CFG)
        cfg["design"]["alpha_grid"] = grid
        with pytest.raises(ValueError, match="alpha_grid"):
            ex.run_alpha_selection(cfg)


def _with_file(path, alpha=None):
    cfg = copy.deepcopy(CFG)
    cfg["design"].update({"alpha": alpha, "alpha_selection_file": str(path)})
    return cfg


def test_benchmark_reads_alpha_from_file(sel, cache, tmp_path):
    res, out = sel
    f = out / "alpha_selected.json"
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    run = ex.run_benchmark(_with_file(f), tmp_path / "b", cache=cache)
    for name in ("provenance.json", "summary.json"):
        info = json.loads((tmp_path / "b" / name).read_text())
        info = info["extra"]["alpha"] if name == "provenance.json" else info["alpha"]
        assert info["value"] == res.alpha and info["alpha_selection_sha256"] == sha
        assert info["source"] == "alpha_selection_file"
    # same alpha given explicitly gives the same numbers as the file
    explicit = copy.deepcopy(CFG)
    explicit["design"]["alpha"] = res.alpha
    again = ex.run_benchmark(explicit, cache=cache)
    cols = [c for c in run.raw.columns if c != "runtime_s"]
    pd.testing.assert_frame_equal(again.raw[cols], run.raw[cols])
    # agreeing explicit alpha plus file is accepted
    ex.resolve_alpha(_with_file(f, res.alpha)["design"])


def test_alpha_mismatch_with_file_raises(sel, bench):
    f = sel[1] / "alpha_selected.json"
    wrong = sel[0].alpha + 1.0
    with pytest.raises(ValueError, match="disagrees"):
        ex.run_benchmark(_with_file(f, wrong))
    with pytest.raises(ValueError, match="disagrees"):
        ex.run_mismatch(_with_file(f, wrong), bench[1])
    cfg = copy.deepcopy(CFG)
    cfg["design"]["alpha"] = None
    with pytest.raises(ValueError, match="alpha_selection_file"):
        ex.run_benchmark(cfg)


def test_mismatch_uses_frozen_alpha(sel, bench, cache, tmp_path):
    f = sel[1] / "alpha_selected.json"
    ex.run_mismatch(_with_file(f), bench[1], tmp_path / "m", cache=cache)
    prov = json.loads((tmp_path / "m" / "provenance.json").read_text())
    summ = json.loads((tmp_path / "m" / "summary.json").read_text())
    assert prov["extra"]["alpha"]["alpha_selection_sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
    assert summ["alpha"]["value"] == sel[0].alpha


def test_alpha_zero_sensitivity_runs_and_is_labelled(cache, tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["design"]["alpha"] = 0.0
    ex.run_benchmark(cfg, tmp_path / "u", cache=cache)
    info = json.loads((tmp_path / "u" / "summary.json").read_text())["alpha"]
    assert info["value"] == 0.0 and info["alpha_selection_sha256"] is None
    assert info["label"].startswith("not selected: sensitivity")


def test_cli_alpha_selection_and_shipped_configs(tmp_path):
    from pravah import cli
    from pravah.config import load_config
    cfg = copy.deepcopy(CFG)
    cfg["experiment"] = "alpha_selection"
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    assert cli.main(["benchmark", "--config", str(tmp_path / "c.json"), "--out", str(tmp_path / "o")]) == 0
    assert json.loads((tmp_path / "o" / "alpha_selected.json").read_text())["alpha"] in (0.0, 1.0, 100.0)
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "configs"
    c = {n: load_config(root / f"{n}.json") for n in ("benchmark", "alpha_selection", "benchmark_unregularized",
                                                      "mismatch", "smoke", "smoke_alpha_selection")}
    assert c["alpha_selection"]["experiment"] == "alpha_selection"
    for n in ("benchmark", "alpha_selection", "mismatch"):
        assert (c[n]["design"]["test_seed"], c[n]["design"]["test_noise_seed"]) == (33, 7000)
    assert c["alpha_selection"]["design"]["layout_seed"] == 77 and c["benchmark"]["design"]["layout_seed"] == 7
    assert c["benchmark"]["design"]["alpha"] is None and c["mismatch"]["design"]["alpha"] is None
    assert c["mismatch"]["design"]["alpha_selection_file"] == c["benchmark"]["design"]["alpha_selection_file"]
    assert c["mismatch"]["mismatch"]["layouts_from"] == "results/benchmark-final"
    assert c["benchmark_unregularized"]["design"]["alpha"] == 0.0
    assert c["benchmark_unregularized"]["design"]["alpha_selection_file"] is None
    assert c["smoke"]["design"]["alpha_selection_file"] is None and c["smoke"]["design"]["alpha"] == 1.0
    for n in c:  # every shipped config passes the ensemble guards
        ex._setup(c[n], need_alpha=False)

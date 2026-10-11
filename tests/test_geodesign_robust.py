"""Tests for the robustness study of the fixed geographic layouts (Stage F2b).  Real geography; synthetic emissions and
meteorology.  Real FiPy solves on the cropped smoke configurations; no solver mocks.  Skipped when data/belauri/manifest.json
is missing."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pravah import design as D
from pravah import geodesign as G
from pravah import geodesign_robust as R
from pravah.cli import main
from pravah.config import load_config, to_dict
from pravah.experiments import block_average
from pravah.inverse import ResponseCache

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "belauri" / "manifest.json"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="data/belauri/manifest.json is missing (run geo prepare)")
SMOKE = ROOT / "configs" / "geodesign_smoke.json"
ROBUST = ROOT / "configs" / "geodesign_robust_smoke.json"


def robust_cfg(refinement: bool = False) -> dict[str, Any]:
    cfg = load_config(ROBUST)
    cfg["region"]["manifest"] = str(MANIFEST)
    cfg["robust"]["refinement"]["enabled"] = refinement
    return cfg


def strict_json(path: Path) -> Any:
    def bad(c: str) -> None:
        raise ValueError(f"non-strict JSON constant {c} in {path}")
    return json.loads(path.read_text(), parse_constant=bad)


@pytest.fixture(scope="module")
def cache() -> ResponseCache:
    return ResponseCache()


@pytest.fixture(scope="module")
def main_run(cache: ResponseCache, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The smoke geodesign study (the design-stage layouts this module re-uses)."""
    cfg = load_config(SMOKE)
    cfg["region"]["manifest"] = str(MANIFEST)
    out = tmp_path_factory.mktemp("main") / "run"
    G.run_geodesign(cfg, out, cache=cache)
    return out


def _forbid(*_a: Any, **_k: Any) -> Any:
    raise AssertionError("a selection / tuning function was called during the robustness evaluation")


@pytest.fixture(scope="module")
def robust_run(main_run: Path, cache: ResponseCache, tmp_path_factory: pytest.TempPathFactory) -> R.RobustResult:
    """The robustness study with every selection function patched to raise: layouts are re-used verbatim."""
    out = tmp_path_factory.mktemp("robust") / "run"
    with pytest.MonkeyPatch.context() as mp:
        for mod, names in ((D, ("greedy_trace", "select_layout", "random_layouts", "spread_layout")),
                           (G, ("make_layouts", "alpha_independent_layouts", "select_alpha", "design_stage"))):
            for n in names:
                mp.setattr(mod, n, _forbid)
        return R.run_geodesign_robust(robust_cfg(), out, layouts_from=main_run, cache=cache)


# ---------------------------------------------------------------------------------------------- reuse and pairing
def test_layouts_reused_verbatim(main_run: Path, robust_run: R.RobustResult) -> None:
    prov = strict_json(robust_run.out_dir / "provenance.json")["extra"]
    assert prov["layouts_reused_verbatim"] is True and prov["status"] == G.PROVISIONAL
    assert prov["layouts_from_sha256"]["layouts.csv"] == R._sha(main_run / "layouts.csv")
    lt = pd.read_csv(main_run / "layouts.csv")
    raw = robust_run.raw
    assert set(raw["layout_id"]) <= set(lt["layout_id"]) and set(raw["regime"]) == {"standard"}
    assert set(raw["layout_id"]) == set(lt[lt["regime"] == "standard"]["layout_id"])
    assert prov["n_fixed_layouts"] == int((lt["regime"] == "standard").sum())


def test_matched_rows_equal_the_main_run(main_run: Path, robust_run: R.RobustResult) -> None:
    main_raw = pd.read_csv(main_run / "test_raw.csv")
    mat = robust_run.raw[robust_run.raw["condition"] == "matched"]
    m = mat.merge(main_raw, on=R.KEYS, suffixes=("", "_main"))
    assert len(m) == len(mat) > 0
    for c in ("rmse", "rmse_local", "rmse_external", "attr_err", "source_rel_err", "inflow_abs_err_bg"):
        np.testing.assert_allclose(m[c], m[f"{c}_main"], rtol=1e-9, atol=1e-12, err_msg=c)
    chk = strict_json(robust_run.out_dir / "provenance.json")["extra"]
    assert chk["matched_vs_main_test_raw"]["compared"] and chk["matched_vs_main_test_raw"]["max_relative_difference"] < 1e-9
    assert chk["design_regeneration_check"]["compared"] and chk["design_regeneration_check"]["max_relative_difference"] < 1e-9


def test_conditions_produce_rows_and_valid_json(robust_run: R.RobustResult) -> None:
    out, raw = robust_run.out_dir, robust_run.raw
    assert set(raw["condition"]) == set(R.CONDITIONS)
    counts = raw.groupby("condition").size()
    assert counts.nunique() == 1 and counts.iloc[0] > 0  # same layouts x cases x reps under every condition
    for n in ("config.json", "provenance.json", "timings.json", "numerical_diffusion.json"):
        strict_json(out / n)
    for n in ("test_raw.csv", "summary.csv", "paired_rows.csv", "paired_mismatch.csv", "contrasts.csv", "ranking_changes.csv",
              "flags_vs_error.csv", "chi2_by_condition.csv", "cases.csv", "numerical_diffusion.csv"):
        assert len(pd.read_csv(out / n)) > 0, n
    for n in ("rmse_by_condition_interior_only.png", "paired_difference.png"):
        assert (out / "figures" / n).stat().st_size > 5000, n
    assert strict_json(out / "timings.json")["peak_memory_mb"] > 0
    s = raw[(raw["alpha_label"] == "selected")]
    for col in ("rmse", "rmse_local", "rmse_external", "attr_err", "source_rel_err", "inflow_abs_err_bg", "err_road", "err_bg"):
        assert np.isfinite(s[col]).all(), col


def test_paired_tables_are_paired_and_descriptive(robust_run: R.RobustResult) -> None:
    pr = pd.read_csv(robust_run.out_dir / "paired_rows.csv")
    assert "matched" not in set(pr["condition"]) and set(pr["condition"]) == set(R.CONDITIONS) - {"matched"}
    np.testing.assert_allclose(pr["diff_rmse"], pr["rmse"] - pr["rmse_matched"])
    ps = pd.read_csv(robust_run.out_dir / "paired_mismatch.csv")
    assert (ps["n_cases_worse"] <= ps["n_cases"]).all() and (ps["n_cases"] == 2).all()
    assert ps["sd_note"].str.contains("never a confidence").all()
    assert {"rmse", "rmse_local", "rmse_external", "attr_err", "ext_frac_abs_err", "source_rel_err", "inflow_abs_err_bg_abs",
            "abs_err_road", "abs_err_bg"} <= set(ps["metric"])
    # the unseen declared wind has a different inference model: no pairing requirement beyond the row keys
    con = pd.read_csv(robust_run.out_dir / "contrasts.csv")
    assert set(con["condition"]) == set(R.CONDITIONS)
    assert {"greedy_total vs spread", "greedy_attribution vs greedy_total", "with_reference vs interior_only"} <= set(con["comparison"])
    rk = pd.read_csv(robust_run.out_dir / "ranking_changes.csv")
    assert "same_sign_as_matched" in rk and "matched" not in set(rk["condition"])


def test_sigma_is_the_validation_scale_in_every_condition(main_run: Path, robust_run: R.RobustResult) -> None:
    vs = strict_json(main_run / "alpha_selected.json")["validation"]["signal_scale_ug_m3"]
    raw = robust_run.raw
    assert np.allclose(raw["sigma_ug_m3"], raw["noise_fraction"] * vs, rtol=1e-12)
    prov = strict_json(robust_run.out_dir / "provenance.json")["extra"]
    assert "VALIDATION" in prov["noise_scale_source"]
    nl = pd.read_csv(main_run / "noise_levels.csv")
    assert np.allclose(sorted(raw["sigma_ug_m3"].unique()), sorted(nl["sigma_ug_m3"].unique()))


def test_chi2_by_condition_table(robust_run: R.RobustResult) -> None:
    t = pd.read_csv(robust_run.out_dir / "chi2_by_condition.csv")
    assert set(t["condition"]) == set(R.CONDITIONS) and {"median_chi2_per_dof", "median_ratio_to_matched"} <= set(t)
    m = t[t["condition"] == "matched"]
    assert np.allclose(m["median_ratio_to_matched"].dropna(), 1.0)
    d = t[t["budget"] == 3]
    assert d["median_chi2_per_dof"].notna().all() and (t[t["budget"] == 1]["n_defined"] == 0).all()
    assert "median_chi2_per_dof" in pd.read_csv(robust_run.out_dir / "summary.csv")
    pr = pd.read_csv(robust_run.out_dir / "paired_rows.csv")
    assert "diff_chi2_per_dof" in pr
    # a wrong model shifts the data-based check upwards for at least one condition (descriptive, 2 smoke cases)
    print(d.groupby("condition")["median_ratio_to_matched"].median())
    assert (d[d["condition"] != "matched"]["median_ratio_to_matched"] > 1.0).any()


def test_flags_cannot_detect_a_wrong_model(robust_run: R.RobustResult) -> None:
    pr = pd.read_csv(robust_run.out_dir / "paired_rows.csv")
    wrong = pr[pr["condition"].isin(["unseen_wind", "hidden_wind_error", "boundary_time_profile", "omitted_facility",
                                     "omitted_external", "refined_truth"])]
    assert wrong["flags_same"].all()  # identical operator and sigma -> identical flags, whatever the data say
    fl = pd.read_csv(robust_run.out_dir / "flags_vs_error.csv")
    w = fl[fl["condition"] == "hidden_wind_error"]
    assert np.allclose(w["frac_flagged"], w["frac_flagged_matched"]) and (w["frac_flags_equal_matched"] == 1.0).all()


# ---------------------------------------------------------------------------------------------- the truths
def test_wind_truth_differs_from_inference_model(robust_run: R.RobustResult) -> None:
    cases = pd.read_csv(robust_run.out_dir / "cases.csv")
    for kind in ("unseen_wind", "hidden_wind_error"):
        c = cases[cases["condition"] == kind]
        assert len(c) == 2 and (c["truth_wind"] != c["nominal_wind"]).all(), kind
    u = cases[cases["condition"] == "unseen_wind"]
    sp = [float(w.split(" at ")[1].split(" m/s")[0]) for t in u["truth_wind"] for w in t.split(";")]
    assert min(sp) >= 2.59 and max(sp) <= 3.51  # outside the design draw U(0.8, 2.5)
    m = cases[cases["condition"] == "matched"]
    assert (m["truth_wind"] == m["nominal_wind"]).all()
    d = cases[cases["condition"] == "unseen_wind_declared"]
    assert (d["truth_wind"].to_numpy() == u["truth_wind"].to_numpy()).all()  # same unseen truth, but declared


def test_unseen_winds_are_outside_design_sectors_and_deterministic() -> None:
    cfg = robust_cfg()
    rc = R.robust_settings(cfg)
    draws = G.draw_cases(cfg, 6, 303, "t")
    a, b = R.unseen_winds(cfg, rc, draws), R.unseen_winds(cfg, rc, draws)
    assert a == b
    half, width = cfg["ensemble"]["sector_half_width_deg"], 360.0 / cfg["ensemble"]["n_sectors"]
    for d, segs in zip(draws, a):
        assert [s[0] for s in segs] == [w[0] for w in d.winds]  # segment times kept
        for _t, f, s in segs:
            off = abs(((f + width / 2) % width) - width / 2)  # distance to the nearest sector centre
            assert abs(off - 22.5) < 1e-9 and off > half
            assert 2.6 <= s <= 3.5


def test_hidden_wind_matches_perturb_wind_and_declared_wind_is_nominal(cache: ResponseCache) -> None:
    cfg = robust_cfg()
    st = G.Setup(cfg, cache)
    rc = R.robust_settings(cfg)
    d = G.draw_cases(cfg, 1, 303, "t")[0]
    case, info = R.variant_case(st, "hidden_wind_error", d, R.unseen_winds(cfg, rc, [d])[0], 3000, rc)
    assert case.inference_fingerprint == D._fingerprint(to_dict(st.scenario(d).scenario))  # the NOMINAL wind is declared
    matched, _ = st.build(d, "test", 3000)
    assert case.inference_fingerprint == matched.inference_fingerprint and case.truth_fingerprint != matched.truth_fingerprint
    np.testing.assert_array_equal(case.noise, matched.noise)  # same noise arrays: rows are paired
    assert not np.allclose(case.truth_readings, matched.truth_readings)


def test_omitted_sources_are_matched_by_kind_block(cache: ResponseCache) -> None:
    cfg = robust_cfg()
    st = G.Setup(cfg, cache)
    rc = R.robust_settings(cfg)
    d = G.draw_cases(cfg, 1, 303, "t")[0]
    for kind, name, group in (("omitted_facility", "facility", "local"), ("omitted_external", "extra_external", "external")):
        case, info = R.variant_case(st, kind, d, R.unseen_winds(cfg, rc, [d])[0], 3000, rc)
        assert case.param_names == ("road", "builtup", "external_patch", "bg")  # inference parameters only
        assert case.q_true.shape == (5,) and case.q_true[3] > 0  # [3 emissions, extra, bg]
        assert list(case.truth_index) == [0, 1, 2, 4]  # emissions by position, inflow after the extra emission
        assert case.unmodelled_mass > 0 and info["unmodelled_emission_share"] == case.unmodelled_mass
        assert case.has_components and group in ("local", "external")
        assert case.q_true[4] == pytest.approx(st.scenario(d).theta[3])
        assert [p.group for p in st.scenario(d).scenario.param_table()][:3] == ["local", "local", "external"]


def test_boundary_profile_truth_differs_only_in_bg_profile(cache: ResponseCache) -> None:
    cfg = robust_cfg()
    st = G.Setup(cfg, cache)
    rc = R.robust_settings(cfg)
    d = G.draw_cases(cfg, 1, 303, "t")[0]
    case, _ = R.variant_case(st, "boundary_time_profile", d, R.unseen_winds(cfg, rc, [d])[0], 3000, rc)
    matched, _ = st.build(d, "test", 3000)
    assert case.unmodelled_mass == 0.0 and case.n_params == matched.n_params
    np.testing.assert_array_equal(case.q_true, matched.q_true)
    # the delayed bg onset means weaker inflow early: the truth readings differ, the inference operator is the same
    np.testing.assert_array_equal(case.A_site, matched.A_site)
    assert not np.allclose(case.truth_readings, matched.truth_readings)


def test_refined_truth_error_field_is_block_averaged(cache: ResponseCache) -> None:
    cfg = robust_cfg()
    st = G.Setup(cfg, cache)
    rc = R.robust_settings(cfg)
    d = G.draw_cases(cfg, 1, 303, "t")[0]
    inf = st.scenario(d).scenario
    fine = R.upsample_scenario(inf, 2, 0.5)
    assert (fine.domain.nx, fine.domain.ny) == (2 * inf.domain.nx, 2 * inf.domain.ny) and fine.times.dt == inf.times.dt / 2
    for s_f, s_c in zip(fine.sources, inf.sources):
        if hasattr(s_c, "weights"):
            w = np.asarray(s_f.weights)
            assert w.shape == (2 * inf.domain.ny, 2 * inf.domain.nx)
            np.testing.assert_allclose(w.reshape(inf.domain.ny, 2, inf.domain.nx, 2).mean(axis=(1, 3)), np.asarray(s_c.weights))
    case, info = R.variant_case(st, "refined_truth", d, R.unseen_winds(cfg, rc, [d])[0], 3000, rc)
    # the quadratic form of the error was built against the block average: nx, ny and times are the inference grid's
    resp = st.cache.get(inf, st.solver)
    total, local, ext = R._split_truth(st, fine, st.scenario(d).theta)
    avg = block_average(total, inf)
    assert avg.shape == resp.base.shape == (len(resp.times), inf.domain.ny, inf.domain.nx)
    np.testing.assert_allclose(avg.values.mean(), total.values.mean(), rtol=1e-10)
    np.testing.assert_allclose((local + ext).values, total.values, rtol=1e-9, atol=1e-12)
    matched, _ = st.build(d, "test", 3000)
    assert case.c0 != matched.c0  # a different error field: scheme contrast
    assert info["max_courant_truth"] == pytest.approx(matched_courant(st, d), rel=1e-9)


def matched_courant(st: G.Setup, d: G.CaseDraw) -> float:
    return max(s for _, _, s in d.winds) * st.dt / st.res


# ---------------------------------------------------------------------------------------------- guards
def test_config_mismatch_with_the_main_run_is_refused(main_run: Path, cache: ResponseCache, tmp_path: Path) -> None:
    cfg = robust_cfg()
    cfg["noise"]["reps"] = 3  # a different study: the regenerated test ensemble would not be the main one
    with pytest.raises(ValueError, match="config differs"):
        R.run_geodesign_robust(cfg, tmp_path / "x", layouts_from=main_run, cache=cache)
    with pytest.raises(ValueError, match="no layouts"):
        R.run_geodesign_robust(robust_cfg() | {"robust": {"layouts_from": None}}, tmp_path / "y", cache=cache)
    with pytest.raises(FileNotFoundError):
        R.run_geodesign_robust(robust_cfg(), tmp_path / "z", layouts_from=tmp_path / "nowhere", cache=cache)


def test_matched_cannot_be_dropped_and_unknown_conditions_fail() -> None:
    cfg = robust_cfg()
    cfg["robust"]["conditions"] = ["unseen_wind"]
    with pytest.raises(ValueError, match="matched"):
        R.robust_settings(cfg)
    cfg["robust"]["conditions"] = ["matched", "bogus"]
    with pytest.raises(ValueError, match="unknown conditions"):
        R.robust_settings(cfg)


def test_numerical_diffusion_table(robust_run: R.RobustResult) -> None:
    nd = strict_json(robust_run.out_dir / "numerical_diffusion.json")
    assert nd["tuned"] is False and "not tuned" in nd["note"]
    tab = pd.read_csv(robust_run.out_dir / "numerical_diffusion.csv")
    assert set(tab["mesh"]) == {"design", "refined"} and (tab["K_eff_over_K"] >= 1.0).all()
    from pravah.geo.scenarios import numerical_diffusion_estimate
    r = tab[(tab["mesh"] == "design") & (tab["speed_label"] == "reference_2.5")].iloc[0]
    assert r["excess_K_eff_over_K_minus_1"] == pytest.approx(numerical_diffusion_estimate(2.5, 500.0, 100.0, 100.0))
    # the finer mesh with the halved time step has less numerical spreading at the same speed
    fine = tab[(tab["mesh"] == "refined") & (tab["speed_label"] == "reference_2.5")].iloc[0]
    assert fine["K_eff_over_K"] < r["K_eff_over_K"]


def test_determinism(main_run: Path, robust_run: R.RobustResult, cache: ResponseCache, tmp_path: Path) -> None:
    out = tmp_path / "again"
    R.run_geodesign_robust(robust_cfg(), out, layouts_from=main_run, cache=cache)
    for n in ("test_raw.csv", "paired_mismatch.csv", "contrasts.csv", "flags_vs_error.csv"):
        a, b = pd.read_csv(robust_run.out_dir / n), pd.read_csv(out / n)
        drop = [c for c in a.columns if "runtime" in c]
        pd.testing.assert_frame_equal(a.drop(columns=drop), b.drop(columns=drop), check_exact=False, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------------------------- refinement
def test_refinement_stability_smoke(main_run: Path, cache: ResponseCache, tmp_path: Path) -> None:
    cfg = robust_cfg(refinement=True)
    cfg["robust"]["conditions"] = ["matched", "refined_truth"]
    out = tmp_path / "ref"
    R.run_geodesign_robust(cfg, out, layouts_from=main_run, cache=cache)
    ov = pd.read_csv(out / "refinement_overlap.csv")
    assert len(ov) > 0 and ov["jaccard"].between(0, 1).all() and (ov["mean_nearest_site_distance_m"] >= 0).all()
    assert set(ov["method"]) == {"greedy_total", "greedy_attribution"} and set(ov["budget"]) == {1, 3}
    lt = pd.read_csv(out / "refinement_layouts_250m.csv")
    assert (lt["design_model"] == "250m").all() and lt["provenance"].eq(G.PROVISIONAL).all()
    raw = pd.read_csv(out / "refinement_test_raw.csv")
    assert set(raw["design_model"]) == {"500m", "250m"}
    gap = pd.read_csv(out / "refinement_gap.csv")
    assert len(gap) > 0 and (gap["cases_won_by_250m_layouts"] <= gap["n_cases"]).all()
    info = strict_json(out / "refinement_info.json")
    assert info["fine_response_sets_design"] > 0 and info["fine_mesh"] == {"resolution_m": 250.0, "dt_s": 50.0}
    assert (out / "figures" / "refinement_sites.png").stat().st_size > 5000
    # the 250 m design layouts are real layouts of the same candidate pool
    cand = set(pd.read_csv(main_run / "candidates.csv")["id"])
    assert all(set(s.split(",")) <= cand for s in lt["sites"])


# ---------------------------------------------------------------------------------------------- CLI
def test_cli_end_to_end(main_run: Path, tmp_path: Path) -> None:
    cfg = robust_cfg(refinement=False)
    cfg["robust"]["layouts_from"] = str(main_run)
    p = tmp_path / "robust.json"
    p.write_text(json.dumps(cfg))
    out = tmp_path / "cli"
    t0 = time.perf_counter()
    assert main(["benchmark", "--config", str(p), "--out", str(out)]) == 0  # fresh response cache
    assert time.perf_counter() - t0 < 180.0
    prov = strict_json(out / "provenance.json")
    assert prov["extra"]["experiment"] == "geodesign_robust" and prov["extra"]["layouts_from"] == str(main_run)
    assert set(pd.read_csv(out / "test_raw.csv")["condition"]) == set(R.CONDITIONS)
    assert prov["extra"]["git_archive"]["dirty"] == prov["git_dirty"]
    if prov["git_dirty"]:
        assert (out / "git_diff.patch").is_file() and (out / "git_status.txt").is_file()
    assert prov["config"]["experiment"] == "geodesign_robust" and not (out / "refinement_gap.csv").exists()
    with pytest.raises(FileExistsError):
        main(["benchmark", "--config", str(p), "--out", str(out)])

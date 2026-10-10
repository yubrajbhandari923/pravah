"""Tests for the geographic sensor-design study (Stage F2a).  Real geography; synthetic emissions and meteorology.

Real FiPy solves on the cropped smoke configuration (configs/geodesign_smoke.json); no solver mocks.  Skipped when
data/belauri/manifest.json is missing."""
from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pravah import design as D
from pravah import geodesign as G
from pravah.cli import main
from pravah.config import load_config
from pravah.inverse import ResponseCache

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "belauri" / "manifest.json"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="data/belauri/manifest.json is missing (run geo prepare)")
SMOKE = ROOT / "configs" / "geodesign_smoke.json"


def smoke_cfg() -> dict[str, Any]:
    cfg = load_config(SMOKE)
    cfg["region"]["manifest"] = str(MANIFEST)
    return cfg


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return smoke_cfg()


@pytest.fixture(scope="module")
def cache() -> ResponseCache:
    return ResponseCache()


@pytest.fixture(scope="module")
def stage(cfg: dict[str, Any], cache: ResponseCache) -> G.DesignStage:
    return G.design_stage(cfg, cache)


@pytest.fixture(scope="module")
def cli_run(cfg: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The smoke study end to end through the CLI, with a fresh response cache."""
    out = tmp_path_factory.mktemp("geodesign_cli") / "run"
    p = tmp_path_factory.mktemp("cfg") / "smoke.json"
    p.write_text(json.dumps(cfg))
    assert main(["benchmark", "--config", str(p), "--out", str(out)]) == 0
    return out


def strict_json(path: Path) -> Any:
    def bad(c: str) -> None:
        raise ValueError(f"non-strict JSON constant {c} in {path}")
    return json.loads(path.read_text(), parse_constant=bad)


# ---------------------------------------------------------------------------------------------- ensembles and seeds
def test_ensemble_guard_raises_on_overlap(cfg: dict[str, Any]) -> None:
    e = cfg["ensembles"]
    draws = {r: (G.draw_cases(cfg, e[r]["n_cases"], e[r]["seed"], r[0]), e[r]["seed"], e[r]["noise_seed"]) for r in e}
    G.check_ensembles(draws)  # the configured ensembles are disjoint
    d, v = draws["design"], draws["validation"]
    with pytest.raises(ValueError, match="seeds must differ"):
        G.check_ensembles({"design": d, "validation": (v[0], d[1], v[2])})
    with pytest.raises(ValueError, match="noise-seed ranges overlap"):
        G.check_ensembles({"design": d, "validation": (v[0], v[1], d[2] + 1)})
    with pytest.raises(ValueError, match="share a"):  # same draws under another seed label and another name
        renamed = [dataclasses.replace(c, name="x" + c.name) for c in d[0]]
        G.check_ensembles({"design": d, "validation": (renamed, v[1], v[2])})
    with pytest.raises(ValueError, match="names overlap"):
        G.check_ensembles({"design": d, "validation": (G.draw_cases(cfg, 2, 555, "d"), 555, 9000)})


def test_draws_follow_the_declared_distribution() -> None:
    cfg = load_config(ROOT / "configs" / "geodesign.json")
    cases = G.draw_cases(cfg, 60, 7, "d")
    assert cases == G.draw_cases(cfg, 60, 7, "d") and cases != G.draw_cases(cfg, 60, 8, "d")
    e, base = cfg["ensemble"], cfg["model"]["base_amplitudes_ug_s"]
    T = cfg["physics"]["T"]
    sectors = set()
    for i, c in enumerate(cases):
        road, built, patch, bg = c.theta
        assert 0.5 * base["road"] <= road <= 1.5 * base["road"] and 0.5 * base["builtup"] <= built <= 1.5 * base["builtup"]
        assert 0.0 <= patch <= 2.0 * base["external"] and 0.0 <= bg <= 0.05
        assert len(c.winds) == (2 if i % 3 == 1 else 1)
        for t, deg, speed in c.winds:
            assert 0.8 <= speed <= 2.5
            centre = round(deg / 45.0) * 45.0
            assert abs(((deg - centre + 180.0) % 360.0) - 180.0) <= 15.0 + 1e-9
            sectors.add(int(round(deg / 45.0)) % 8)
        if len(c.winds) == 2:
            t = c.winds[1][0]
            assert T / 3 <= t <= 2 * T / 3 and t % 3600.0 == 0.0
    assert sectors == set(range(8))  # all eight sectors occur in 60 cases (up to the +-15 deg jitter)
    patches = [c.theta[2] / base["external"] for c in cases]
    assert min(patches) < 0.3 and max(patches) > 1.7  # nearly local-only and external-dominated cases both occur


# ---------------------------------------------------------------------------------------------- candidates
@pytest.fixture(scope="module")
def setup(cfg: dict[str, Any], cache: ResponseCache) -> G.Setup:
    return G.Setup(cfg, cache)


def test_candidate_table_geometry_and_flags(setup: G.Setup) -> None:
    import shapely
    t, geo = setup.table, setup.geo
    assert t["id"].is_unique and set(t["category"]) == {G.INTERIOR, G.REF_DOMESTIC, G.REF_HYPOTHETICAL}
    pts = {c: shapely.points(g["utm_x"].to_numpy(), g["utm_y"].to_numpy()) for c, g in t.groupby("category")}
    assert shapely.contains(geo.study, pts[G.INTERIOR]).all()
    assert (~shapely.intersects(geo.study, pts[G.REF_DOMESTIC])).all()
    assert shapely.contains(geo.nepal, pts[G.REF_DOMESTIC]).all()
    assert (~shapely.intersects(geo.nepal, pts[G.REF_HYPOTHETICAL])).all()
    dom = t[t["category"] == G.REF_DOMESTIC]
    assert not dom["in_polygon"].any() and dom["in_nepal_adm0"].all()
    assert t[t["category"] == G.INTERIOR]["in_polygon"].all()
    hyp = t[t["category"] == G.REF_HYPOTHETICAL]
    assert (~hyp["in_nepal_adm0"]).all() and hyp["hypothetical"].all() and (~hyp["available_default"]).all()
    assert (t["on_or_near_road"] == (t["dist_road_m"] <= 100.0)).all()
    assert np.isfinite(t["dist_road_m"]).all() and np.isfinite(t["dist_facility_m"]).all()
    assert t["assumption_flags"].str.contains("not permission/power/security/connectivity").all()
    # local coordinates are UTM minus the cropped region's south-west corner
    gx, gy = geo.region.georef.x0, geo.region.georef.y0
    assert np.allclose(t["local_x"], t["utm_x"] - gx) and np.allclose(t["local_y"], t["utm_y"] - gy)


def test_hypothetical_excluded_by_default_and_ids_stable(cfg: dict[str, Any], setup: G.Setup) -> None:
    assert not setup.hyp and G.REF_HYPOTHETICAL not in set(setup.cands.categories)
    assert len(setup.cands) == int((setup.table["category"] != G.REF_HYPOTHETICAL).sum())
    with_h = G.table_to_candidates(setup.table, include_hypothetical=True)
    assert G.REF_HYPOTHETICAL in set(with_h.categories) and len(with_h) == len(setup.table)
    again = G.build_candidate_table(setup.geo, cfg)  # deterministic
    pd.testing.assert_frame_equal(again, setup.table)
    cfg2 = copy.deepcopy(cfg)
    cfg2["region"]["crop_buffer_m"] = None  # the full prepared domain: interior IDs do not change
    full = G.build_candidate_table(G.load_geography(cfg2), cfg2)
    ids = lambda t, c: list(t[t["category"] == c]["id"])  # noqa: E731
    assert ids(full, G.INTERIOR) == ids(setup.table, G.INTERIOR)
    assert set(ids(setup.table, G.REF_DOMESTIC)) <= set(ids(full, G.REF_DOMESTIC))


def test_candidates_geojson_and_manifest_sha(setup: G.Setup) -> None:
    meta = {"manifest_sha256": setup.geo.manifest_sha256}
    gj = G.candidates_geojson(setup.table, meta)
    assert gj["type"] == "FeatureCollection" and gj["metadata"] == meta
    assert len(gj["features"]) == len(setup.table)
    from pravah.geo.regionio import lonlat_box
    w, s, e, n = lonlat_box(setup.geo.region.level(500.0).grid.bounds, setup.geo.epsg)  # the simulation domain
    bw, bs, be, bn = setup.geo.manifest["study"]["bbox_lonlat"]
    for f in gj["features"]:
        lon, lat = f["geometry"]["coordinates"]
        assert f["geometry"]["type"] == "Point" and w <= lon <= e and s <= lat <= n
        if f["properties"]["category"] == G.INTERIOR:
            assert bw <= lon <= be and bs <= lat <= bn
    json.dumps(gj, allow_nan=False)
    import hashlib
    assert setup.geo.manifest_sha256 == hashlib.sha256(setup.geo.manifest_path.read_bytes()).hexdigest()
    assert set(setup.geo.raw_sha256) == {"study", "adm0_NPL", "osm_roads", "osm_amenity"}


def test_heldout_points(setup: G.Setup) -> None:
    import shapely
    h = setup.heldout
    assert len(h["random"]) == 20
    gx, gy = setup.geo.region.georef.x0, setup.geo.region.georef.y0
    assert shapely.contains_xy(setup.geo.study, h["random"].x + gx, h["random"].y + gy).all()
    d = np.hypot(h["random"].x[:, None] - setup.cands.x[None, :], h["random"].y[:, None] - setup.cands.y[None, :])
    assert d.min() > 1.0  # held-out points are not candidates
    b = h["block"]
    assert shapely.contains_xy(setup.geo.study, b.x + gx, b.y + gy).all() and len(b) > 5
    x0, _, x1, _ = setup.geo.study.bounds
    third = (x1 - x0) / 3.0  # block = the east third of the lattice (its bounding box is within 1 km of the polygon's)
    assert b.x.min() + gx >= x1 - third - 1000.0
    again = G.heldout_points(setup.geo, setup.cfg, setup.table)
    assert np.array_equal(again["random"].x, h["random"].x) and np.array_equal(again["block"].y, b.y)


# ---------------------------------------------------------------------------------------------- constraints and layouts
def test_constraints_respected_in_every_layout(setup: G.Setup, stage: G.DesignStage) -> None:
    cats = np.asarray(setup.cands.categories)
    radius = float(setup.cfg["heldout"]["block"]["exclusion_radius_m"])
    b = setup.heldout["block"]
    assert stage.layouts
    for r in stage.layouts:
        assert len(r.layout) == r.budget == len(set(r.layout)) and list(r.layout) == sorted(r.layout)
        n_ref = int((cats[list(r.layout)] != G.INTERIOR).sum())
        if r.setting == G.SETTING_A:
            assert n_ref == 0
        else:
            assert n_ref <= 2 and (cats[list(r.layout)] != G.REF_HYPOTHETICAL).all()  # reference counts in the budget
        if r.regime == G.BLOCK:
            for i in r.layout:
                assert np.hypot(b.x - setup.cands.x[i], b.y - setup.cands.y[i]).min() >= radius * (1 - 1e-9)
        if r.order:
            assert sorted(r.order) == list(r.layout)
    assert {r.method for r in stage.layouts} == set(G.METHODS)
    assert {r.regime for r in stage.layouts} == {G.STANDARD, G.BLOCK}
    # with_reference allows (and the greedy methods use) reference sites in some layouts
    assert any(cats[list(r.layout)].tolist().count(G.REF_DOMESTIC) for r in stage.layouts if r.setting == G.SETTING_B)


def test_hypothetical_setting_is_separate_and_capped(cfg: dict[str, Any], cache: ResponseCache) -> None:
    c2 = copy.deepcopy(cfg)
    c2["settings"]["hypothetical"] = True
    st = G.Setup(c2, cache)
    assert st.settings[-1] == G.SETTING_C and G.REF_HYPOTHETICAL in set(st.cands.categories)
    cats = np.asarray(st.cands.categories)
    lays, _ = G.alpha_independent_layouts(st, G.SETTING_B, G.STANDARD)
    assert all((cats[list(r.layout)] != G.REF_HYPOTHETICAL).all() for r in lays)  # B never uses hypothetical sites
    lays_c, _ = G.alpha_independent_layouts(st, G.SETTING_C, G.STANDARD)
    assert all(int((cats[list(r.layout)] == G.REF_HYPOTHETICAL).sum()) <= 2 for r in lays_c)
    assert any((cats[list(r.layout)] == G.REF_HYPOTHETICAL).any() for r in lays_c)


def test_infeasible_budgets_reported_when_exclusion_blocks(cfg: dict[str, Any], setup: G.Setup, cache: ResponseCache) -> None:
    c2 = copy.deepcopy(cfg)
    c2["heldout"]["block"]["exclusion_radius_m"] = 8000.0  # leaves almost no interior site
    st = G.Setup(c2, cache)
    lays, infeas = G.alpha_independent_layouts(st, G.SETTING_A, G.BLOCK)
    assert set(infeas) == set(st.budgets) and not lays and all("exceeds" in v for v in infeas.values())
    direct = D.infeasible_budgets(st.cands, st.constraints(G.SETTING_A, G.BLOCK), st.budgets)
    assert direct == infeas
    ok, none = G.alpha_independent_layouts(setup, G.SETTING_A, G.STANDARD)
    assert not none and {r.budget for r in ok} == set(setup.budgets)


# ---------------------------------------------------------------------------------------------- selection, leakage, alpha
def test_alpha_and_layouts_independent_of_test_ensemble(cfg: dict[str, Any], cache: ResponseCache, stage: G.DesignStage,
                                                         tmp_path: Path) -> None:
    c2 = copy.deepcopy(cfg)
    c2["ensembles"]["test"] = {"n_cases": 3, "seed": 99999, "noise_seed": 55555}
    other = G.design_stage(c2, cache)
    assert json.dumps(other.alpha, sort_keys=True, default=str) == json.dumps(stage.alpha, sort_keys=True, default=str)
    assert other.layouts == stage.layouts  # selected layouts do not depend on test truths or seeds
    c3 = copy.deepcopy(cfg)
    del c3["ensembles"]["test"]  # the design stage does not even read the test block
    assert G.design_stage(c3, cache).layouts == stage.layouts
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    from pravah.io import write_json
    write_json(p1, stage.alpha)
    write_json(p2, other.alpha)
    assert p1.read_bytes() == p2.read_bytes()
    text = json.dumps(stage.alpha)  # nothing about the test ensemble is recorded in the frozen file
    assert not any(str(v) in text for v in ("99999", "55555", "303", "3000"))
    assert stage.alpha["uses_test_ensemble"] is False
    # alpha is the argmin of the validation criterion per noise fraction, from the pre-declared grid
    for f, tab in stage.alpha["tables"].items():
        best = min(tab, key=lambda r: (r["criterion_mean_rmse"], r["alpha"]))
        assert stage.alpha["alpha_by_noise_fraction"][f] == best["alpha"] and best["alpha"] in cfg["alpha"]["grid"]


def test_role_guards(stage: G.DesignStage, setup: G.Setup) -> None:
    assert {c.role for c in stage.design.cases} == {"design"} and {c.role for c in stage.validation.cases} == {"validation"}
    with pytest.raises(ValueError, match="leakage guard"):
        D.attribution_scales(stage.validation.cases)
    with pytest.raises(ValueError, match="leakage guard"):
        D.greedy_trace(setup.cands, 1, stage.validation.cases, setup.constraints(G.SETTING_A, G.STANDARD),
                       setup.opts(0.01, 0.0, (1.0, 1.0)), "total")
    with pytest.raises(ValueError, match="leakage guard"):
        G.evaluate_test(setup, stage.layouts, G.Ensemble("test", stage.validation.cases, [], 1.0, {0.05: 0.01}), {0.05: 0.0},
                        (1.0, 1.0))


def test_inverse_never_receives_truth_tags(stage: G.DesignStage, setup: G.Setup) -> None:
    c = stage.design.cases[0]
    opts = setup.opts(stage.design.sigma[0.05], 0.01, D.attribution_scales(stage.design.cases))
    lay = stage.layouts[0].layout
    ref = D.fit_layout(c, lay, 0, opts)
    stripped = dataclasses.replace(c, objectives={"total": c.objectives["total"]}, component_ms=None, fraction=None,
                                   external_fraction_true=None, q_true=c.q_true * 7.0, unmodelled_mass=0.0)
    got = D._solve(stripped, lay, 0, opts)  # the fit uses only noisy readings, the model operator and the options
    assert np.array_equal(got.q_hat, ref["q_hat"])


def test_greedy_is_a_heuristic_checked_by_exhaustive_search(stage: G.DesignStage, setup: G.Setup) -> None:
    """Tiny fixture: 8 interior candidates, budget 2: compare greedy with the exhaustive minimum of the design score."""
    idx = [i for i, c in enumerate(setup.cands.categories) if c == G.INTERIOR][:8]
    allowed = np.zeros(len(setup.cands), dtype=bool)
    allowed[idx] = True
    cons = D.Constraints(allowed)
    opts = setup.opts(stage.design.sigma[0.05], stage.alpha_by_f[0.05], D.attribution_scales(stage.design.cases))
    for obj in ("total", "attribution"):
        tr = D.greedy_trace(setup.cands, 2, stage.design.cases, cons, opts, obj)
        best = min(D.score_layout(tuple(c), stage.design.cases, opts, obj)["value"] for c in itertools.combinations(idx, 2))
        assert tr.scores[-1] >= best * (1 - 1e-12)  # greedy can never beat the exhaustive optimum (and need not equal it)
        assert tr.scores[-1] == pytest.approx(D.score_layout(tr.layout, stage.design.cases, opts, obj)["value"])


def test_greedy_layouts_are_prefixes_of_the_trace(stage: G.DesignStage, setup: G.Setup) -> None:
    scales = D.attribution_scales(stage.design.cases)
    f = 0.05
    opts = setup.opts(stage.design.sigma[f], stage.alpha_by_f[f], scales)
    for r in (r for r in stage.layouts if r.method == "greedy_total" and r.noise_fraction == f and r.regime == G.STANDARD
              and r.setting == G.SETTING_B):
        tr = D.greedy_trace(setup.cands, r.budget, stage.design.cases, setup.constraints(r.setting, r.regime), opts, "total")
        assert tr.layout == r.layout and tr.order == r.order
        assert r.alpha == stage.alpha_by_f[f] and r.objective == "total"


# ---------------------------------------------------------------------------------------------- component metrics
def test_component_and_attribution_metrics(stage: G.DesignStage, setup: G.Setup) -> None:
    for c in stage.design.cases:
        assert c.has_components and c.fraction is not None
        assert c.fraction(np.zeros(c.n_params)) is None  # zero field: external fraction undefined (None), never 0 or nan
    scales = D.attribution_scales(stage.design.cases)
    assert all(math.isfinite(s) and s > 0 for s in scales)
    c = stage.design.cases[0]
    opts = setup.opts(stage.design.sigma[0.05], 0.01, scales)
    f = D.fit_layout(c, stage.layouts[-1].layout, 0, opts)
    for k in ("mse_local", "mse_external", "mse_attribution", "rmse", "source_rel_err", "inflow_rmse"):
        assert math.isfinite(f[k]), k
    assert f["mse_attribution"] == pytest.approx(0.5 * (f["mse_local"] / scales[0] + f["mse_external"] / scales[1]))
    assert f["external_fraction_error"] is None or math.isfinite(f["external_fraction_error"])


# ---------------------------------------------------------------------------------------------- end to end
def test_smoke_cli_outputs_valid(cli_run: Path, cfg: dict[str, Any]) -> None:
    need = ["config.json", "provenance.json", "timings.json", "candidates.csv", "candidates.geojson", "alpha_selected.json",
            "validation_raw.csv", "test_raw.csv", "summary.csv", "paired.csv", "layouts.csv", "layouts.geojson",
            "design_scores.csv", "infeasible_budgets.json", "noise_levels.csv", "cases.csv"]
    for n in need:
        assert (cli_run / n).is_file(), n
    for n in ("selected_sites.png", "error_vs_count_interior_only.png", "error_vs_count_with_reference.png",
              "attribution_vs_total.png", "heldout_block_map.png"):
        assert (cli_run / "figures" / n).stat().st_size > 5000, n
    for n in [n for n in need if n.endswith(".json")]:
        strict_json(cli_run / n)
    prov = strict_json(cli_run / "provenance.json")
    ex = prov["extra"]
    import hashlib
    assert ex["alpha_selected_sha256"] == hashlib.sha256((cli_run / "alpha_selected.json").read_bytes()).hexdigest()
    assert ex["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest() and ex["status"] == G.PROVISIONAL
    assert prov["seeds"]["test"]["scenario_seed"] == 303 and "git_diff_hash" in prov
    tim = strict_json(cli_run / "timings.json")
    assert tim["peak_memory_mb"] > 0 and "test_evaluation_s" in tim["stages_s"]
    cand = strict_json(cli_run / "candidates.geojson")
    assert cand["metadata"]["manifest_sha256"] == ex["manifest_sha256"] and cand["metadata"]["label"] == G.LABEL
    lay = strict_json(cli_run / "layouts.geojson")
    lt = pd.read_csv(cli_run / "layouts.csv")
    assert len(lay["features"]) == int(lt["n_sites"].sum())
    assert (lt["provenance"] == G.PROVISIONAL).all() and lt["config_hash"].nunique() == 1
    assert set(lt["objective"]) >= {"total", "attribution"}
    inf = strict_json(cli_run / "infeasible_budgets.json")
    assert set(inf["infeasible"]) == {f"{s}|{r}" for s in ("interior_only", "with_reference") for r in ("standard", "block_holdout")}
    assert not (cli_run / "test_seed").exists()
    with pytest.raises(FileExistsError):
        G.run_geodesign(cfg, cli_run)  # refuses a non-empty directory


def test_smoke_tables_content(cli_run: Path, cfg: dict[str, Any]) -> None:
    raw = pd.read_csv(cli_run / "test_raw.csv")
    nl = pd.read_csv(cli_run / "noise_levels.csv")
    cases = pd.read_csv(cli_run / "cases.csv")
    # noise: sigma = fraction x signal scale per ensemble; seeds are disjoint across ensembles
    assert np.allclose(nl["sigma_ug_m3"], nl["noise_fraction"] * nl["signal_scale_ug_m3"])
    assert np.allclose(nl["sigma_pct_of_signal_scale"], 100.0 * nl["noise_fraction"])
    assert cases["noise_seed"].is_unique and set(cases["role"]) == {"design", "validation", "test"}
    assert set(raw["case"]) == set(cases[cases["role"] == "test"]["case"])
    assert set(raw["alpha_label"]) == {G.SELECTED, G.ALPHA_ZERO, G.STRONG}
    assert set(raw.loc[raw["alpha_label"] == G.ALPHA_ZERO, "alpha"]) == {0.0}
    assert set(raw.loc[raw["alpha_label"] == G.STRONG, "alpha"]) == {cfg["alpha"]["strong_prior"]}
    sel = raw[raw["alpha_label"] == G.SELECTED]
    for col in ("rmse", "rmse_local", "rmse_external", "attr_err", "source_rel_err", "inflow_abs_err_bg", "heldout_random_rmse",
                "heldout_block_rmse", "err_road", "err_builtup", "err_external_patch", "err_bg"):
        assert np.isfinite(sel[col]).all(), col
    assert (sel["rank"] >= 0).all() and sel["id_flags"].notna().all()
    # external fraction error is NaN exactly where the estimated or true fraction is undefined
    undefined = sel["external_fraction_est"].isna() | sel["external_fraction_true"].isna()
    assert (sel["external_fraction_error"].isna() == undefined).all()
    assert np.allclose(sel["external_fraction_error"][~undefined], (sel["external_fraction_est"] - sel["external_fraction_true"])[~undefined])
    # per-kind errors: emission relative error and inflow absolute error are separate columns in their own units
    assert "inflow_abs_err_bg" in raw and "source_rel_err" in raw
    # raw rows exist per (case, layout, rep): never only means
    reps = cfg["noise"]["reps"]
    assert len(sel[(sel["method"] == "spread") & (sel["noise_fraction"] == 0.05) & (sel["setting"] == "interior_only") &
                   (sel["regime"] == "standard") & (sel["budget"] == 3)]) == len(set(raw["case"])) * reps
    assert sel[sel["method"] == "random"]["layout_id"].nunique() >= cfg["methods"]["n_random"]
    # the noise reps / cases of the test ensemble differ from validation
    val = pd.read_csv(cli_run / "validation_raw.csv")
    assert set(val["case"]).isdisjoint(set(raw["case"])) and set(val["alpha"]) == set(cfg["alpha"]["grid"])


def test_paired_and_summary_tables(cli_run: Path) -> None:
    p = pd.read_csv(cli_run / "paired.csv")
    s = pd.read_csv(cli_run / "summary.csv")
    assert {"greedy_total vs spread", "greedy_total vs random_mean", "greedy_attribution vs greedy_total",
            "with_reference vs interior_only", "consecutive_budgets"} <= set(p["comparison"])
    ga = p[p["comparison"] == "greedy_attribution vs greedy_total"]
    assert set(ga["metric"]) == {"rmse", "attr_err"}  # both metrics, as required
    assert (p["a_wins_cases"] <= p["n_cases"]).all() and (p["n_cases"] == 2).all() and p["sd_note"].str.contains("never a confidence").all()
    cb = p[p["comparison"] == "consecutive_budgets"]
    assert np.allclose(cb["per_added_sensor"], cb["mean_diff"] / (cb["budget"] - cb["budget_previous"]))
    assert s["sd_note"].str.contains("descriptive").all()
    ref = s[(s["method"] == "greedy_total") & (s["setting"] == "interior_only") & (s["regime"] == "standard") &
            (s["alpha_label"] == "selected") & (s["noise_fraction"] == 0.05)]
    assert set(ref["budget"]) == {1, 3} and (ref["n_rows"] == 4).all()


def test_runs_are_deterministic(cli_run: Path, cfg: dict[str, Any], cache: ResponseCache, stage: G.DesignStage,
                                tmp_path: Path) -> None:
    seen: list[str] = []
    out = tmp_path / "again"
    G.run_geodesign(cfg, out, progress=seen.append, cache=cache)  # warm cache for design/validation, fresh test solves
    for n in ("test_raw.csv", "validation_raw.csv", "summary.csv", "paired.csv", "layouts.csv", "design_scores.csv"):
        a, b = pd.read_csv(cli_run / n), pd.read_csv(out / n)
        drop = [c for c in a.columns if "runtime" in c]
        pd.testing.assert_frame_equal(a.drop(columns=drop), b.drop(columns=drop), check_exact=False, rtol=1e-9, atol=1e-12)
    assert (cli_run / "alpha_selected.json").read_bytes() == (out / "alpha_selected.json").read_bytes()
    for n in ("candidates.csv", "layouts.geojson", "infeasible_budgets.json"):
        assert (cli_run / n).read_text() == (out / n).read_text(), n
    # ordering guarantee: alpha is selected and layouts are frozen before the test ensemble is built
    i_alpha = next(i for i, m in enumerate(seen) if m.startswith("alpha selected"))
    i_test = next(i for i, m in enumerate(seen) if m.startswith("test case") or "TEST ensemble" in m)
    assert i_alpha < i_test and not any(m.startswith("test case") for m in seen[:i_test])
    assert (out / "alpha_selected.json").stat().st_mtime <= (out / "test_raw.csv").stat().st_mtime

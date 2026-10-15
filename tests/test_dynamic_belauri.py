"""DYN-08 preparation: ERA5 episode selector (synthetic record), ``dynamic.expand`` / ``truth_scenario`` on a Belauri-like RasterSource scenario,
padded-grid population regrid, candidate-site mapping, seed registration guard and the cost model.  Real-data tests skip when the data is absent."""
from pathlib import Path

import numpy as np
import pytest

from pravah import dynamic as dyn
from pravah.config import InflowBasis, WindSegment
from pravah.era5 import Era5Wind
from pravah.era5_episodes import SPLITS, clip_speed, make_windows, select_windows, speed_median, split_of, strata_table, stratum
from pravah.forward import simulate
from pravah.inverse import compute_responses
from pravah.studies.belauri import dynamic_belauri as db

ROOT = Path(__file__).resolve().parents[1]
STD = ROOT / "data/belauri/manifest.json"
PAD = ROOT / "configs/belauri_padded_geography/manifest.json"
POP = ROOT / "data/population/belauri_worldpop_2025_CN_250m.npy"


def _record(months=(1, 7)) -> Era5Wind:
    t = np.concatenate([np.datetime64(f"2023-{m:02d}-01T00") + np.arange(31 * 24) * np.timedelta64(1, "h") for m in months])
    t = t[t.astype("datetime64[M]") == t.astype("datetime64[M]")]
    rng = np.random.default_rng(0)
    n = len(t)
    u = (np.cumsum(rng.normal(0, 0.4, n)) % 6 - 3)[:, None, None] * np.ones((n, 2, 2))
    v = (np.cumsum(rng.normal(0, 0.4, n)) % 6 - 3)[:, None, None] * np.ones((n, 2, 2))
    return Era5Wind(t, np.array([80.0, 81.0]), np.array([28.0, 29.0]), u, v)


def test_windows_aligned_nonoverlapping_and_split_by_day():
    w = make_windows(_record(months=(1,)), 80.5, 28.5)
    assert len(w) == 93 and all(x.start.endswith(("T00", "T08", "T16")) for x in w)  # 31 days x 3 windows
    assert [x.index for x in w] == list(range(0, 31 * 24, 8))
    assert {x.split for x in w} == set(SPLITS)
    assert all(split_of(x.day) == x.split for x in w)
    assert split_of(10) == "design" and split_of(11) == "validation" and split_of(21) == "test"


def test_selection_is_seeded_stratified_and_split_pure():
    w = make_windows(_record(), 80.5, 28.5)
    med = speed_median(w)
    a = select_windows(w, "test", 8, 41100, med)
    b = select_windows(w, "test", 8, 41100, med)
    assert [x.start for x in a] == [x.start for x in b] and len({x.start for x in a}) == 8
    assert all(x.split == "test" for x in a)
    assert [x.start for x in a] != [x.start for x in select_windows(w, "test", 8, 41101, med)]
    n_strata = len(strata_table(w, med)["test"])
    assert len({stratum(x, med) for x in a}) == min(8, n_strata)  # round robin: every stratum before any repeat
    assert select_windows(w, "test", 10 ** 6, 1, med) == [] or len(select_windows(w, "test", 10 ** 6, 1, med)) == sum(x.split == "test" for x in w)


def test_clip_speed_keeps_direction():
    u, v = np.array([3.0, 0.5]), np.array([4.0, 0.0])
    cu, cv, n = clip_speed(u, v, 2.5)
    assert n == 1 and np.isclose(np.hypot(cu[0], cv[0]), 2.5) and np.isclose(cu[0] / cv[0], 3 / 4) and cu[1] == 0.5


@pytest.mark.skipif(not PAD.exists(), reason="padded geography missing")
def test_expand_and_truth_scenario_on_belauri_rastersource():
    """The readiness gap: expand/truth_scenario on a real RasterSource + Gaussian external patch + split inflow bases (2000 m padded level)."""
    from pravah.config import RasterSource
    from pravah.geo.prepare import load_region
    from pravah.geo.scenarios import belauri_scenario
    reg = load_region(PAD)
    bases = (InflowBasis("bg_wsw", ("west", "south"), ((0.0, 1.0),)), InflowBasis("bg_ene", ("east", "north"), ((0.0, 1.0),)))
    case = belauri_scenario(reg, 2000.0, (WindSegment(0.0, 1.2, 0.4), WindSegment(3600.0, -0.5, 1.0)), duration_s=7200.0, dt_s=450.0, output_interval_s=900.0,
                            K=100.0, lam=4.6e-5, H=500.0, amplitudes={"road": 1.5e6, "builtup": 1e6, "external": 5e5, "bg_wsw": 0.05, "bg_ene": 0.05},
                            bases=bases, external_profile=((0.0, 1.0),), origin_tol=0.25)
    sc, nb = case.scenario, 3
    assert any(isinstance(s, RasterSource) for s in sc.sources) and sc.n_params == 5
    ex = dyn.expand(sc, nb)
    assert ex.n_params == 5 * nb and sum(isinstance(s, RasterSource) for s in ex.sources) == 2 * nb
    assert [s.name for s in ex.sources[:nb]] == [f"road@{b}" for b in range(nb)]
    r = compute_responses(ex)
    hist = np.array([[1.0, 0.0, 2.0], [0.5, 0.5, 0.5], [0.0, 1.0, 1.0], [0.0, 0.05, 0.0], [0.02, 0.0, 0.03]]) * np.array([1e6, 1e6, 5e5, 1.0, 1.0])[:, None]
    direct = simulate(dyn.truth_scenario(sc, hist, dyn.bin_edges(7200.0, nb)), np.ones(5)).concentration.values
    np.testing.assert_allclose(r.field(hist.reshape(-1)).values - r.base, direct, rtol=1e-6, atol=1e-12)
    assert np.abs(direct).max() > 0
    tr = dyn.truth_scenario(sc, hist, dyn.bin_edges(7200.0, nb))
    assert [type(s) for s in tr.sources] == [type(s) for s in sc.sources]  # geometry untouched, only profiles replaced


@pytest.mark.skipif(not (STD.exists() and PAD.exists() and POP.exists()), reason="geography / population extract missing")
def test_padded_population_regrid_is_exact_and_conserving():
    from pravah.geo.prepare import load_region
    from pravah.population import mask_to_study, regrid_counts
    std, pad = load_region(STD), load_region(PAD)
    src = np.load(POP)
    c500, rep = regrid_counts(src, std.level(250.0).grid, pad.level(500.0).grid)
    c250, _ = regrid_counts(src, std.level(250.0).grid, pad.level(250.0).grid)
    assert rep["lost_outside"] == 0.0 and c500.shape == (68, 68) and np.isclose(c500.sum(), src.sum())
    np.testing.assert_allclose(c250.reshape(68, 2, 68, 2).sum(axis=(1, 3)), c500, atol=1e-9)  # aligned edges: exact
    sf = np.asarray(pad.level(500.0).arrays["study_fraction"], dtype=float)
    m = mask_to_study(c500, sf, float((c500 * sf).sum()))
    assert np.isclose(m.sum(), (c500 * sf).sum()) and m[sf == 0].sum() == 0
    assert 54000 < m.sum() < 55000  # study-polygon total (54357 at 500 m), NOT the 184030 grid total
    assert np.isclose(c500.sum(), 184029.57, atol=0.01)


@pytest.mark.skipif(not (STD.exists() and PAD.exists()), reason="geography missing")
def test_candidate_sites_map_to_distinct_cells_at_500m():
    cfg = db.load_cfg()
    S = db.get_study(cfg)
    s = S.sites
    assert s["n_table"] == 54 and len(s["ids"]) == 54 and s["dropped_shared_cell"] == 0
    assert len(set(zip(s["iy"].tolist(), s["ix"].tolist()))) == 54
    assert s["ix"].min() >= 0 and s["ix"].max() < 68 and s["iy"].max() < 68
    sf = np.asarray(S.region.level(500.0).arrays["study_fraction"])
    assert (sf[s["iy"], s["ix"]] > 0).all()  # every interior site lies in a cell that overlaps the polygon


def test_registered_seeds_are_disjoint_from_spent_ranges():
    cfg = db.load_cfg()
    assert db.seed_conflicts(cfg) == []
    bad = {**cfg, "seeds": {**cfg["seeds"], "test": 7300}}
    assert db.seed_conflicts(bad)
    assert db.seed_conflicts({**cfg, "seeds": {"validation": 6501}})


def test_cost_model_requires_measured_constants_and_adds_up():
    cfg = db.load_cfg()
    assert "error" in db.cost_model({**cfg, "cost": {**cfg["cost"], "fit_M2_ms": None}})
    m = db.cost_model(cfg)
    assert np.isclose(sum(m["parts_core_h"].values()), m["total_core_h"]) and m["n_episodes"]["test"] == 192 and m["n_response_sets"] == 30
    assert np.isclose(m["total_with_contingency"], m["total_core_h"] * cfg["cost"]["contingency"])
    assert m["total_with_contingency"] < 45.0  # the budget this design was sized for
    twice = db.cost_model({**cfg, "cost": {**cfg["cost"], "solve_500_dt50_s": 2 * cfg["cost"]["solve_500_dt50_s"]}})
    assert twice["total_core_h"] > m["total_core_h"]

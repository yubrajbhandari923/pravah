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
    cfg = db.load_cfg()  # v1: its ranges are now SPENT (RUN-0099), so the v1 config must be rejected
    assert any("test 43000" in c for c in db.seed_conflicts(cfg)) and any("validation 46000" in c for c in db.seed_conflicts(cfg))
    bad = {**cfg, "seeds": {**cfg["seeds"], "test": 7300}}
    assert db.seed_conflicts(bad)
    assert db.seed_conflicts({**cfg, "seeds": {"validation": 6501}})


def test_v2_seeds_fresh_and_reserved_for_v2_only():
    v2 = db.load_cfg("configs/dyn08_v2.json")
    assert v2["name"] == "dyn08_v2" and db.seed_conflicts(v2) == []
    s, n = v2["seeds"], len(v2["families"])
    assert (s["test"], s["validation"], s["wind_selection"], s["pool_layouts"]) == (44000, 45000, 41100, 41300)
    assert s["test"] + 12 * n * 2 - 1 == 44191 and s["validation"] + 6 * n - 1 == 45047  # 192 test / 48 validation episodes
    for lo, hi in ((43000, 43191), (46000, 46047), (49000, 49999)):  # v1 spent ranges
        assert any(a <= lo and hi <= b for a, b in db.USED_SEEDS)
        assert not (s["test"] <= hi and s["test"] + 999 >= lo) and not (s["validation"] <= hi and s["validation"] + 999 >= lo)
    assert db.seed_conflicts({**v2, "name": "dyn08_other"})  # 44000-45999 belong to v2 only
    assert db.seed_conflicts({**v2, "seeds": {**s, "test": 43100}})


def _groups(rng, n_layouts=10, n_ep=96, layout_sd=0.2, shift=1.0):
    ep = np.exp(rng.normal(0, 0.5, n_ep))  # shared episode difficulty
    def grp(mult):
        return {i: dict(enumerate(ep * np.exp(rng.normal(0, layout_sd)) * mult * np.exp(rng.normal(0, 0.1, n_ep)))) for i in range(n_layouts)}
    return list(grp(1.0).values()), list(grp(shift).values())


def test_two_level_bootstrap_null_rarely_meets_criterion_and_real_shift_does():
    wind_of = {e: f"w{e % 6}" for e in range(96)}
    rng = np.random.default_rng(7)
    met_null = 0
    for _ in range(25):  # identical-distribution layout groups with large layout-to-layout spread (sd 0.2)
        a, b = _groups(rng)
        c = db.paired_contrast(a, b, 0.0025, wind_of, 0.1, True, n_boot=1000)
        assert c is not None
        met_null += c["effect_criterion_met"]
    assert met_null <= 1
    hits = 0
    for _ in range(10):  # a real 30 % shift of group B (A better by ~23 %), layout sd 0.1
        a, b = _groups(rng, layout_sd=0.1, shift=1 / 0.7)
        c = db.paired_contrast(a, b, 0.0025, wind_of, 0.1, True, n_boot=1000)
        assert c is not None
        hits += c["effect_criterion_met"]
        assert c["ci_method"].startswith("two-level") and c["rel_change"] < -0.1
    assert hits >= 9


def test_two_level_ci_is_wider_than_episode_only_ci():
    a, b = _groups(np.random.default_rng(3))
    A, B = np.array([list(d.values()) for d in a]), np.array([list(d.values()) for d in b])
    lo2, hi2 = db.two_level_boot_ci(A, B, 0.0025, 2000)
    lo1, hi1 = db.boot_ci(A.mean(0) - B.mean(0), 0.0025, 2000)
    assert hi2 - lo2 > 1.5 * (hi1 - lo1)
    # a single layout vs a population: the layout level of A is fixed, the draw is reproducible
    m = db.two_level_boot_diffs(A[:1], B, 500, seed=1)
    assert m.shape == (500,) and np.array_equal(m, db.two_level_boot_diffs(A[:1], B, 500, seed=1))


def test_select_worst_is_argmax_of_fixed_pool_and_enforces_ratio():
    n, k = 20, 3
    risk = lambda s: float(sum(s)) ** 2 + 1.0  # noqa: E731  larger index sums = worse
    named = {"spread": (0, 1, 2), "dyn_opt": (3, 4, 5)}
    w = db.select_worst(risk, named, n, k, 500, 41300, 1.5)
    assert w["pool_size"] == 502 and w["risk"] == max(risk(tuple(w["subset"])), 0) and w["ratio_to_median"] >= 1.5
    again = db.select_worst(risk, named, n, k, 500, 41300, 1.5)
    assert again["subset"] == w["subset"]  # fixed pool: reproducible
    assert w["risk"] >= risk((17, 18, 19)) * 0.5 and w["risk"] > risk((3, 4, 5))
    assert db.select_worst(risk, {**named, "bad": (17, 18, 19)}, n, k, 500, 41300, 1.5)["risk"] >= risk((17, 18, 19))
    with pytest.raises(db.WorstLayoutCheckFailed, match="pool median"):
        db.select_worst(lambda s: 1.0, named, n, k, 50, 41300, 1.5)  # constant risk: no layout is 1.5x the median
    assert np.isfinite(db.select_worst(lambda s: float("inf") if 0 in s else 1.0 + sum(s), named, n, k, 50, 1, 1.0)["risk"])  # non-finite excluded


def test_history_bin0_is_filled_and_flagged():
    S = object.__new__(db.Study)
    S.T, S.nb, S.REF = 28800.0, 6, np.ones(db.NP)
    obs_t = np.arange(9000.0, 28801.0, 1800.0)  # t_exclude 7200 s: nothing in bin 0 (0-4800 s)
    assert S.acausal_bins(obs_t) == [0]
    n_site = 3
    ot = np.repeat(obs_t, n_site)
    rng = np.random.default_rng(0)
    A = rng.uniform(0.0, 1.0, (len(ot), db.NP * S.nb))
    y, sig = A @ rng.uniform(0.5, 1.5, db.NP * S.nb), np.full(len(ot), 0.1)
    for m in db.HIST_MODELS:
        th = S._history_estimate(m, A, y, sig, ot, 1.0, 1.0)
        assert th.shape == (db.NP, 6) and np.abs(th[:, 1]).max() > 0
        np.testing.assert_array_equal(th[:, 0], th[:, 1])  # bin 0 = bin-1 estimate in BOTH arms
    assert S.acausal_bins(np.arange(1800.0, 28801.0, 1800.0)) == []  # with early observations nothing is filled


def test_v2_summarise_gates_controls():
    cfg = {**db.load_cfg("configs/dyn08_v2.json"), "ks": [3], "noise": [0.1]}
    rng = np.random.default_rng(1)
    rows = []
    layouts = ["spread", "worst", "dyn_opt", "lowpop"] + [f"{g}#{i}" for g in db.CTL_GROUPS for i in range(4)] + [f"random#{i}" for i in range(3)]
    for e in range(24):
        for lay in layouts:
            base = {"worst": 2.0, "lowpop": 1.0}.get(lay, 1.0) * np.exp(rng.normal(0, 0.1)) * (1 + 0.05 * (e % 6))
            for m in ("M1", "M2"):
                rows.append({"episode": e, "wind": f"w{e % 6}", "valid": True, "k": 3, "noise": 0.1, "layout": lay, "model": m, "jpop": float(base)})
    res = db.summarise(rows, cfg)
    assert res["control_a_required"] == 1 and res["control_a_passed"] and res["control_b_passed"] and res["controls_all_passed"]
    assert [c["gating"] for c in res["reported_layouts"]] == [None]
    assert any(c["contrast"].startswith("dyn_opt-random") and c["ci_method"].startswith("two-level") for c in res["summary"])
    assert all("[this layout only]" in c["contrast"] for c in res["summary"] if c["contrast"].startswith("dyn_opt-spread"))
    for r in rows:  # worst no longer worse than spread: control (a') fails
        if r["layout"] == "worst":
            r["jpop"] = 1.0
    assert db.summarise(rows, cfg)["controls_all_passed"] is False


def test_cost_model_requires_measured_constants_and_adds_up():
    cfg = db.load_cfg()
    assert "error" in db.cost_model({**cfg, "cost": {**cfg["cost"], "fit_M2_ms": None}})
    m = db.cost_model(cfg)
    assert np.isclose(sum(m["parts_core_h"].values()), m["total_core_h"]) and m["n_episodes"]["test"] == 192 and m["n_response_sets"] == 30
    assert np.isclose(m["total_with_contingency"], m["total_core_h"] * cfg["cost"]["contingency"])
    assert m["total_with_contingency"] < 45.0  # the budget this design was sized for
    twice = db.cost_model({**cfg, "cost": {**cfg["cost"], "solve_500_dt50_s": 2 * cfg["cost"]["solve_500_dt50_s"]}})
    assert twice["total_core_h"] > m["total_core_h"]

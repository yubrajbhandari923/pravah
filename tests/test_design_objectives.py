"""Tests for the generalised design machinery (design.py): full parameter vector (emission + inflow), component
objectives, categories/constraints, spatial-block held-out evaluation.  Real forward solves on a tiny 20x20 problem
(T=3600 s, output 600 s); no solver mocks."""
import dataclasses
import itertools

import numpy as np
import pytest

from pravah import design
from pravah.attribution import (base_parts, contributions, external_fraction, external_part)
from pravah.config import (Boundary, Domain, GaussianSource, InflowBasis, Initial, Scenario, StudyRegion, Times,
                           Transport, WindSegment)
from pravah.design import (Candidates, Constraints, DesignOptions, InfeasibleError, assert_disjoint,
                           attribution_scales, build_case, candidate_grid, evaluate_layouts, fit_layout,
                           greedy_trace, heldout_blocks, infeasible_budgets, objective_value, random_layouts,
                           score_layout, select_layout, spread_layout)
from pravah.inverse import InferenceOptions, ResponseCache, fit_sources, reconstruct_linear
from pravah.metrics import field_mse, population_weight
from pravah.observations import Observations, _wrap

DOM = Domain(10_000.0, 10_000.0, 20, 20)
SRC = (GaussianSource("L1", 4500.0, 5000.0, 700.0), GaussianSource("L2", 6500.0, 6500.0, 700.0),
       GaussianSource("E1", 700.0, 2500.0, 700.0, group="external"))
BASES = (InflowBasis("west", ("west",)), InflowBasis("south", ("south",)))
STUDY = StudyRegion(3000.0, 3000.0, 9000.0, 9000.0)
CANDS = candidate_grid(DOM, 3)
TH = {"a": np.array([1.0e5, 1.4e5, 1.2e5, 0.03, 0.02]), "b": np.array([1.5e5, 0.8e5, 0.9e5, 0.02, 0.04]),
      "c": np.array([0.8e5, 1.1e5, 1.6e5, 0.05, 0.01]), "d": np.array([1.2e5, 1.2e5, 0.7e5, 0.01, 0.03])}


def scen(u=1.0, v=0.3, sources=SRC, bases=BASES) -> Scenario:
    return Scenario(domain=DOM, sources=sources, transport=Transport(wind=(WindSegment(0.0, u, v),)),
                    boundary=Boundary("open", 0.02, bases), initial=Initial("uniform", 0.01),
                    times=Times(duration=3600.0, dt=120.0, output_interval=600.0), study=STUDY)


INF = scen()
W = population_weight(DOM)
OPT = DesignOptions(noise_std=0.05, alpha=0.01, weight=W, t_exclude=1200.0)


@pytest.fixture(scope="module")
def cache():
    return ResponseCache()


def make(cache, name, role, truth_scn, theta, seed, opt=OPT, cands=CANDS, n_noise=2, comps=True):
    resp = cache.get(truth_scn)
    tc = contributions(resp, theta, base_parts(truth_scn, cache=cache))
    truth_components = {"local": tc["local"], "external": external_part(tc)} if comps else None
    return build_case(name, role, truth_scn, theta, INF, cands, opt, seed, n_noise, cache,
                      truth_field=resp.field(theta), truth_components=truth_components), tc


@pytest.fixture(scope="module")
def cases(cache):
    out = {}
    for name, role, scn, key, seed in (("d1", "design", INF, "a", 11), ("d2", "design", scen(0.7, 0.6), "b", 12),
                                       ("t1", "test", scen(1.2, 0.0), "c", 101), ("t2", "test", INF, "d", 102)):
        out[name], out["tc_" + name] = make(cache, name, role, scn, TH[key], seed)
    return out


@pytest.fixture(scope="module")
def opt_s(cases):
    return dataclasses.replace(OPT, attribution_scales=attribution_scales([cases["d1"], cases["d2"]]))


# ---------------------------------------------------------------- parameters
def test_full_parameter_vector_shapes(cases):
    c = cases["d1"]
    assert c.n_params == 5 and c.n_src == 3 and c.A_site.shape[2] == 5 and c.G.shape == (5, 5)
    assert set(c.objectives) == {"total", "local", "external"} and c.q_true.shape == (5,)
    assert c.unmodelled_mass == 0.0


def test_parameter_match_by_name_and_kind(cache):
    resp_q = TH["a"]
    with pytest.raises(ValueError, match="name and kind"):  # inflow bases renamed in the truth
        build_case("x", "design", scen(bases=(InflowBasis("west2", ("west",)), BASES[1])), resp_q, INF, CANDS, OPT, 1,
                   1, cache)
    with pytest.raises(ValueError, match="entry per truth parameter"):
        build_case("x", "design", INF, resp_q[:3], INF, CANDS, OPT, 1, 1, cache)
    with pytest.raises(ValueError, match="at least as many"):
        build_case("x", "design", scen(bases=BASES[:1]), resp_q[:4], INF, CANDS, OPT, 1, 1, cache)
    with pytest.raises(ValueError, match="at least as many"):
        build_case("x", "design", scen(sources=SRC[:2]), resp_q[1:4], INF, CANDS, OPT, 1, 1, cache)


def test_extra_emission_source_with_inflow_bases_omitted_source(cache):
    """Belauri 'omitted source': the truth has an extra emission source while the model has 2 inflow bases."""
    truth = scen(sources=SRC + (GaussianSource("X", 7500.0, 2500.0, 600.0, group="external"),))
    th = np.array([1e5, 1.2e5, 1.1e5, 5e4, 0.03, 0.02])  # [L1, L2, E1, X, west, south]
    c, _ = make(cache, "om", "test", truth, th, 8)
    assert c.n_params == 5 and list(c.truth_index) == [0, 1, 2, 4, 5]
    assert c.unmodelled_mass == pytest.approx(5e4 / 4.7e5)
    f = fit_layout(c, (0, 4, 8), 0, OPT)
    assert set(f["inflow_abs_err"]) == {"west", "south"}
    assert f["inflow_abs_err"]["west"] == pytest.approx(f["q_hat"][3] - 0.03)
    assert f["source_rel_err"] == pytest.approx(np.linalg.norm(f["q_hat"][:3] - th[:3]) / np.linalg.norm(th[:3]))
    # an extra truth inflow basis is unmodelled too (and never counted in the emission mass)
    t2 = scen(bases=BASES + (InflowBasis("north", ("north",)),))
    c2, _ = make(cache, "xb", "test", t2, np.append(TH["a"], 0.04), 9)
    assert c2.unmodelled_mass == 0.0 and list(c2.truth_index) == [0, 1, 2, 3, 4]


def test_extra_truth_parameters_unmodelled_and_per_kind_errors(cache):
    inf = scen(bases=())
    truth = scen(sources=SRC + (GaussianSource("X", 7500.0, 2500.0, 600.0, group="external"),), bases=())
    th = np.array([1e5, 1e5, 1e5, 5e4])
    c = build_case("u", "test", truth, th, inf, CANDS, DesignOptions(noise_std=0.05), 3, 1, cache)
    assert c.n_params == 3 and c.unmodelled_mass == pytest.approx(5e4 / 3.5e5)
    f = fit_layout(c, (0, 4, 8), 0, DesignOptions(noise_std=0.05))
    assert f["inflow_abs_err"] == {} and np.isnan(f["inflow_rmse"])
    # inference with inflow bases: errors reported per kind, in their own units
    c2, _ = make(cache, "k", "design", INF, TH["a"], 5)
    f2 = fit_layout(c2, (0, 4, 8), 0, OPT)
    assert set(f2["inflow_abs_err"]) == {"west", "south"} and len(f2["q_hat"]) == 5
    d = f2["q_hat"][3:] - TH["a"][3:]
    assert f2["inflow_rmse"] == pytest.approx(np.sqrt(np.mean(d ** 2)))
    assert f2["source_rel_err"] == pytest.approx(np.linalg.norm(f2["q_hat"][:3] - TH["a"][:3]) / np.linalg.norm(TH["a"][:3]))


# ---------------------------------------------------------------- fast objectives vs brute force
@pytest.mark.parametrize("name", ["d1", "d2", "t1"])
@pytest.mark.parametrize("layout", [(4,), (0, 8), (1, 3, 5), (0, 2, 6, 8)])
def test_objectives_match_brute_force(cache, cases, opt_s, name, layout):
    case, tc = cases[name], cases["tc_" + name]
    truth_scn = {"d1": INF, "d2": scen(0.7, 0.6), "t1": scen(1.2, 0.0)}[name]
    rep = 1
    fast = fit_layout(case, layout, rep, opt_s)
    y = case.truth_readings[list(layout)] + opt_s.noise_std * case.noise[rep, list(layout)]
    sl = CANDS.sensor_layout(layout)
    obs = Observations(_wrap(y.T, sl, np.array(case.obs_times)), np.full(len(layout), opt_s.noise_std), sl,
                       case.obs_times)
    fit = fit_sources(INF, obs, InferenceOptions(alpha=opt_s.alpha), cache)
    assert np.allclose(fast["q_hat"], fit.q_hat, rtol=1e-8, atol=1e-6)
    resp = cache.get(INF)
    est = contributions(resp, fit.q_hat, base_parts(INF, cache=cache))
    truth = resp.field(case.q_true) if truth_scn is INF else cache.get(truth_scn).field(case.q_true)
    brute_total = field_mse(reconstruct_linear(resp, fit), truth, W, 1200.0)
    brute_loc = field_mse(est["local"], tc["local"], W, 1200.0)
    brute_ext = field_mse(external_part(est), external_part(tc), W, 1200.0)
    s_l, s_e = opt_s.attribution_scales or (1.0, 1.0)
    brute_att = 0.5 * (brute_loc / s_l + brute_ext / s_e)
    for key, brute in (("mse_w", brute_total), ("mse_local", brute_loc), ("mse_external", brute_ext),
                       ("mse_attribution", brute_att)):
        assert abs(fast[key] - brute) <= 1e-8 * brute, key
    for key, brute in (("total", brute_total), ("local", brute_loc), ("external", brute_ext),
                       ("attribution", brute_att)):
        assert abs(objective_value(case, fast["q_hat"], key, opt_s) - brute) <= 1e-8 * brute
    # external fraction: the attribution.external_fraction definition, same weights / window
    fe = external_fraction(est, INF, 1200.0, weights=W)
    ft = external_fraction(tc, truth_scn, 1200.0, weights=W)
    assert fast["external_fraction_est"] == pytest.approx(fe, rel=1e-8)
    assert fast["external_fraction_true"] == pytest.approx(ft, rel=1e-6)
    assert fe is not None and ft is not None
    assert fast["external_fraction_error"] == pytest.approx(fe - ft, abs=1e-6)


def test_total_objective_is_legacy_form_and_components_need_truth(cache, cases):
    c = cases["d1"]
    assert c.objectives["total"].c0 == c.c0 and np.array_equal(c.objectives["total"].G, c.G)
    plain, _ = make(cache, "p", "design", INF, TH["a"], 21, comps=False)
    assert set(plain.objectives) == {"total"} and not plain.has_components
    with pytest.raises(ValueError, match="truth_components"):
        score_layout((0,), [plain], OPT, objective="local")
    with pytest.raises(ValueError, match="attribution_scales"):
        score_layout((0,), [c], OPT, objective="attribution")
    f = fit_layout(c, (0, 4), 0, OPT)
    assert np.isnan(f["mse_attribution"])
    with pytest.raises(ValueError, match="keys"):
        resp = cache.get(INF)
        build_case("z", "design", INF, TH["a"], INF, CANDS, OPT, 1, 1, cache,
                   truth_components={"local": resp.dataarray(resp.base, "l")})


def test_background_and_initial_cancel(cases):
    """With the exact parameters the component errors vanish: local and external compare only the fields the
    parameters generate (the known initial and background parts are identical in model and truth)."""
    c = cases["d1"]
    for k in ("total", "local", "external"):
        assert c.objectives[k](c.q_true) == pytest.approx(0.0, abs=1e-12 * max(c.objectives[k].c0, 1e-30) + 1e-18)


# ---------------------------------------------------------------- scales and role guards
def test_attribution_scales_design_only(cache, cases, opt_s):
    d = [cases["d1"], cases["d2"]]
    s = attribution_scales(d)
    ms = [c.component_ms or {} for c in d]
    assert s == (pytest.approx(np.mean([m["local"] for m in ms])), pytest.approx(np.mean([m["external"] for m in ms])))
    with pytest.raises(ValueError, match="leakage"):
        attribution_scales(d + [cases["t1"]])
    plain, _ = make(cache, "p", "design", INF, TH["a"], 21, comps=False)
    with pytest.raises(ValueError, match="truth_components"):
        attribution_scales([plain])
    with pytest.raises(ValueError, match="leakage"):
        select_layout(CANDS, 2, [cases["d1"], cases["t1"]], Constraints(), opt_s, objective="attribution")
    with pytest.raises(ValueError, match="leakage"):
        score_layout((0,), [cases["t1"]], opt_s, objective="local")
    with pytest.raises(ValueError, match="leakage"):
        evaluate_layouts({"m": [(0,)]}, [cases["d1"]], opt_s)


def test_scales_and_layouts_independent_of_test_cases(cache, cases, opt_s):
    d = [cases["d1"], cases["d2"]]
    lay = select_layout(CANDS, 3, d, Constraints(), opt_s, objective="attribution")
    other, _ = make(cache, "t9", "test", scen(0.5, -0.8), 3 * TH["c"], 777)
    assert_disjoint(d, [other])
    evaluate_layouts({"g": [lay]}, [other, cases["t1"]], opt_s)
    assert attribution_scales(d) == opt_s.attribution_scales
    assert select_layout(CANDS, 3, d, Constraints(), opt_s, objective="attribution") == lay


# ---------------------------------------------------------------- selection
def single_scores(dcs, opt, objective):
    return [score_layout((i,), dcs, opt, objective)["value"] for i in range(len(CANDS))]


@pytest.mark.parametrize("objective", ["total", "local", "external", "attribution"])
def test_greedy_first_pick_is_exhaustive_argmin(cases, opt_s, objective):
    d = [cases["d1"], cases["d2"]]
    tr = greedy_trace(CANDS, 2, d, Constraints(), opt_s, objective)
    singles = single_scores(d, opt_s, objective)
    assert tr.order[0] == int(np.argmin(singles)) and tr.objective == objective
    assert tr.scores[0] == pytest.approx(min(singles), rel=1e-12)
    assert select_layout(CANDS, 2, d, Constraints(), opt_s, objective=objective) == tr.layout


def test_default_objective_is_total(cases, opt_s):
    d = [cases["d1"], cases["d2"]]
    s = score_layout((1, 4), d, opt_s)
    assert s["objective"] == "total" and s["value"] == s["mse_w"] and s["rmse"] == pytest.approx(np.sqrt(s["mse_w"]))
    assert select_layout(CANDS, 3, d, Constraints(), opt_s) == select_layout(CANDS, 3, d, Constraints(), opt_s,
                                                                             objective="total")


def test_attribution_greedy_differs_from_total_and_gap_to_exhaustive(cache):
    # external-dominated truths (weak local emission): the total error is driven by the external part, the
    # normalised attribution objective weighs the (small) local field equally -> different layouts
    d = [make(cache, "e1", "design", INF, np.array([0.3e5, 0.4e5, 3e5, 0.03, 0.02]), 31)[0],
         make(cache, "e2", "design", scen(0.7, 0.6), np.array([0.5e5, 0.2e5, 2e5, 0.02, 0.04]), 32)[0]]
    opt_s = dataclasses.replace(OPT, attribution_scales=attribution_scales(d))
    report = []
    differs = False
    for budget in (1, 2, 3):
        gt = greedy_trace(CANDS, budget, d, Constraints(), opt_s, "total")
        ga = greedy_trace(CANDS, budget, d, Constraints(), opt_s, "attribution")
        differs |= ga.layout != gt.layout
        best = min(itertools.combinations(range(9), budget),
                   key=lambda lay: score_layout(lay, d, opt_s, "attribution")["value"])
        bs = score_layout(best, d, opt_s, "attribution")["value"]
        assert ga.scores[-1] >= bs * (1 - 1e-12)  # greedy can never beat the optimum; the gap is only reported
        report.append(f"budget {budget}: total-greedy {gt.layout}, attribution-greedy {ga.layout} "
                      f"({ga.scores[-1]:.6g}), exhaustive {best} ({bs:.6g}), gap {ga.scores[-1] / bs - 1:.3%}")
    print("\n" + "\n".join(report))
    assert differs, "attribution and total greedy coincide for every budget in this fixture:\n" + "\n".join(report)


# ---------------------------------------------------------------- categories and constraints
CAT = Candidates(CANDS.ids, CANDS.x, CANDS.y, category=tuple("reference" if i in (0, 1, 2, 3) else "interior"
                                                            for i in range(9)), meta={"lon": list(range(9))})


def n_ref(lay, cands=CAT):
    return sum(cands.categories[i] == "reference" for i in lay)


def test_candidates_category_defaults_and_validation():
    assert CANDS.category == ("interior",) * 9 and CANDS.meta is None
    with pytest.raises(ValueError, match="category"):
        Candidates(("a", "b"), np.zeros(2), np.zeros(2), category=("x",))
    with pytest.raises(ValueError):
        Constraints(max_per_category={"reference": -1})


def test_max_per_category_enforced_everywhere(cases, opt_s):
    k = Constraints(max_per_category={"reference": 2})
    for budget in (1, 3, 5):
        for lay in random_layouts(CAT, budget, 40, k, seed=2) + [spread_layout(CAT, budget, k)]:
            assert len(lay) == budget and n_ref(lay) <= 2
        lay = select_layout(CAT, budget, [cases["d1"], cases["d2"]], k, opt_s, objective="attribution")
        assert len(lay) == budget and n_ref(lay) <= 2
    # the cap binds: the unconstrained spread layout uses 2 reference sites, a cap of 1 allows only one
    assert n_ref(spread_layout(CAT, 5)) == 2
    assert n_ref(spread_layout(CAT, 5, Constraints(max_per_category={"reference": 1}))) == 1
    # the cap counts inside the budget: budget 8 needs 3 reference sites when only 5 interior sites exist
    with pytest.raises(InfeasibleError, match="max_per_category"):
        random_layouts(CAT, 8, 2, k, 0)
    with pytest.raises(InfeasibleError):
        spread_layout(CAT, 8, k)
    with pytest.raises(InfeasibleError):
        design.check_feasible(CAT, 8, k)
    f = design.feasible_sites(CAT, k, (0, 1))
    assert set(f) == {4, 5, 6, 7, 8}  # references 2 and 3 are exhausted
    assert set(infeasible_budgets(CAT, k, [1, 5, 7, 8, 9])) == {8, 9}  # 5 interior + 2 reference = 7 sites
    assert infeasible_budgets(CAT, Constraints(), [1, 9]) == {}


def test_max_per_category_zero_forbids_category():
    k = Constraints(max_per_category={"reference": 0})
    assert set(design.feasible_sites(CAT, k)) == {4, 5, 6, 7, 8}


def test_exclusion_radius_respected_and_reported():
    pts = ((CANDS.x[8], CANDS.y[8]),)
    k = Constraints(exclusion_radius_m=4000.0, exclusion_points=pts)
    f = design.feasible_sites(CANDS, k)
    d = np.hypot(CANDS.x[f] - pts[0][0], CANDS.y[f] - pts[0][1])
    assert (d >= 4000.0).all() and 8 not in f and 7 not in f and 5 not in f  # neighbours are 3333 m away
    for lay in random_layouts(CANDS, 4, 20, k, seed=4) + [spread_layout(CANDS, 4, k)]:
        assert set(lay) <= set(f)
    allowed_n = len(f)
    bad = infeasible_budgets(CANDS, k, [1, allowed_n, allowed_n + 1, 9])
    assert set(bad) == {allowed_n + 1, 9}
    assert Constraints(exclusion_radius_m=0.0, exclusion_points=pts).exclusion_radius_m == 0.0
    assert len(design.feasible_sites(CANDS, Constraints(exclusion_radius_m=0.0, exclusion_points=pts))) == 9
    # distance exactly equal to the radius is allowed (>=), like min_separation
    exact = Constraints(exclusion_radius_m=float(np.hypot(CANDS.x[7] - pts[0][0], CANDS.y[7] - pts[0][1])),
                        exclusion_points=pts)
    assert 7 in design.feasible_sites(CANDS, exact)


def test_heldout_blocks_deterministic_and_disjoint():
    g = candidate_grid(DOM, 3)
    a = heldout_blocks(g, 3, 3, 4)
    assert np.array_equal(a[0], [4]) and np.array_equal(a[1], [0, 1, 2, 3, 5, 6, 7, 8])
    assert all(np.array_equal(x, y) for x, y in zip(a, heldout_blocks(g, 3, 3, 4)))
    fine = candidate_grid(DOM, 8)
    for which in (0, (1, 4), [2, 3, 5]):
        inside, rest = heldout_blocks(fine, 2, 3, which)
        assert not set(inside) & set(rest) and len(inside) + len(rest) == len(fine)
        assert (np.diff(inside) > 0).all() and (np.diff(rest) > 0).all()
    all_blocks = [heldout_blocks(fine, 2, 3, w)[0] for w in range(6)]
    assert sorted(np.concatenate(all_blocks).tolist()) == list(range(len(fine)))  # blocks partition the points
    inside, _ = heldout_blocks(fine, 2, 2, 3)  # upper right block
    assert (fine.x[inside] >= fine.x.mean()).all() and (fine.y[inside] >= fine.y.mean()).all()
    with pytest.raises(ValueError):
        heldout_blocks(g, 2, 2, 4)
    with pytest.raises(ValueError):
        heldout_blocks(g, 0, 2, 0)


# ---------------------------------------------------------------- evaluation
def test_evaluate_block_vs_random_components_categories(cache):
    fine = candidate_grid(DOM, 6)
    blk, rest = heldout_blocks(fine, 2, 2, 3)
    block = Candidates(tuple(fine.ids[i] for i in blk), fine.x[blk], fine.y[blk])
    rnd = Candidates(tuple(fine.ids[i] for i in rest[::5]), fine.x[rest[::5]], fine.y[rest[::5]])
    opt = dataclasses.replace(OPT, eval_sites={"block": block, "random": rnd})
    cc = design.Candidates(CAT.ids, CAT.x, CAT.y, CAT.category)
    cs = []
    for name, role, scn, key, seed in (("d1", "design", INF, "a", 11), ("d2", "design", scen(0.7, 0.6), "b", 12)):
        cs.append(make(cache, name, role, scn, TH[key], seed, opt=opt, cands=cc)[0])
    t, _ = make(cache, "t1", "test", scen(1.2, 0.0), TH["c"], 101, opt=opt, cands=cc)
    opt_s = dataclasses.replace(opt, attribution_scales=attribution_scales(cs))
    f = fit_layout(t, (0, 4, 8), 0, opt_s)
    assert set(f["heldout_by_group"]) == {"block", "random"} and np.isfinite(f["heldout_rmse"])
    n_b, n_r = len(block), len(rnd)
    pooled = np.sqrt((n_b * f["heldout_by_group"]["block"] ** 2 + n_r * f["heldout_by_group"]["random"] ** 2)
                     / (n_b + n_r))
    assert f["heldout_rmse"] == pytest.approx(pooled, rel=1e-12)
    # block RMSE brute force: noiseless readings of the estimated and the true field at the block sites
    resp = cache.get(INF)
    est = resp.field(f["q_hat"])
    tr = cache.get(scen(1.2, 0.0)).field(TH["c"])
    from pravah.metrics import sensor_rmse
    assert f["heldout_by_group"]["block"] == pytest.approx(sensor_rmse(est, tr, block.sensor_layout(), t.obs_times),
                                                           rel=1e-9)
    lays = {"greedy": [select_layout(cc, 3, cs, Constraints(max_per_category={"reference": 1}), opt_s,
                                     objective="attribution")],
            "random": random_layouts(cc, 3, 3, Constraints(max_per_category={"reference": 2}), 1)}
    df = evaluate_layouts(lays, [t], opt_s, n_noise=2)
    assert list(df.columns) == design.COLUMNS + ["heldout_block_rmse", "heldout_random_rmse"] \
        + design.COMPONENT_COLUMNS + ["inflow_rmse", "n_reference_sites", "category_counts"]
    assert len(df) == 4 * 2 and (df["n_reference_sites"] <= 2).all()
    assert (df[df["method"] == "greedy"]["n_reference_sites"] <= 1).all()
    row = df.iloc[0]
    assert row["category_counts"].count("=") >= 1 and np.isfinite(row["mse_attribution"])
    assert np.isfinite(df["heldout_block_rmse"]).all() and np.isfinite(df["heldout_random_rmse"]).all()


def test_legacy_columns_unchanged_without_new_features(cache):
    sc = scen(bases=())
    inf = sc
    opt = DesignOptions(noise_std=0.05)
    th = np.array([1e5, 1e5, 1e5])
    c = build_case("l", "test", sc, th, inf, CANDS, opt, 4, 1, cache)
    df = evaluate_layouts({"m": [(0, 4)]}, [c], opt)
    assert list(df.columns) == design.COLUMNS
    assert not c.has_components and c.n_params == c.n_src == 3

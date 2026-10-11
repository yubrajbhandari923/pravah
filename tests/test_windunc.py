"""EXP-019 wind-uncertainty toy: guards (no truth leak into C, pairing, freeze, disjoint seeds) and the maths."""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from pravah import attribution as attr
from pravah import windunc as W
from pravah.config import load_config
from pravah.metrics import field_mse
from pravah.population import pop_weighted_rmse

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def tiny_cfg() -> dict:
    cfg = load_config(CONFIGS / "wind_error_smoke.json")
    cfg["wind_error"]["prior"] = {"direction_deg": [-10, 0, 10], "speed_factor": [1.0]}
    cfg["wind_error"]["alpha"] = 1.0
    return cfg


@pytest.fixture(scope="module")
def st(tmp_path_factory: pytest.TempPathFactory) -> W.Study:
    return W.Study(tiny_cfg(), tmp_path_factory.mktemp("bank"))


@pytest.fixture(scope="module")
def case(st: W.Study) -> W.Case:
    return W.draw_cases("test", 1, 99, 9900, list(st.base_theta), (0.5, 1.5), len(st.cands), len(st.times))[0]


# ---------------------------------------------------------------- helpers
def test_rotation_and_vector_error() -> None:
    u, v = W.rotate_wind(1.0, 0.0, 90.0)
    assert (round(u, 12), round(v, 12)) == (0.0, 1.0)
    u, v = W.rotate_wind(1.0, 0.3, 30.0, 1.2)
    assert math.hypot(u, v) == pytest.approx(1.2 * math.hypot(1.0, 0.3))
    for d in (5, 10, 15, 30):
        assert W.vector_error(d) == pytest.approx(2 * math.sin(math.radians(d) / 2))
        du = np.subtract(W.rotate_wind(1.0, 0.3, d), (1.0, 0.3))
        assert W.vector_error(d) == pytest.approx(np.linalg.norm(du) / math.hypot(1.0, 0.3))
    assert W.vector_error(0, 1.2) == pytest.approx(0.2)
    assert W.vector_error(0, 0.7) == pytest.approx(0.3)


def test_conditions_labels() -> None:
    grid = load_config(CONFIGS / "wind_error.json")["wind_error"]["grid"]
    c = W.conditions(grid)
    assert W.key(0, 1) in c and set(c[W.key(0, 1)]) == {"direction", "speed", "joint"}
    assert sum("joint" in b for b in c.values()) == 9
    assert len(c) == 9 + 5 - 1 + 4  # axes shared with the 3x3 joint block


# ---------------------------------------------------------------- weights
def test_member_weights_sum_and_one_parameter_chi2() -> None:
    assert W.member_weights(np.array([1000.0, 1002.0, 1010.0])).sum() == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    nt, ns, sigma = 6, 3, 0.1
    x = [rng.uniform(0.5, 1.5, (ns, nt)), rng.uniform(0.5, 1.5, (ns, nt))]  # two member operators, one parameter
    truth = 2.0 * x[0]
    y_sn = truth + sigma * rng.standard_normal((ns, nt))
    ens = W.Ensemble(((0.0, 1.0), (10.0, 1.0)), np.zeros((2, ns, nt)), np.stack(x)[..., None], ("emission",), ("S",), ("u",))
    fit = W.fit_ensemble(y_sn.T.reshape(-1), range(ns), ens, W.FitOptions(sigma, 0.0))
    chi2 = []
    for xm in x:
        a = max(float(np.sum(xm * y_sn) / np.sum(xm * xm)), 0.0)  # analytic 1-parameter least squares
        assert fit.thetas[len(chi2), 0] == pytest.approx(a, rel=1e-6)
        chi2.append(float(np.sum((y_sn - a * xm) ** 2) / sigma**2))
    assert fit.chi2 == pytest.approx(chi2, rel=1e-6)
    w = np.exp(-0.5 * (np.array(chi2) - min(chi2)))
    assert fit.weights == pytest.approx(w / w.sum())
    assert fit.weights.sum() == pytest.approx(1.0)
    assert fit.weights[0] > fit.weights[1]  # the member that generated the data wins


# ---------------------------------------------------------------- no truth leak
def _reach(obj: object, seen: set[int] | None = None) -> set[int]:
    """ids of every object reachable from obj through attributes, containers and arrays' bases."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return seen
    seen.add(id(obj))
    if isinstance(obj, dict):
        kids = [*obj.keys(), *obj.values()]
    elif isinstance(obj, (list, tuple, set, frozenset)):
        kids = list(obj)
    elif isinstance(obj, np.ndarray):
        kids = [obj.base] if obj.base is not None else []
    elif hasattr(obj, "__dict__"):
        kids = list(vars(obj).values())
    elif hasattr(obj, "__slots__"):
        kids = [getattr(obj, s) for s in getattr(obj, "__slots__") if hasattr(obj, s)]
    else:
        kids = []
    for k in kids:
        _reach(k, seen)
    return seen


def test_strategy_c_cannot_reach_true_wind(st: W.Study, case: W.Case) -> None:
    assert list(inspect.signature(W.strategy_c).parameters) == ["y", "idx", "ens", "opts"]
    truth_key = W.key(30.0, 1.0)  # outside the declared prior on purpose
    assert truth_key not in st.ens_prior.keys
    world = [st.truth_ens(truth_key), st.bank.resp(truth_key), st.bank.resp(truth_key).g, case,
             st.truth_ens(truth_key).A, st.truth_ens(truth_key).b, st.readings(case, truth_key)]
    idx = (3, 10, 20)
    y = st.observe(case, truth_key, idx)
    args = (y, idx, st.ens_prior, W.FitOptions(st.sigma, st.alpha))
    reach = _reach(args)
    assert all(id(o) not in reach for o in world)
    assert not np.shares_memory(y, st.readings(case, truth_key))
    fit = W.strategy_c(*args)
    assert fit.weights.sum() == pytest.approx(1.0)
    # negative control: the walker does see a truth object when it is smuggled into the ensemble
    leaky = {"ens": st.ens_prior, "oops": st.truth_ens(truth_key)}
    assert id(st.truth_ens(truth_key)) in _reach(leaky)
    # the ensemble is built before, and independent of, any case or truth wind
    assert st.ens_prior.keys == tuple(st.prior)


def test_strategy_c_source_names_no_truth() -> None:
    src = inspect.getsource(W.strategy_c)
    body = src.split('"""')[2]
    assert "truth" not in body and "readings" not in body


# ---------------------------------------------------------------- pairing
def test_shared_noise_across_winds_and_strategies(st: W.Study, case: W.Case) -> None:
    idx = (2, 7, 30)
    k1, k2 = W.key(5.0, 1.0), W.key(-10.0, 1.15)
    n1 = st.observe(case, k1, idx) - st.readings(case, k1)[list(idx)].T.reshape(-1)
    n2 = st.observe(case, k2, idx) - st.readings(case, k2)[list(idx)].T.reshape(-1)
    assert n1 == pytest.approx(n2)  # identical noise realisation for every wind error
    again = W.draw_cases("test", 1, 99, 9900, list(st.base_theta), (0.5, 1.5), len(st.cands), len(st.times))[0]
    assert np.array_equal(again.noise, case.noise) and np.array_equal(again.theta, case.theta)
    assert n1 == pytest.approx(st.sigma * case.noise[list(idx)].T.reshape(-1))


# ---------------------------------------------------------------- quadratic forms == library metrics
def test_gram_forms_match_library_metrics(st: W.Study, case: W.Case) -> None:
    tk = W.key(5.0, 1.0)
    gram = st.gram([tk])
    for ens in (st.ens_prior, st.ens_nominal):
        f = W.Forms(gram, ens.keys, tk, st.kinds, st.groups)
        rng = np.random.default_rng(1)
        thetas = case.theta * rng.uniform(0.6, 1.4, (len(ens.keys), len(case.theta)))
        w = rng.dirichlet(np.ones(len(ens.keys)))
        th_e = (w[:, None] * thetas).reshape(-1)
        est = cast(Any, sum(wm * st.bank.resp(k).field(t) for wm, k, t in zip(w, ens.keys, thetas)))
        truth = st.bank.resp(tk).field(case.theta)
        assert f.mse("unif", "total", th_e, case.theta) == pytest.approx(field_mse(est, truth), rel=1e-8)
        err = np.asarray((est - truth).values)
        assert math.sqrt(f.mse("pop", "total", th_e, case.theta)) == pytest.approx(
            pop_weighted_rmse(err, st.counts, times=st.bank.resp(tk).times), rel=1e-8)
        # attribution.py: external fraction (explicit unit weights: this scenario has no study region)
        parts = {"initial": np.zeros_like(err), "background": np.zeros_like(err)}

        def frac(th: np.ndarray, keys: tuple, wts: np.ndarray) -> float | None:
            c = [attr.contributions(st.bank.resp(k), t, parts) for k, t in zip(keys, th)]
            mix = {tag: cast(Any, sum(wm * ci[tag] for wm, ci in zip(wts, c))) for tag in attr.TAGS}
            return attr.external_fraction(mix, st.scn, weights=np.ones(err.shape[1:]))

        assert f.ext_fraction_est(th_e) == pytest.approx(frac(thetas, ens.keys, w), rel=1e-8)
        assert f.ext_fraction_true(case.theta) == pytest.approx(frac(case.theta[None], (tk,), np.ones(1)), rel=1e-8)


def test_evaluate_rows_complete(st: W.Study, case: W.Case) -> None:
    k = W.key(10.0, 1.0)
    rows = st.evaluate(case, k, (3, 10, 20, 40), st.gram([k]))
    assert [r["strategy"] for r in rows] == ["A", "B", "C"]
    for r in rows:
        for c in ("rmse_total", "rmse_pop", "rmse_heldout", "chi2_per_dof", "chi2_p", "cov90", "ext_frac_err", "flag_any", "relerr_S1", "bias_rel_mean"):
            assert np.isfinite(r[c])
    assert rows[0]["rmse_total"] <= rows[1]["rmse_total"] * 1.5  # oracle is not much worse than a wrong wind


# ---------------------------------------------------------------- guards
def test_freeze_guard(tmp_path: Path) -> None:
    frozen = {"alpha": 1.0, "prior_keys": [[0.0, 1.0]], "layouts": {"3": {"greedy": (1, 2, 3)}}}
    p = tmp_path / "freeze.json"
    h = W.write_freeze(p, frozen)
    assert W.assert_frozen(p, json.loads(json.dumps(frozen))) == h
    with pytest.raises(AssertionError):
        W.assert_frozen(p, {**frozen, "alpha": 2.0})
    with pytest.raises(AssertionError):
        W.assert_frozen(p, {**frozen, "prior_keys": [[0.0, 1.0], [10.0, 1.0]]})


def test_disjoint_seeds() -> None:
    mk = lambda role, seed, ns: W.draw_cases(role, 2, seed, ns, [1.0, 2.0], (0.5, 1.5), 4, 3)  # noqa: E731
    d, v, t = mk("design", 1, 100), mk("validation", 2, 200), mk("test", 3, 300)
    W.assert_disjoint_cases(d, v, t)
    with pytest.raises(ValueError, match="noise_seed"):
        W.assert_disjoint_cases(d, mk("test", 3, 100))
    with pytest.raises(ValueError, match="seed"):
        W.assert_disjoint_cases(d, mk("test", 1, 300))


CTRL = ("alpha_sensitivity", "offset_prior", "model_error", "base_wind_45", "base_wind_90")  # model_error_baseline shares seeds on purpose


def test_profile_configs_use_disjoint_seed_ranges() -> None:
    seen: list[tuple[str, set[int]]] = []
    names = ("wind_error", "wind_error_pilot", "wind_error_smoke", *(f"wind_error_ctrl_{c}{x}" for c in CTRL for x in ("", "_smoke")))
    for name in names:
        we = load_config(CONFIGS / f"{name}.json")["wind_error"]
        n = {"design": we["n_design"], "validation": we["n_validation"], "test": we["n_test"]}
        s = we["seeds"]
        ids = {s[r] * 1000 + k for r in n for k in range(n[r])} | {s[f"{r}_noise"] + k for r in n for k in range(n[r])}
        for other, o in seen:
            assert not ids & o, f"{name} shares seeds with {other}"
        seen.append((name, ids))
    for name in names:
        s = load_config(CONFIGS / f"{name}.json")["wind_error"]["seeds"]
        assert len(set(s.values())) == len(s)
    base = "wind_error_ctrl_model_error"
    for sfx in ("", "_smoke"):
        assert load_config(CONFIGS / f"{base}_baseline{sfx}.json")["wind_error"]["seeds"] == load_config(CONFIGS / f"{base}{sfx}.json")["wind_error"]["seeds"]


def test_prior_declared_and_truth_extremes_outside() -> None:
    we = load_config(CONFIGS / "wind_error.json")["wind_error"]
    assert max(abs(d) for d in we["prior"]["direction_deg"]) == 20 and 30 in we["grid"]["direction_deg"]
    assert we["alpha"] is None  # selected on validation scenarios at run time


# ---------------------------------------------------------------- statistics
def test_paired_bootstrap() -> None:
    rows = [{"cond": "c", "strategy": s, "case": f"k{i}", "m": base + (1.0 if s == "B" else 0.0)}
            for i in range(8) for s, base in (("B", i * 0.1), ("C", i * 0.1))]
    out = W.paired_bootstrap(pd.DataFrame(rows), "C", "B", "m", ["cond"], n_boot=200)
    r = out.iloc[0]
    assert r.diff_mean == pytest.approx(-1.0) and r.ci_lo == pytest.approx(-1.0) and r.ci_hi == pytest.approx(-1.0) and r.excludes_0
    assert "percentile bootstrap" in r.interval


def test_confounding_demo_zero_rotation_is_null(st: W.Study) -> None:
    out = W.wind_amplitude_confounding(st, (3, 10, 20, 40), st.base_theta, [0.0])
    assert out["rotations"][0]["signal_change_sigma"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- EXP-023 controls (opt-in; default unchanged)
def ctl_study(tmp: Path, **control: Any) -> W.Study:
    cfg = tiny_cfg()
    cfg["wind_error"]["control"] = {"name": "t", **control}
    return W.Study(cfg, tmp)


def test_default_run_unchanged(tmp_path: Path) -> None:
    """The default smoke config still writes the freeze hash and the rows recorded before the controls existed."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("wind_error_script", root / "scripts" / "wind_error.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = load_config(CONFIGS / "wind_error_smoke.json")
    assert "control" not in cfg["wind_error"] and cfg["experiment"] == "EXP-019"
    mod.run(cfg, tmp_path / "out", tmp_path / "work", False)
    assert json.loads((tmp_path / "out" / "freeze.json").read_text())["hash"] == "0a38565582788d62"  # hash of the pre-controls smoke run
    ref = root / "results" / "phase3-wind-error-toy" / "smoke" / "raw"
    if ref.exists():  # untracked local artefact: compare every column except wall-clock when present
        for f in sorted(ref.glob("*.csv")):
            a, b = pd.read_csv(f).drop(columns="runtime_s"), pd.read_csv(tmp_path / "out" / "raw" / f.name).drop(columns="runtime_s")
            pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_offset_prior_has_no_truth_member() -> None:
    cfg = load_config(CONFIGS / "wind_error_ctrl_offset_prior.json")
    we = cfg["wind_error"]
    prior = set(W.prior_keys(we["prior"]["direction_deg"], we["prior"]["speed_factor"]))
    truths = set(W.conditions(we["grid"])) | {W.key(d, 1 + p / 100) for d, p in we["validation"]["winds"]}
    assert truths and not truths & prior
    assert min(abs(t[0] - m[0]) for t in truths for m in prior) == 2.5
    assert we["control"]["off_prior"]


def test_assert_off_prior_raises(tmp_path: Path) -> None:
    st = ctl_study(tmp_path, off_prior=True)  # tiny prior {-10, 0, 10} x {1.0}
    assert st.assert_off_prior([W.key(5.0)]) == []
    with pytest.raises(ValueError, match="prior members"):
        st.assert_off_prior([W.key(10.0)])
    assert ctl_study(tmp_path).assert_off_prior([W.key(10.0)]) == [W.key(10.0)]  # reported, not fatal, without the flag


def test_k_mismatch_truth_path(tmp_path: Path, st: W.Study, case: W.Case) -> None:
    m = ctl_study(tmp_path, truth_K_factor=1.2)
    k = W.key(10.0)
    assert m.truth_bank is not m.bank and m.bank.nominal.transport.K == m.truth_bank.nominal.transport.K / 1.2
    assert st.truth_bank is st.bank and st.known_ens(k) is st.truth_ens(k)  # default: A's model is the truth operator
    assert np.array_equal(m.known_ens(k).A, st.truth_ens(k).A)  # estimators keep the nominal K
    assert not np.allclose(m.truth_ens(k).A, st.truth_ens(k).A, rtol=1e-3)  # truth readings come from K x 1.2
    assert not np.allclose(m.readings(case, k), st.readings(case, k), rtol=1e-3)
    # Gram: truth fields come from the truth bank
    gram = m.gram([k])
    f = W.Forms(gram, m.ens_nominal.keys, k, m.kinds, m.groups)
    est = m.bank.resp(m.nominal).field(case.theta)
    truth = m.truth_bank.resp(k).field(case.theta)
    assert f.mse("unif", "total", case.theta, case.theta) == pytest.approx(field_mse(est, truth), rel=1e-8)
    # a model error makes the zero-wind-error fit inconsistent with the data: chi2/dof rises
    idx = (3, 10, 20, 40)
    k0 = W.key(0.0)
    r0 = st.evaluate(case, k0, idx, st.gram([k0]), ("B",))[0]
    r1 = m.evaluate(case, k0, idx, m.gram([k0]), ("B",))[0]
    assert r1["chi2_per_dof"] > r0["chi2_per_dof"]


def test_base_wind_rotation(tmp_path: Path, st: W.Study) -> None:
    r = ctl_study(tmp_path, base_rotation_deg=90.0)
    w0, w1 = st.scn.transport.wind[0], r.scn.transport.wind[0]
    assert (w1.u, w1.v) == pytest.approx(W.rotate_wind(w0.u, w0.v, 90.0))
    # wind keys are relative to the rotated base: key (10 deg) = 100 deg from the original wind
    wk = W.wind_scenario(r.scn, W.key(10.0)).transport.wind[0]
    assert (wk.u, wk.v) == pytest.approx(W.rotate_wind(w0.u, w0.v, 100.0))
    assert r.nominal == W.key(0.0) and (st.scn.transport.wind == W.build_scenario(tiny_cfg()).transport.wind)


def test_alpha_override_per_strategy(st: W.Study, case: W.Case) -> None:
    k, idx = W.key(10.0), (3, 10, 20, 40)
    gram = st.gram([k])
    base = st.evaluate(case, k, idx, gram)
    try:
        st.alpha_by = {"A": st.alpha, "B": st.alpha, "C": st.alpha}
        same = st.evaluate(case, k, idx, gram)
        st.alpha_by = {"B": 100.0}
        diff = st.evaluate(case, k, idx, gram)
    finally:
        st.alpha_by = {}
    assert [r["rmse_total"] for r in same] == [r["rmse_total"] for r in base]
    assert diff[1]["rmse_total"] != base[1]["rmse_total"] and diff[0]["rmse_total"] == base[0]["rmse_total"]

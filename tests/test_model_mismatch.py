"""E13 contract (docs/research/model-mismatch-design.md §2, §11): truth/inverse separation, physical truth-side sources,
noise/bias processes as declared, missing data masked and never imputed."""
import numpy as np
import pytest

from pravah import benchmarks as B
from pravah import model_mismatch as M
from pravah.experiment_config import ExperimentConfig, Split, Winds
from pravah.experiments import cross_objective as X
from pravah.sensitivity import anomaly_fields


@pytest.fixture(scope="module")
def tiny():
    c = ExperimentConfig.default("EXP-E13-test")
    c.model.grid, c.model.spinup_h = (16, 16), 6
    c.design.n_sources = (2,)
    c.split = Split(train=(1, 1), validation=(2, 2), test=(3, 3))
    c.winds = Winds(train=["veering"], heldout=c.winds.heldout)
    sc = B.generate_scenario(c, 0, 2)
    F, G3 = X.responses(sc)
    sol = M.truth_solves(c, 0, 2, sc.truth.source_xy, M.SPECS)
    tfields = {k: M.unit_field(c, *v, sc.truth.winds) for k, v in sol.items()}
    Gs = {k: X.site_responses(sc, f[None])[:, :, 0] for k, f in tfields.items()}
    return c, sc, F, G3, tfields, Gs


def test_sweep_solve_keys():
    c = ExperimentConfig.default("EXP-E13-test")
    sxy = np.array([[3000., 3000.], [7000., 6000.]])
    keys = {k for k, _ in M.truth_solves(c, 0, 2, sxy, M.SWEEP_SPECS)}
    assert keys == {"shift800", "missing", "diurnal0.5"}   # design §10: 2K + 1 truth-side solves per scenario


def test_amp_none_and_linearity(tiny):
    c, sc, F, *_ = tiny
    S = B.source_basis(tuple(c.model.grid), tuple(c.model.domain_m), sc.truth.source_xy[0], 400.) / c.model.mixing_depth_m
    T = len(sc.truth.winds)
    np.testing.assert_array_equal(anomaly_fields(c.model, sc.truth.winds, S, T), F[0])
    f2 = anomaly_fields(c.model, sc.truth.winds, S, T, amp=np.full(T, 2.))
    assert np.allclose(f2, 2 * F[0], rtol=1e-6, atol=1e-12 * np.abs(F[0]).max())


def test_oracle_world_is_e07_and_truth_untouched(tiny):
    c, sc, F, G3, tf_, Gs = tiny
    f0, st0 = sc.truth.field.copy(), sc.truth.site_truth.copy()
    sig = X.site_sigma(sc, False)
    T = len(sc.truth.winds)
    tf, Y = M.world(sc, "oracle", tf_, Gs, sig, 1)
    np.testing.assert_array_equal(Y, X.readings(sc, sig, "white", 1, T))
    np.testing.assert_array_equal(tf, sc.truth.field.reshape(T, -1) - 5.0)
    for name in M.SPECS:
        tfm, Ym = M.world(sc, name, tf_, Gs, sig, 1)
        if name != "oracle":
            assert not np.allclose(np.nan_to_num(Ym), Y), name  # the world really differs
    np.testing.assert_array_equal(sc.truth.field, f0)
    np.testing.assert_array_equal(sc.truth.site_truth, st0)
    # the estimator inputs come from nominal F only: same for every spec by construction (no spec argument exists)
    F2, G2 = X.responses(sc)
    np.testing.assert_array_equal(F2, F)


def test_truth_side_sources_physical(tiny):
    c, sc, F, G3, tf_, Gs = tiny
    dx, dy = np.asarray(c.model.domain_m) / c.model.grid
    for d in (400., 800.):
        cs = M.shifted_centres(c, 0, 2, sc.truth.source_xy, d)
        assert np.allclose(np.linalg.norm(cs - sc.truth.source_xy, axis=1), d, atol=1e-6) or (cs >= 1200).all()
        for k in range(2):
            phi = B.source_basis(tuple(c.model.grid), tuple(c.model.domain_m), cs[k], 400.)
            assert phi.sum() * dx * dy == pytest.approx(1.0)
    for key, f in tf_.items():
        assert f.min() >= -1e-9 * f.max(), key   # upwind: no negative concentrations from a positive source
    xy, a = M.extra_source(c, 0, 2, sc.truth.source_xy)
    assert np.linalg.norm(sc.truth.source_xy - xy, axis=1).min() >= c.sources.min_separation_m and a > 0
    g = M.diurnal_amp(0, 2, 1000)
    assert g.min() >= .5 - 1e-12 and g.max() <= 1.5 + 1e-12 and abs(g.mean() - 1) < .01


def test_noise_and_bias_processes(tiny):
    c, sc, F, G3, tf_, Gs = tiny
    sig = X.site_sigma(sc, False)
    T = len(sc.truth.winds)
    st = np.asarray(sc.truth.site_truth)[:, :T]
    _, Yw = M.world(sc, "oracle", tf_, Gs, sig, 0)
    e = M.world(sc, "ar1", tf_, Gs, sig, 0)[1] - st
    assert abs(e.std() / sig[0] - 1) < .1
    assert abs(np.corrcoef(e[:, 1:].ravel(), e[:, :-1].ravel())[0, 1] - .8) < .05
    b = M.world(sc, "sitebias1", tf_, Gs, sig, 0)[1] - Yw
    assert np.allclose(b, b[:, :1]) and abs(b[:, 0].std() - 1) < .2   # constant in time, sd 1
    cm = M.world(sc, "common1", tf_, Gs, sig, 0)[1] - Yw
    assert np.allclose(cm, 1.0)
    dr = M.world(sc, "drift1", tf_, Gs, sig, 0)[1] - Yw
    assert np.allclose(dr[:, 0], 0) and (dr[:, -1] > .45).all() and (np.diff(dr, axis=1) >= -1e-12).all()
    # background offset and common-mode bias are the same data at the sensors (H3), but not the same truth
    tb, Yb = M.world(sc, "bgoff1", tf_, Gs, sig, 0)
    np.testing.assert_allclose(Yb, Yw + 1.0)
    assert np.allclose(tb - M.world(sc, "common1", tf_, Gs, sig, 0)[0], 1.0)


def test_dropout_masked_never_imputed(tiny):
    c, sc, F, G3, tf_, Gs = tiny
    T = len(sc.truth.winds)
    m = M.outage_mask(0, 2, 2000, 400, .5)
    assert abs((~m).mean() - .5) < .05
    sig = X.site_sigma(sc, False)
    _, Y = M.world(sc, "drop0.5", tf_, Gs, sig, 0)
    obs = np.isfinite(Y)
    assert 0.3 < (~obs).mean() < 0.7
    _, Yw = M.world(sc, "oracle", tf_, Gs, sig, 0)
    np.testing.assert_array_equal(Y[obs], Yw[obs])
    # masked MAP == MAP on the observed rows only (zeroed rows contribute nothing)
    S = [10, 50, 90]
    A = 1e6 * G3[S] / sig[S][:, None, None]
    z = ((Y[S] - 5) / sig[S][:, None])[..., None]
    issue = np.arange(30, T - 1)
    u, C = M.map_masked(A, z, issue)
    for t in (0, 20):
        h = np.arange(issue[t] + 1)
        Ao = np.concatenate([A[i][h][obs[S[i]][h]] for i in range(3)])
        zo = np.concatenate([z[i, h, 0][obs[S[i]][h]] for i in range(3)])
        N = np.eye(2) + Ao.T @ Ao
        np.testing.assert_allclose(u[t, :, 0], np.linalg.solve(N, Ao.T @ zo), rtol=1e-8)
        np.testing.assert_allclose(C[t], np.linalg.inv(N), rtol=1e-8)
    assert np.isfinite(u).all()


def test_bg_column_recovers_offset(tiny):
    c, sc, F, G3, tf_, Gs = tiny
    sig = X.site_sigma(sc, False)
    S = list(range(0, 256, 9))
    T = len(sc.truth.winds)
    A = 1e6 * G3[S] / sig[S][:, None, None]
    Y = np.stack([M.world(sc, "bgoff1", tf_, Gs, sig, r)[1][S] for r in range(2)], 2)
    z = (Y - 5) / sig[S][:, None, None]
    u, _ = M.map_masked(A, z, [T - 1], None, bg_col=M.BG_PRIOR_SD / sig[S])
    assert u[0, -1].mean() == pytest.approx(1.0, abs=.05)     # offset recovered (units of BG_PRIOR_SD = 1 ug m^-3)
    np.testing.assert_allclose(u[0, :2].mean(1), sc.truth.strengths_ug_s / 1e6, rtol=.1)


def test_field_silent_failure_rule_needs_oracle_relative_condition():
    import importlib.util, pandas as pd
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sp = importlib.util.spec_from_file_location("e13", root / "scripts/e13_model_mismatch.py")
    E13 = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(E13)
    s, _, cex = E13.summarise(pd.read_csv(root / "reports/quarto/tables/e13-local-rows.csv"))
    f = cex[cex.kind == "field_silent_failure"]
    assert len(f) == 29 and (f.D_forecast > 1.5).all() and (f.D_fit < 1.1).all()   # audit: 65 unconditioned, 29 oracle-relative

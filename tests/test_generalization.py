"""E16 generalization: cell validity and units, paired seeds across cells, cache-name isolation, the exact background
share, split isolation / causality of both estimators, and the predefined verdict statistics."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pravah import benchmarks as B
from pravah.experiment_config import ExperimentConfig, Split
from pravah.experiments import cross_objective as X
from pravah.experiments import forecast_design as FD
from pravah.experiments import generalization as G

_p = Path(__file__).resolve().parents[1] / "scripts" / "e16_generalization.py"
_spec = importlib.util.spec_from_file_location("e16_driver", _p)
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


@pytest.mark.parametrize("cell", list(G.CELLS))
def test_cells_valid_and_applied(cell):
    cfg = G.apply_cell(ExperimentConfig.default("t"), G.CELLS[cell])
    for key, v in G.CELLS[cell].items():
        sec, attr = key.split(".")
        assert getattr(getattr(cfg, sec), attr) == v
        assert G.RANGES[key][2]  # every axis field has a declared unit


@pytest.mark.parametrize("ov", [{"model.removal_per_s": 1 / 600}, {"model.diffusivity_m2_s": 0.0},
                                {"sources.background_ug_m3": -1.0}, {"model.advection_scheme": "lax"}, {"model.grid": (8, 8)}])
def test_rejects_invalid(ov):
    with pytest.raises(ValueError):
        G.validate_cell(ov)


def test_units_and_dimensionless_groups():
    c = ExperimentConfig.default("t")
    c.model.grid = (32, 32)
    nom = G.dimensionless(c)
    assert nom["K_eff_m2_s"] > 10 * c.model.diffusivity_m2_s  # upwind at 32^2: numerical diffusion dominates
    lo = G.dimensionless(G.apply_cell(ExperimentConfig.default("t"), G.CELLS["diff_lo"]))
    assert lo["K_eff_m2_s"] == 50.0
    hi = G.dimensionless(G.apply_cell(ExperimentConfig.default("t"), G.CELLS["loss_hi"]))
    assert np.isclose(hi["Da"] / G.dimensionless(ExperimentConfig.default("t"))["Da"], 6.0)  # 12 h -> 2 h e-folding


@pytest.mark.parametrize("pre", ["local", "sweep"])
def test_paired_seeds_across_cells(pre):
    """Same seed -> same strengths, winds and noise/placement streams in every cell; layouts too except src_sparse."""
    for K in D.BASE[pre]["Ks"]:
        for seed in D.BASE[pre]["seeds"][:3]:
            ref = B.scenario_drivers(D.cfg_of(pre, "nominal"), seed, K)
            for cell in D.CELLS[pre]:
                cfg = D.cfg_of(pre, cell)
                d = B.scenario_drivers(cfg, seed, K)
                np.testing.assert_array_equal(d[1], ref[1])
                np.testing.assert_array_equal(d[3], ref[3])
                assert cfg.seeds.streams(seed) == D.cfg_of(pre, "nominal").seeds.streams(seed)
                if cell == "src_sparse":
                    sep = np.linalg.norm(d[2][:, None] - d[2][None], axis=2)[np.triu_indices(K, 1)]
                    assert sep.min() >= 3000.0
                else:
                    np.testing.assert_array_equal(d[2], ref[2])


@pytest.mark.parametrize("pre", ["local", "sweep"])
def test_cache_names_isolate_cells(pre):
    """Field caches differ between cells whose physics differ (no stale reuse); unique run ids across the matrix."""
    names = {}
    for cell in D.CELLS[pre]:
        if cell in D.SHARE:
            continue
        names[cell] = {n for n, _ in D.E7.scenario_fields(D.scale_name(pre, cell), D.WIND, 2, 0)}
    cells = list(names)
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            assert not names[a] & names[b], (a, b)
    jobs = [(c, K, s) for c in D.CELLS[pre] for K in D.BASE[pre]["Ks"] for s in D.BASE[pre]["seeds"]]
    assert len(set(jobs)) == len(jobs)


def test_sweep_nominal_reuses_e07_fields():
    """64^2 nominal = E07/E08 sweep model: identical cache names, so the cluster's E07/E08 fields are reused."""
    a = D.E7.scenario_fields(D.scale_name("sweep", "nominal"), D.WIND, 4, 3)
    b = D.E7.scenario_fields("sweep", D.WIND, 4, 3)
    assert [n for n, _ in a] == [n for n, _ in b]


def _tiny(bg=5.0):
    c = ExperimentConfig.default("t")
    c.model.grid, c.model.spinup_h = (16, 16), 2
    c.split = Split((1, 1), (2, 2), (3, 3))
    c.sources.background_ug_m3 = bg
    c.validate()
    return c


def test_background_share_is_exact():
    """bg_hi reuses nominal fields: truth(bg2) = truth(bg1) + (bg2 - bg1), sites/readings consistent."""
    s1 = B.generate_scenario(_tiny(5.0), 0, 2)
    s2 = B.generate_scenario(_tiny(20.0), 0, 2)
    np.testing.assert_allclose(s2.truth.field, s1.truth.field + 15.0, rtol=0, atol=1e-12)
    sh = B.generate_scenario(_tiny(20.0), 0, 2, truth_field=s1.truth.field + 15.0)
    np.testing.assert_allclose(sh.noise_sigma, s2.noise_sigma, rtol=1e-12)


def test_estimators_causal_and_designs_training_only():
    rng = np.random.default_rng(0)
    k, H, K = 3, 120, 2
    A, z = rng.normal(size=(k, H, K)), rng.normal(size=(k, H))
    issue = np.arange(40, 100)
    for W in G.ESTIMATORS.values():
        u = FD.estimate(A, z, issue, W)
        z2 = z.copy()
        z2[:, 70:] += 100.0  # future readings
        u2 = FD.estimate(A, z2, issue, W)
        np.testing.assert_array_equal(u[issue < 70], u2[issue < 70])
    np.testing.assert_allclose(FD.estimate(A, z, issue, 24), X.estimate_series(A, z, issue), atol=1e-12)
    sc = B.generate_scenario(_tiny(), 0, 2)
    G3 = rng.normal(size=(sc.n_candidate_sites, len(sc.truth.winds), 2))
    di = X.design_input(sc, G3, X.site_sigma(sc, False))
    assert di.G3_train.shape[1] == len(sc.split_hours("train"))


def test_verdicts_and_kstar():
    assert G.verdict(.1, .3, .2) == "holds" and G.verdict(-.3, -.1, -.2) == "reversed"
    assert G.verdict(-.1, .3, .1) == "weakened" and G.verdict(-.3, .1, -.1) == "flipped (indicative)"
    rows = []
    for cell in ("nominal", "x"):
        for seed in range(10):
            rows.append(dict(cell=cell, K=2, seed=seed, k=0, strategy="background_only", estimator="-", status="ok", forecast_rmse=1.0))
            for k in G.KS:
                for st in G.STRATEGIES:
                    for e in G.ESTIMATORS:
                        good = {"source_near": 1, "dopt": 2, "coverage": 3, "random": 4}[st] if (cell, e) != ("x", "map24") \
                            else {"source_near": 2, "dopt": 1, "coverage": 3, "random": 4}[st]
                        rows.append(dict(cell=cell, K=2, seed=seed, k=k, strategy=st, estimator=e, status="ok",
                                         forecast_rmse=good * 0.15 / k + 0.001 * seed))
    d = G.with_nrmse(pd.DataFrame(rows))
    vt = G.verdict_table(G.contrast_rows(d)).set_index(["conclusion", "cell"])
    assert vt.loc[("C1_sn_vs_dopt_map24", "nominal"), "verdict"] == "holds"
    assert vt.loc[("C1_sn_vs_dopt_map24", "x"), "verdict"] == "reversed"
    _, ks = G.kendall_table(d)
    assert ks.set_index(["estimator", "cell"]).loc[("map24", "x"), "frac_same_top"] == 0.0
    kk, kv = G.kstar_table(d)
    assert set(kk[kk.strategy == "source_near"].kstar) == {2} and set(kk[kk.strategy == "random"].kstar) == {4}
    assert (kv.verdict == "holds").all()


@pytest.mark.parametrize("pre", ["local", "sweep"])
def test_vanleer_courant_over_every_calendar(pre):
    """Every VanLeer cell satisfies Courant <= 1 over the whole calendar of every seed (dt 75 failed at seeds 2 and 4)."""
    for cell in D.CELLS[pre]:
        cfg = D.cfg_of(pre, cell)
        if cfg.model.advection_scheme != "vanleer":
            continue
        dx = cfg.model.domain_m[0] / cfg.model.grid[0]
        for K in D.BASE[pre]["Ks"]:
            for seed in D.BASE[pre]["seeds"]:
                w = B.scenario_drivers(cfg, seed, K)[3]
                assert np.abs(w).max() * cfg.model.dt_s / dx <= 1.0, (cell, K, seed)

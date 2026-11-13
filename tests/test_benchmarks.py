"""Contract tests for pravah.observations / pravah.benchmarks (T-010). Independent of the implementation.

Contract
- observations.bilinear_sample(c, xy, domain): c[nx, ny] cell-centre values ((i+.5)dx, (j+.5)dy); xy (n, 2) metres.
  Exact at cell centres, bilinear between, constant extension between wall and first centre, ValueError outside the domain.
- observations.noise_field(seed, n_sites, n_hours, sigma, kind, rho=0.8): (n_sites, n_hours); sigma scalar or per-site
  (heteroscedastic) marginal std; kind white|ar1 draw from SeedSequence(seed).spawn(2)[0|1]; same seed -> same array.
- benchmarks.generate_scenario(cfg, replicate, n_sources, *, lattice_n, heteroscedastic) -> Scenario holding the complete
  truth (field, sources, winds, all-candidate readings) privately; Scenario.view(split, site_ids, purpose, kind) is the ONLY
  inference-facing API and returns a read-only InferenceView with no truth, no strengths, no unselected sites.
- Split enforcement: LeakError when test data feed fitting purposes or evaluation touches non-test splits.
"""
import dataclasses

import numpy as np
import pytest

from pravah import benchmarks as B
from pravah import observations as O
from pravah.experiment_config import ExperimentConfig, LeakError, Split


# ---------------------------------------------------------------- observations
def test_sample_exact_at_cell_centres():
    rng = np.random.default_rng(0)
    c = rng.random((8, 8))
    dx = 100.0
    ij = np.array([[0, 0], [3, 5], [7, 7]])
    xy = (ij + 0.5) * dx
    np.testing.assert_allclose(O.bilinear_sample(c, xy, (800.0, 800.0)), c[ij[:, 0], ij[:, 1]], rtol=0, atol=1e-15)


def test_sample_between_cells_is_bilinear():
    c = np.add.outer(2.0 * np.arange(6), 3.0 * np.arange(6))  # linear field: bilinear must be exact
    xy = np.array([[150.0, 250.0], [212.5, 333.0]])           # dx = 100 -> centres at 50, 150, ...
    expect = 2.0 * (xy[:, 0] / 100 - 0.5) + 3.0 * (xy[:, 1] / 100 - 0.5)
    np.testing.assert_allclose(O.bilinear_sample(c, xy, (600.0, 600.0)), expect, atol=1e-12)
    mid = O.bilinear_sample(c, np.array([[100.0, 100.0]]), (600.0, 600.0))  # midpoint of 4 cells = their mean
    assert mid[0] == pytest.approx(c[:2, :2].mean())


def test_sample_wall_band_and_outside():
    c = np.arange(16.0).reshape(4, 4)
    assert O.bilinear_sample(c, np.array([[1.0, 1.0]]), (400.0, 400.0))[0] == c[0, 0]  # constant extension
    for bad in ([-1.0, 10.0], [10.0, 401.0], [400.5, 0.0]):
        with pytest.raises(ValueError):
            O.bilinear_sample(c, np.array([bad]), (400.0, 400.0))


def test_noise_same_seed_identical_different_seed_differs():
    a = O.noise_field(7, 5, 50, 1.0, "white")
    assert a.shape == (5, 50)
    np.testing.assert_array_equal(a, O.noise_field(7, 5, 50, 1.0, "white"))
    assert not np.array_equal(a, O.noise_field(8, 5, 50, 1.0, "white"))
    assert not np.array_equal(a, O.noise_field(7, 5, 50, 1.0, "ar1"))  # separate child streams


def test_noise_heteroscedastic_scales_per_site():
    sig = np.array([0.1, 1.0, 10.0])
    e = O.noise_field(1, 3, 20000, sig, "white")
    np.testing.assert_allclose(e.std(axis=1), sig, rtol=0.03)
    assert (np.abs(e.mean(axis=1)) < 5 * sig / np.sqrt(20000)).all()


def test_noise_ar1_marginal_sigma_and_lag1_correlation():
    e = O.noise_field(3, 4, 40000, 2.0, "ar1", rho=0.8)
    assert e.std() == pytest.approx(2.0, rel=0.03)
    lag1 = np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in e])
    assert lag1 == pytest.approx(0.8, abs=0.02)
    assert abs(np.corrcoef(e[0], e[1])[0, 1]) < 0.03  # spatially independent


def test_noise_rejects_bad_input():
    for kw in (dict(sigma=-1.0), dict(sigma=np.ones(3)), dict(kind="pink"), dict(rho=1.0)):
        args = dict(seed=0, n_sites=4, n_hours=5, sigma=1.0, kind="ar1", rho=0.8) | kw
        with pytest.raises(ValueError):
            O.noise_field(**args)


# ---------------------------------------------------------------- benchmarks
def _cfg():
    cfg = ExperimentConfig.default("EXP-003")
    cfg.model.grid = (16, 16)
    cfg.model.spinup_h = 2
    cfg.split = Split(train=(1, 2), validation=(3, 3), test=(4, 4))
    cfg.design.n_sources = (2,)
    cfg.validate()
    return cfg


@pytest.fixture(scope="module")
def sc():
    return B.generate_scenario(_cfg(), 0, 2, lattice_n=16, heteroscedastic=False)


@pytest.fixture(scope="module")
def sc_het():
    return B.generate_scenario(_cfg(), 0, 2, lattice_n=16, heteroscedastic=True)


def _net(sc, k):
    """First k admissible sites that respect the 1250 m minimum separation (greedy)."""
    out = []
    for i in sc.admissible_sites():
        if all(np.hypot(*(sc.site_xy[i] - sc.site_xy[j])) >= 1250.0 for j in out):
            out.append(int(i))
        if len(out) == k:
            return np.array(out)
    raise AssertionError("not enough separated sites")


def test_truth_is_complete_and_physical(sc):
    t = sc.truth
    assert t.field.shape == (96, 16, 16)               # 4 days hourly, [hour, i, j]
    assert np.isfinite(t.field).all() and (t.field >= 0).all()
    assert t.field.max() > 5.0                         # sources raise c above background
    assert t.source_xy.shape == (2, 2) and (t.strengths_ug_s > 0).all()
    assert t.winds.shape == (96, 2)
    assert t.site_truth.shape == (256, 96)                # all 16x16 candidate sites x hours
    assert sc.n_candidate_sites == 256


def test_source_basis_normalised_discrete():
    phi = B.source_basis((64, 64), (10000.0, 10000.0), (5000.0, 5000.0), 400.0)
    assert phi.sum() * (10000 / 64) ** 2 == pytest.approx(1.0, abs=1e-12)
    corner = B.source_basis((64, 64), (10000.0, 10000.0), (50.0, 50.0), 400.0)  # clipped at the walls
    assert corner.sum() * (10000 / 64) ** 2 == pytest.approx(1.0, abs=1e-12)
    assert corner.min() >= 0


def test_scenario_matches_frozen_manifest():
    """Source cells and strengths must reproduce what T-009 recorded for s00_K2 (default config)."""
    import yaml
    from pathlib import Path
    y = yaml.safe_load((Path(__file__).parents[1] / "configs" / "benchmark-v1.yaml").read_text())["scenarios"]["s00_K2"]
    cells, a = B.draw_sources(ExperimentConfig.default("EXP-003"), 0, 2)
    assert [list(map(int, c)) for c in cells] == y["sources_lattice_ab"]
    np.testing.assert_allclose(a, y["strength_ug_s"], rtol=1e-9, atol=0.05)  # manifest rounded to 0.01 ug/s after A1


def test_same_seed_same_scenario_and_fair_noise():
    a = B.generate_scenario(_cfg(), 0, 2, lattice_n=16)
    b = B.generate_scenario(_cfg(), 0, 2, lattice_n=16)
    np.testing.assert_array_equal(a.truth.field, b.truth.field)
    ids = _net(a, 6)
    va = a.view("train", ids[:4], "tuning", "white")
    vb = b.view("train", ids[2:6], "tuning", "white")  # two designs sharing sites ids[2], ids[3]
    np.testing.assert_array_equal(va.readings[2:], vb.readings[:2])  # identical noise at shared sites
    other = B.generate_scenario(_cfg(), 1, 2, lattice_n=16)
    assert not np.array_equal(other.truth.field, a.truth.field)


def test_readings_equal_truth_plus_noise_at_site(sc):
    ids = _net(sc, 3)
    v = sc.view("train", ids, "tuning", "white")
    hrs = v.hours
    resid = v.readings - sc.truth.site_truth[ids][:, hrs]
    np.testing.assert_allclose(resid, sc._noise["white"][ids][:, hrs], atol=1e-12)
    assert resid.std() == pytest.approx(sc.noise_sigma[ids].mean(), rel=0.5)


def test_heteroscedastic_sigma_varies_by_site(sc, sc_het):
    assert np.ptp(sc.noise_sigma) == 0 and sc.noise_sigma[0] > 0
    assert np.ptp(sc_het.noise_sigma) > 0 and (sc_het.noise_sigma > 0).all()


def test_admissible_sites_exclude_ring_and_near_sources(sc):
    ids = sc.admissible_sites()
    xy = sc.site_xy[ids]
    assert 0 < len(ids) < sc.n_candidate_sites
    assert xy.min() > 625.0 / 2 * 1.0 and xy.max() < 10000 - 625.0 / 2  # outer ring dropped
    d = np.linalg.norm(xy[:, None] - sc.truth.source_xy[None], axis=2)
    assert (d > 400.0).all()


def test_view_has_no_truth_and_is_read_only(sc):
    ids = _net(sc, 3)
    v = sc.view("validation", ids, "tuning", "ar1")
    assert dataclasses.is_dataclass(v) and v.readings.shape == (3, 24)
    names = {f.name for f in dataclasses.fields(v)}
    assert not names & {"truth", "field", "strengths_ug_s", "site_truth", "noise", "scenario", "_noise"}
    assert not any(isinstance(getattr(v, n), (B.Scenario, B.Truth)) for n in names)
    assert v.source_xy.shape == (2, 2)                  # locations known to the estimator (protocol)
    with pytest.raises(ValueError):
        v.readings[0, 0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.split = "test"
    assert v.readings.base is None or not np.shares_memory(v.readings, sc._noise["ar1"])


def test_view_only_contains_selected_sites_and_split_hours(sc):
    ids = _net(sc, 2)
    v = sc.view("train", ids, "model_selection", "white")
    assert v.readings.shape == (2, 48) and v.hours.tolist() == list(range(48))
    assert v.site_ids.tolist() == list(ids)
    assert sc.view("validation", ids, "tuning", "white").hours.tolist() == list(range(48, 72))


def test_split_enforcement(sc):
    ids = _net(sc, 2)
    for purpose in ("tuning", "placement", "model_selection"):
        with pytest.raises(LeakError):
            sc.view("test", ids, purpose, "white")
    for split in ("train", "validation"):
        with pytest.raises(LeakError):
            sc.view(split, ids, "evaluation", "white")
    assert sc.view("test", ids, "evaluation", "white").hours.tolist() == list(range(72, 96))
    with pytest.raises(ValueError):
        sc.view("holdout", ids, "tuning", "white")


def test_winds_respect_split_regimes(sc):
    from pravah.experiment_config import Winds
    w = Winds()
    assert set(sc.truth.wind_regimes["train"]) <= set(w.train) and set(sc.truth.wind_regimes["validation"]) <= set(w.train)
    assert set(sc.truth.wind_regimes["test"]) <= set(w.heldout)
    sp = np.hypot(*sc.truth.winds.T)
    assert sp.min() >= 0.2 - 1e-12 and sp.max() <= 5.0 + 1e-12


def test_view_rejects_inadmissible_and_crowded_sites(sc):
    allids = np.arange(sc.n_candidate_sites)
    inad = np.setdiff1d(allids, sc.admissible_sites())
    with pytest.raises(ValueError):
        sc.view("train", inad[:1], "tuning", "white")
    with pytest.raises(ValueError):
        sc.view("train", [], "tuning", "white")
    ids = sc.admissible_sites()
    d = np.linalg.norm(sc.site_xy[ids][:, None] - sc.site_xy[ids][None], axis=2)
    a, b = np.argwhere((d > 0) & (d < 1250.0))[0]       # 625 m neighbours exist on the 16x16 lattice
    with pytest.raises(ValueError):
        sc.view("train", [ids[a], ids[b]], "tuning", "white")


def test_truth_id_links_to_config(sc):
    from pravah.experiment_config import truth_id
    assert sc.truth_id == truth_id(_cfg(), 0, 2)


def test_constant_westerly_is_exactly_constant():
    w, used = B.make_winds(["constant_westerly"], 150, seed=3)
    assert used == ["constant_westerly"] * 4 and np.allclose(w, [3.0, 0.0])

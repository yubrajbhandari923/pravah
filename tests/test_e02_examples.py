"""Contract tests for scripts/e02_examples.py (T-011). Written before the script.

Contract
- build_examples() -> {name: Scenario} for exactly: "separated2", "collinear2", "changing4"; K = 2, 2, 4 sources;
  fixed source cells (documented override of the random draw); fresh truth_id per scenario; deterministic.
- mass_budget(sc) -> dict(emitted_ug, removed_ug, stored_ug, outflow_ug, outflow_frac, source_rate_err) with
  source_rate_err = |sum(S) dx dy H - sum(a)| / sum(a) at roundoff and 0 <= outflow_frac <= 1 (units: ug, ug s^-1).
- leak_checks(sc) -> list of (case, ok) where every case, including the forbidden ones, behaved as specified.
- same_truth_across_designs(sc) -> True iff two different admissible designs leave truth untouched and agree
  on noise at shared sites.
- splits_table(examples) -> rows with schema_version, config_hash, truth_id, split, days, wind_regimes.
"""
import hashlib
import importlib.util
import pathlib

import numpy as np
import pytest

root = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e02", root / "scripts" / "e02_examples.py")
e02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e02)


@pytest.fixture(scope="module")
def ex():
    return e02.build_examples()


def test_three_named_examples(ex):
    assert set(ex) == {"separated2", "collinear2", "changing4"}
    assert [ex[k].n_sources for k in ("separated2", "collinear2", "changing4")] == [2, 2, 4]
    assert len({s.truth_id for s in ex.values()}) == 3


def test_geometry(ex):
    sep = ex["separated2"].truth.source_xy
    col = ex["collinear2"].truth.source_xy
    assert np.hypot(*(sep[0] - sep[1])) >= 4000          # well separated: > 4 sigma_footprint*2.5
    assert abs(col[0, 1] - col[1, 1]) < 1e-9             # collinear along x = the steady westerly wind
    assert ex["collinear2"].truth.wind_regimes["train"] == ["steady_westerly"] * 2


def test_changing_winds_actually_change(ex):
    w = ex["changing4"].truth.winds
    th = np.degrees(np.arctan2(-w[:, 0], -w[:, 1])) % 360
    assert len(set(ex["changing4"].truth.wind_regimes["train"])) >= 2
    assert np.ptp(th[:96]) > 20


def test_mass_budget(ex):
    for sc in ex.values():
        m = e02.mass_budget(sc)
        assert m["source_rate_err"] < 1e-12
        assert m["stored_ug"] >= 0 and m["emitted_ug"] > 0
        assert 0 <= m["outflow_frac"] <= 1


def test_leak_checks_all_behave(ex):
    for sc in ex.values():
        res = e02.leak_checks(sc)
        assert len(res) >= 5 and all(ok for _, ok in res), res


def test_same_truth_across_designs(ex):
    assert all(e02.same_truth_across_designs(sc) for sc in ex.values())


def test_splits_table(ex):
    rows = e02.splits_table(ex)
    assert {r["split"] for r in rows} == {"train", "validation", "test"}
    assert len(rows) == 9 and len({r["config_hash"] for r in rows}) >= 1
    assert all(r["schema_version"] == 1 for r in rows)


def test_deterministic():
    a, b = e02.build_examples(["separated2"]), e02.build_examples(["separated2"])
    h = lambda s: hashlib.sha256(s.truth.field.tobytes()).hexdigest()
    assert h(a["separated2"]) == h(b["separated2"])

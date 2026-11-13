"""E07 sweep driver: the cached two-stage path (one solve per task, then designs/evaluation from disk) must reproduce the
single-process path bit for bit, and cache names must change with anything that changes a field."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from pravah import benchmarks as B
from pravah.experiments import cross_objective as X

_p = Path(__file__).resolve().parents[1] / "scripts" / "e07_cross_objective.py"
_spec = importlib.util.spec_from_file_location("e07_driver", _p)
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)
E.SCALES["tiny"] = dict(grid=(16, 16), dt=300, scheme="upwind", spinup=6, split=((1, 1), (2, 2), (3, 4)), Ks=(2,), ks=(2, 4),
                        seeds=(0,), R=2)


def _same(a, b):
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def test_cached_two_stage_equals_single_process(tmp_path):
    wind, K, seed = "veering", 2, 0
    sc = B.generate_scenario(E.make_cfg("tiny", wind), seed, K)  # old path: everything in one process
    F, G3 = X.responses(sc)
    rows0, swaps0, _ = E.evaluate("tiny", wind, K, seed, sc, F, G3, {}, 0.0, 0.0)

    fl = E.scenario_fields("tiny", wind, K, seed)
    assert len(fl) == K + 1 and len({n for n, _ in fl}) == K + 1
    (tmp_path / "fields").mkdir()
    for t in fl:
        assert E.run_field((t, str(tmp_path))) is not None
        assert E.run_field((t, str(tmp_path))) is None  # cached: skipped
    rows1, swaps1, tm = E.run_scenario(("tiny", wind, K, seed, str(tmp_path)))

    assert len(rows0) == len(rows1) and swaps0 == swaps1
    for r0, r1 in zip(rows0, rows1):
        assert r0.keys() == r1.keys()
        assert all(_same(r0[c], r1[c]) for c in r0 if c != "design_s"), r0  # design_s is a wall-clock time
    assert tm["linearity_max_abs"] == float(np.abs(sc.cfg.sources.background_ug_m3
                                                   + np.einsum("k,khxy->hxy", sc.truth.strengths_ug_s, F) - sc.truth.field).max())
    assert len(list((tmp_path / "scen").glob("*.pkl"))) == 1  # stage-2 result cached for reruns

    # sweep-style subset (strategies, estimators): exactly the matching rows of the full run
    E.SCALES["tiny_sub"] = {**E.SCALES["tiny"], "strategies": ("dopt", "podqr"), "estimators": ("map",)}
    rows2, swaps2, _ = E.evaluate("tiny_sub", wind, K, seed, sc, F, G3, {}, 0.0, 0.0)
    keep = [r for r in rows0 if r["strategy"] in ("random", "dopt", "podqr", "background_only", "climatology")
            and r["estimator"] in ("map", "-")]
    assert [{**x, "scale": "tiny"} for x in swaps2] == swaps0 and len(rows2) == len(keep)
    for r0, r2 in zip(keep, rows2):
        assert all(_same(r0[c], r2[c]) for c in r0 if c not in ("design_s", "scale")), r0


def test_cache_names_track_field_content():
    base = [n for n, _ in E.scenario_fields("tiny", "veering", 2, 0)]
    assert base == [n for n, _ in E.scenario_fields("tiny", "veering", 2, 0)]
    assert set(base).isdisjoint(n for n, _ in E.scenario_fields("tiny", "veering", 2, 1))  # other seed: winds/sources differ
    E.SCALES["tiny_vl"] = {**E.SCALES["tiny"], "scheme": "vanleer"}
    assert set(base).isdisjoint(n for n, _ in E.scenario_fields("tiny_vl", "veering", 2, 0))


def test_truth_field_must_match_calendar():
    cfg = E.make_cfg("tiny", "veering")
    with pytest.raises(ValueError, match="truth_field shape"):
        B.generate_scenario(cfg, 0, 2, truth_field=np.zeros((5, 16, 16)))


def test_exclusion_removes_exactly_sites_within_radius():
    wind, K, seed, r = "veering", 2, 0, 1200.0
    sc = B.generate_scenario(E.make_cfg("tiny", wind), seed, K)
    F, G3 = X.responses(sc)
    adm = sc.admissible_sites()
    d = np.linalg.norm(sc.site_xy[:, None] - sc.truth.source_xy[None], axis=2).min(1)
    keep = X.exclude_near_sources(sc.site_xy, adm, sc.truth.source_xy, r)
    assert keep.tolist() == [i for i in adm if d[i] > r] and 0 < len(keep) < len(adm)  # exactly the radius; order kept

    rows0, swaps0, _ = E.evaluate("tiny", wind, K, seed, sc, F, G3, {}, 0.0, 0.0)
    # a radius inside the admissibility mask (> 1 sigma = 400 m) removes nothing, so nothing else may change
    rows_s, swaps_s, _ = E.evaluate("tiny", wind, K, seed, sc, F, G3, {}, 0.0, 0.0, exclude_m=300.0)
    assert swaps_s == swaps0 and len(rows_s) == len(rows0)
    assert all(all(_same(a[c], b[c]) for c in a if c != "design_s") for a, b in zip(rows0, rows_s))

    rows1, _, tm = E.evaluate("tiny", wind, K, seed, sc, F, G3, {}, 0.0, 0.0, exclude_m=r)
    assert tm["n_cand"] == len(keep) and len(rows1) == len(rows0)
    sites = [int(i) for x in rows1 if isinstance(x.get("sites"), str) for i in x["sites"].split()]
    assert sites and set(sites) <= set(keep.tolist())
    base = lambda rows: [x for x in rows if x["k"] == 0]  # design-free baselines untouched
    assert base(rows1) == base(rows0)
    assert [x["sigma_med"] for x in rows1 if "sigma_med" in x] == [x["sigma_med"] for x in rows0 if "sigma_med" in x]

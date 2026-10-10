"""Phase-2 data contracts: origin tags, raster sources, study region, inflow bases, parameter table."""
import json

import numpy as np
import pytest

from pravah.config import (Boundary, Domain, GaussianSource, InflowBasis, RasterSource, Scenario, StudyMask, StudyRegion,
                           Times, Transport, WindSegment, array_sha256, scenario_from_dict, to_dict)
from pravah.inverse import cache_key
from pravah.config import SolverOptions
from pravah.sources import check_origin_tags, origin_mass_fractions, source_fields, study_weights

DOM = Domain(lx=8000.0, ly=8000.0, nx=16, ny=16)
TIMES = Times(duration=3600.0, dt=120.0, output_interval=600.0)


def _raster(seed: int = 0, group: str = "local") -> RasterSource:
    w = np.random.default_rng(seed).random((16, 16))
    return RasterSource.from_array("roads", group, w)


def test_raster_source_is_normalised_like_a_gaussian():
    sc = Scenario(domain=DOM, sources=(_raster(), GaussianSource("g", 4000, 4000, 500)), times=TIMES)
    f = source_fields(sc)
    assert np.allclose(f.sum(axis=(1, 2)) * DOM.dx * DOM.dy, 1.0, rtol=0, atol=1e-12)


def test_to_dict_uses_hashes_not_arrays_and_is_json():
    r = _raster()
    sc = Scenario(domain=DOM, sources=(r,), times=TIMES, study=StudyMask.from_array(np.ones((16, 16))),
                  transport=Transport(wind=(WindSegment.from_streamfunction(0.0, np.zeros((17, 17)), 1.0, 0.0),)))
    d = to_dict(sc)
    json.dumps(d)
    assert d["sources"][0]["sha256"] == r.sha256 and "weights" not in d["sources"][0]
    assert d["study"]["sha256"] == array_sha256(np.ones((16, 16))) and "psi" not in d["transport"]["wind"][0]


def test_loader_verifies_hash_and_shapes(tmp_path):
    w = np.random.default_rng(1).random((16, 16))
    np.save(tmp_path / "roads.npy", w)
    cfg = {"domain": {"lx": 8000.0, "ly": 8000.0, "nx": 16, "ny": 16},
           "sources": [{"type": "raster", "name": "roads", "group": "local", "file": "roads.npy",
                        "sha256": array_sha256(w)}],
           "times": {"duration": 3600.0, "dt": 120.0, "output_interval": 600.0},
           "study": {"type": "rectangle", "x0": 2000, "y0": 2000, "x1": 6000, "y1": 6000},
           "boundary": {"bases": [{"name": "west", "sides": ["west"], "profile": [[0, 0], [1800, 1]]}]}}
    sc = scenario_from_dict(cfg, tmp_path)
    assert isinstance(sc.sources[0], RasterSource) and sc.n_params == 2
    bad = json.loads(json.dumps(cfg))
    bad["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        scenario_from_dict(bad, tmp_path)
    np.save(tmp_path / "small.npy", np.ones((4, 4)))
    bad2 = json.loads(json.dumps(cfg))
    bad2["sources"][0].update(file="small.npy", sha256=array_sha256(np.ones((4, 4))))
    with pytest.raises(ValueError, match="shape"):
        scenario_from_dict(bad2, tmp_path)


def test_study_region_weights_are_exact_overlap():
    sc = Scenario(domain=DOM, sources=(GaussianSource("g", 4000, 4000, 500),), times=TIMES,
                  study=StudyRegion(1250.0, 2000.0, 6250.0, 5750.0))
    w = study_weights(sc)
    assert w is not None
    assert w.sum() * DOM.dx * DOM.dy == pytest.approx(5000.0 * 3750.0, rel=1e-12)  # polygon area is preserved
    assert w.max() <= 1.0 and w.min() >= 0.0
    assert study_weights(Scenario(domain=DOM, sources=(GaussianSource("g", 4000, 4000, 500),), times=TIMES)) is None


def test_origin_tags_must_match_geometry():
    inside = GaussianSource("a", 4000, 4000, 300, group="local")
    outside_tagged_local = GaussianSource("b", 500, 500, 200, group="local")
    study = StudyRegion(2000.0, 2000.0, 6000.0, 6000.0)
    check_origin_tags(Scenario(domain=DOM, sources=(inside, GaussianSource("e", 500, 500, 200, group="external")),
                               times=TIMES, study=study))
    with pytest.raises(ValueError, match="tagged local"):
        check_origin_tags(Scenario(domain=DOM, sources=(inside, outside_tagged_local), times=TIMES, study=study))
    with pytest.raises(ValueError, match="tagged external"):
        check_origin_tags(Scenario(domain=DOM, sources=(GaussianSource("x", 4000, 4000, 300, group="external"),),
                                   times=TIMES, study=study))
    assert origin_mass_fractions(Scenario(domain=DOM, sources=(inside,), times=TIMES, study=study))[0] > 0.99


def test_param_table_orders_emissions_then_inflow_with_units():
    sc = Scenario(domain=DOM, times=TIMES,
                  sources=(GaussianSource("l", 4000, 4000, 400), GaussianSource("e", 500, 500, 300, group="external")),
                  boundary=Boundary(bases=(InflowBasis("W", ("west",)), InflowBasis("S", ("south",)))))
    t = sc.param_table()
    assert [(p.name, p.kind, p.units) for p in t] == [
        ("l", "emission", "microgram/s"), ("e", "emission", "microgram/s"),
        ("W", "inflow", "microgram/m^3"), ("S", "inflow", "microgram/m^3")]
    assert sc.n_params == 4 and sc.n_emission == 2


def test_invalid_contract_inputs_rejected():
    with pytest.raises(ValueError):
        GaussianSource("a", 1, 1, 100, group="martian")
    with pytest.raises(ValueError):
        InflowBasis("b", ("up",))
    with pytest.raises(ValueError):
        Boundary(mode="periodic", bases=(InflowBasis("b"),))
    with pytest.raises(ValueError):
        Scenario(domain=DOM, times=TIMES, sources=(GaussianSource("a", 1, 1, 100), GaussianSource("a", 2, 2, 100)))
    with pytest.raises(ValueError):
        StudyMask.from_array(np.full((16, 16), 1.5))
    with pytest.raises(ValueError):
        Scenario(domain=DOM, times=TIMES, sources=(RasterSource.from_array("r", "local", np.ones((4, 4))),))


def test_cache_key_depends_on_raster_content_and_group():
    def key(r: RasterSource) -> str:
        return cache_key(Scenario(domain=DOM, sources=(r,), times=TIMES), SolverOptions(), 1e5)
    assert key(_raster(0)) != key(_raster(1))
    assert key(_raster(0, "local")) != key(_raster(0, "external"))
    assert key(_raster(0)) == key(_raster(0))

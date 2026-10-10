import json
import pytest
from pravah.config import Scenario, Domain, Boundary, Transport, WindSegment, load_config, scenario_from_dict, to_dict

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_reference_config_roundtrip():
    cfg = load_config(ROOT / "configs" / "reference.json")
    sc = scenario_from_dict(cfg["scenario"])
    assert sc == scenario_from_dict(to_dict(sc))
    assert sc.domain.dx == 200.0 and len(sc.sources) == 3
    assert sc.output_times[-1] == 14400.0 and len(sc.output_times) == 49


def test_closed_with_wind_rejected():
    with pytest.raises(ValueError):
        Scenario(boundary=Boundary(mode="closed"))


def test_invalid_values_rejected():
    for bad in (lambda: Domain(nx=1), lambda: Transport(H=0), lambda: Boundary(mode="x"),
                lambda: Transport(wind=(WindSegment(5.0, 1, 0),))):
        with pytest.raises(ValueError):
            bad()


def test_extends_merge(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"scenario": {"domain": {"nx": 10, "ny": 10}}, "q": [1]}))
    (tmp_path / "b.json").write_text(json.dumps({"extends": "a.json", "scenario": {"domain": {"nx": 20}}}))
    cfg = load_config(tmp_path / "b.json")
    assert cfg["scenario"]["domain"] == {"nx": 20, "ny": 10} and cfg["q"] == [1]


def test_periodic_background_rejected():
    from pravah.config import Boundary
    with pytest.raises(ValueError, match="background"):
        Scenario(boundary=Boundary(mode="periodic", background=1.0))
    Scenario(boundary=Boundary(mode="periodic"))

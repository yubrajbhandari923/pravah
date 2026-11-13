"""Independent oracle for the experiment contract (protocol: docs/research/preregistered-protocol.md)."""
import json
import pytest

from pravah import experiment_config as ec


def cfg():
    return ec.ExperimentConfig.default("EXP-001")


def test_defaults_match_preregistered_protocol():
    c = cfg()
    assert c.model.grid == (64, 64) and c.model.dt_s == 300
    assert c.model.units == {"length": "m", "time": "s", "concentration": "ug m^-3",
                             "emission_rate": "ug s^-1", "areal_flux": "ug m^-2 s^-1"}
    assert c.design.n_sources == (2, 4, 8) and c.design.sensor_budgets == (2, 4, 8, 16, 32)
    assert c.split.train == (1, 30) and c.split.validation == (31, 42) and c.split.test == (43, 60)
    assert c.seeds.n_replicates == 20
    assert c.seeds.streams(3) == {"truth": 1003, "noise": 2003, "placement": 3003}


def test_exactly_one_primary_metric_with_units():
    c = cfg()
    prim = [m for m in c.metrics if m.primary]
    assert len(prim) == 1 and prim[0].name == "forecast_rmse_1h_preassim"
    assert all(m.unit and m.definition for m in c.metrics)
    c.metrics.append(ec.MetricDef("x", "ug m^-3", "d", primary=True))
    with pytest.raises(ec.ConfigError):
        c.validate()


@pytest.mark.parametrize("bad", [(1, 30, 30, 42, 43, 60), (1, 30, 31, 42, 40, 60), (1, 30, 32, 42, 43, 60)])
def test_split_must_be_contiguous_ordered_disjoint(bad):
    a, b, c_, d, e, f = bad
    with pytest.raises(ec.ConfigError):
        ec.Split(train=(a, b), validation=(c_, d), test=(e, f)).validate()


def test_json_roundtrip_and_hash_stable(tmp_path):
    c = cfg()
    p = tmp_path / "c.json"
    ec.dump_config(c, p)
    c2 = ec.load_config(p)
    assert c2 == c and c2.config_hash() == c.config_hash()
    c2.model.diffusivity_m2_s = 51.0
    assert c2.config_hash() != c.config_hash()


def test_load_rejects_unknown_keys_and_bad_values(tmp_path):
    d = json.loads(cfg().to_json())
    d["surprise"] = 1
    with pytest.raises(ec.ConfigError):
        ec.ExperimentConfig.from_dict(d)
    d = json.loads(cfg().to_json())
    d["model"]["diffusivity_m2_s"] = -1
    with pytest.raises(ec.ConfigError):
        ec.ExperimentConfig.from_dict(d)


def test_truth_id_deterministic_and_sensitive():
    c = cfg()
    assert ec.truth_id(c, 4, 2) == ec.truth_id(c, 4, 2)
    assert ec.truth_id(c, 4, 2) != ec.truth_id(c, 5, 2)
    assert ec.truth_id(c, 4, 2) != ec.truth_id(c, 4, 4)
    with pytest.raises(ec.ConfigError):
        ec.truth_id(c, 20, 2)  # replicate index out of range
    with pytest.raises(ec.ConfigError):
        ec.truth_id(c, 0, 3)  # K_s not in design


def test_no_leak_guard():
    s = cfg().split
    ec.check_no_leak(s, "tuning", days=[1, 42])
    ec.check_no_leak(s, "evaluation", days=[43, 60])
    for purpose in ("tuning", "placement", "model_selection"):
        with pytest.raises(ec.LeakError):
            ec.check_no_leak(s, purpose, days=[43])
    with pytest.raises(ec.LeakError):
        ec.check_no_leak(s, "evaluation", days=[42])
    with pytest.raises(ec.LeakError):
        ec.check_no_leak(s, "tuning", days=[1], wind_regimes=["steady_southerly"])
    with pytest.raises(ec.ConfigError):
        ec.check_no_leak(s, "bogus", days=[1])


def test_heldout_wind_regimes_disjoint_from_training():
    c = cfg()
    assert not set(c.winds.train) & set(c.winds.heldout)
    c.winds.heldout.append(c.winds.train[0])
    with pytest.raises(ec.ConfigError):
        c.validate()


def test_resource_profiles():
    c = cfg()
    loc, hpc = c.resources["local"], c.resources["hpc"]
    assert loc.backend == "local" and hpc.backend == "slurm"
    assert loc.cpus >= 1 and hpc.walltime_h > 0 and hpc.mem_gb > 0
    assert hpc.registers_via == "RESEARCH_RUN_DIR"
    bad = ec.ResourceProfile("hpc", "slurm", cpus=0, mem_gb=1, walltime_h=1)
    with pytest.raises(ec.ConfigError):
        bad.validate()


def test_manifest_hashes_and_detects_tampering(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    m = ec.DataManifest.build([f], root=tmp_path)
    assert m.entries[0]["path"] == "a.txt" and len(m.entries[0]["sha256"]) == 64
    assert m.verify(tmp_path) == []
    f.write_text("y")
    assert m.verify(tmp_path) == ["a.txt"]
    f.unlink()
    assert m.verify(tmp_path) == ["a.txt"]
    with pytest.raises(ec.ConfigError):
        ec.DataManifest.build([tmp_path / "missing"], root=tmp_path)


def test_provenance_and_output_registration(tmp_path):
    c = cfg()
    m = ec.DataManifest(entries=[])
    prov = ec.write_provenance(tmp_path, c, command=["python", "x.py"], replicate=2,
                               n_sources=4, run_id="RUN-007", manifest=m)
    on_disk = json.loads((tmp_path / "outputs" / "provenance.json").read_text())
    assert on_disk == prov
    for k in ("command", "git_sha", "seeds", "config_hash", "config", "machine", "python",
              "truth_id", "run_id", "manifest", "units"):
        assert k in prov
    assert prov["seeds"] == {"truth": 1002, "noise": 2002, "placement": 3002}
    assert prov["truth_id"] == ec.truth_id(c, 2, 4)
    arts = [json.loads(l)["path"] for l in (tmp_path / "artifacts.jsonl").read_text().splitlines()]
    assert arts == [str((tmp_path / "outputs" / "provenance.json").resolve())]
    ec.log_metric(tmp_path, "rmse", 0.5, step=1)
    rec = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[-1])
    assert rec == {"name": "rmse", "value": 0.5, "step": 1}
    f = tmp_path / "outputs" / "t.csv"
    f.write_text("a")
    ec.register_artifact(tmp_path, f)
    assert str(f.resolve()) in (tmp_path / "artifacts.jsonl").read_text()
    with pytest.raises(ec.ConfigError):
        ec.register_artifact(tmp_path, tmp_path / "outputs" / "nope.csv")
    with pytest.raises(ec.ConfigError):
        ec.log_metric(tmp_path, "bad", float("nan"))

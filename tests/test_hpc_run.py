"""Orchestrator tests with mock phases (python -c): sequence, resume, hash invalidation, missing-script skip, failure, manifest, bundle."""
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hpc_run", ROOT / "scripts/hpc_run.py")
hr = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(hr)  # type: ignore[union-attr]

WRITE = "import os,sys;os.makedirs(sys.argv[1],exist_ok=True);open(sys.argv[1]+'/o.txt','a').write('x')"


def cfg(tmp, phases):
    root = f"results/_hpctest_{tmp.parent.name + tmp.name if tmp.name == 'sub' else tmp.name}"
    c = tmp / "c.json"
    c.write_text(json.dumps({"name": "t", "results_root": root, "phases": phases}))
    return c, ROOT / root


def ph(name, code="pass", fail=False, script=None):
    cmd = ["{python}", script] if script else ["{python}", "-c", "import sys;sys.exit(3)" if fail else code]
    return {"name": name, "command": cmd, "requires": [], "done_marker": f"{{out}}/{name}.done", "est_minutes": 1}


def run(c, *args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hpc_run", "--config", str(c), *args])
    return hr.main()


@pytest.fixture(autouse=True)
def cleanup(tmp_path):
    yield
    import shutil
    for r in ROOT.glob(f"results/_hpctest_*{tmp_path.name}*"):
        shutil.rmtree(r, ignore_errors=True)


def test_run_resume_and_hash(tmp_path, monkeypatch):
    p1 = ph("a", "import os;os.makedirs('results/_hpctest_%s',exist_ok=True);open('results/_hpctest_%s/a.txt','w').write('1')" % ((tmp_path.name,) * 2))
    c, root = cfg(tmp_path, [p1, ph("b")])
    monkeypatch.setenv("PRAVAH_WORKERS", "4")
    assert run(c, monkeypatch=monkeypatch) == 0
    m = json.loads((root / "hpc_manifest.json").read_text())
    assert [p["phase"] for p in m["phases"]] == ["a", "b"] and all(p["exit_code"] == 0 and p["workers"] == 4 for p in m["phases"])
    assert any(o["path"].endswith("a.txt") and len(o["sha256"]) == 64 for o in m["phases"][0]["outputs"])
    assert m["phases"][0]["versions"]["numpy"] and "diff_sha256" in m["phases"][0]["git"]
    b = root / "phase3-hpc-bundle.tar.gz"
    assert b.exists() and m["bundle"]["sha256"] == hr.sha256(b)
    assert any(n.endswith("hpc_manifest.json") for n in tarfile.open(b).getnames())
    start = {p["phase"]: p["start"] for p in m["phases"]}
    assert run(c, monkeypatch=monkeypatch) == 0  # resume: nothing re-run
    m2 = json.loads((root / "hpc_manifest.json").read_text())
    assert {p["phase"]: p["start"] for p in m2["phases"]} == start
    c.write_text(c.read_text().replace('"pass"', '"pass;pass"'))  # phase b changed -> re-run b only
    assert run(c, monkeypatch=monkeypatch) == 0
    m3 = json.loads((root / "hpc_manifest.json").read_text())
    assert {p["phase"]: p["start"] for p in m3["phases"]}["a"] == start["a"] and {p["phase"]: p["start"] for p in m3["phases"]}["b"] >= start["b"]
    assert m3["phases"][-1]["phase_config_sha256"] != m["phases"][1]["phase_config_sha256"]


def test_failure_stops_and_missing_script_skips(tmp_path, monkeypatch, capsys):
    c, root = cfg(tmp_path, [ph("a", fail=True), ph("b")])
    assert run(c, monkeypatch=monkeypatch) == 3
    m = json.loads((root / "hpc_manifest.json").read_text())
    assert [(p["phase"], p["exit_code"]) for p in m["phases"]] == [("a", 3)]
    assert not (root / "a.done").exists()
    (tmp_path / "sub").mkdir()
    c, root = cfg(tmp_path / "sub", [ph("s", script="scripts/does_not_exist.py"), ph("b")])
    assert run(c, monkeypatch=monkeypatch) == 0
    assert "script missing" in capsys.readouterr().out
    assert [p["phase"] for p in json.loads((root / "hpc_manifest.json").read_text())["phases"]] == ["b"]


def test_dry_run_only_and_workers(tmp_path, monkeypatch, capsys):
    c, root = cfg(tmp_path, [ph("a"), ph("b")])
    assert run(c, "--dry-run", monkeypatch=monkeypatch) == 0 and not root.exists()
    assert run(c, "--only", "b", "--no-bundle", monkeypatch=monkeypatch) == 0
    assert [p["phase"] for p in json.loads((root / "hpc_manifest.json").read_text())["phases"]] == ["b"]
    monkeypatch.delenv("PRAVAH_WORKERS", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "12")
    assert hr.n_workers(None) == 12 and hr.n_workers(5) == 5
    monkeypatch.setenv("PRAVAH_WORKERS", "7")
    assert hr.n_workers(None) == 7

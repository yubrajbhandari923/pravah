"""HPC smoke-test script: correct arithmetic passes, corrupted arithmetic fails, run-dir output registers artifacts."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/hpc_smoke.py"
CONFIG = ROOT / "configs/hpc_smoke.json"


def run(out=None, run_dir=None, fail=False):
    cmd = [sys.executable, str(SCRIPT), "--config", str(CONFIG)]
    if out:
        cmd += ["--out", str(out)]
    if fail:
        cmd += ["--fail"]
    env = {"PATH": "/usr/bin:/bin", "RESEARCH_RUN_DIR": str(run_dir)} if run_dir else None
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)


def test_expected_values_are_the_closed_forms():
    cfg = json.loads(CONFIG.read_text())
    n = cfg["n"]
    assert cfg["expected"]["sum"] == n * (n + 1) // 2
    assert cfg["expected"]["sum_squares"] == n * (n + 1) * (2 * n + 1) // 6


def test_correct_run_passes_and_records_execution(tmp_path):
    r = run(out=tmp_path / "o")
    assert r.returncode == 0, r.stderr
    d = json.loads((tmp_path / "o/result.json").read_text())
    assert d["ok"] and all(c["ok"] for c in d["checks"])
    assert d["execution"]["hostname"] and d["execution"]["python_version"]
    assert d["input"]["sha256"]


def test_corrupted_arithmetic_is_reported_as_failure(tmp_path):
    r = run(out=tmp_path / "o", fail=True)
    assert r.returncode != 0
    assert json.loads((tmp_path / "o/result.json").read_text())["ok"] is False


def test_run_dir_output_registers_artifacts_for_collection(tmp_path):
    # Writing under $RESEARCH_RUN_DIR and listing files in artifacts.jsonl is what `research hpc collect` reads.
    rd = tmp_path / "run"
    rd.mkdir()
    assert run(run_dir=rd).returncode == 0
    out = rd / "results/hpc-smoke"
    assert (out / "result.json").is_file()
    lines = [json.loads(x) for x in (rd / "artifacts.jsonl").read_text().splitlines()]
    assert {Path(a["path"]).name for a in lines} == {"result.json", "result.json.sha256"}
    assert all(a["sha256"] for a in lines)

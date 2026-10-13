"""data_provenance is optional and backward compatible: historical provenance.json files lack it."""
import json
import subprocess
import sys

import pytest

from pravah import io


def test_field_is_omitted_unless_given():
    assert "data_provenance" not in io.provenance({}, {})
    p = io.provenance({}, {}, data=io.data_provenance("synthetic"))
    assert p["data_provenance"] == {"kind": "synthetic", "datasets": []}


def test_unknown_kind_is_rejected_and_datasets_kept():
    with pytest.raises(ValueError):
        io.data_provenance("measured")
    d = io.data_provenance("geography_only", [{"role": "geography_manifest", "path": "m.json", "sha256": "ab"}])
    assert d["datasets"][0]["sha256"] == "ab"


def test_cli_simulate_records_synthetic(tmp_path):
    r = subprocess.run([sys.executable, "-m", "pravah", "simulate", "--config", "configs/smoke.json", "--out", str(tmp_path / "o")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    assert json.loads((tmp_path / "o" / "provenance.json").read_text())["data_provenance"]["kind"] == "synthetic"

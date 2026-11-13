"""Verify summary tables against data/MANIFEST.json, then render the Quarto report.

Usage: python reports/quarto/build.py [--no-render]
Manifest: {"entries":[{"path","sha256","bytes"}]} paths relative to reports/quarto/data, built with
pravah.experiment_config.DataManifest. An empty/missing manifest is fine only if data/ has no tables.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1] / "src"))
from pravah.experiment_config import DataManifest  # noqa: E402

data = here / "data"
mf = data / "MANIFEST.json"
m = DataManifest(json.loads(mf.read_text())["entries"] if mf.exists() else [])
listed = {e["path"] for e in m.entries}
unlisted = sorted(str(p.relative_to(data)) for p in data.rglob("*")
                  if p.is_file() and p.name != "MANIFEST.json" and str(p.relative_to(data)) not in listed)
bad = m.verify(data)
if bad or unlisted:
    sys.exit(f"manifest mismatch: changed/missing={bad} unlisted={unlisted}")
if "--no-render" in sys.argv:
    sys.exit(0)
if not shutil.which("quarto"):
    sys.exit("quarto not installed; manifest verified, render skipped")
sys.exit(subprocess.call(["quarto", "render", str(here)]))

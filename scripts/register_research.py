"""Optional Research Panel adapter (pravah itself never imports it).

    .research/bin/research run exec EXP-001 -- python -m pravah ...   # makes RUN-001
    python scripts/register_research.py RUN-001 EXP-001 results/<dir>

Registers the run directory's files as artifacts and logs every top-level numeric value of its
summary.json / metrics.json / inverse_metrics.json as a metric, via the project-local CLI.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CLI = str(Path(__file__).resolve().parents[1] / ".research" / "bin" / "research")
SUMMARIES = ("summary.json", "metrics.json", "inverse_metrics.json", "verification.json")
SKIP = ("provenance.json",)


def research(*args: str) -> None:
    subprocess.run([CLI, *args], check=True)


def flat_numbers(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[prefix + k] = float(v)
        elif isinstance(v, dict) and len(prefix) < 12:
            out.update(flat_numbers(v, f"{prefix}{k}."))
    return out


def main(run: str, exp: str, results: str) -> None:
    root = Path(results)
    files = sorted(p for p in root.iterdir() if p.is_file() and p.name not in SKIP)
    research("artifact", "add", "--run", run, "--experiment", exp, *map(str, files))
    for name in SUMMARIES:
        f = root / name
        if f.exists() and isinstance((data := json.loads(f.read_text())), dict):
            for k, v in list(flat_numbers(data).items())[:60]:
                research("metric", "add", k, repr(v), "--run", run, "--experiment", exp)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])

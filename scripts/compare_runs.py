"""Compare two capture directories (scripts/capture_regression.py) for deterministic equality.

    python scripts/compare_runs.py results/solver-regression/before results/solver-regression/after --out comparison.json

Compared: NetCDF fields (values and coordinates, exact), CSV tables (exact, NaN == NaN) and JSON results (exact).
Excluded as volatile: runtime/timing columns and keys, timestamps, git state, filesystem paths, provenance.json,
timings.json, config.json, manifest.json, figures.  Anything else that differs is reported; the exit code is 1 then.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

SKIP_FILES = {"manifest.json", "provenance.json", "timings.json", "config.json", "smoke_mismatch.local.json"}
VOLATILE_KEYS = {"runtime_s", "wall_s", "timestamp_utc", "created_utc", "git_revision", "git_dirty", "layouts_from",
                 "config_hash", "alpha_selection_file", "peak_memory_mb", "package_versions", "backend"}


def _vol(name: str) -> bool:
    return name in VOLATILE_KEYS or "runtime" in name or name.endswith("_s") and name.startswith("wall")


def diff_json(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for k in sorted(set(a) | set(b)):
            if _vol(k):
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'a' if k in a else 'b'}")
            else:
                out += diff_json(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff_json(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def diff_csv(pa: Path, pb: Path) -> list[str]:
    a, b = pd.read_csv(pa), pd.read_csv(pb)
    keep = [c for c in a.columns if not _vol(c)]
    if list(a.columns) != list(b.columns):
        return [f"columns differ: {list(a.columns)} vs {list(b.columns)}"]
    return [] if a[keep].equals(b[keep]) else [f"{len(keep)} compared columns differ"]


def diff_nc(pa: Path, pb: Path) -> list[str]:
    with xr.open_dataset(pa) as a, xr.open_dataset(pb) as b:
        out = [f"variable {v} differs" for v in a.data_vars if v not in b or not np.array_equal(a[v].values, b[v].values)]
        out += [f"coord {c} differs" for c in a.coords if c not in b.coords or not np.array_equal(a[c].values, b[c].values)]
        return out + [f"variable {v} only in b" for v in b.data_vars if v not in a.data_vars]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A, B = Path(args.before), Path(args.after)
    files = sorted(p.relative_to(A) for p in A.rglob("*") if p.is_file() and p.suffix in {".csv", ".json", ".nc"}
                   and p.name not in SKIP_FILES and "figures" not in p.parts)
    report: dict[str, Any] = {"before": str(A), "after": str(B), "compared": {}, "n_differences": 0}
    for rel in files:
        pa, pb = A / rel, B / rel
        if not pb.exists():
            diffs = ["missing in after"]
        elif rel.suffix == ".nc":
            diffs = diff_nc(pa, pb)
        elif rel.suffix == ".csv":
            diffs = diff_csv(pa, pb)
        else:
            diffs = diff_json(json.loads(pa.read_text()), json.loads(pb.read_text()))
        report["compared"][str(rel)] = {"identical": not diffs, "differences": diffs[:20]}
        report["n_differences"] += len(diffs)
    report["n_files"] = len(files)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    bad = [f for f, v in report["compared"].items() if not v["identical"]]
    print(f"{len(files)} files compared, {len(bad)} differ")
    for f in bad:
        print(" ", f, report["compared"][f]["differences"][:3])
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

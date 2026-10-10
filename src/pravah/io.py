"""Run directories, JSON/CSV/NetCDF output and provenance records."""
from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def make_run_dir(path: str | Path, overwrite: bool = False) -> Path:
    """Create ``path``. An existing non-empty directory is refused unless ``overwrite`` (files are then kept and
    overwritten only when re-written; nothing is deleted)."""
    p = Path(path)
    if p.exists() and not p.is_dir():
        raise FileExistsError(f"{p} exists and is not a directory")
    if p.is_dir() and any(p.iterdir()) and not overwrite:
        raise FileExistsError(f"run directory {p} is not empty; pass overwrite=True to reuse it")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    """Strict JSON: NaN/inf become null, numpy types are converted."""
    Path(path).write_text(json.dumps(_clean(obj), indent=2, allow_nan=False) + "\n")


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                           cwd=Path(__file__).resolve().parent)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def provenance(config: dict[str, Any], seeds: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    rev = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if rev else None
    return {
        "config": config,
        "seeds": seeds,
        "versions": {p: _version(p) for p in ("numpy", "scipy", "xarray", "fipy", "pravah")},
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": rev,
        "git_dirty": None if status is None else bool(status),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "extra": extra or {},
    }


def save_field(path: str | Path, data: xr.DataArray | xr.Dataset) -> None:
    ds = data.to_dataset(name=data.name or "field") if isinstance(data, xr.DataArray) else data
    ds.to_netcdf(path)


def write_csv(path: str | Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)

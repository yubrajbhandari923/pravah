"""Run directories, JSON/CSV/NetCDF output and provenance records."""
from __future__ import annotations

import datetime as dt
import hashlib
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


def _dirty_info(rev: str | None) -> dict[str, Any]:
    """Additive: short sha256 of ``git diff HEAD`` and the list of changed/untracked files (None outside a git repo),
    so a run from a dirty tree can be identified."""
    if not rev:
        return {"git_diff_hash": None, "git_dirty_files": None, "git_dirty_n_files": None}
    diff = _git("diff", "HEAD")
    status = _git("status", "--porcelain")
    files = [ln[3:] for ln in status.splitlines()] if status else []
    h = hashlib.sha256(diff.encode()).hexdigest()[:12] if diff else None
    return {"git_diff_hash": h, "git_dirty_files": files[:200], "git_dirty_n_files": len(files)}


GIT_PATCH_CAP = 5 * 1024 * 1024


def archive_git_state(out_dir: str | Path, cap: int = GIT_PATCH_CAP, cwd: str | Path | None = None) -> dict[str, Any]:
    """Additive.  When the git tree is dirty, write ``git_diff.patch`` (``git diff HEAD``, capped at ``cap`` bytes; cut and
    flagged if longer) and ``git_status.txt`` (``git status --porcelain``, which also lists untracked files that the patch
    does not contain) into ``out_dir`` and return their sha256 and sizes for provenance.json.  Clean tree: {"dirty": False};
    no git / no repository: {"dirty": None}.  ``cwd`` (default: this package) selects the repository."""
    def run(*args: str) -> bytes | None:
        try:
            r = subprocess.run(["git", *args], capture_output=True, timeout=60, cwd=cwd or Path(__file__).resolve().parent)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout if r.returncode == 0 else None
    status = run("status", "--porcelain")
    if status is None:
        return {"dirty": None}
    if not status.strip():
        return {"dirty": False}
    diff = run("diff", "HEAD") or b""
    cut = diff[:cap]
    out = Path(out_dir)
    (out / "git_diff.patch").write_bytes(cut)
    (out / "git_status.txt").write_bytes(status)
    return {"dirty": True, "git_diff_patch_sha256": hashlib.sha256(cut).hexdigest(), "git_diff_patch_bytes": len(cut),
            "git_diff_patch_full_bytes": len(diff), "git_diff_patch_truncated": len(diff) > cap,
            "git_status_sha256": hashlib.sha256(status).hexdigest(),
            "note": "git_diff.patch is `git diff HEAD` (tracked files only; untracked files are listed in git_status.txt)"}


DATA_KINDS = ("synthetic", "geography_only", "observed")


def data_provenance(kind: str, datasets: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Scientific input identity of a run: ``kind`` is synthetic (nothing external), geography_only (real geometry, synthetic
    emissions/meteorology) or observed (measurements); ``datasets`` lists only inputs whose identity is known
    (``{"role", "path", "sha256"}``). Execution metadata stays with Research OS."""
    if kind not in DATA_KINDS:
        raise ValueError(f"data kind {kind!r} not in {DATA_KINDS}")
    return {"kind": kind, "datasets": datasets or []}


def provenance(config: dict[str, Any], seeds: dict[str, Any], extra: dict[str, Any] | None = None,
               data: dict[str, Any] | None = None) -> dict[str, Any]:
    rev = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if rev else None
    return {
        "config": config,
        "seeds": seeds,
        "versions": {p: _version(p) for p in ("numpy", "scipy", "xarray", "fipy", "pandas", "matplotlib", "netCDF4", "pravah")},
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": rev,
        "git_dirty": None if status is None else bool(status),
        **_dirty_info(rev),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "extra": extra or {},
        **({"data_provenance": data} if data is not None else {}),
    }


def save_field(path: str | Path, data: xr.DataArray | xr.Dataset) -> None:
    ds = data.to_dataset(name=data.name or "field") if isinstance(data, xr.DataArray) else data
    ds.to_netcdf(path)


def write_csv(path: str | Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)

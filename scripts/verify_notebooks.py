"""Execute notebooks from a clean kernel (this interpreter's python3 kernel) and report pass/fail.

Usage: .venv/bin/python scripts/verify_notebooks.py [notebook.ipynb ...] [--write-outputs-to DIR]
With no notebook arguments every notebooks/*.ipynb is run. Stored notebooks are never modified.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_S = 600


def make_kernel_manager() -> KernelManager:
    """Kernel manager that launches THIS interpreter (sys.executable), whatever the python3 kernelspec says."""
    km = KernelManager(kernel_name="python3")
    km.kernel_cmd = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    return km


def run_one(path: Path, out_dir: Path | None) -> bool:
    nb = nbformat.read(path, as_version=4)  # a copy in memory; the file on disk is never written
    t0 = time.perf_counter()
    try:
        client = NotebookClient(nb, timeout=TIMEOUT_S, km=make_kernel_manager(),
                                resources={"metadata": {"path": str(path.parent)}})
        client.execute(cwd=str(path.parent))
        ok, msg = True, ""
    except Exception as exc:  # report any failure and keep going with other notebooks
        ok, msg = False, f"{type(exc).__name__}: {str(exc)[-1500:]}"
    dt = time.perf_counter() - t0
    print(f"{'PASS' if ok else 'FAIL'}  {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}  {dt:6.1f} s")
    if msg:
        print(msg)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out_dir / path.name)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("notebooks", nargs="*", type=Path)
    ap.add_argument("--write-outputs-to", type=Path, default=None, metavar="DIR")
    args = ap.parse_args()
    paths = [p.resolve() for p in args.notebooks] or sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not paths:
        print("no notebooks found")
        return 1
    print(f"interpreter {sys.executable}")
    results = [run_one(p, args.write_outputs_to) for p in paths]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

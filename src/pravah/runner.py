"""Small helpers shared by the study drivers: progress reporting, figure saving, frozen-hash checks, output guards.

Only behaviour that was identical in every copy lives here; the study-specific payloads (what is hashed, what the run guard records)
stay with the study that owns them.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .io import write_json

Progress: TypeAlias = Callable[[str], None] | None


def say(progress: Progress, msg: str) -> None:
    if progress is not None:
        progress(msg)


def save_fig(out: Path, fig: Figure, name: str, figures: list[str]) -> None:
    """Write ``fig`` to ``out/figures/name`` (dpi 110, tight box), close it and append ``name`` to ``figures``."""
    d = out / "figures"
    d.mkdir(exist_ok=True)
    fig.savefig(d / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    figures.append(name)


def check_frozen(out: Path, now: str) -> str:
    """Freeze guard: the hash recorded in ``out/provenance.json`` before test evaluation must equal ``now``."""
    rec = json.loads((out / "provenance.json").read_text())["frozen_config_hash"]
    if rec != now:
        raise RuntimeError(f"freeze guard: frozen config hash changed ({rec} -> {now})")
    return now


def guard_out(out: Path, rec: dict) -> None:
    """Output dir must be empty (new) or carry a ``run_guard.json`` equal to ``rec`` (resume). Writes the guard for a new dir."""
    gp = out / "run_guard.json"
    if out.exists() and any(out.iterdir()):
        if not gp.exists():
            raise RuntimeError(f"{out} is not empty and has no run_guard.json; use a new --out")
        if json.loads(gp.read_text()) != rec:
            raise RuntimeError(f"{out} was made with a different frozen config or run selection; use a new --out")
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(gp, rec)

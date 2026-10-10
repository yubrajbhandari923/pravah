"""Numerical refinement studies for the forward model. Usage: python scripts/verification.py --out DIR

Writes DIR/verification.json and DIR/verification.csv (NaN -> null in JSON).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from pravah.config import (Boundary, Domain, Initial, Scenario, Times, Transport, WindSegment, load_config,
                           scenario_from_dict)
from pravah.config import SolverOptions
from pravah.forward import FIPY  # physics verification OF the FiPy reference: FiPy is used explicitly
from pravah.sources import cell_centres, quadrature_ratio

ROOT = Path(__file__).resolve().parents[1]
LAM = 1.0 / 21600.0


def fourier_error(nx: int, dt: float, T: float = 1800.0) -> float:
    """Periodic advection-diffusion-decay of a Gaussian vs the exact translated/spread/decayed solution."""
    L, s0, K, u, v = 10000.0, 1000.0, 100.0, 1.0, 0.3
    sc = Scenario(domain=Domain(L, L, nx, nx), transport=Transport((WindSegment(0.0, u, v),), K, LAM, 500.0),
                  boundary=Boundary("periodic"), initial=Initial("gaussian", 1.0, 5000.0, 5000.0, s0),
                  times=Times(T, dt, T))
    c = FIPY(sc, [0.0, 0.0, 0.0]).concentration.values[-1]
    x, y = cell_centres(sc)
    X, Y = np.meshgrid(x, y)
    s2 = s0**2 + 2 * K * T
    exact = sum(np.exp(-((X - 5000 - u * T - i * L) ** 2 + (Y - 5000 - v * T - j * L) ** 2) / (2 * s2))
                for i in range(-2, 3) for j in range(-2, 3)) * s0**2 / s2 * math.exp(-LAM * T)
    return float(np.linalg.norm(c - exact) / np.linalg.norm(exact))


def keff_excess(nx: int, dt: float, conv: str, T: float = 3600.0) -> tuple[float, float]:
    """Measured K_eff/K - 1 and theory, from the decay of the m=1 Fourier mode of 1 + cos(kx) (periodic, lam=0,
    u=1 along x). Theory: spatial part + u^2 dt/(2K); spatial = (Pe/2)coth(Pe/2) - 1 (exponential) or Pe/2 (upwind)."""
    L, K, u = 10000.0, 100.0, 1.0
    sc = Scenario(domain=Domain(L, L, nx, 4), transport=Transport((WindSegment(0.0, u, 0.0),), K, 0.0, 500.0),
                  boundary=Boundary("periodic"), times=Times(T, dt, T))
    x = (np.arange(nx) + 0.5) * L / nx
    k = 2 * math.pi / L
    c = FIPY(sc, [0, 0, 0], SolverOptions(conv), initial_field=np.tile(1 + np.cos(k * x), (4, 1)))
    a = abs(np.mean(c.concentration.values[-1] * np.exp(-1j * k * x)[None, :]))
    pe = u * (L / nx) / K
    spatial = 0.5 * pe / math.tanh(0.5 * pe) - 1 if conv == "exponential" else 0.5 * pe
    return -math.log(a / 0.5) / (k * k * T) / K - 1.0, spatial + u * u * dt / (2 * K)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def final_mass(sc: Scenario, c: np.ndarray) -> float:
    return sc.transport.H * sc.domain.dx * sc.domain.dy * float(c.sum())


def run(out: Path) -> list[dict[str, Any]]:
    cfg = load_config(ROOT / "configs" / "reference.json")
    ref: Scenario = scenario_from_dict(cfg["scenario"])
    q: list[float] = cfg["q"]
    rows: list[dict[str, Any]] = []

    for nx in (20, 40, 80):  # dt fixed small so the time error stays below the spatial error
        rows.append({"study": "fourier_spatial", "label": f"nx={nx},dt=15", "nx": nx, "dt": 15.0,
                     "value": fourier_error(nx, 15.0), "metric": "relative_L2_error_vs_exact"})
    for dt in (60.0, 30.0, 15.0, 7.5):
        rows.append({"study": "fourier_time", "label": f"nx=40,dt={dt}", "nx": 40, "dt": dt,
                     "value": fourier_error(40, dt), "metric": "relative_L2_error_vs_exact"})

    # dt refinement on the reference case: compare successive solutions at the common output times
    fields: dict[float, np.ndarray] = {}
    for dt in (60.0, 30.0, 15.0):
        sc = dc.replace(ref, times=dc.replace(ref.times, dt=dt))
        r = FIPY(sc, q)
        fields[dt] = r.concentration.values
        rows.append({"study": "dt_refinement", "label": f"dt={dt}", "dt": dt, "metric": "final_mass_ug",
                     "value": final_mass(sc, fields[dt][-1]), "runtime_s": r.runtime_s,
                     "n_steps": r.diagnostics["n_steps"], "min_concentration": r.diagnostics["min_concentration"],
                     "max_rel_budget_residual": r.diagnostics["max_rel_budget_residual"]})
    for a, b in ((60.0, 30.0), (30.0, 15.0)):
        rows.append({"study": "dt_refinement", "label": f"dt {a} vs {b}", "metric": "field_RMSE_change_ug_m3",
                     "value": rmse(fields[a], fields[b]),
                     "rel_to_max": rmse(fields[a], fields[b]) / float(fields[b].max()),
                     "mass_change_rel": final_mass(ref, fields[a][-1]) / final_mass(ref, fields[b][-1]) - 1})

    # grid refinement: identical continuous sources on 50x50 and 100x100; 100-grid block-averaged onto 50-grid
    sc100 = dc.replace(ref, domain=dc.replace(ref.domain, nx=100, ny=100))
    r50, r100 = FIPY(ref, q), FIPY(sc100, q)
    c100 = r100.concentration.values
    c100_on_50 = c100.reshape(c100.shape[0], 50, 2, 50, 2).mean(axis=(2, 4))
    c50 = r50.concentration.values
    rows.append({"study": "grid_refinement", "label": "50 vs 100", "metric": "field_RMSE_ug_m3",
                 "value": rmse(c50, c100_on_50), "rel_to_max": rmse(c50, c100_on_50) / float(c100.max()),
                 "final_mass_50": final_mass(ref, c50[-1]), "final_mass_100": final_mass(sc100, c100[-1]),
                 "mass_change_rel": final_mass(ref, c50[-1]) / final_mass(sc100, c100[-1]) - 1,
                 "max_50": float(c50.max()), "max_100": float(c100.max()),
                 "runtime_50_s": r50.runtime_s, "runtime_100_s": r100.runtime_s})
    for nx, sc in ((50, ref), (100, sc100)):
        for j, ratio in enumerate(quadrature_ratio(sc)):
            rows.append({"study": "quadrature_ratio", "label": f"nx={nx},source={j}", "nx": nx,
                         "metric": "discrete/continuous", "value": float(ratio)})
    for conv in ("exponential", "upwind"):  # numerical diffusion of the baseline scheme (documented model error)
        for nx in (25, 50, 100):
            for dt in (60.0, 30.0, 15.0, 10.0):
                meas, th = keff_excess(nx, dt, conv)
                rows.append({"study": "numerical_diffusion", "label": f"{conv},nx={nx},dt={dt}", "nx": nx, "dt": dt,
                             "convection": conv, "metric": "Keff/K-1_measured", "value": meas,
                             "theory": th, "keff_ratio": 1 + meas})
    return rows


def clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    out: Path = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    rows = run(out)
    (out / "verification.json").write_text(json.dumps([clean(r) for r in rows], indent=2, allow_nan=False))
    cols = sorted({k for r in rows for k in r})
    with open(out / "verification.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["study"], r["label"], r["metric"], r["value"])


if __name__ == "__main__":
    main()

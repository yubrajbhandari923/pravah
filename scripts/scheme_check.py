"""Numerical-scheme check of the forward model: exponential vs upwind vs vanleer convection (task T-026).

    python scripts/scheme_check.py [--out results/phase3-scheme]

Writes scheme_check.csv (long format: study, scheme, nx, dx_m, dt_s, courant, metric, value).  Studies:
  gauss_tied / gauss_fixed  periodic 2-D Gaussian vs the exact translating, spreading, decaying solution, nx = 20/40/80,
                            dt tied to dx (dt = 0.06 dx) or fixed (7.5 s); rel_L2, Keff_over_K, min value, order
  courant                   dx = 250 m, u = 1 m/s, v = 0, Courant 0.25 ... 4 (dt = C dx / u)
  open_steady               open boundaries, uniform wind, steady sources, nx = 25 vs an exponential nx = 100 reference
                            (cell-averaged): rel_L2_vs_ref, min value, max relative budget residual
Keff/K = 1 + (var_num - var_exact) / (2 K T) with var the isotropic second moment about the centroid, so the exact
field's own periodic-truncation bias cancels.  Not a verification of K itself."""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

from pravah.config import Boundary, Domain, Initial, Scenario, SolverOptions, Times, Transport, WindSegment
from pravah.forward import simulate
from pravah.sources import cell_centres

SCHEMES = ("exponential", "upwind", "vanleer")
LAM, K, L, S0 = 1.0 / 21600.0, 100.0, 10000.0, 1000.0
ROOT = Path(__file__).resolve().parents[1]


def _periodic(scheme: str, nx: int, dt: float, T: float, u: float, v: float) -> dict[str, float]:
    sc = Scenario(domain=Domain(L, L, nx, nx), transport=Transport((WindSegment(0.0, u, v),), K, LAM, 500.0),
                  boundary=Boundary("periodic", 0.0), initial=Initial("gaussian", 1.0, 5000.0, 5000.0, S0),
                  times=Times(T, dt, T))
    res = simulate(sc, [0.0, 0.0, 0.0], SolverOptions(convection=scheme))
    c = res.concentration.values[-1]
    x, y = cell_centres(sc)
    X, Y = np.meshgrid(x, y)
    s2 = S0**2 + 2 * K * T
    ex = sum(np.exp(-((X - 5000 - u * T - i * L) ** 2 + (Y - 5000 - v * T - j * L) ** 2) / (2 * s2))
             for i in range(-2, 3) for j in range(-2, 3)) * S0**2 / s2 * math.exp(-LAM * T)

    def var(f: np.ndarray) -> float:  # isotropic second moment about the centroid, field recentred (periodic)
        i0, j0 = np.unravel_index(np.argmax(f), f.shape)
        g = np.roll(np.roll(f, nx // 2 - i0, 0), nx // 2 - j0, 1)
        m = g.sum()
        cx, cy = (g.sum(0) * x).sum() / m, (g.sum(1) * y).sum() / m
        return float((g * ((X - cx) ** 2 + (Y - cy) ** 2)).sum() / m / 2)

    return {"rel_L2": float(np.linalg.norm(c - ex) / np.linalg.norm(ex)),
            "Keff_over_K": 1 + (var(c) - var(ex)) / (2 * K * T), "min_value": float(c.min()),
            "max_rel_budget_residual": float(res.diagnostics["max_rel_budget_residual"]),
            "courant": float(res.diagnostics.get("max_courant", (abs(u) + abs(v)) * dt / (L / nx)))}


def _open(scheme: str, nx: int, dt: float) -> tuple[np.ndarray, dict[str, float]]:
    sc = Scenario(domain=Domain(L, L, nx, nx), transport=Transport((WindSegment(0.0, 1.0, 0.3),), K, LAM, 500.0),
                  boundary=Boundary("open", 0.5), times=Times(7200.0, dt, 7200.0))
    r = simulate(sc, [50.0, 80.0, 30.0], SolverOptions(convection=scheme))
    return r.concentration.values[-1], {"min_value": float(r.concentration.values.min()),
                                        "max_rel_budget_residual": float(r.diagnostics["max_rel_budget_residual"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "results" / "phase3-scheme"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    def add(study: str, scheme: str, nx: int, dt: float, metric: str, value: float, courant: float = float("nan")) -> None:
        rows.append({"study": study, "scheme": scheme, "nx": nx, "dx_m": L / nx, "dt_s": dt, "courant": courant,
                     "metric": metric, "value": value})

    t0 = time.time()
    for study, dt_of in (("gauss_tied", lambda nx: 0.06 * L / nx), ("gauss_fixed", lambda nx: 7.5)):
        for s in SCHEMES:
            errs = []
            for nx in (20, 40, 80):
                r = _periodic(s, nx, dt_of(nx), 1800.0, 1.0, 0.3)
                errs.append(r["rel_L2"])
                for k in ("rel_L2", "Keff_over_K", "min_value", "max_rel_budget_residual"):
                    add(study, s, nx, dt_of(nx), k, r[k], r["courant"])
            for a, b, nx in zip(errs, errs[1:], (40, 80)):
                add(study, s, nx, dt_of(nx), "order_L2", math.log2(a / b))
    for s in SCHEMES:
        for C in (0.25, 0.5, 1.0, 2.0, 4.0):
            r = _periodic(s, 40, 250.0 * C, 4000.0, 1.0, 0.0)
            for k in ("rel_L2", "Keff_over_K", "min_value", "max_rel_budget_residual"):
                add("courant", s, 40, 250.0 * C, k, r[k], C)
    ref, _ = _open("exponential", 100, 30.0)
    ref_c = ref.reshape(25, 4, 25, 4).mean((1, 3))
    for s in SCHEMES:
        c, d = _open(s, 25, 120.0)
        add("open_steady", s, 25, 120.0, "rel_L2_vs_ref", float(np.linalg.norm(c - ref_c) / np.linalg.norm(ref_c)))
        for k, val in d.items():
            add("open_steady", s, 25, 120.0, k, val)
    with open(out / "scheme_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out / 'scheme_check.csv'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

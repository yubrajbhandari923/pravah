"""convection='vanleer' (FiPy VanLeerConvectionTerm, explicit TVD, sub-stepped to Courant <= 0.5) vs the legacy schemes.
Small grids.  Reference numbers (results/phase3-scheme/scheme_check.csv, scripts/scheme_check.py) are quoted in comments."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pravah.config import (Boundary, Domain, Initial, Scenario, SolverOptions, Times, Transport, WindSegment)
from pravah.forward import FIPY, FipyForward, simulate
from pravah.inverse import ResponseCache, cache_key
from pravah.sources import cell_centres

LAM, K, L, S0 = 1.0 / 21600.0, 100.0, 10000.0, 1000.0
SCHEMES = ("exponential", "upwind", "vanleer")


def _gauss(scheme: str, nx: int, dt: float, T: float = 1800.0, u: float = 1.0, v: float = 0.3):
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
    return float(np.linalg.norm(c - ex) / np.linalg.norm(ex)), res


def _open(scheme: str, nx: int, dt: float):
    sc = Scenario(domain=Domain(L, L, nx, nx), transport=Transport((WindSegment(0.0, 1.0, 0.3),), K, LAM, 500.0),
                  boundary=Boundary("open", 0.5), times=Times(3600.0, dt, 3600.0))
    return simulate(sc, [50.0, 80.0, 30.0], SolverOptions(convection=scheme))


def test_gaussian_error_and_order_beat_legacy():
    # dt fixed 7.5 s (at 15 s the first-order time error floors the vanleer order at the finest level).  Measured: L2 0.042/0.0080/0.0023 (order 2.4, 1.8) vs exponential
    # 0.152/0.057/0.019 (order 1.4, 1.6) and upwind 0.23/0.14/0.077 (order 0.8).
    e = {s: [_gauss(s, n, 7.5)[0] for n in (20, 40, 80)] for s in SCHEMES}
    for a, b in zip(e["vanleer"], e["exponential"]):
        assert a < 0.5 * b
    assert all(x > 1.5 for x in (math.log2(e["vanleer"][0] / e["vanleer"][1]), math.log2(e["vanleer"][1] / e["vanleer"][2])))
    assert e["upwind"][2] > e["exponential"][2]


def test_budget_closure_and_positivity():
    for res in (_gauss("vanleer", 40, 15.0)[1], _open("vanleer", 25, 120.0)):
        assert res.diagnostics["max_rel_budget_residual"] < 1e-12  # measured ~1e-15 .. 3e-13
        assert res.diagnostics["n_negative"] == 0, res.diagnostics["min_concentration"]
        assert res.diagnostics["min_concentration"] >= 0.0


@pytest.mark.parametrize("C", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_courant_sweep_stable_and_more_accurate(C):
    # dx = 250 m, u = 1 m/s so Courant = dt/250.  Bare explicit vanleer (no sub-stepping) went unstable at C >= 2
    # (min -1.4e-3, -0.37); with sub-stepping it is stable and its error is flat at ~0.006 for C >= 0.5.
    ev, rv = _gauss("vanleer", 40, 250.0 * C, T=4000.0, v=0.0)
    ee, _ = _gauss("exponential", 40, 250.0 * C, T=4000.0, v=0.0)
    assert rv.diagnostics["min_concentration"] >= 0.0
    assert ev < 0.5 * ee
    assert rv.diagnostics["max_substeps"] == max(1, math.ceil(C / 0.5))


def test_open_boundary_closer_to_fine_reference():
    ref = _open("exponential", 100, 30.0).concentration.values[-1].reshape(25, 4, 25, 4).mean((1, 3))
    err = {s: float(np.linalg.norm(_open(s, 25, 120.0).concentration.values[-1] - ref) / np.linalg.norm(ref))
           for s in ("exponential", "vanleer")}
    assert err["vanleer"] < err["exponential"]  # measured 0.0163 vs 0.0447 (T = 2 h)


def test_legacy_schemes_unchanged_diagnostics():
    for s in ("exponential", "upwind"):
        d = _open(s, 25, 120.0).diagnostics
        assert "convection" not in d and "max_substeps" not in d


def test_cache_keys_distinct_and_linear_route_refused():
    sc = Scenario(domain=Domain(L, L, 10, 10), transport=Transport((WindSegment(0.0, 1.0, 0.3),), K, LAM, 500.0),
                  boundary=Boundary("open", 0.0), times=Times(600.0, 60.0, 600.0))
    keys = {cache_key(sc, SolverOptions(convection=s), 1e5) for s in SCHEMES}
    keys |= {cache_key(sc, SolverOptions(), 1e5, FipyForward(convection=s)) for s in SCHEMES}
    assert len(keys) == 6  # solver option and forward override are both distinct, never shared
    assert FipyForward(convection="vanleer").info.settings["convection"] == "vanleer"
    assert FipyForward(convection="vanleer").info.capabilities["linear"] is False
    for args in ((SolverOptions(convection="vanleer"), FIPY), (SolverOptions(), FipyForward(convection="vanleer"))):
        with pytest.raises(ValueError, match="vanleer"):
            ResponseCache().get(sc, args[0], forward=args[1])
    with pytest.raises(ValueError):
        SolverOptions(convection="bogus")

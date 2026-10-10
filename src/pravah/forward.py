"""FiPy advection-diffusion-removal forward model (see docs/implementation-plan.md, "FiPy discretisation").

    TransientTerm + ExponentialConvectionTerm(u_int) == DiffusionTerm(K) - ImplicitSourceTerm(lam + a_out + a_in)
                                                         + S_q + S_in

``u_int`` is the wind on interior faces and ZERO on exterior faces, so FiPy itself contributes no boundary flux;
open-boundary inflow/outflow is handled explicitly (and the budget uses the very same coefficients).
Backward Euler in time. Concentrations are never clipped.
"""
from __future__ import annotations

import math
import time as _time
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import xarray as xr
import fipy as _fipy

from .config import FieldResult, Scenario, SolverOptions, WindSegment
from .sources import cell_centres, emission_profile, source_fields

fp: Any = _fipy  # FiPy is untyped; use it through Any

_TOL = 1e-9  # s; times closer than this are the same breakpoint


def step_times(scenario: Scenario) -> np.ndarray:
    """Time points of the implicit steps. Breakpoints = output times, wind-segment starts, emission-profile
    breakpoints (inside the run); each interval between breakpoints is split in ceil(len/dt) equal steps."""
    T, dt = scenario.times.duration, scenario.times.dt
    pts = list(scenario.output_times) + [w.t_start for w in scenario.transport.wind]
    pts += [t for s in scenario.sources for t, _ in s.profile]
    pts = sorted(p for p in pts if 0.0 <= p <= T + _TOL)
    bps: list[float] = []
    for p in pts:
        if not bps or p - bps[-1] > _TOL:
            bps.append(min(p, T))
    out = [bps[0]]
    for a, b in zip(bps, bps[1:]):
        n = max(1, math.ceil((b - a) / dt - 1e-9))
        out.extend(np.linspace(a, b, n + 1)[1:])
    return np.array(out)


def _wind_at(scenario: Scenario, t: float) -> int:
    idx = 0
    for i, w in enumerate(scenario.transport.wind):
        if t >= w.t_start - _TOL:
            idx = i
    return idx


def _boundary_coefficients(scenario: Scenario, w: WindSegment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell (ny, nx) a_out [1/s], a_in [1/s], S_in [ug/m^3/s] for the open-boundary faces, classified per face
    by the sign of u.n for this wind segment."""
    d, K, cb = scenario.domain, scenario.transport.K, scenario.boundary.background
    a_out, a_in, s_in = (np.zeros((d.ny, d.nx)) for _ in range(3))
    # (cell slice, u.n, cell size normal to the face): face area / cell volume = 1 / size
    sides = [
        (np.s_[:, 0], -w.u, d.dx), (np.s_[:, -1], w.u, d.dx),
        (np.s_[0, :], -w.v, d.dy), (np.s_[-1, :], w.v, d.dy),
    ]
    for sl, un, h in sides:
        if un > 0:  # outflow: upwind export of the boundary-cell value, no diffusive flux
            a_out[sl] += un / h
        elif un < 0:  # inflow: face value prescribed c_b (advective + Dirichlet diffusive flux)
            a_in[sl] += 2 * K / h**2
            s_in[sl] += (-un + 2 * K / h) * cb / h
    return a_out, a_in, s_in


class _Segment:
    """FiPy objects for one wind segment, built once; only the explicit source variable changes per step."""

    def __init__(self, scenario: Scenario, solver: SolverOptions, w: WindSegment, source: Any, grid: Any) -> None:
        K = scenario.transport.K
        if solver.convection == "exponential" and K <= 0:
            raise ValueError("exponential convection needs K > 0 (use convection='upwind' for K = 0)")
        d = scenario.domain
        self.periodic = scenario.boundary.mode == "periodic"
        if self.periodic:
            self.a_out = self.a_in = self.s_in = np.zeros((d.ny, d.nx))
        else:
            self.a_out, self.a_in, self.s_in = _boundary_coefficients(scenario, w)
        face_u = fp.FaceVariable(mesh=grid, rank=1, value=[[w.u], [w.v]])
        if not self.periodic:
            face_u = face_u * ~grid.exteriorFaces  # zero on exterior faces
        self.source = source
        coef = fp.CellVariable(mesh=grid, value=scenario.transport.lam + (self.a_out + self.a_in).reshape(-1))
        conv_cls = fp.ExponentialConvectionTerm if solver.convection == "exponential" else fp.UpwindConvectionTerm
        self.eq = (fp.TransientTerm() + conv_cls(coeff=face_u)
                   == fp.DiffusionTerm(coeff=K) - fp.ImplicitSourceTerm(coeff=coef) + source)


def _validate_q(scenario: Scenario, q: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(q, dtype=float)
    if arr.ndim != 1 or arr.size != len(scenario.sources):
        raise ValueError(f"q must be 1-D with {len(scenario.sources)} entries, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("q must be finite and >= 0")
    return arr


def _initial_field(scenario: Scenario) -> np.ndarray:
    ini = scenario.initial
    x, y = cell_centres(scenario)
    shape = (scenario.domain.ny, scenario.domain.nx)
    if ini.kind == "zero":
        return np.zeros(shape)
    if ini.kind == "uniform":
        return np.full(shape, ini.value)
    r2 = (x[None, :] - ini.x) ** 2 + (y[:, None] - ini.y) ** 2
    return ini.value * np.exp(-r2 / (2 * ini.sigma**2))


def simulate(scenario: Scenario, q: Sequence[float] | np.ndarray,
             solver: SolverOptions = SolverOptions()) -> FieldResult:
    """Run the forward model for emission rates q [ug/s] (one per source). Raises ValueError for invalid input and
    RuntimeError if the solution is not finite."""
    t_wall = _time.perf_counter()
    qa = _validate_q(scenario, q)
    d, tr, T = scenario.domain, scenario.transport, scenario.times
    if scenario.boundary.mode == "closed" and any(w.u != 0 or w.v != 0 for w in tr.wind):
        raise ValueError("closed boundaries require zero wind")
    if scenario.boundary.mode == "periodic":
        grid: Any = fp.PeriodicGrid2D(dx=d.dx, dy=d.dy, nx=d.nx, ny=d.ny)
    else:
        grid = fp.Grid2D(dx=d.dx, dy=d.dy, nx=d.nx, ny=d.ny)
    shape = (d.ny, d.nx)
    dA = d.dx * d.dy
    f = source_fields(scenario)  # (n_src, ny, nx)
    var = fp.CellVariable(mesh=grid, value=_initial_field(scenario).reshape(-1), hasOld=True)
    source = fp.CellVariable(mesh=grid, value=0.0)
    lu = fp.LinearLUSolver(tolerance=solver.tolerance)
    segments: dict[int, _Segment] = {}

    times = step_times(scenario)
    out_times = np.array(scenario.output_times)
    frames = [var.value.reshape(shape).copy()]
    n_out = len(out_times) - 1
    keys = ("emission", "removal", "inflow", "outflow", "mass_change")
    acc = {k: np.zeros(n_out) for k in keys}
    i_out = 0
    for t0, t1 in zip(times[:-1], times[1:]):
        h = float(t1 - t0)
        tm = 0.5 * (t0 + t1)
        k_w = _wind_at(scenario, tm)
        seg = segments.get(k_w)
        if seg is None:
            seg = segments[k_w] = _Segment(scenario, solver, tr.wind[k_w], source, grid)
        s_vec = np.array([emission_profile(s, tm) for s in scenario.sources])
        s_q = np.tensordot(qa * s_vec, f, axes=1) / tr.H  # (ny, nx), ug/m^3/s
        source.setValue((s_q + seg.s_in).reshape(-1))
        c_old = var.value.reshape(shape).copy()
        var.updateOld()
        seg.eq.solve(var=var, dt=h, solver=lu)
        c = var.value.reshape(shape)
        while i_out < n_out and tm > out_times[i_out + 1]:
            i_out += 1
        # budget terms from the same implicit (n+1) boundary coefficients used in the equation
        acc["emission"][i_out] += h * float(np.dot(qa, s_vec))
        acc["removal"][i_out] += h * tr.H * tr.lam * float(c.sum()) * dA
        acc["inflow"][i_out] += h * tr.H * dA * float((seg.s_in - seg.a_in * c).sum())
        acc["outflow"][i_out] += h * tr.H * dA * float((seg.a_out * c).sum())
        acc["mass_change"][i_out] += tr.H * dA * float((c - c_old).sum())
        if t1 >= out_times[i_out + 1] - _TOL and i_out < n_out:
            frames.append(c.copy())
    data = np.stack(frames)
    if not np.all(np.isfinite(data)):
        raise RuntimeError("non-finite concentrations in forward solution")

    x, y = cell_centres(scenario)
    conc = xr.DataArray(data, dims=("time", "y", "x"), coords={"time": out_times, "x": x, "y": y},
                        name="concentration", attrs={"units": "microgram/m^3"})
    budget = xr.Dataset(
        {k: ("time", v) for k, v in acc.items()},
        coords={"time": out_times[1:], "t_start": ("time", out_times[:-1])},
        attrs={"units": "microgram", "mass": "M = H * sum(c * cell_area)"},
    )
    budget["residual"] = budget["mass_change"] - (budget["emission"] - budget["removal"] + budget["inflow"]
                                                 - budget["outflow"])
    scale = np.maximum.reduce([np.abs(budget[k].values) for k in keys])
    scale = np.maximum(scale, np.finfo(float).tiny)
    resid = np.abs(budget["residual"].values)
    diagnostics = {
        "min_concentration": float(data.min()),
        "n_negative": int((data < 0).sum()),
        "n_steps": int(len(times) - 1),
        "max_abs_budget_residual": float(resid.max()),
        "max_rel_budget_residual": float((resid / scale).max()),
    }
    return FieldResult(concentration=conc, budget=budget, status="ok",
                       runtime_s=_time.perf_counter() - t_wall, diagnostics=cast(dict[str, Any], diagnostics))

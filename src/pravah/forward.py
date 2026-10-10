"""FiPy advection-diffusion-removal forward model (see docs/implementation-plan.md, "FiPy discretisation").

    TransientTerm + ExponentialConvectionTerm(u_int) == DiffusionTerm(K) - ImplicitSourceTerm(lam + a_out + a_in)
                                                         + S_q + S_in

``u_int`` is the wind on interior faces and ZERO on exterior faces, so FiPy itself contributes no boundary flux;
open-boundary inflow/outflow is handled explicitly (and the budget uses the very same coefficients).
Backward Euler in time. Concentrations are never clipped.
"""
from __future__ import annotations

import dataclasses
import importlib
import math
import time as _time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import xarray as xr
import fipy as _fipy

from . import __version__
from .config import FieldResult, Scenario, SolverOptions, WindSegment
from .sources import cell_centres, emission_profile, source_fields

fp: Any = _fipy  # FiPy is untyped; use it through Any

_TOL = 1e-9  # s; times closer than this are the same breakpoint
_FIPY_VERSION = str(getattr(_fipy, "__version__", "unknown"))


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
             solver: SolverOptions = SolverOptions(), initial_field: np.ndarray | None = None) -> FieldResult:
    """Run the forward model for emission rates q [ug/s] (one per source). Raises ValueError for invalid input and
    RuntimeError if the solution is not finite.

    ``initial_field`` (shape (ny, nx), for verification of the linear operator only) overrides ``scenario.initial``;
    it may be negative. Ordinary configs leave it None; it is not part of any cache key (the inverse module never
    passes it)."""
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
    if initial_field is None:
        c0 = _initial_field(scenario)
    else:
        c0 = np.asarray(initial_field, dtype=float)
        if c0.shape != shape or not np.all(np.isfinite(c0)):
            raise ValueError(f"initial_field must be finite with shape {shape}")
    var = fp.CellVariable(mesh=grid, value=c0.reshape(-1), hasOld=True)
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
        "backend": FIPY.info.to_dict(),  # provenance only; FipyForward overwrites it with its own settings
    }
    return FieldResult(concentration=conc, budget=budget, status="ok",
                       runtime_s=_time.perf_counter() - t_wall, diagnostics=cast(dict[str, Any], diagnostics))


@dataclass(frozen=True)
class BackendInfo:
    """Identity of a forward solver, recorded in cache keys, provenance and ``FieldResult.diagnostics['backend']``.

    ``capabilities`` (all False unless declared): ``linear`` = superposition of source responses is valid (required
    by the cached-response route, never assumed), ``budget`` = results carry a mass budget, ``boundary_bases`` =
    supports boundary-inflow basis parameters (False for FiPy until the phase-2 physics lands)."""
    name: str
    version: str
    settings: dict[str, Any]
    capabilities: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "settings": dict(self.settings),
                "capabilities": dict(self.capabilities)}


class ForwardSolver(Protocol):
    """Interface a replacement solver (e.g. a neural surrogate for a fixed family of scenarios) must satisfy.

    It is called exactly like ``simulate`` and must return a ``FieldResult`` whose ``concentration`` has dims
    ("time", "y", "x"), coords time = ``scenario.output_times``, x/y = cell centres (``sources.cell_centres``),
    attrs["units"] == "microgram/m^3", and finite values.  ``budget`` may be None; ``status`` is a string.
    A solver only needs to support the scenarios it was built for.  Optionally give it an ``info`` attribute
    (``BackendInfo``) or, minimally, a ``forward_id`` string (e.g. a weights hash) and a ``capabilities`` dict;
    see ``forward_info``.  The cached-response route needs ``capabilities['linear'] is True``.
    ``check_field_result`` validates the contract."""

    def __call__(self, scenario: Scenario, q: Sequence[float] | np.ndarray,
                 solver: SolverOptions = ...) -> FieldResult: ...


@dataclass(frozen=True)
class FipyForward:
    """The FiPy reference solver as a ``ForwardSolver``.  ``convection`` / ``tolerance`` (when not None) override the
    corresponding fields of the per-call ``SolverOptions`` and are recorded in ``info.settings``.  With no overrides
    it is exactly ``simulate``."""
    convection: str | None = None
    tolerance: float | None = None

    @property
    def info(self) -> BackendInfo:
        return BackendInfo("fipy", f"fipy {_FIPY_VERSION}; pravah {__version__}",
                           {"convection": self.convection, "tolerance": self.tolerance},
                           {"linear": True, "budget": True, "boundary_bases": False})

    @property
    def forward_id(self) -> str:
        return f"fipy:convection={self.convection}:tolerance={self.tolerance}"

    def __call__(self, scenario: Scenario, q: Sequence[float] | np.ndarray,
                 solver: SolverOptions = SolverOptions()) -> FieldResult:
        over = {k: v for k, v in (("convection", self.convection), ("tolerance", self.tolerance)) if v is not None}
        res = simulate(scenario, q, dataclasses.replace(solver, **over) if over else solver)
        res.diagnostics["backend"] = self.info.to_dict()
        return res


FIPY = FipyForward()  # THE default forward solver of every library entry point and the CLI


def forward_info(forward: ForwardSolver) -> BackendInfo:
    """``forward.info`` if it has one; ``simulate`` is FIPY; otherwise derived from identity: name = module.qualname,
    version 'unknown', settings = its ``forward_id`` (if any), capabilities from its ``capabilities`` dict.  An
    undeclared replacement is NOT assumed linear."""
    if forward is simulate:
        return FIPY.info
    info = getattr(forward, "info", None)
    if isinstance(info, BackendInfo):
        return info
    obj: Any = forward
    name = getattr(obj, "__qualname__", None)
    if name is None:
        obj = type(forward)
        name = obj.__qualname__
    caps = {"linear": False, "budget": False, "boundary_bases": False}
    caps.update({str(k): bool(v) for k, v in dict(getattr(forward, "capabilities", {})).items()})
    return BackendInfo(f"{obj.__module__}.{name}", "unknown", {"forward_id": str(getattr(forward, "forward_id", ""))},
                       caps)


def get_forward(spec: str | None, kwargs: dict[str, Any] | None = None) -> ForwardSolver:
    """Resolve a solver spec.  None or 'fipy' -> FIPY (``FipyForward(**kwargs)`` when kwargs are given).
    'module:callable' -> the callable is imported and treated as a FACTORY: it is called once with ``**kwargs`` and
    must return the solver object (a ``ForwardSolver``).  ValueError for malformed specs, import failures, factory
    errors and non-callable results."""
    if spec is None or spec == "fipy":
        if not kwargs:
            return FIPY
        try:
            return FipyForward(**kwargs)
        except TypeError as e:
            raise ValueError(f"invalid kwargs for the fipy solver: {e}") from e
    mod, sep, attr = spec.partition(":")
    if not sep or not mod or not attr:
        raise ValueError(f"solver spec {spec!r} must be 'fipy' or 'module:callable'")
    try:
        factory = getattr(importlib.import_module(mod), attr)
    except ImportError as e:
        raise ValueError(f"cannot import solver module {mod!r} from spec {spec!r}: {e}") from e
    except AttributeError as e:
        raise ValueError(f"module {mod!r} has no attribute {attr!r} (solver spec {spec!r})") from e
    if not callable(factory):
        raise ValueError(f"solver spec {spec!r} does not name a callable factory")
    try:
        solver = factory(**(kwargs or {}))
    except Exception as e:
        raise ValueError(f"solver factory {spec!r} failed with kwargs {kwargs or {}}: {e}") from e
    if not callable(solver):
        raise ValueError(f"solver factory {spec!r} returned a non-callable {type(solver).__name__}")
    return cast(ForwardSolver, solver)


def check_field_result(res: FieldResult, scenario: Scenario) -> None:
    """Raise ValueError (clear message) if ``res`` violates the ``ForwardSolver`` output contract for ``scenario``."""
    da = res.concentration
    if tuple(da.dims) != ("time", "y", "x"):
        raise ValueError(f"concentration dims must be ('time', 'y', 'x'), got {tuple(da.dims)}")
    x, y = cell_centres(scenario)
    for name, want in (("time", np.asarray(scenario.output_times, dtype=float)), ("y", y), ("x", x)):
        if name not in da.coords:
            raise ValueError(f"concentration is missing coordinate {name!r}")
        got = np.asarray(da[name].values, dtype=float)
        if got.shape != want.shape or not np.allclose(got, want, rtol=0, atol=1e-9):
            raise ValueError(f"coordinate {name!r} differs from the scenario's "
                             f"{'output times' if name == 'time' else 'cell centres'} (shape {got.shape} vs "
                             f"{want.shape}, max |diff| "
                             f"{np.max(np.abs(got - want)) if got.shape == want.shape else float('nan'):.3g})")
    if da.attrs.get("units") != "microgram/m^3":
        raise ValueError(f"concentration attrs['units'] must be 'microgram/m^3', got {da.attrs.get('units')!r}")
    if not np.all(np.isfinite(np.asarray(da.values, dtype=float))):
        raise ValueError("concentration contains non-finite values")
    if not isinstance(res.status, str):
        raise ValueError(f"status must be a str, got {type(res.status).__name__}")

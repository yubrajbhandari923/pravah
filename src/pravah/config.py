"""Configuration dataclasses (frozen, JSON-serialisable) and the shared public result types.

Conventions (used everywhere):
* x, y in metres; t in seconds; concentration in microgram/m^3.
* Grid cells are indexed k = i + j*nx (FiPy order), so ``values.reshape(ny, nx)`` is ``[y, x]``.
* Arrays carry coordinates at cell centres: x = (i+0.5)*dx, y = (j+0.5)*dy.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import xarray as xr

SCHEMA_VERSION = "1"  # bump whenever forward-relevant behaviour changes (part of every cache key)


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@dataclass(frozen=True)
class Domain:
    lx: float = 10_000.0  # m
    ly: float = 10_000.0  # m
    nx: int = 50
    ny: int = 50

    def __post_init__(self) -> None:
        _need(self.lx > 0 and self.ly > 0, "domain lengths must be positive")
        _need(self.nx >= 2 and self.ny >= 2, "need at least 2 cells per direction")

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny


@dataclass(frozen=True)
class GaussianSource:
    """Spatial profile f(x,y) ~ exp(-r^2/2 sigma^2), normalised on the grid so sum(f*cell_area)=1 (1/m^2).

    ``profile`` is a dimensionless piecewise-constant temporal factor: [(t_start, value), ...]
    (first t_start must be 0). Default constant 1.
    """
    name: str
    x: float
    y: float
    sigma: float
    profile: tuple[tuple[float, float], ...] = ((0.0, 1.0),)

    def __post_init__(self) -> None:
        _need(self.sigma > 0, f"source {self.name}: sigma must be positive")
        _need(len(self.profile) >= 1 and self.profile[0][0] == 0.0, "profile must start at t=0")
        ts = [p[0] for p in self.profile]
        _need(all(b > a for a, b in zip(ts, ts[1:])), "profile times must increase")
        _need(all(v >= 0 and math.isfinite(v) for _, v in self.profile), "profile values must be >= 0")


@dataclass(frozen=True)
class WindSegment:
    """Wind (u, v) in m/s applying from t_start (s) until the next segment starts."""
    t_start: float
    u: float
    v: float


@dataclass(frozen=True)
class Transport:
    wind: tuple[WindSegment, ...] = (WindSegment(0.0, 1.0, 0.3),)
    K: float = 100.0  # m^2/s
    lam: float = 1.0 / 21600.0  # 1/s
    H: float = 500.0  # m

    def __post_init__(self) -> None:
        _need(len(self.wind) >= 1 and self.wind[0].t_start == 0.0, "wind must start at t=0")
        ts = [w.t_start for w in self.wind]
        _need(all(b > a for a, b in zip(ts, ts[1:])), "wind segment times must increase")
        _need(self.K >= 0 and self.lam >= 0 and self.H > 0, "need K>=0, lam>=0, H>0")


@dataclass(frozen=True)
class Boundary:
    mode: str = "open"  # open | periodic | closed
    background: float = 0.0  # inflow concentration c_b (open mode), microgram/m^3

    def __post_init__(self) -> None:
        _need(self.mode in ("open", "periodic", "closed"), f"unknown boundary mode {self.mode!r}")
        _need(self.background >= 0, "background must be >= 0")


@dataclass(frozen=True)
class Initial:
    kind: str = "zero"  # zero | uniform | gaussian
    value: float = 0.0  # uniform level, or Gaussian peak (microgram/m^3)
    x: float = 0.0
    y: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        _need(self.kind in ("zero", "uniform", "gaussian"), f"unknown initial kind {self.kind!r}")
        _need(self.value >= 0 and self.sigma > 0, "initial value >= 0 and sigma > 0 required")


@dataclass(frozen=True)
class Times:
    duration: float = 14_400.0  # s
    dt: float = 60.0  # nominal step, s
    output_interval: float = 300.0  # s

    def __post_init__(self) -> None:
        _need(self.duration > 0 and self.dt > 0 and self.output_interval > 0, "times must be positive")
        _need(self.output_interval <= self.duration, "output_interval must not exceed duration")
        ratio = self.duration / self.output_interval
        _need(abs(ratio - round(ratio)) <= 1e-9 * max(1.0, ratio),
              f"duration ({self.duration}) must be an integer multiple of output_interval ({self.output_interval})")


@dataclass(frozen=True)
class SolverOptions:
    convection: str = "exponential"  # exponential | upwind
    tolerance: float = 1e-12  # relative residual tolerance of the linear solve

    def __post_init__(self) -> None:
        _need(self.convection in ("exponential", "upwind"), "convection must be 'exponential' or 'upwind'")


def _default_sources() -> tuple[GaussianSource, ...]:
    return (
        GaussianSource("S1", 2500.0, 2500.0, 500.0),
        GaussianSource("S2", 4000.0, 6000.0, 500.0),
        GaussianSource("S3", 6500.0, 3500.0, 500.0),
    )


@dataclass(frozen=True)
class Scenario:
    """Everything the forward model needs EXCEPT emission rates q (microgram/s), which are passed separately."""
    domain: Domain = field(default_factory=Domain)
    sources: tuple[GaussianSource, ...] = field(default_factory=_default_sources)
    transport: Transport = field(default_factory=Transport)
    boundary: Boundary = field(default_factory=Boundary)
    initial: Initial = field(default_factory=Initial)
    times: Times = field(default_factory=Times)

    def __post_init__(self) -> None:
        _need(len(self.sources) >= 1, "need at least one source")
        if self.boundary.mode == "closed":
            _need(all(w.u == 0 and w.v == 0 for w in self.transport.wind),
                  "closed boundaries require zero wind (a uniform wind cannot be tangent to every wall)")
        if self.boundary.mode == "periodic":
            _need(len(self.transport.wind) >= 1, "periodic mode needs a wind")
            _need(self.boundary.background == 0, "background is ignored in periodic mode; it must be 0")

    @property
    def output_times(self) -> list[float]:
        n = int(round(self.times.duration / self.times.output_interval))
        return [i * self.times.output_interval for i in range(n + 1)]


@dataclass
class FieldResult:
    """Solver-independent simulation output. No FiPy objects."""
    concentration: xr.DataArray  # (time, y, x), microgram/m^3
    budget: xr.Dataset | None  # per-output-interval mass budget (microgram); None for non-physical fields
    status: str  # "ok" or description of a problem
    runtime_s: float
    diagnostics: dict[str, Any] = field(default_factory=dict)  # min_concentration, n_negative, n_steps, max_budget_residual...


def to_dict(obj: Any) -> Any:
    """Dataclass tree -> plain JSON-able dict."""
    return json.loads(json.dumps(dataclasses.asdict(obj)))


def _wind(d: list[dict[str, Any]]) -> tuple[WindSegment, ...]:
    return tuple(WindSegment(**w) for w in d)


def scenario_from_dict(d: dict[str, Any]) -> Scenario:
    d = dict(d)
    tr = dict(d.get("transport", {}))
    if "wind" in tr:
        tr["wind"] = _wind(tr["wind"])
    srcs = tuple(
        GaussianSource(**{**s, "profile": tuple(tuple(p) for p in s["profile"])} if "profile" in s else s)
        for s in d["sources"]
    ) if "sources" in d else _default_sources()
    return Scenario(
        domain=Domain(**d.get("domain", {})),
        sources=srcs,
        transport=Transport(**tr),
        boundary=Boundary(**d.get("boundary", {})),
        initial=Initial(**d.get("initial", {})),
        times=Times(**d.get("times", {})),
    )


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON config. An optional top-level "extends": "other.json" (path relative to this file) is deep-merged
    under this file's values (dicts merge recursively; lists and scalars are replaced)."""
    path = Path(path)
    cfg = json.loads(path.read_text())
    base = cfg.pop("extends", None)
    return cfg if base is None else _merge(load_config(path.parent / base), cfg)

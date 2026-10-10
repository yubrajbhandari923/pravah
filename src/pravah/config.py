"""Configuration dataclasses (frozen, JSON-serialisable) and the shared public result types.

Conventions (used everywhere):
* x, y in metres; t in seconds; concentration in microgram/m^3.
* Grid cells are indexed k = i + j*nx (FiPy order), so ``values.reshape(ny, nx)`` is ``[y, x]``.
* Arrays carry coordinates at cell centres: x = (i+0.5)*dx, y = (j+0.5)*dy.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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
    group: str = "local"  # origin tag: "local" (inside the study region) or "external" (outside it)

    def __post_init__(self) -> None:
        _need(self.sigma > 0, f"source {self.name}: sigma must be positive")
        _check_profile(self.profile)
        _need(self.group in GROUPS, f"source {self.name}: group must be one of {GROUPS}")


GROUPS = ("local", "external")  # origin tags; "local" = emitted inside the study region, "external" = outside it


def _check_profile(profile: tuple[tuple[float, float], ...]) -> None:
    _need(len(profile) >= 1 and profile[0][0] == 0.0, "profile must start at t=0")
    ts = [p[0] for p in profile]
    _need(all(b > a for a, b in zip(ts, ts[1:])), "profile times must increase")
    _need(all(v >= 0 and math.isfinite(v) for _, v in profile), "profile values must be >= 0")


def array_sha256(a: np.ndarray) -> str:
    """sha256 of the float64 C-order bytes of ``a`` (the identity of a gridded input in cache keys and manifests)."""
    return hashlib.sha256(np.ascontiguousarray(a, dtype=float).tobytes()).hexdigest()


@dataclass(frozen=True)
class RasterSource:
    """Gridded emission shape on the simulation grid, e.g. road- or settlement-associated (synthetic spatial prior).

    ``weights`` (ny, nx) >= 0 is normalised on the grid by ``sources.source_fields`` so sum(f * cell_area) = 1 (1/m^2),
    exactly like a Gaussian source. Only ``sha256`` (of the weights) enters ``to_dict`` / cache keys; the array itself
    is excluded. ``profile`` as for GaussianSource.
    """
    name: str
    group: str
    weights: np.ndarray = field(compare=False, repr=False)
    sha256: str
    profile: tuple[tuple[float, float], ...] = ((0.0, 1.0),)

    def __post_init__(self) -> None:
        w = np.asarray(self.weights)
        _need(w.ndim == 2 and bool(np.all(np.isfinite(w))) and bool(np.all(w >= 0)) and float(w.sum()) > 0,
              f"raster source {self.name}: weights must be a finite, non-negative 2-D array with a positive sum")
        _need(self.group in GROUPS, f"source {self.name}: group must be one of {GROUPS}")
        _need(self.sha256 == array_sha256(w), f"raster source {self.name}: sha256 does not match the weights")
        _check_profile(self.profile)

    @classmethod
    def from_array(cls, name: str, group: str, weights: np.ndarray,
                   profile: tuple[tuple[float, float], ...] = ((0.0, 1.0),)) -> RasterSource:
        w = np.asarray(weights, dtype=float)
        return cls(name, group, w, array_sha256(w), profile)


Source = GaussianSource | RasterSource


def source_signature(src: Source) -> tuple[Any, ...]:
    """Hashable identity of a source's spatial definition (used to check that inference sources match truth sources)."""
    if isinstance(src, GaussianSource):
        return ("gaussian", src.x, src.y, src.sigma, src.group)
    return ("raster", src.sha256, src.group)


@dataclass(frozen=True)
class StudyRegion:
    """Rectangular study region (m, simulation coordinates): defines local/external origin and where metrics are
    evaluated. It is a reporting mask / weight only -- never a wall of the PDE."""
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        _need(self.x1 > self.x0 and self.y1 > self.y0, "study region must have positive extent")


@dataclass(frozen=True)
class StudyMask:
    """Study region as a per-cell area fraction in [0, 1] (e.g. polygon-cell intersection area / cell area)."""
    weights: np.ndarray = field(compare=False, repr=False)
    sha256: str

    def __post_init__(self) -> None:
        w = np.asarray(self.weights)
        _need(w.ndim == 2 and bool(np.all(np.isfinite(w))) and bool(np.all((w >= 0) & (w <= 1))) and float(w.sum()) > 0,
              "study mask must be a 2-D array of area fractions in [0, 1] with a positive sum")
        _need(self.sha256 == array_sha256(w), "study mask: sha256 does not match the weights")

    @classmethod
    def from_array(cls, weights: np.ndarray) -> StudyMask:
        w = np.asarray(weights, dtype=float)
        return cls(w, array_sha256(w))


@dataclass(frozen=True)
class Param:
    """One entry of the parameter vector theta that ``simulate`` / the inverse problem use."""
    name: str
    kind: str  # "emission" (microgram/s) or "inflow" (boundary-basis amplitude, microgram/m^3)
    units: str
    group: str  # origin tag for emissions ("local"/"external"); "boundary" for inflow bases


@dataclass(frozen=True)
class WindSegment:
    """Wind (u, v) in m/s applying from t_start (s) until the next segment starts.

    Optional ``psi`` (ny+1, nx+1) [m^2/s] at cell corners defines a spatially varying, exactly divergence-free flow
    through face fluxes u_face = d(psi)/dy, v_face = -d(psi)/dx; then (u, v) are only the nominal/domain-mean wind used
    for display and perturbation helpers and the PDE uses ``psi``. Only ``psi_sha256`` enters ``to_dict``.
    """
    t_start: float
    u: float
    v: float
    psi: np.ndarray | None = field(default=None, compare=False, repr=False)
    psi_sha256: str = ""

    def __post_init__(self) -> None:
        if self.psi is not None:
            p = np.asarray(self.psi)
            _need(p.ndim == 2 and bool(np.all(np.isfinite(p))), "stream function must be a finite 2-D array")
            _need(self.psi_sha256 == array_sha256(p), "wind segment: psi_sha256 does not match psi")
        else:
            _need(self.psi_sha256 == "", "psi_sha256 given without psi")

    @classmethod
    def from_streamfunction(cls, t_start: float, psi: np.ndarray, u: float, v: float) -> WindSegment:
        p = np.asarray(psi, dtype=float)
        return cls(t_start, u, v, p, array_sha256(p))


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


SIDES = ("west", "east", "south", "north")


@dataclass(frozen=True)
class InflowBasis:
    """Unit concentration shape on the outer boundary for unresolved distant sources: face concentration
    a * profile(t) on the faces of ``sides`` that are INFLOW under the current wind (u.n < 0); outflow faces and
    other sides are unaffected. The amplitude ``a`` is a parameter in microgram/m^3 (not an emission rate)."""
    name: str
    sides: tuple[str, ...] = SIDES
    profile: tuple[tuple[float, float], ...] = ((0.0, 1.0),)

    def __post_init__(self) -> None:
        _need(len(self.sides) >= 1 and all(s in SIDES for s in self.sides) and len(set(self.sides)) == len(self.sides),
              f"basis {self.name}: sides must be unique members of {SIDES}")
        _check_profile(self.profile)


@dataclass(frozen=True)
class Boundary:
    mode: str = "open"  # open | periodic | closed
    background: float = 0.0  # static inflow concentration c_b (open mode), microgram/m^3, known (not estimated)
    bases: tuple[InflowBasis, ...] = ()  # inflow-concentration bases (parameters), open mode only

    def __post_init__(self) -> None:
        _need(self.mode in ("open", "periodic", "closed"), f"unknown boundary mode {self.mode!r}")
        _need(self.background >= 0, "background must be >= 0")
        _need(len(self.bases) == 0 or self.mode == "open", "inflow bases need boundary mode 'open'")
        _need(len({b.name for b in self.bases}) == len(self.bases), "inflow basis names must be unique")


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


def _default_sources() -> tuple[Source, ...]:
    return (
        GaussianSource("S1", 2500.0, 2500.0, 500.0),
        GaussianSource("S2", 4000.0, 6000.0, 500.0),
        GaussianSource("S3", 6500.0, 3500.0, 500.0),
    )


@dataclass(frozen=True)
class Scenario:
    """Everything the forward model needs EXCEPT emission rates q (microgram/s), which are passed separately."""
    domain: Domain = field(default_factory=Domain)
    sources: tuple[Source, ...] = field(default_factory=_default_sources)
    transport: Transport = field(default_factory=Transport)
    boundary: Boundary = field(default_factory=Boundary)
    initial: Initial = field(default_factory=Initial)
    times: Times = field(default_factory=Times)
    study: StudyRegion | StudyMask | None = None  # defines local/external origin and metric weights; never a PDE wall

    def __post_init__(self) -> None:
        _need(len(self.sources) >= 1, "need at least one source")
        shape = (self.domain.ny, self.domain.nx)
        for s in self.sources:
            if isinstance(s, RasterSource):
                _need(np.asarray(s.weights).shape == shape, f"raster source {s.name}: weights shape must be {shape}")
        if isinstance(self.study, StudyMask):
            _need(np.asarray(self.study.weights).shape == shape, f"study mask shape must be {shape}")
        corner = (self.domain.ny + 1, self.domain.nx + 1)
        for w in self.transport.wind:
            if w.psi is not None:
                _need(np.asarray(w.psi).shape == corner, f"stream function shape must be {corner}")
        _need(len({s.name for s in self.sources}) == len(self.sources), "source names must be unique")
        if self.boundary.mode == "closed":
            _need(all(w.u == 0 and w.v == 0 for w in self.transport.wind),
                  "closed boundaries require zero wind (a uniform wind cannot be tangent to every wall)")
        if self.boundary.mode == "periodic":
            _need(len(self.transport.wind) >= 1, "periodic mode needs a wind")
            _need(self.boundary.background == 0, "background is ignored in periodic mode; it must be 0")

    @property
    def n_emission(self) -> int:
        return len(self.sources)

    @property
    def n_params(self) -> int:
        return len(self.sources) + len(self.boundary.bases)

    def param_table(self) -> list[Param]:
        """theta = [emission amplitudes (microgram/s) in source order, then inflow-basis amplitudes (microgram/m^3)].
        Amplitudes of different kinds have different units and must never be compared by raw magnitude."""
        return ([Param(s.name, "emission", "microgram/s", s.group) for s in self.sources]
                + [Param(b.name, "inflow", "microgram/m^3", "boundary") for b in self.boundary.bases])

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


def _drop_arrays(items: list[tuple[str, Any]]) -> dict[str, Any]:
    return {k: v for k, v in items if not isinstance(v, np.ndarray)}


def to_dict(obj: Any) -> Any:
    """Dataclass tree -> plain JSON-able dict. Gridded arrays (raster sources, study mask, stream function) are
    represented by their sha256 only; ``scenario_from_dict`` needs a ``file`` entry to rebuild them."""
    return json.loads(json.dumps(dataclasses.asdict(obj, dict_factory=_drop_arrays)))


def _load_array(entry: dict[str, Any], base_dir: Path, key: str = "file", sha_key: str = "sha256") -> np.ndarray:
    """Read a prepared .npy/.npz array named by ``entry[key]`` (relative to base_dir) and verify its sha256."""
    path = base_dir / entry[key]
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        arr = arr[arr.files[0]]
    arr = np.asarray(arr, dtype=float)
    want = entry.get(sha_key)
    if want is not None and array_sha256(arr) != want:
        raise ValueError(f"{path}: sha256 {array_sha256(arr)} does not match the declared {want}")
    return arr


def _profile(entry: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    return tuple((float(t), float(v)) for t, v in entry["profile"]) if "profile" in entry else ((0.0, 1.0),)


def _wind(d: list[dict[str, Any]], base_dir: Path) -> tuple[WindSegment, ...]:
    out = []
    for w in d:
        if "psi_file" in w:
            psi = _load_array(w, base_dir, "psi_file", "psi_sha256")
            out.append(WindSegment.from_streamfunction(w["t_start"], psi, w["u"], w["v"]))
        else:
            out.append(WindSegment(w["t_start"], w["u"], w["v"]))
    return tuple(out)


def _source(s: dict[str, Any], base_dir: Path) -> Source:
    s = dict(s)
    kind = s.pop("type", "gaussian")
    if kind == "raster":
        return RasterSource.from_array(s["name"], s.get("group", "local"), _load_array(s, base_dir), _profile(s))
    _need(kind == "gaussian", f"unknown source type {kind!r}")
    s["profile"] = _profile(s)
    return GaussianSource(**s)


def scenario_from_dict(d: dict[str, Any], base_dir: Path | str = ".") -> Scenario:
    """Build a Scenario from a config dict. Gridded inputs are named by ``file`` (relative to ``base_dir``) with an
    optional ``sha256`` that is verified: raster sources ``{"type": "raster", "name", "group", "file", "sha256"}``, study
    region ``{"type": "rectangle", x0, y0, x1, y1}`` or ``{"type": "mask", "file", "sha256"}``, wind segments with
    ``psi_file``/``psi_sha256``, boundary ``bases``."""
    base = Path(base_dir)
    d = dict(d)
    tr = dict(d.get("transport", {}))
    if "wind" in tr:
        tr["wind"] = _wind(tr["wind"], base)
    bd = dict(d.get("boundary", {}))
    if "bases" in bd:
        bd["bases"] = tuple(InflowBasis(b["name"], tuple(b.get("sides", SIDES)), _profile(b)) for b in bd["bases"])
    srcs = tuple(_source(s, base) for s in d["sources"]) if "sources" in d else _default_sources()
    study: StudyRegion | StudyMask | None = None
    if "study" in d and d["study"] is not None:
        st = dict(d["study"])
        kind = st.pop("type", "rectangle")
        if kind == "mask":
            study = StudyMask.from_array(_load_array(st, base))
            _need(st.get("sha256") in (None, study.sha256), "study mask sha256 mismatch")
        else:
            _need(kind == "rectangle", f"unknown study type {kind!r}")
            study = StudyRegion(**st)
    return Scenario(
        domain=Domain(**d.get("domain", {})),
        sources=srcs,
        transport=Transport(**tr),
        boundary=Boundary(**bd),
        initial=Initial(**d.get("initial", {})),
        times=Times(**d.get("times", {})),
        study=study,
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

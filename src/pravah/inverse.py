"""Source-amplitude inversion: recover the emission rates q [ug/s] from sensor readings.

Everything except q (wind, K, lam, H, source geometry, initial state, boundary forcing) is KNOWN to the inversion.

Linear structure.  For a fixed scenario the model is affine in q:  c(q) = base + sum_j q_j g_j, where ``base`` is the
zero-emission response (configured initial condition and inflow background) and ``g_j`` is the response to a unit
emission of source j with zero initial state and zero background.  Sensors read ``y = b + A q + noise`` with
``b = sample(base)`` and ``A[:, j] = sample(g_j)``.  Observations are flattened time-major, then sensor:
row = it * n_sensors + isensor, i.e. ``values.reshape(-1)`` of a (time, sensor) array.

Responses are computed once per scenario (cached, see ``ResponseCache``) with reference amplitude ``q_ref`` (only a
numerical scaling, linearity makes the result independent of it) and then reused for every sensor layout, noise
draw and fit.  Any future nonlinear physics must make ``is_linear`` return False; the cached route then refuses.

Identifiability warning: optimiser ``success`` only means the optimiser converged.  It says nothing about whether the
data determine q.  Read ``rank``, ``condition``, ``singular_values``, ``effective_rank`` and ``q_std`` of the FitResult.  A well fitting field with
a rank-deficient A means different q give (almost) the same readings: accurate reconstruction != correct attribution.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import xarray as xr
import fipy as _fipy
from scipy import optimize

from . import __version__
from .config import SCHEMA_VERSION, FieldResult, Initial, Scenario, SolverOptions, to_dict
from .forward import simulate
from .observations import Observations, SensorLayout, sample

_FIPY_VERSION = str(_fipy.__version__)  # type: ignore[attr-defined]
Q_REF_DEFAULT = 1e5  # ug/s; numerical scaling of the unit-response runs only


def is_linear(scenario: Scenario) -> bool:
    """True if the concentration is an affine function of q for this scenario (all current physics: the PDE, its
    boundary treatment and the removal are linear).  Any nonlinear term added to the forward model (e.g. a
    concentration-dependent removal) MUST make this return False for the affected scenarios."""
    return True


def _check_linear(scenario: Scenario) -> None:
    if not is_linear(scenario):
        raise ValueError("scenario is not linear in the emission rates: the cached-response (superposition) route "
                         "is invalid; use method='direct' without the response cache")


@dataclass(frozen=True)
class Responses:
    """Base (q=0) field and per-unit-emission responses of one scenario on the output grid."""
    base: np.ndarray  # (nt, ny, nx) ug/m^3
    g: np.ndarray  # (n_src, nt, ny, nx) ug/m^3 per (ug/s)
    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    q_ref: float

    def _da(self, values: np.ndarray, name: str) -> xr.DataArray:
        return xr.DataArray(values, dims=("time", "y", "x"), coords={"time": self.times, "x": self.x, "y": self.y},
                            name=name, attrs={"units": "microgram/m^3"})

    def field(self, q: np.ndarray | list[float] | tuple[float, ...]) -> xr.DataArray:
        """Concentration base + sum_j q_j g_j as a (time, y, x) DataArray."""
        qa = np.asarray(q, dtype=float)
        if qa.shape != (self.g.shape[0],):
            raise ValueError(f"q must have shape ({self.g.shape[0]},), got {qa.shape}")
        return self._da(self.base + np.tensordot(qa, self.g, axes=1), "concentration")

    def operator(self, layout: SensorLayout, times: np.ndarray | list[float] | tuple[float, ...]
                 ) -> tuple[np.ndarray, np.ndarray]:
        """(b, A): b (nt*ns,) zero-emission readings, A (nt*ns, n_src); rows time-major then sensor.  Uses
        observations.sample, so it is identical to sampling the direct field."""
        b = sample(self._da(self.base, "base"), layout, times).values.reshape(-1)
        cols = [sample(self._da(gj, "g"), layout, times).values.reshape(-1) for gj in self.g]
        return np.asarray(b, dtype=float), np.stack(cols, axis=1)


def cache_key(scenario: Scenario, solver: SolverOptions, q_ref: float) -> str:
    payload = {"scenario": to_dict(scenario), "solver": to_dict(solver), "schema": SCHEMA_VERSION, "q_ref": q_ref,
               "pravah": __version__, "fipy": _FIPY_VERSION}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ResponseCache:
    """Responses per scenario: in-memory dict, plus optional .npz files in ``directory``.
    ``hits``/``misses`` count lookups served from memory-or-disk vs computed."""

    def __init__(self, directory: Path | str | None = None) -> None:
        self.directory = None if directory is None else Path(directory)
        self._mem: dict[str, Responses] = {}
        self.hits = 0
        self.misses = 0

    def get(self, scenario: Scenario, solver: SolverOptions = SolverOptions(), q_ref: float = Q_REF_DEFAULT
            ) -> Responses:
        _check_linear(scenario)
        if not q_ref > 0:
            raise ValueError("q_ref must be positive")
        key = cache_key(scenario, solver, q_ref)
        if key in self._mem:
            self.hits += 1
            return self._mem[key]
        path = None if self.directory is None else self.directory / f"{key}.npz"
        if path is not None and path.exists():
            with np.load(path) as z:
                resp = Responses(z["base"], z["g"], z["times"], z["x"], z["y"], float(z["q_ref"]))
            self.hits += 1
        else:
            resp = compute_responses(scenario, solver, q_ref)
            self.misses += 1
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, base=resp.base, g=resp.g, times=resp.times, x=resp.x, y=resp.y, q_ref=resp.q_ref)
        self._mem[key] = resp
        return resp


def compute_responses(scenario: Scenario, solver: SolverOptions = SolverOptions(),
                      q_ref: float = Q_REF_DEFAULT) -> Responses:
    """n_src + 1 forward solves: base (q=0, configured IC/background) and unit responses (zero IC, zero bg)."""
    _check_linear(scenario)
    n = len(scenario.sources)
    base_da = simulate(scenario, np.zeros(n), solver).concentration
    zero = dataclasses.replace(scenario, initial=Initial(),
                               boundary=dataclasses.replace(scenario.boundary, background=0.0))
    g = []
    for j in range(n):
        q = np.zeros(n)
        q[j] = q_ref
        g.append(simulate(zero, q, solver).concentration.values / q_ref)
    return Responses(np.asarray(base_da.values, dtype=float), np.stack(g), np.asarray(base_da["time"].values),
                     np.asarray(base_da["x"].values), np.asarray(base_da["y"].values), q_ref)


@dataclass(frozen=True)
class InferenceOptions:
    """method 'linear' (cached operator + lsq_linear) or 'direct' (least_squares on forward solves).
    q_scale: scale of each q_j (ug/s); the optimiser works in theta = q / q_scale >= 0 (default 1e5 each).
    alpha, prior: Tikhonov alpha * ||theta - prior/q_scale||^2 (prior in ug/s, default zeros).
    heldout_ids: sensors excluded from the fit."""
    method: Literal["linear", "direct"] = "linear"
    q_scale: tuple[float, ...] | None = None
    alpha: float = 0.0
    prior: tuple[float, ...] | None = None
    heldout_ids: tuple[str, ...] = ()
    solver: SolverOptions = field(default_factory=SolverOptions)
    q_ref: float = Q_REF_DEFAULT

    def __post_init__(self) -> None:
        if self.method not in ("linear", "direct"):
            raise ValueError("method must be 'linear' or 'direct'")
        if not self.alpha >= 0:
            raise ValueError("alpha must be >= 0")


@dataclass
class FitResult:
    q_hat: np.ndarray
    success: bool  # optimiser convergence ONLY; not identifiability
    message: str
    weighted_residual: float  # ||r||_2 over data rows only (unit weights if all sigma == 0)
    n_obs: int
    singular_values: np.ndarray  # of the scaled, whitened, unregularised observation matrix
    rank: int  # singular values > max(shape) * eps * s_max (purely relative: also inspect the size of s_max itself)
    condition: float  # s_max / s_min, inf if rank < n_src
    effective_rank: int  # number of whitened, q_scale-scaled singular values > 1: directions the data constrain
    #   to better than one q_scale (relative to noise).  Unlike ``rank`` it is meaningful with real noise.
    q_std: np.ndarray  # posterior std of each q_j [ug/s] = q_scale_j * sqrt(diag((Aw^T Aw + alpha I)^-1)).  Linear-
    #   Gaussian approximation: ignores the q >= 0 bound (unreliable when q_hat is at 0) and assumes sigma is correct
    #   (if all sigma == 0 it is in units of the data, ug/m^3, per unit weight).  inf for directions that neither
    #   the data nor the regularisation constrain.
    reg_singular_values: np.ndarray  # same, with the Tikhonov rows stacked
    reg_condition: float
    method: str
    q_scale: np.ndarray
    alpha: float
    prior: np.ndarray
    heldout_ids: tuple[str, ...]
    runtime_s: float


def _svd_diag(M: np.ndarray) -> tuple[np.ndarray, int, float]:
    s = np.linalg.svd(M, compute_uv=False)
    if s.size == 0 or s[0] == 0.0:
        return s, 0, float("inf")
    rank = int(np.sum(s > max(M.shape) * np.finfo(float).eps * s[0]))
    cond = float(s[0] / s[-1]) if rank == M.shape[1] and s[-1] > 0 else float("inf")
    return s, rank, cond


def _posterior_std(A_stack: np.ndarray) -> np.ndarray:
    """sqrt(diag((M^T M)^-1)) via SVD of M; inf for coordinates touching the numerical null space of M."""
    _, s, vt = np.linalg.svd(A_stack, full_matrices=True)
    n = A_stack.shape[1]
    s_full = np.zeros(n)
    s_full[:s.size] = s
    tiny = s_full <= max(A_stack.shape) * np.finfo(float).eps * (s[0] if s.size and s[0] > 0 else 1.0)
    inv2 = np.where(tiny, 0.0, 1.0 / np.where(tiny, 1.0, s_full) ** 2)
    var = (vt.T**2) @ inv2
    null_weight = (vt.T**2) @ tiny.astype(float)
    return np.where(null_weight > 1e-12, np.inf, np.sqrt(var))


def _vec(v: tuple[float, ...] | None, n: int, default: float, name: str) -> np.ndarray:
    out = np.full(n, default) if v is None else np.asarray(v, dtype=float)
    if out.shape != (n,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be {n} finite numbers")
    return out


def fit_sources(scenario: Scenario, obs: Observations, options: InferenceOptions = InferenceOptions(),
                cache: ResponseCache | None = None) -> FitResult:
    """Fit q >= 0 to the readings of the non-held-out sensors.  Truth is not an input.

    Residual r = (b + A q - y) / sigma (sigma per sensor; if all sigma are 0, unit weights, so r is in ug/m^3; a mix
    of zero and non-zero sigma raises ValueError).  With alpha > 0 the rows sqrt(alpha) * (theta - prior/q_scale) are
    appended.  'direct' calls simulate at every optimiser iterate (finite-difference Jacobian, no caching); its
    diagnostic matrix A is taken from the cached responses (n_src + 1 solves, the same cost as finite differences
    and exact thanks to linearity).  'success' is optimiser convergence, NOT identifiability: check rank/condition.
    """
    t0 = _time.perf_counter()
    n = len(scenario.sources)
    ids = list(obs.layout.ids)
    unknown = [s for s in options.heldout_ids if s not in ids]
    if unknown:
        raise ValueError(f"unknown heldout_ids: {unknown}")
    keep = [i for i, s in enumerate(ids) if s not in options.heldout_ids]
    if not keep:
        raise ValueError("all sensors are held out")
    layout = obs.layout.select(keep)
    times = np.asarray(obs.times if obs.times else obs.values["time"].values, dtype=float)
    y = np.asarray(obs.values.transpose("time", "sensor").values, dtype=float)[:, keep]
    sigma = np.broadcast_to(np.asarray(obs.noise_std, dtype=float)[keep], y.shape).reshape(-1)  # time-major, sensor
    yv = y.reshape(-1)
    w = _whitening(sigma)  # raises early on a mix of zero and non-zero sigma

    _check_linear(scenario)
    cache = cache if cache is not None else ResponseCache()
    resp = cache.get(scenario, options.solver, options.q_ref)
    b, A = resp.operator(layout, times)
    if options.method == "linear":
        diag = solve_linear(b, A, yv, sigma, options)
        q_hat, success, message, r_norm = diag.q_hat, diag.success, diag.message, diag.weighted_residual
    else:
        diag = _linear_system(b, A, yv, sigma, options)  # diagnostics from the cached operator (see docstring)
        q_scale, sa, th_prior = diag.q_scale, diag.sa, diag.prior / diag.q_scale

        def fun(th: np.ndarray) -> np.ndarray:
            pred = sample(simulate(scenario, th * q_scale, options.solver).concentration, layout, times)
            r = (np.asarray(pred.values, dtype=float).reshape(-1) - yv) * w
            return np.concatenate([r, sa * (th - th_prior)])

        ls = optimize.least_squares(fun, np.ones(n), bounds=(0.0, np.inf), x_scale=1.0, ftol=1e-14, xtol=1e-14,
                                    gtol=1e-14, max_nfev=200)
        q_hat, success, message = np.asarray(ls.x, dtype=float) * q_scale, bool(ls.success), str(ls.message)
        r_norm = float(np.linalg.norm(np.asarray(ls.fun, dtype=float)[:A.shape[0]]))
    return FitResult(
        q_hat=q_hat, success=success, message=message, weighted_residual=r_norm,
        n_obs=A.shape[0], singular_values=diag.singular_values, rank=diag.rank, condition=diag.condition,
        effective_rank=diag.effective_rank, q_std=diag.q_std, reg_singular_values=diag.reg_singular_values,
        reg_condition=diag.reg_condition, method=options.method, q_scale=diag.q_scale, alpha=float(options.alpha),
        prior=diag.prior, heldout_ids=tuple(options.heldout_ids), runtime_s=_time.perf_counter() - t0)


def _whitening(sigma: np.ndarray) -> np.ndarray:
    """Row weights 1/sigma; if all sigma are 0, unit weights; a mix of zero and non-zero raises ValueError."""
    sigma = np.asarray(sigma, dtype=float)
    if np.all(sigma == 0):
        return np.ones_like(sigma)
    if np.any(sigma == 0):
        raise ValueError("mixed zero and non-zero noise_std among the used sensors is not supported")
    return 1.0 / sigma


@dataclass(frozen=True)
class _System:
    """Scaled, whitened, regularised least-squares system and its identifiability diagnostics (no optimisation)."""
    Aw: np.ndarray  # scaled, whitened, unregularised
    rhs: np.ndarray
    A_stack: np.ndarray  # Aw with the Tikhonov rows appended
    q_scale: np.ndarray
    prior: np.ndarray
    sa: float  # sqrt(alpha)
    singular_values: np.ndarray
    rank: int
    condition: float
    effective_rank: int
    q_std: np.ndarray
    reg_singular_values: np.ndarray
    reg_condition: float


def _linear_system(b: np.ndarray, A: np.ndarray, y: np.ndarray, sigma: np.ndarray,
                   options: InferenceOptions) -> _System:
    n = A.shape[1]
    q_scale = _vec(options.q_scale, n, 1e5, "q_scale")
    prior = _vec(options.prior, n, 0.0, "prior")
    if np.any(q_scale <= 0):
        raise ValueError("q_scale must be positive")
    w = _whitening(sigma)
    Aw = (A * w[:, None]) * q_scale[None, :]
    rhs = (np.asarray(y, dtype=float) - b) * w
    sa = float(np.sqrt(options.alpha))
    A_stack = np.vstack([Aw, sa * np.eye(n)])
    s, rank, cond = _svd_diag(Aw)
    s_reg, _, cond_reg = _svd_diag(A_stack)
    return _System(Aw, rhs, A_stack, q_scale, prior, sa, s, rank, cond, int(np.sum(s > 1.0)),
                   q_scale * _posterior_std(A_stack), s_reg, cond_reg)


@dataclass(frozen=True)
class LinearSolution:
    """Result of ``solve_linear``: the bounded solution plus the diagnostics that FitResult reports."""
    q_hat: np.ndarray
    success: bool  # optimiser convergence ONLY
    message: str
    weighted_residual: float
    singular_values: np.ndarray
    rank: int
    condition: float
    effective_rank: int
    q_std: np.ndarray
    reg_singular_values: np.ndarray
    reg_condition: float
    q_scale: np.ndarray
    prior: np.ndarray


def solve_linear(b: np.ndarray, A: np.ndarray, y: np.ndarray, sigma: np.ndarray,
                 options: InferenceOptions = InferenceOptions()) -> LinearSolution:
    """The linear route of ``fit_sources`` on an explicit operator: min over q >= 0 of
    ||(b + A q - y)/sigma||^2 + alpha ||theta - prior/q_scale||^2 with theta = q / q_scale (bounded-variable LS).

    b, y: (m,); A: (m, n_src); sigma: (m,) per-row noise std (all 0 -> unit weights).  Row order is irrelevant.
    ``fit_sources`` (method 'linear') and the sensor-design code both call this, so they use identical maths."""
    sy = _linear_system(b, A, y, sigma, options)
    res = optimize.lsq_linear(sy.A_stack, np.concatenate([sy.rhs, sy.sa * (sy.prior / sy.q_scale)]),
                              bounds=(0.0, np.inf), method="bvls", tol=1e-14, max_iter=1000)
    theta = np.asarray(res.x, dtype=float)
    return LinearSolution(
        q_hat=theta * sy.q_scale, success=bool(res.success), message=str(res.message),
        weighted_residual=float(np.linalg.norm(sy.Aw @ theta - sy.rhs)), singular_values=sy.singular_values,
        rank=sy.rank, condition=sy.condition, effective_rank=sy.effective_rank, q_std=sy.q_std,
        reg_singular_values=sy.reg_singular_values, reg_condition=sy.reg_condition, q_scale=sy.q_scale,
        prior=sy.prior)


def reconstruct(scenario: Scenario, fit: FitResult, solver: SolverOptions = SolverOptions()) -> FieldResult:
    """True forward run with the fitted amplitudes."""
    return simulate(scenario, fit.q_hat, solver)


def reconstruct_linear(responses: Responses, fit: FitResult) -> xr.DataArray:
    """Fast reconstruction from cached responses (same field as ``reconstruct`` up to solver round-off)."""
    return responses.field(cast(Any, fit.q_hat))

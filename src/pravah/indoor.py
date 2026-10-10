"""One-way coupled indoor box model, separate from the outdoor inverse problem.

    dc_in/dt = P*a*c_out(t) - (a + k_dep)*c_in + q_in(t)/V

P dimensionless penetration factor in [0, 1]; a air-exchange rate [1/s] >= 0; k_dep indoor deposition/removal
[1/s] >= 0; V volume [m^3] > 0; q_in indoor emission [microgram/s] >= 0 (piecewise constant); c_out is the outdoor
concentration interpolated at the building from a pravah field (via ``observations.sample``); c_in in microgram/m^3.

ONE-WAY COUPLING: the outdoor concentration is NOT modified by the indoor box. No indoor exhaust is added to any
outdoor source; the building only reads the outdoor field.

Exact solution per interval [0, h] with c_out = u0 + s*t, constant q, lam = a + k_dep, E = exp(-lam h):
    c(h) = c(0) E + f0 * h*phi1(x) + P*a*s * h^2*phi2(x),   f0 = P*a*u0 + q/V,  x = lam*h,
    phi1(x) = (1 - e^-x)/x,   phi2(x) = (x - 1 + e^-x)/x^2   (both -> 1 and 1/2 at x = 0: no division by zero).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import xarray as xr

from .observations import SensorLayout, sample


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@dataclass(frozen=True)
class IndoorBox:
    name: str
    x: float
    y: float
    P: float
    a: float
    k_dep: float
    V: float

    def __post_init__(self) -> None:
        _need(all(math.isfinite(v) for v in (self.x, self.y, self.P, self.a, self.k_dep, self.V)),
              "IndoorBox parameters must be finite")
        _need(0.0 <= self.P <= 1.0, "penetration P must be in [0, 1]")
        _need(self.a >= 0, "air-exchange rate a must be >= 0")
        _need(self.k_dep >= 0, "k_dep must be >= 0")
        _need(self.V > 0, "volume V must be > 0")

    @property
    def lam(self) -> float:
        return self.a + self.k_dep


def _phi(x: float) -> tuple[float, float]:
    """phi1(x) = (1-e^-x)/x and phi2(x) = (x-1+e^-x)/x^2, by Taylor series for small x (no cancellation)."""
    if x < 0.5:
        p1 = p2 = 0.0
        term1, term2 = 1.0, 0.5  # (-x)^k/(k+1)! and (-x)^k/(k+2)!
        for k in range(25):
            p1 += term1
            p2 += term2
            term1 *= -x / (k + 2)
            term2 *= -x / (k + 3)
        return p1, p2
    e = math.exp(-x)
    return (1 - e) / x, (x - 1 + e) / (x * x)


def indoor_response(box: IndoorBox, c_out_times: np.ndarray, c_out_values: np.ndarray,
                    q_profile: tuple[tuple[float, float], ...] = ((0.0, 1.0),), c0: float = 0.0) -> np.ndarray:
    """Exact c_in at ``c_out_times`` for piecewise-linear c_out and piecewise-constant q.

    ``q_profile`` is ((t_start, q [microgram/s]), ...) with increasing t_start, first t_start <= c_out_times[0].
    c_in(c_out_times[0]) = c0. Interval splits at q breakpoints are exact (c_out is linear across them).
    """
    t = np.asarray(c_out_times, dtype=float)
    u = np.asarray(c_out_values, dtype=float)
    _need(t.ndim == 1 and t.shape == u.shape and t.size >= 1, "c_out times/values must be 1-D and equal length")
    _need(bool(np.all(np.diff(t) > 0)), "c_out times must be strictly increasing")
    _need(len(q_profile) >= 1, "q_profile must be non-empty")
    qt = np.array([p[0] for p in q_profile], dtype=float)
    qv = np.array([p[1] for p in q_profile], dtype=float)
    _need(bool(np.all(np.diff(qt) > 0)), "q_profile start times must be strictly increasing")
    _need(bool(np.all(qv >= 0)), "indoor emission q must be >= 0")
    _need(bool(qt[0] <= t[0]), "first q_profile start must not be after the first time")
    grid = np.union1d(t, qt[(qt > t[0]) & (qt < t[-1])])
    ug = np.interp(grid, t, u)
    c = np.empty(grid.size)
    c[0] = c0
    for n in range(grid.size - 1):
        h = grid[n + 1] - grid[n]
        q = qv[np.searchsorted(qt, grid[n], side="right") - 1]
        s = (ug[n + 1] - ug[n]) / h
        x = box.lam * h
        p1, p2 = _phi(x)
        f0 = box.P * box.a * ug[n] + q / box.V
        c[n + 1] = c[n] * math.exp(-x) + f0 * h * p1 + box.P * box.a * s * h * h * p2
    return c[np.searchsorted(grid, t)]


def outdoor_at_building(field: xr.DataArray, box: IndoorBox,
                        times: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(times, c_out) at the building; the building must lie inside the cell-centre hull (else ValueError)."""
    tt = np.asarray(field["time"].values, dtype=float) if times is None else np.asarray(times, dtype=float)
    lay = SensorLayout((box.name,), (box.x,), (box.y,))
    return tt, np.asarray(sample(field, lay, tt).values[:, 0], dtype=float)


def indoor_linear_parts(box: IndoorBox, field: xr.DataArray,
                        times: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(c_in_outdoor_driven, g_in): c_in = c_in_outdoor_driven + q_in*g_in for constant q_in from times[0], c0 = 0.

    g_in has units (microgram/m^3)/(microgram/s).
    """
    tt, co = outdoor_at_building(field, box, times)
    driven = indoor_response(box, tt, co, ((float(tt[0]), 0.0),))
    g = indoor_response(box, tt, np.zeros_like(co), ((float(tt[0]), 1.0),))
    return driven, g


@dataclass(frozen=True)
class IndoorFit:
    q_hat: float  # microgram/s, clipped at 0
    std: float  # 1/sqrt(g.g/sigma^2 + alpha): unclipped posterior/least-squares std
    singular_value: float  # ||g_in||/sigma
    identifiable: bool


def fit_indoor_emission(box: IndoorBox, y_in: np.ndarray, sigma: float | np.ndarray, c_in_outdoor_driven: np.ndarray,
                        g_in: np.ndarray, alpha: float = 0.0, min_singular_value: float = 1e-6) -> IndoorFit:
    """Nonnegative weighted LS of one amplitude q: min ||(y - driven - q g)/sigma||^2 + alpha q^2, q >= 0.

    Closed form q = max(0, sum(g r/sigma^2) / (sum(g^2/sigma^2) + alpha)).
    """
    _need(alpha >= 0, "alpha must be >= 0")
    y = np.asarray(y_in, dtype=float)
    g = np.asarray(g_in, dtype=float)
    s = np.broadcast_to(np.asarray(sigma, dtype=float), y.shape)
    _need(bool(np.all(s > 0)), "sigma must be > 0")
    w = 1.0 / s**2
    r = y - np.asarray(c_in_outdoor_driven, dtype=float)
    gg = float(np.sum(w * g * g))
    sv = math.sqrt(gg)
    ident = sv > min_singular_value
    denom = gg + alpha
    if denom <= 0:
        return IndoorFit(0.0, math.inf, sv, False)
    return IndoorFit(max(0.0, float(np.sum(w * g * r)) / denom), 1.0 / math.sqrt(denom), sv, ident)


def paired_observations(box: IndoorBox, field: xr.DataArray, times: np.ndarray, q_true: float, sigma_in: float,
                        sigma_out: float, seed: int) -> dict[str, np.ndarray]:
    """Synthetic paired indoor/outdoor readings with iid Gaussian noise (default_rng(seed)).

    Keys: times, c_out_true, c_in_true, y_out, y_in, driven, g_in.
    """
    _need(q_true >= 0 and sigma_in >= 0 and sigma_out >= 0, "q_true and sigmas must be >= 0")
    tt, co = outdoor_at_building(field, box, times)
    driven, g = indoor_linear_parts(box, field, tt)
    c_in = driven + q_true * g
    rng = np.random.default_rng(seed)
    y_out = co + sigma_out * rng.standard_normal(co.shape)
    y_in = c_in + sigma_in * rng.standard_normal(c_in.shape)
    return {"times": tt, "c_out_true": co, "c_in_true": c_in, "y_out": y_out, "y_in": y_in, "driven": driven,
            "g_in": g}


def misattribution_demo(box: IndoorBox, field: xr.DataArray, times: np.ndarray,
                        q_in: float) -> dict[str, np.ndarray]:
    """Decompose the indoor reading and show what an analyst who ignores indoor emission would infer.

    reading = from_outdoor + from_indoor. The naive outdoor-equivalent is reading / F with F = P a/(a+k_dep)
    (steady-state indoor/outdoor ratio from outdoor sources alone); it overstates the true c_out by
    from_indoor / F (plus the lag error of the dynamic response). Needs P*a > 0.
    """
    _need(box.P * box.a > 0 and box.lam > 0, "misattribution demo needs P*a > 0")
    tt, co = outdoor_at_building(field, box, times)
    driven, g = indoor_linear_parts(box, field, tt)
    emis = q_in * g
    f = box.P * box.a / box.lam
    reading = driven + emis
    return {"times": tt, "c_out_true": co, "from_outdoor": driven, "from_indoor": emis, "reading": reading,
            "outdoor_equivalent": reading / f, "overstatement": emis / f}


@dataclass(frozen=True)
class Ambiguity:
    cosine: float
    sv_indoor_only: np.ndarray
    sv_paired: np.ndarray


def indoor_only_ambiguity(box: IndoorBox, times: np.ndarray, sigma: float = 1.0, shape: np.ndarray | None = None,
                          sigma_out: float | None = None) -> Ambiguity:
    """Identifiability of (outdoor amplitude A, indoor emission q) with c_out = A*shape(t) (default shape 1).

    Columns: indoor response per unit A, and g_in. Indoor-only design matrix has rows t_i / sigma; the paired design
    adds outdoor-sensor rows [shape/sigma_out, 0]. Returns singular values (descending) and the cosine between
    the two indoor columns (1 = fully ambiguous). With constant shape and constant q from t0 the columns are exactly
    proportional, so indoor-only has smallest singular value ~0 while the paired design does not.
    """
    tt = np.asarray(times, dtype=float)
    sh = np.ones_like(tt) if shape is None else np.asarray(shape, dtype=float)
    so = sigma if sigma_out is None else sigma_out
    _need(sigma > 0 and so > 0, "sigmas must be > 0")
    col_a = indoor_response(box, tt, sh, ((float(tt[0]), 0.0),))
    col_q = indoor_response(box, tt, np.zeros_like(tt), ((float(tt[0]), 1.0),))
    a_mat = np.column_stack([col_a, col_q]) / sigma
    p_mat = np.vstack([a_mat, np.column_stack([sh, np.zeros_like(sh)]) / so])
    denom = float(np.linalg.norm(col_a) * np.linalg.norm(col_q))
    cos = float(col_a @ col_q / denom) if denom > 0 else 0.0
    return Ambiguity(cos, np.linalg.svd(a_mat, compute_uv=False), np.linalg.svd(p_mat, compute_uv=False))

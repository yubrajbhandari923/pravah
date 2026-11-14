"""E12 meteorological mismatch: perturbed model winds for the inverse/forecast model (design: wind-mismatch-design.md).

The truth is never touched: `perturb_winds` returns a new array from a copy of the true winds, and `model_responses`
drives every unit-source solve with the MODEL wind. Only the oracle spec (magnitude 0) reproduces the true wind.
Convention (benchmarks.make_winds): ux = -s sin(th), uy = -s cos(th), th = direction the wind blows FROM (deg).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .experiments import cross_objective as X

FAMILIES = ("oracle", "dir", "spd", "lag", "ar")
AR_TAU_H, AR_SEED = 12.0, 7000


@dataclass(frozen=True)
class WindSpec:
    family: str
    magnitude: float = 0.0   # dir: deg veer; spd: speed factor; lag: hours; ar: marginal sd deg

    @property
    def name(self):
        return "oracle" if self.oracle else f"{self.family}{self.magnitude:g}"

    @property
    def oracle(self):
        return self.family == "oracle"


ORACLE = WindSpec("oracle")
PILOT_SPECS = (ORACLE, WindSpec("dir", 10), WindSpec("dir", 30), WindSpec("spd", .8), WindSpec("spd", .6),
               WindSpec("lag", 2), WindSpec("lag", 6), WindSpec("ar", 15), WindSpec("ar", 30))
SWEEP_SPECS = (ORACLE, WindSpec("dir", 10), WindSpec("dir", 30), WindSpec("spd", .6), WindSpec("lag", 6), WindSpec("ar", 30))


def to_polar(w):
    w = np.asarray(w, float)
    return np.hypot(w[:, 0], w[:, 1]), np.degrees(np.arctan2(-w[:, 0], -w[:, 1]))


def from_polar(s, th):
    t = np.radians(th)
    return np.stack([-s * np.sin(t), -s * np.cos(t)], 1)


def ar_path(n, replicate, tau_h=AR_TAU_H):
    """Unit-variance AR(1) path (n,), correlation exp(-1/tau) per hour; one stream per replicate, shared by magnitudes."""
    rng = np.random.default_rng(np.random.SeedSequence([AR_SEED, replicate]))
    phi, xi = np.exp(-1 / tau_h), rng.standard_normal(n)
    e = np.empty(n)
    e[0] = xi[0]
    for h in range(1, n):
        e[h] = phi * e[h - 1] + np.sqrt(1 - phi ** 2) * xi[h]
    return e


def perturb_winds(winds, spec: WindSpec, replicate=0):
    """Model wind (hours, 2) for `spec` from the true hourly winds; always a new array (the input is never written)."""
    w = np.array(winds, float, copy=True)
    if spec.oracle:
        return w
    s, th = to_polar(w)
    if spec.family == "dir":
        return from_polar(s, th + spec.magnitude)
    if spec.family == "spd":
        return from_polar(s * spec.magnitude, th)
    if spec.family == "lag":
        L = int(spec.magnitude)
        return w[np.maximum(np.arange(len(w)) - L, 0)]
    if spec.family == "ar":
        return from_polar(s, th + spec.magnitude * ar_path(len(w), replicate))
    raise ValueError(f"unknown wind perturbation family {spec.family!r}")


def wind_error(true, model):
    """(mean |angle difference| deg, mean speed ratio) between two wind series."""
    s0, t0 = to_polar(true)
    s1, t1 = to_polar(model)
    d = np.abs((t1 - t0 + 180) % 360 - 180)
    return float(d.mean()), float(np.mean(s1 / s0))


def model_responses(sc, spec: WindSpec):
    """(F_m (K, hours, nx, ny), G3_m (n_sites, hours, K), model winds): unit responses of the known source locations under
    the MODEL wind. Never uses strengths or the truth field; the true wind enters only through perturb_winds."""
    wm = perturb_winds(sc.truth.winds, spec, sc.replicate)
    F = np.stack([X.source_field(sc.cfg, xy, wm, len(wm)) for xy in sc.truth.source_xy])
    return F, X.site_responses(sc, F), wm

"""Gaussian emission source fields and temporal profiles.

Sources are defined by continuous parameters (x, y, sigma) and sampled at cell centres, so the same source can be
put on grids of different resolution. The sampled field is normalised on the grid so ``sum(f * dx * dy) = 1``.
"""
from __future__ import annotations

import math

import numpy as np

from .config import GaussianSource, Scenario


def cell_centres(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinates (x, y), each 1-D ascending, in metres."""
    d = scenario.domain
    return (np.arange(d.nx) + 0.5) * d.dx, (np.arange(d.ny) + 0.5) * d.dy


def _gaussian_unnormalised(scenario: Scenario, src: GaussianSource) -> np.ndarray:
    x, y = cell_centres(scenario)
    gx = np.exp(-((x - src.x) ** 2) / (2 * src.sigma**2))
    gy = np.exp(-((y - src.y) ** 2) / (2 * src.sigma**2))
    return gy[:, None] * gx[None, :]  # (ny, nx)


def source_fields(scenario: Scenario) -> np.ndarray:
    """Normalised source shapes f_j, shape (n_src, ny, nx), units 1/m^2, with sum(f_j * dx * dy) = 1."""
    dA = scenario.domain.dx * scenario.domain.dy
    out = np.empty((len(scenario.sources), scenario.domain.ny, scenario.domain.nx))
    for j, src in enumerate(scenario.sources):
        g = _gaussian_unnormalised(scenario, src)
        total = g.sum() * dA
        if not total > 0:
            raise ValueError(f"source {src.name}: Gaussian is numerically zero on the grid")
        out[j] = g / total
    return out


def emission_profile(src: GaussianSource, t: float) -> float:
    """Piecewise-constant temporal factor s(t) (right-continuous: a breakpoint belongs to the later segment)."""
    value = src.profile[0][1]
    for t_start, v in src.profile:
        if t >= t_start:
            value = v
    return value


def quadrature_ratio(scenario: Scenario) -> np.ndarray:
    """Per source: discrete unnormalised Gaussian integral / continuous integral truncated to the domain.

    The continuous value is 2 pi sigma^2 * [erf-difference in x] * [erf-difference in y]. A ratio of 1 means the
    cell-centre sum represents the source exactly; the normalisation to 1 hides any deviation, so report it.
    """
    d = scenario.domain
    dA = d.dx * d.dy
    ratios = np.empty(len(scenario.sources))
    for j, src in enumerate(scenario.sources):
        s2 = math.sqrt(2.0) * src.sigma
        ix = 0.5 * (math.erf((d.lx - src.x) / s2) - math.erf(-src.x / s2))
        iy = 0.5 * (math.erf((d.ly - src.y) / s2) - math.erf(-src.y / s2))
        exact = 2 * math.pi * src.sigma**2 * ix * iy
        ratios[j] = _gaussian_unnormalised(scenario, src).sum() * dA / exact
    return ratios

"""Scenario builders for the REAL Belauri region (Stage E).  Real geography; synthetic emissions and meteorology.

What is real and what is not
* Real: the municipality polygon (as the cell-area fraction ``study_fraction``), the road / built-up / facility
  geometry (the source SHAPES ``w_road`` / ``w_builtup`` / ``w_facility``, clipped to the polygon) and the position of
  the external patch.
* Synthetic: every emission rate (``amplitudes``, micrograms per second), the temporal onset of the external patch and
  all meteorology (uniform or stream-function winds prescribed here; K, lam, H are nominal values).  DEM, hillshade and
  slope are CONTEXT ONLY: nothing in this module reads them, so they cannot change any number.

Coordinates are local metres (origin = south-west corner of the prepared domain, x east, y north); ``crop_region``
re-bases them to the cropped domain's own south-west corner and records the offset, so the physical sources and the
study region stay at the same absolute (UTM) positions.

Wind directions use the METEOROLOGICAL convention: ``from_deg`` is the compass bearing the air comes FROM (clockwise
from north); the air moves toward ``from_deg + 180``.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import (Boundary, Domain, GaussianSource, InflowBasis, Initial, Scenario, StudyMask, Times, Transport,
                      WindSegment)
from ..sources import check_origin_tags
from ..winds import eddy_streamfunction  # noqa: F401  (moved to pravah.winds; re-exported here)
from .prepare import GeoRef, GridLevel, Region
from .regionio import GridSpec

LABEL = "Real geography; synthetic emissions and meteorology."
LOCAL_ORIGIN_TOL = 0.06  # local rasters are clipped to the polygon but boundary cells count in full: the manifest's
# origin_mass_inside is 0.949 (road), 0.977 (built-up), 0.968 (facility) at 500 m, so a tolerance of 6 % is documented here
# (the library default 1 % is for synthetic Gaussians).  External sources must keep <= this fraction inside.
LOCAL_LAYERS = ("road", "builtup", "facility")
Offset = tuple[float, float]
WindFactory = Callable[[Domain, Offset], tuple[WindSegment, ...]]  # (domain, offset of its SW corner in the standard domain)


# ---------------------------------------------------------------------------------------------- numerical diffusion
def numerical_diffusion_estimate(u: float, dx: float, dt: float, K: float) -> float:
    """Excess effective diffusivity K_eff/K - 1 of the baseline scheme (exponential convection + backward Euler) for a
    Fourier mode and wind ALONG a grid axis:  (Pe/2) coth(Pe/2) - 1 + u^2 dt / (2 K),  Pe = |u| dx / K
    (docs/specification.md; reproduced to 3 % by scripts/verification.py).  For an oblique wind it is only a scalar-speed
    approximation.  Report it SEPARATELY from the physical K."""
    if not K > 0:
        raise ValueError("K must be positive")
    pe = abs(u) * dx / K
    space = 0.0 if pe < 1e-12 else 0.5 * pe / math.tanh(0.5 * pe) - 1.0
    return space + u * u * dt / (2.0 * K)


# ---------------------------------------------------------------------------------------------- winds
def toward_vector(from_deg: float, speed: float) -> tuple[float, float]:
    """(u, v) m/s of air that comes FROM the compass bearing ``from_deg`` (clockwise from north)."""
    th = math.radians(from_deg)
    return -speed * math.sin(th), -speed * math.cos(th)


def uniform_wind(segments: Sequence[tuple[float, float]], speed: float) -> WindFactory:
    """Piecewise-constant uniform wind: ``segments`` = [(t_start_s, from_deg), ...] (first t_start must be 0)."""
    segs = tuple(segments)

    def make(domain: Domain, offset: Offset) -> tuple[WindSegment, ...]:
        return tuple(WindSegment(t, *toward_vector(f, speed)) for t, f in segs)
    return make


def eddy_wind(from_deg: float, speed: float, centre: tuple[float, float], sigma: float, amplitude: float) -> WindFactory:
    u, v = toward_vector(from_deg, speed)

    def make(domain: Domain, offset: Offset) -> tuple[WindSegment, ...]:
        return (WindSegment.from_streamfunction(0.0, eddy_streamfunction(domain, offset, u, v, centre, sigma, amplitude),
                                                u, v),)
    return make


def build_wind(name: str, spec: dict[str, Any], default_speed: float, duration_s: float,
               centre: tuple[float, float]) -> WindFactory:
    """Wind scenario ``name`` from its config entry.  ``spec`` keys: ``kind`` 'uniform' (default) | 'reversal' |
    'streamfunction'; ``from_deg`` (compass bearing the wind comes from); optional ``speed``; 'reversal' also
    ``second_from_deg`` (default from_deg + 180) switching at ``duration_s / 2``; 'streamfunction' also
    ``eddy_sigma_m`` and ``eddy_amplitude_m2_s`` (eddy centred at ``centre``).  SYNTHETIC meteorology, not terrain physics."""
    speed = float(spec.get("speed", default_speed))
    kind = spec.get("kind", "uniform")
    f0 = float(spec["from_deg"])
    if kind == "uniform":
        return uniform_wind([(0.0, f0)], speed)
    if kind == "reversal":
        return uniform_wind([(0.0, f0), (duration_s / 2.0, float(spec.get("second_from_deg", f0 + 180.0)) % 360.0)], speed)
    if kind == "streamfunction":
        return eddy_wind(f0, speed, centre, float(spec["eddy_sigma_m"]), float(spec["eddy_amplitude_m2_s"]))
    raise ValueError(f"wind scenario {name!r}: unknown kind {kind!r}")


# ---------------------------------------------------------------------------------------------- cropping
def _cells_with(a: np.ndarray, thresh: float = 0.0) -> tuple[int, int, int, int]:
    jj, ii = np.nonzero(a > thresh)
    return int(ii.min()), int(ii.max()) + 1, int(jj.min()), int(jj.max()) + 1  # [i0, i1) x [j0, j1)


def crop_region(region: Region, buffer_m: float) -> Region:
    """A Region on a SMALLER outer domain: the study-polygon bounding box (cells with ``study_fraction`` > 0) expanded by
    ``buffer_m`` (rounded outward to whole COARSEST cells so every resolution level is cropped consistently), clipped to
    the prepared domain.  Arrays are sliced on whole cells, so source weights and the study mask are unchanged and the
    georeference is shifted by the offset: physical positions are identical.  The external patch centre is re-based to the
    new origin.  Raises if a local source layer would lose weight or the external patch would be cut (mass within 3 sigma
    box < 99.9 % of the full-domain Gaussian)."""
    if not buffer_m >= 0:
        raise ValueError("buffer_m must be >= 0")
    res = sorted(region.levels)
    coarse = region.levels[res[-1]]
    i0, i1, j0, j1 = _cells_with(coarse.arrays["study_fraction"])
    nb = int(math.ceil(buffer_m / coarse.grid.dx - 1e-9)), int(math.ceil(buffer_m / coarse.grid.dy - 1e-9))
    c0, c1 = max(i0 - nb[0], 0), min(i1 + nb[0], coarse.grid.nx)
    r0, r1 = max(j0 - nb[1], 0), min(j1 + nb[1], coarse.grid.ny)
    off = (c0 * coarse.grid.dx, r0 * coarse.grid.dy)
    levels: dict[float, GridLevel] = {}
    for rs, lv in region.levels.items():
        g = lv.grid
        fx, fy = coarse.grid.dx / g.dx, coarse.grid.dy / g.dy
        a0, a1, b0, b1 = int(round(c0 * fx)), int(round(c1 * fx)), int(round(r0 * fy)), int(round(r1 * fy))
        arrays: dict[str, np.ndarray] = {}
        for name, arr in lv.arrays.items():
            if arr.ndim == 2 and arr.shape == g.shape:
                arrays[name] = np.ascontiguousarray(arr[b0:b1, a0:a1])
            elif name == "profile_ns_z":
                arrays[name] = np.ascontiguousarray(arr[b0:b1])
            elif name == "profile_ew_z":
                arrays[name] = np.ascontiguousarray(arr[a0:a1])
            else:
                arrays[name] = arr
        for name in ("study_fraction", "w_road", "w_builtup", "w_facility"):
            if name in arrays and not np.isclose(arrays[name].sum(), lv.arrays[name].sum(), rtol=1e-12, atol=0.0):
                raise ValueError(f"crop (buffer {buffer_m} m) would cut weight of {name!r} at {rs} m")
        ng = dataclasses.replace(g, x0=g.x0 + off[0], y0=g.y0 + off[1], nx=a1 - a0, ny=b1 - b0)
        info = dict(lv.info)
        if "centroid_cell" in info:
            info["centroid_cell"] = {"j": info["centroid_cell"]["j"] - b0, "i": info["centroid_cell"]["i"] - a0}
        info["cropped_from"] = {"nx": g.nx, "ny": g.ny, "cell_offset": [a0, b0]}
        levels[rs] = GridLevel(rs, ng, arrays, info)
    man = dict(region.manifest)
    p = dict(man["external_patch"])
    cx, cy = p["centre_local_m"]
    p["centre_local_m"] = [cx - off[0], cy - off[1]]
    man["external_patch"] = p
    gr = region.georef
    out = Region(region.name, GeoRef(gr.epsg, gr.x0 + off[0], gr.y0 + off[1]), levels, man)
    kept = _gaussian_mass_in_box(cx, cy, float(p["sigma_m"]), off, (coarse.grid.dx * (c1 - c0), coarse.grid.dy * (r1 - r0)),
                                 (coarse.grid.lx, coarse.grid.ly))
    if kept < 0.999:
        raise ValueError(f"crop (buffer {buffer_m} m) cuts the external patch: only {kept:.4f} of its Gaussian mass remains")
    return out


def _gaussian_mass_in_box(cx: float, cy: float, sigma: float, off: Offset, size: tuple[float, float],
                          full: tuple[float, float]) -> float:
    """Gaussian mass inside the crop box relative to the mass inside the full prepared domain (continuous integrals)."""
    def part(c: float, lo: float, hi: float) -> float:
        s = math.sqrt(2.0) * sigma
        return 0.5 * (math.erf((hi - c) / s) - math.erf((lo - c) / s))
    crop = part(cx, off[0], off[0] + size[0]) * part(cy, off[1], off[1] + size[1])
    whole = part(cx, 0.0, full[0]) * part(cy, 0.0, full[1])
    return crop / whole


def crop_offset_m(full: Region, cropped: Region) -> Offset:
    """Offset (m) of ``cropped``'s south-west corner in ``full``'s local coordinates (from the georeferences)."""
    return cropped.georef.x0 - full.georef.x0, cropped.georef.y0 - full.georef.y0


# ---------------------------------------------------------------------------------------------- the scenario
@dataclass(frozen=True, eq=False)
class BelauriCase:
    """One forward problem on the real region.  ``theta`` is aligned with ``scenario.param_table()`` (sources, then
    bases); ``labels`` has one dict per parameter (name, kind, units, group, true_value); ``georef`` records the CRS, the
    SW corner in UTM metres, the offset in the standard domain and the resolution."""
    scenario: Scenario
    theta: np.ndarray
    labels: list[dict[str, Any]]
    georef: dict[str, Any]
    region: Region


def belauri_scenario(region: Region, resolution_m: float, wind: WindFactory | tuple[WindSegment, ...], *,
                     groups: Sequence[str] = ("road", "builtup", "external"), crop_buffer_m: float | None = None,
                     duration_s: float, dt_s: float, output_interval_s: float, K: float, lam: float, H: float,
                     amplitudes: dict[str, float], bases: Sequence[InflowBasis] = (),
                     external_profile: tuple[tuple[float, float], ...] = ((0.0, 0.0), (7200.0, 1.0)),
                     origin_tol: float = LOCAL_ORIGIN_TOL) -> BelauriCase:
    """Build the forward problem.  ``groups`` selects the source layers: 'road' | 'builtup' | 'facility' (local rasters),
    'external' (Gaussian patch, external tag, onset ``external_profile``).  ``amplitudes`` maps each group (and each
    basis name) to its synthetic rate (microgram/s; basis: microgram/m^3).  ``crop_buffer_m`` None keeps the prepared
    (standard) outer domain; otherwise ``crop_region`` is used.  Boundary open, zero background, zero initial field.
    Origin tags are verified with ``check_origin_tags`` (tolerance ``origin_tol``)."""
    full = region
    reg = full if crop_buffer_m is None else crop_region(full, crop_buffer_m)
    offset = crop_offset_m(full, reg)
    dom = reg.domain(resolution_m)
    srcs: list[Any] = []
    for g in groups:
        if g == "external":
            srcs.append(dataclasses.replace(reg.external_gaussian(external_profile), name="external_patch"))
        elif g in LOCAL_LAYERS:
            srcs.append(reg.raster_source(resolution_m, g))
        else:
            raise ValueError(f"unknown source group {g!r}; use {LOCAL_LAYERS + ('external',)}")
    missing = [g for g in groups if g not in amplitudes]
    if missing or any(b.name not in amplitudes for b in bases):
        raise ValueError(f"amplitudes missing for {missing + [b.name for b in bases if b.name not in amplitudes]}")
    theta = np.array([float(amplitudes[g]) for g in groups] + [float(amplitudes[b.name]) for b in bases])
    winds = wind(dom, offset) if callable(wind) else wind
    sc = Scenario(domain=dom, sources=tuple(srcs), transport=Transport(winds, K, lam, H),
                  boundary=Boundary("open", 0.0, tuple(bases)), initial=Initial(),
                  times=Times(duration_s, dt_s, output_interval_s), study=reg.study_mask(resolution_m))
    check_origin_tags(sc, tol=origin_tol)
    labels = [{"name": p.name, "kind": p.kind, "units": p.units, "group": p.group, "true_value": float(v)}
              for p, v in zip(sc.param_table(), theta)]
    gref = {"epsg": reg.georef.epsg, "x0_utm": reg.georef.x0, "y0_utm": reg.georef.y0, "offset_in_standard_domain_m": list(offset),
            "resolution_m": resolution_m, "crop_buffer_m": crop_buffer_m, "nx": dom.nx, "ny": dom.ny,
            "lx_m": dom.lx, "ly_m": dom.ly}
    return BelauriCase(sc, theta, labels, gref, reg)


def study_weights_of(case: BelauriCase) -> np.ndarray:
    """The study mask (cell area fractions) of the case."""
    st = case.scenario.study
    assert isinstance(st, StudyMask)
    return np.asarray(st.weights, dtype=float)

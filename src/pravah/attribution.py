"""Origin attribution and identifiability diagnostics for SYNTHETIC experiments (the truth is known).

Pure functions on ``inverse.Responses`` + a parameter vector theta; ``fit_sources`` never sees any of this (tags
cannot leak into the inversion).

Tags of the total concentration c = sum of
* ``local``            emission parameters whose source group is "local";
* ``external_sources`` emission parameters whose source group is "external";
* ``boundary``         inflow-basis parameters (distant unresolved sources entering through the open boundary);
* ``background``       the static inflow background (known, zero amplitude parameters);
* ``initial``          the configured initial field (transported/decayed).
The attribution follows the ORIGIN of the mass, not where it currently is: a local plume that leaves the study region
and returns after a wind reversal stays ``local``.

Quantities are CONCENTRATION-derived.  ``concentration_external_fraction`` is
    F_ext = sum_w (external_sources + boundary + background) / sum_w total,
sum_w = sum over study-region cells (area-fraction weights) times trapezoid time weights over output times
>= t_exclude (``metrics.quadrature_weights``).  "External" means "originating outside the study region" and includes
the static background; the configured initial field is reported separately (``origin_fractions``).  An inward mass-flux
fraction is a different quantity and is never substituted.  F is None (JSON null) when the time-mean study-weighted
total M is below eps_rel * scale, scale = max over the window of the study-weighted mean total concentration.

Emission amplitudes (ug/s) and inflow amplitudes (ug/m^3) are never compared by raw magnitude: errors are reported per
kind and per group.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import xarray as xr

from .config import Initial, Scenario, SolverOptions
from .forward import FIPY, ForwardSolver
from .inverse import FitResult, ResponseCache, Responses, run_forward
from .metrics import field_rmse, quadrature_weights
from .sources import study_weights

TAGS = ("local", "external_sources", "boundary", "background", "initial")
UNIDENT_NORM = 1.0  # whitened, scaled column norm below this: data constrain the parameter worse than one q_scale
UNIDENT_STD_RATIO = 10.0  # q_std / q_scale above this: posterior std exceeds ten scales
CONFOUND_COS = 0.99  # |cos| between whitened scaled columns above this: confounded pair
COND_MAX = 1e6  # condition number above this counts as rank deficient (in addition to numerical rank / effective rank)


def base_parts(scenario: Scenario, solver: SolverOptions = SolverOptions(), forward: ForwardSolver = FIPY,
               cache: ResponseCache | None = None) -> dict[str, np.ndarray]:
    """Split of ``Responses.base`` (all amplitudes 0): 'initial' = configured IC only (static background 0) and
    'background' = static background only (zero IC), each (nt, ny, nx).  Their sum equals the base field up to solver
    round-off.  Memoised in ``cache.extra`` when a ResponseCache is given."""
    key = f"base_parts:{scenario!r}:{solver!r}:{forward!r}"
    if cache is not None and key in cache.extra:
        return cache.extra[key]
    zeros = np.zeros(scenario.n_params)
    no_bg = dataclasses.replace(scenario, boundary=dataclasses.replace(scenario.boundary, background=0.0))
    no_ic = dataclasses.replace(scenario, initial=Initial())
    parts = {"initial": np.asarray(run_forward(forward, no_bg, zeros, solver).concentration.values, dtype=float),
             "background": np.asarray(run_forward(forward, no_ic, zeros, solver).concentration.values, dtype=float)}
    if cache is not None:
        cache.extra[key] = parts
    return parts


def contributions(resp: Responses, theta: np.ndarray | list[float] | tuple[float, ...],
                  parts: dict[str, np.ndarray]) -> dict[str, xr.DataArray]:
    """Tagged fields (see module docstring); their sum equals ``resp.field(theta)`` to round-off."""
    th = np.asarray(theta, dtype=float)
    n = resp.g.shape[0]
    if th.shape != (n,) or len(resp.param_kinds) != n:
        raise ValueError(f"theta must have shape ({n},) and resp must carry a parameter table")
    out = {t: np.zeros_like(resp.base) for t in TAGS}
    for j in range(n):
        if resp.param_kinds[j] == "inflow":
            tag = "boundary"
        else:
            tag = "local" if resp.param_groups[j] == "local" else "external_sources"
        out[tag] = out[tag] + th[j] * resp.g[j]
    out["background"] = np.asarray(parts["background"], dtype=float)
    out["initial"] = np.asarray(parts["initial"], dtype=float)
    return {t: resp.dataarray(v, t) for t, v in out.items()}


def external_part(contrib: dict[str, xr.DataArray]) -> xr.DataArray:
    """Origin outside the study region: external_sources + boundary + background."""
    return contrib["external_sources"] + contrib["boundary"] + contrib["background"]


def total_field(contrib: dict[str, xr.DataArray]) -> xr.DataArray:
    return sum((contrib[t] for t in TAGS[1:]), contrib[TAGS[0]])


def _arr(da: xr.DataArray) -> np.ndarray:
    return np.asarray(da.transpose("time", "y", "x").values, dtype=float)


def _study_w(scenario: Scenario, weights: np.ndarray | None) -> np.ndarray:
    if weights is not None:
        return np.asarray(weights, dtype=float)
    w = study_weights(scenario)
    if w is None:
        raise ValueError("scenario has no study region: the origin fractions need study weights")
    return w


def origin_fractions(contrib: dict[str, xr.DataArray], scenario: Scenario, t_exclude: float = 0.0,
                     eps_rel: float = 1e-3, weights: np.ndarray | None = None) -> dict[str, float | None]:
    """{'local', 'external', 'initial'}: study-weighted, time-integrated share of the total (sum 1 when defined),
    each None when the total is ~0 (see module docstring).  external = external_sources + boundary + background."""
    total = _arr(total_field(contrib))
    times = np.asarray(contrib[TAGS[0]]["time"].values, dtype=float)
    ny, nx = total.shape[1:]
    sel, dt, w, den = quadrature_weights(times, ny, nx, _study_w(scenario, weights), t_exclude)
    tot = total[sel]
    mean_t = np.einsum("tyx,yx->t", tot, w) / w.sum()  # study-weighted mean total concentration per output time
    scale = float(mean_t.max())
    d_tot = float(np.einsum("t,tyx,yx->", dt, tot, w) / den)  # time-mean of the same
    if not (scale > 0 and d_tot >= eps_rel * scale and d_tot > 0):
        return {"local": None, "external": None, "initial": None}
    parts = {"local": _arr(contrib["local"]), "external": _arr(external_part(contrib)),
             "initial": _arr(contrib["initial"])}
    return {k: float(np.einsum("t,tyx,yx->", dt, v[sel], w) / den / d_tot) for k, v in parts.items()}


def external_fraction(contrib: dict[str, xr.DataArray], scenario: Scenario, t_exclude: float = 0.0,
                      eps_rel: float = 1e-3, weights: np.ndarray | None = None) -> float | None:
    """concentration_external_fraction F_ext (module docstring); None when undefined (total ~ 0)."""
    return origin_fractions(contrib, scenario, t_exclude, eps_rel, weights)["external"]


def component_errors(contrib_est: dict[str, xr.DataArray], contrib_true: dict[str, xr.DataArray], scenario: Scenario,
                     t_exclude: float = 0.0, eps_rel: float = 1e-3) -> dict[str, float | None]:
    """Study-weighted RMSE (same quadrature as ``metrics.field_rmse``) of the total, local and external (external_sources
    + boundary + background) fields, the two external fractions and their difference (None if either is undefined)."""
    w = _study_w(scenario, None)
    out: dict[str, float | None] = {
        "rmse_total": field_rmse(total_field(contrib_est), total_field(contrib_true), w, t_exclude),
        "rmse_local": field_rmse(contrib_est["local"], contrib_true["local"], w, t_exclude),
        "rmse_external": field_rmse(external_part(contrib_est), external_part(contrib_true), w, t_exclude)}
    fe = external_fraction(contrib_est, scenario, t_exclude, eps_rel)
    ft = external_fraction(contrib_true, scenario, t_exclude, eps_rel)
    out["external_fraction_est"], out["external_fraction_true"] = fe, ft
    out["external_fraction_error"] = None if fe is None or ft is None else fe - ft
    return out


def _rel_l2(d: np.ndarray, t: np.ndarray) -> float | None:
    nt = float(np.linalg.norm(t))
    return float(np.linalg.norm(d) / nt) if nt > 0 else None


def emission_error(theta_hat: np.ndarray, theta_true: np.ndarray, scenario: Scenario) -> dict[str, Any]:
    """Relative L2 errors over EMISSION parameters only (key 'all_emission' and one per group, e.g. 'local',
    'external'; None if that group's truth is zero).  Inflow amplitudes (ug/m^3) are reported separately under
    'inflow': {'abs_error_ug_m3': {name: error}, 'relative_l2': ...}; they are never mixed with emissions (ug/s)."""
    th, tt = np.asarray(theta_hat, dtype=float), np.asarray(theta_true, dtype=float)
    tab = scenario.param_table()
    if th.shape != (len(tab),) or tt.shape != th.shape:
        raise ValueError(f"theta_hat and theta_true must have shape ({len(tab)},)")
    d = th - tt
    em = np.array([p.kind == "emission" for p in tab])
    out: dict[str, Any] = {"all_emission": _rel_l2(d[em], tt[em]) if em.any() else None}
    for g in sorted({p.group for p in tab if p.kind == "emission"}):
        m = np.array([p.kind == "emission" and p.group == g for p in tab])
        out[g] = _rel_l2(d[m], tt[m])
    inf = np.array([p.kind == "inflow" for p in tab])
    out["inflow"] = {"abs_error_ug_m3": {p.name: float(d[i]) for i, p in enumerate(tab) if p.kind == "inflow"},
                     "relative_l2": _rel_l2(d[inf], tt[inf]) if inf.any() else None}
    return out


def identifiability_report(fit: FitResult) -> dict[str, Any]:
    """Identifiability summary with explicit flags (nothing is hidden behind a prior).

    Whitened, scaled columns a_j = A_j * q_scale_j / sigma (norm ``column_norm``); flags:
    * 'rank_deficient': numerical rank < n_params or condition > 1e6 (the unregularised problem is singular or nearly so);
    * 'weakly_constrained': effective rank < n_params (some direction is constrained worse than one q_scale at the stated
      noise); this depends on the q_scale choice, so it is a statement about the scaled problem, not a singularity;
    * 'unidentifiable:<name>': column norm < 1 (the data constrain the parameter worse than one q_scale at the stated
      noise) or q_std / q_scale > 10.  With all sigma == 0 the norm is in ug/m^3 and the 1.0 threshold is not
      meaningful; use a realistic noise_std;
    * 'confounded:<a>,<b>': |cos angle| between a_a and a_b > 0.99 (columns with negligible norm are skipped).  This is
      PAIRWISE: a column lying in the span of two or more others (without being close to any one) is not flagged."""
    n = len(fit.q_hat)
    names = fit.param_names or tuple(f"p{j}" for j in range(n))
    kinds = fit.param_kinds or ("emission",) * n
    units = fit.param_units or ("",) * n
    norms = fit.column_norms
    flags: list[str] = []
    if fit.rank < n or fit.condition > COND_MAX:
        flags.append("rank_deficient")
    if fit.effective_rank < n:
        flags.append("weakly_constrained")
    params = []
    for j in range(n):
        ratio = float(fit.q_std[j] / fit.q_scale[j])
        params.append({"name": names[j], "kind": kinds[j], "units": units[j], "q_hat": float(fit.q_hat[j]),
                       "q_std": float(fit.q_std[j]), "q_scale": float(fit.q_scale[j]), "q_std_over_scale": ratio,
                       "column_norm": float(norms[j])})
        if norms[j] < UNIDENT_NORM or ratio > UNIDENT_STD_RATIO:
            flags.append(f"unidentifiable:{names[j]}")
    big = float(norms.max()) if norms.size else 0.0
    for a in range(n):
        for b in range(a + 1, n):
            if min(norms[a], norms[b]) > 1e-8 * big and abs(fit.column_cosines[a, b]) > CONFOUND_COS:
                flags.append(f"confounded:{names[a]},{names[b]}")
    return {"singular_values": [float(s) for s in fit.singular_values], "rank": fit.rank,
            "effective_rank": fit.effective_rank, "condition": fit.condition, "params": params, "flags": flags}

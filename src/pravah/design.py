"""Sensor-placement machinery: candidates, constraints, layouts, cached design cases, fast fits, greedy selection.

Speed.  The forward model is affine in q, so for every case the unit responses are computed ONCE (``ResponseCache``)
and only arrays are kept in a ``DesignCase``: truth readings and the operator (b, A) at every candidate site, a
standard-normal common-random-number noise array, and the quadratic form of the weighted regional field error.  A
fit of any layout is then a tiny bounded least-squares problem (``inverse.solve_linear``, the very routine used by
``fit_sources``) and its field error MSE_w costs O(n_src^2): with d0 = base - truth,
``E(q) = c0 + 2 q.h + q'Gq`` (exact, same weights as ``metrics.field_mse``).

Leakage.  The fit sees only the noisy readings at the layout's sites (y = truth_readings + sigma * noise).  Three
roles exist.  Layout selection (``select_layout`` / ``greedy_trace``) accepts only ``role == 'design'`` cases;
``evaluate_layouts`` accepts only ``role == 'test'`` cases (override with allow_design=True for diagnostics);
``evaluate_validation`` accepts only ``role == 'validation'`` cases and exists for the Tikhonov-strength selection
(``experiments.run_alpha_selection``).  ``assert_disjoint`` checks that the design, test and validation cases share
no noise seed, name or truth fingerprint.

Baked at build time: obs_times, t_exclude, weight and eval_sites of the ``DesignOptions`` passed to ``build_case``
(fit-time options must carry the same values; this is checked).  noise_std, alpha, prior, q_scale are applied at fit
time, so one built case can be fitted with many alphas (``dataclasses.replace(options, alpha=a)``) without any
forward solve.  Row order of the operator is time-major then sensor, as in ``inverse``.

Parameters and objectives.  A case uses ALL n_params columns of the inference ``Responses`` (emission amplitudes,
then inflow-basis amplitudes); ``q_true`` is the TRUTH parameter vector theta_true, matched BY KIND BLOCK (first emissions, first inflow bases) and by
name (``scenario.param_table()``) to the first n_params truth parameters; further truth parameters are
'unmodelled'.  ``DesignCase.objectives`` holds named quadratic forms E(theta) = c0 + 2 theta.h + theta'G theta:
'total' (the weighted regional field error, as always) and, if ``truth_components`` is given to ``build_case``,
'local' and 'external' (see ``build_case``); 'attribution' = 0.5 (E_local / s_local + E_external / s_external) is
formed at fit time from the options' ``attribution_scales`` (``attribution_scales(design_cases)``, design cases only).

Feasibility.  Choosing `budget` sites with pairwise distance >= min_separation is a maximum-independent-set problem
(NP-hard).  ``InfeasibleError`` is raised exactly when budget > number of allowed sites; otherwise detection is
heuristic: a deterministic index-order greedy construction plus ``N_FEASIBILITY_ORDERS`` seeded random orders.
"""
from __future__ import annotations

import hashlib
import json
import time as _time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr

from .attribution import base_parts, external_fraction
from .config import source_signature, Domain, Scenario, SolverOptions, to_dict
from .inverse import InferenceOptions, ResponseCache, run_forward, solve_linear
from .forward import FIPY, ForwardSolver
from .metrics import quadrature_weights, source_error
from .observations import SensorLayout, candidate_hull, sample

Layout = tuple[int, ...]  # sorted, unique candidate indices
N_FEASIBILITY_ORDERS = 200
MAX_RANDOM_ATTEMPTS = 200
ROLES = ("design", "test", "validation")
OBJECTIVES = ("total", "local", "external", "attribution")
DEFAULT_CATEGORY = "interior"
FRACTION_EPS_REL = 1e-3  # same default as attribution.external_fraction


class InfeasibleError(ValueError):
    """No layout satisfies the budget and constraints (exact if budget > #allowed, heuristic otherwise)."""


# ------------------------------------------------------------------ candidates and constraints
@dataclass(frozen=True, eq=False)
class Candidates:
    ids: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray
    category: tuple[str, ...] | None = None  # one label per site (default all 'interior'), e.g. 'reference'
    meta: dict[str, Any] | None = None  # passthrough (lon/lat, flags, ...); never interpreted here

    def __post_init__(self) -> None:
        x, y = np.asarray(self.x, dtype=float), np.asarray(self.y, dtype=float)
        if not (len(self.ids) == x.size == y.size) or x.ndim != 1 or y.ndim != 1:
            raise ValueError("ids, x, y must be 1-D with equal lengths")
        if self.category is None:
            object.__setattr__(self, "category", (DEFAULT_CATEGORY,) * len(self.ids))
        elif len(self.category) != len(self.ids):
            raise ValueError("category needs one label per candidate")
        else:
            object.__setattr__(self, "category", tuple(str(c) for c in self.category))
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("candidate ids must be unique")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("candidate coordinates must be finite")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def categories(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], self.category)

    def sensor_layout(self, layout: Layout | None = None) -> SensorLayout:
        idx = range(len(self)) if layout is None else layout
        return SensorLayout(tuple(self.ids[i] for i in idx), tuple(float(self.x[i]) for i in idx),
                            tuple(float(self.y[i]) for i in idx))


def candidate_grid(domain: Domain, n: int = 10) -> Candidates:
    """n x n sites at ((i+0.5) Lx/n, (j+0.5) Ly/n) (500..9500 m for the default); index k = i + j*n.  Must lie in
    the cell-centre hull (``observations.candidate_hull``), else ValueError (e.g. n larger than the grid)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    xs = (np.arange(n) + 0.5) * domain.lx / n
    ys = (np.arange(n) + 0.5) * domain.ly / n
    (x0, x1), (y0, y1) = candidate_hull(domain)
    tol = 1e-9 * max(domain.lx, domain.ly)
    if xs[0] < x0 - tol or xs[-1] > x1 + tol or ys[0] < y0 - tol or ys[-1] > y1 + tol:
        raise ValueError(f"candidate grid n={n} lies outside the cell-centre hull [{x0}, {x1}] x [{y0}, {y1}]")
    X, Y = np.meshgrid(xs, ys)
    ids = tuple(f"c{k:03d}" for k in range(n * n))
    return Candidates(ids, X.reshape(-1), Y.reshape(-1))


@dataclass(frozen=True, eq=False)
class Constraints:
    """allowed: bool mask over candidates (None = all); min_separation: minimum pairwise distance [m] (>= comparison).
    max_per_category: cap on the number of chosen sites per ``Candidates.category`` label, e.g. {'reference': 2}
    (labels not listed are unlimited; reference sites count in the budget).  exclusion_radius_m / exclusion_points:
    a site closer than the radius (strictly <) to any exclusion point (e.g. held-out evaluation points) is not allowed,
    which separates design sites from held-out sites spatially."""
    allowed: np.ndarray | None = None
    min_separation: float = 0.0
    max_per_category: dict[str, int] | None = None
    exclusion_radius_m: float = 0.0
    exclusion_points: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.min_separation >= 0:
            raise ValueError("min_separation must be >= 0")
        if not self.exclusion_radius_m >= 0:
            raise ValueError("exclusion_radius_m must be >= 0")
        if self.max_per_category is not None and any(int(v) != v or v < 0 for v in self.max_per_category.values()):
            raise ValueError("max_per_category values must be integers >= 0")
        if any(len(p) != 2 for p in self.exclusion_points):
            raise ValueError("exclusion_points must be (x, y) pairs")


def _allowed(candidates: Candidates, constraints: Constraints) -> np.ndarray:
    if constraints.allowed is None:
        ok = np.ones(len(candidates), dtype=bool)
    else:
        a = np.asarray(constraints.allowed, dtype=bool)
        if a.shape != (len(candidates),):
            raise ValueError(f"allowed must have shape ({len(candidates)},), got {a.shape}")
        ok = a.copy()
    if constraints.exclusion_radius_m > 0:
        for px, py in constraints.exclusion_points:
            d = np.hypot(candidates.x - px, candidates.y - py)
            ok &= d >= constraints.exclusion_radius_m * (1 - 1e-12)
    return ok


def _category_full(candidates: Candidates, constraints: Constraints, chosen: tuple[int, ...] | Layout) -> np.ndarray:
    """Mask of sites whose category has already reached its cap among `chosen`."""
    full = np.zeros(len(candidates), dtype=bool)
    if constraints.max_per_category:
        cats = np.asarray(candidates.categories)
        counts = Counter(candidates.categories[c] for c in chosen)
        for cat, cap in constraints.max_per_category.items():
            if counts.get(cat, 0) >= cap:
                full |= cats == cat
    return full


def feasible_sites(candidates: Candidates, constraints: Constraints, chosen: Layout | tuple[int, ...] = ()
                   ) -> np.ndarray:
    """Indices (ascending) of allowed, not-yet-chosen sites at distance >= min_separation from every chosen site and
    whose category cap (max_per_category) is not yet reached."""
    ok = _allowed(candidates, constraints)
    for c in chosen:
        ok[c] = False
        if constraints.min_separation > 0:
            d = np.hypot(candidates.x - candidates.x[c], candidates.y - candidates.y[c])
            ok &= d >= constraints.min_separation * (1 - 1e-12)
    ok &= ~_category_full(candidates, constraints, chosen)
    return np.flatnonzero(ok)


def _construct(candidates: Candidates, budget: int, constraints: Constraints, order: np.ndarray) -> Layout | None:
    chosen: list[int] = []
    for s in order:
        if len(chosen) == budget:
            break
        if s in feasible_sites(candidates, constraints, tuple(chosen)):
            chosen.append(int(s))
    return tuple(sorted(chosen)) if len(chosen) == budget else None


def check_feasible(candidates: Candidates, budget: int, constraints: Constraints = Constraints()) -> None:
    """Raise InfeasibleError if no layout of `budget` sites is found (see module docstring for exactness)."""
    if budget < 1:
        raise ValueError("budget must be >= 1")
    allowed = _allowed(candidates, constraints)
    n_allowed = int(allowed.sum())
    if budget > n_allowed:
        raise InfeasibleError(f"budget {budget} exceeds the {n_allowed} allowed sites")
    if constraints.max_per_category:
        per_cat = Counter(c for c, a in zip(candidates.categories, allowed) if a)
        capacity = sum(min(n, constraints.max_per_category.get(c, n)) for c, n in per_cat.items())
        if budget > capacity:
            raise InfeasibleError(f"budget {budget} exceeds the {capacity} sites allowed by max_per_category")
    if constraints.min_separation == 0:
        return
    rng = np.random.default_rng(0)
    orders = [np.arange(len(candidates))] + [rng.permutation(len(candidates)) for _ in range(N_FEASIBILITY_ORDERS)]
    if all(_construct(candidates, budget, constraints, o) is None for o in orders):
        raise InfeasibleError(f"no layout of {budget} sites with min_separation {constraints.min_separation} found "
                              f"({len(orders)} construction orders tried; heuristic)")


def infeasible_budgets(candidates: Candidates, constraints: Constraints, budgets: list[int] | tuple[int, ...]
                       ) -> dict[int, str]:
    """{budget: reason} for every budget that ``check_feasible`` rejects (exact for the allowed-site and category
    capacity tests, heuristic under min_separation); feasible budgets are absent."""
    out: dict[int, str] = {}
    for b in budgets:
        try:
            check_feasible(candidates, int(b), constraints)
        except InfeasibleError as e:
            out[int(b)] = str(e)
    return out


def heldout_blocks(points: Candidates, n_blocks_x: int, n_blocks_y: int, which: int | tuple[int, ...] | list[int]
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic spatial-block split of evaluation points.  The bounding box of `points` is cut into
    n_blocks_x * n_blocks_y equal rectangles (block id = ix + iy * n_blocks_x; points on the upper edge belong to the
    last block).  Returns (heldout, rest): ascending index arrays, disjoint, of the points inside the block(s) `which`
    and outside them.  Use the held-out points as ``Constraints.exclusion_points`` to separate them from design sites."""
    if n_blocks_x < 1 or n_blocks_y < 1:
        raise ValueError("n_blocks_x and n_blocks_y must be >= 1")
    if len(points) == 0:
        raise ValueError("no points")
    ids = {int(which)} if isinstance(which, (int, np.integer)) else {int(w) for w in which}
    if any(not 0 <= w < n_blocks_x * n_blocks_y for w in ids):
        raise ValueError(f"block ids must lie in 0..{n_blocks_x * n_blocks_y - 1}")

    def block(v: np.ndarray, n: int) -> np.ndarray:
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            return np.zeros(v.size, dtype=int)
        return np.minimum(((v - lo) / (hi - lo) * n).astype(int), n - 1)

    bid = block(points.x, n_blocks_x) + n_blocks_x * block(points.y, n_blocks_y)
    inside = np.isin(bid, sorted(ids))
    return np.flatnonzero(inside), np.flatnonzero(~inside)


def random_layouts(candidates: Candidates, budget: int, n: int, constraints: Constraints = Constraints(),
                   seed: int = 0) -> list[Layout]:
    """n layouts, each from a seeded permutation of the sites, accepting sequentially every site that is feasible
    (allowed, separated from those already accepted) until `budget` are chosen; if a permutation falls short another
    is drawn (up to MAX_RANDOM_ATTEMPTS, then InfeasibleError).  Deterministic per seed; layouts may repeat."""
    check_feasible(candidates, budget, constraints)
    rng = np.random.default_rng(seed)
    out: list[Layout] = []
    for _ in range(n):
        for _attempt in range(MAX_RANDOM_ATTEMPTS):
            lay = _construct(candidates, budget, constraints, rng.permutation(len(candidates)))
            if lay is not None:
                out.append(lay)
                break
        else:
            raise InfeasibleError("random construction failed to reach the budget")
    return out


def spread_layout(candidates: Candidates, budget: int, constraints: Constraints = Constraints()) -> Layout:
    """Farthest-point layout: start at the feasible site nearest the centre of the candidates' bounding box (= the
    domain centre for a symmetric grid), then repeatedly add the feasible site that maximises the distance to its
    nearest chosen site.  All ties go to the lowest index.  Deterministic.  InfeasibleError if the construction
    gets stuck under min_separation."""
    check_feasible(candidates, budget, constraints)
    cx = 0.5 * (candidates.x.min() + candidates.x.max())
    cy = 0.5 * (candidates.y.min() + candidates.y.max())
    chosen: list[int] = []
    for _ in range(budget):
        feas = feasible_sites(candidates, constraints, tuple(chosen))
        if feas.size == 0:
            raise InfeasibleError("farthest-point construction got stuck under min_separation")
        if not chosen:
            score = -np.hypot(candidates.x[feas] - cx, candidates.y[feas] - cy)
        else:
            dx = candidates.x[feas][:, None] - candidates.x[chosen][None, :]
            dy = candidates.y[feas][:, None] - candidates.y[chosen][None, :]
            score = np.hypot(dx, dy).min(axis=1)
        best = np.flatnonzero(score >= score.max() - 1e-12 * max(abs(score.max()), 1.0))
        chosen.append(int(feas[best[0]]))
    return tuple(sorted(chosen))


# ------------------------------------------------------------------ cases
@dataclass(frozen=True, eq=False)
class DesignOptions:
    """noise_std [ug/m^3] (sigma); alpha/prior/q_scale/solver as in InferenceOptions (prior/q_scale: one entry per
    parameter); obs_times default 600, 1200, ..., T (t=0 excluded); weight (ny, nx) for MSE_w; eval_sites for held-out
    RMSE: one ``Candidates`` (column heldout_rmse) or a dict name -> Candidates of groups (heldout_rmse pools all
    groups, ``heldout_<name>_rmse`` per group, e.g. names 'block' and 'random'); attribution_scales = (s_local,
    s_external) [ug^2/m^6] for the 'attribution' objective (``attribution_scales`` helper, design cases only)."""
    noise_std: float
    obs_times: tuple[float, ...] | None = None
    alpha: float = 0.0
    prior: tuple[float, ...] | None = None
    q_scale: tuple[float, ...] | None = None
    t_exclude: float = 0.0
    weight: np.ndarray | None = None
    eval_sites: Candidates | dict[str, Candidates] | None = None
    solver: SolverOptions = field(default_factory=SolverOptions)
    attribution_scales: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not (np.isfinite(self.noise_std) and self.noise_std >= 0):
            raise ValueError("noise_std must be finite and >= 0")
        if self.attribution_scales is not None:
            sl, se = self.attribution_scales
            if not (np.isfinite(sl) and np.isfinite(se) and sl > 0 and se > 0):
                raise ValueError("attribution_scales must be two finite positive numbers")
        if isinstance(self.eval_sites, dict) and (not self.eval_sites or any(len(c) == 0 for c in self.eval_sites.values())):
            raise ValueError("eval_sites groups must be non-empty")

    def inference(self) -> InferenceOptions:
        return InferenceOptions("linear", self.q_scale, self.alpha, self.prior, (), self.solver)


def _fingerprint(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=list).encode()).hexdigest()[:16]


def _times(options: DesignOptions, scenario: Scenario) -> tuple[float, ...]:
    if options.obs_times is not None:
        return tuple(float(t) for t in options.obs_times)
    T = scenario.times.duration
    return tuple(float(t) for t in np.arange(600.0, T + 1e-9, 600.0))


def _eval_groups(options: DesignOptions) -> list[tuple[str, Candidates]] | None:
    return list(options.eval_sites.items()) if isinstance(options.eval_sites, dict) else None


def _eval_all(options: DesignOptions) -> Candidates | None:
    """The evaluation sites as one Candidates (groups concatenated, ids prefixed '<group>:')."""
    ev = options.eval_sites
    if ev is None or isinstance(ev, Candidates):
        return ev
    return Candidates(tuple(f"{n}:{i}" for n, c in ev.items() for i in c.ids),
                      np.concatenate([c.x for c in ev.values()]), np.concatenate([c.y for c in ev.values()]))


def _build_key(options: DesignOptions, obs_times: tuple[float, ...]) -> str:
    w = None if options.weight is None else hashlib.sha256(np.ascontiguousarray(options.weight).tobytes()).hexdigest()
    ev = options.eval_sites
    if ev is None:
        e: Any = None
    elif isinstance(ev, Candidates):
        e = (ev.ids, ev.x.tolist(), ev.y.tolist())
    else:
        e = [(n, c.ids, c.x.tolist(), c.y.tolist()) for n, c in ev.items()]
    return _fingerprint(obs_times, options.t_exclude, w, e)


@dataclass(frozen=True, eq=False)
class QuadForm:
    """E(theta) = c0 + 2 theta.h + theta'G theta over the full parameter vector (exact weighted MSE of a field)."""
    c0: float
    h: np.ndarray  # (n_params,)
    G: np.ndarray  # (n_params, n_params)

    def __call__(self, theta: np.ndarray) -> float:
        th = np.asarray(theta, dtype=float)
        return max(float(self.c0 + 2.0 * th @ self.h + th @ self.G @ th), 0.0)


@dataclass(frozen=True, eq=False)
class _Fraction:
    """Linear-in-theta pieces of the study-weighted external fraction of the ESTIMATED field (see build_case):
    S_t(theta) = tot0 + theta @ totg is sum_cells w * total[t] for the used output times, likewise ext."""
    tot0: np.ndarray  # (nsel,)
    totg: np.ndarray  # (n_params, nsel)
    ext0: np.ndarray
    extg: np.ndarray
    dt: np.ndarray  # trapezoid weights of the used times
    wsum: float
    den: float

    def __call__(self, theta: np.ndarray) -> float | None:
        s_tot = self.tot0 + theta @ self.totg
        s_ext = self.ext0 + theta @ self.extg
        scale = float((s_tot / self.wsum).max())
        d_tot = float(self.dt @ s_tot / self.den)
        if not (scale > 0 and d_tot >= FRACTION_EPS_REL * scale and d_tot > 0):
            return None
        return float(self.dt @ s_ext / self.den / d_tot)


@dataclass(frozen=True, eq=False)
class DesignCase:
    """One truth scenario + inference model, reduced to arrays (see module docstring).  No truth fields."""
    name: str
    role: str  # 'design' | 'test' | 'validation'
    q_true: np.ndarray  # theta_true: follows the TRUTH scenario's parameters (emissions, then inflow bases)
    noise_seed: int
    truth_fingerprint: str  # truth scenario + q_true
    inference_fingerprint: str
    build_key: str
    candidate_ids: tuple[str, ...]
    obs_times: tuple[float, ...]
    n_src: int  # emission sources of the inference model
    truth_readings: np.ndarray  # (n_site, nt) noiseless
    b_site: np.ndarray  # (n_site, nt)
    A_site: np.ndarray  # (n_site, nt, n_params)
    noise: np.ndarray  # (n_noise, n_site, nt) standard normal
    eval_truth: np.ndarray | None  # (n_eval, nt)
    b_eval: np.ndarray | None
    A_eval: np.ndarray | None
    c0: float  # c0, h, G: the 'total' objective (== objectives['total'])
    h: np.ndarray  # (n_params,)
    G: np.ndarray  # (n_params, n_params)
    n_params: int = 0  # parameters of the inference model (n_src emissions + inflow bases)
    param_kinds: tuple[str, ...] = ()
    param_names: tuple[str, ...] = ()
    objectives: dict[str, QuadForm] = field(default_factory=dict)  # 'total' (+ 'local', 'external' with components)
    unmodelled_mass: float = 0.0  # share of the truth EMISSION in truth sources the model does not have
    category: tuple[str, ...] = ()  # category of every candidate site
    eval_groups: tuple[tuple[str, int, int], ...] | None = None  # (name, start, stop) rows of the eval arrays
    component_ms: dict[str, float] | None = None  # mean square of the TRUE local / external field over the window
    fraction: _Fraction | None = None
    external_fraction_true: float | None = None
    truth_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))  # truth param of each model param

    @property
    def n_noise(self) -> int:
        return self.noise.shape[0]

    @property
    def has_components(self) -> bool:
        return self.component_ms is not None

    def field_error(self, q: np.ndarray) -> float:
        """Exact MSE_w of the field base + sum theta_j g_j against the case truth (O(n_params^2))."""
        q = np.asarray(q, dtype=float)
        return max(float(self.c0 + 2.0 * q @ self.h + q @ self.G @ q), 0.0)


def _check_coords(da: xr.DataArray, resp: Any, what: str, hint: str = "") -> None:
    for dim, coord in (("time", resp.times), ("y", resp.y), ("x", resp.x)):
        v = np.asarray(da[dim].values, dtype=float)
        if v.shape != coord.shape or not np.allclose(v, coord, rtol=0, atol=1e-9 * max(1.0, np.abs(coord).max())):
            raise ValueError(f"{what} and inference responses differ in coordinate {dim!r}" + hint)


def build_case(name: str, role: str, truth_scenario: Scenario, q_true: np.ndarray, inference_scenario: Scenario,
               candidates: Candidates, options: DesignOptions, noise_seed: int, n_noise: int,
               cache: ResponseCache, error_field: xr.DataArray | None = None,
               truth_field: xr.DataArray | None = None, forward: ForwardSolver = FIPY,
               truth_components: dict[str, xr.DataArray] | None = None) -> DesignCase:
    """Run one truth simulation and precompute everything fits need.

    truth_scenario may differ from inference_scenario (wind, resolution, extra sources).  ``q_true`` is the TRUTH
    parameter vector theta_true (one entry per ``truth_scenario.param_table()`` entry: emissions, then inflow bases);
    the first truth emissions and the first truth inflow bases are matched by position and name with the inference emissions and inflow bases
    (ValueError otherwise; emission sources must also have the same shape and group).  Further truth parameters are
    'unmodelled' (``unmodelled_mass`` = their share of the truth EMISSION; inflow amplitudes never enter it).
    Sensors read the NATIVE truth field.  The regional error compares the inference model with `error_field`
    (default: the native truth), which must be on the inference model's (time, y, x) coordinates exactly; for a refined
    truth the caller passes a block-averaged field.  Different output times raise ValueError (no time interpolation).
    ``truth_field`` may pass an already computed ``simulate(truth_scenario, q_true, options.solver).concentration``
    (saves the solve; the caller guarantees it).  ``forward`` (default: the FiPy reference) is used for both the
    inference responses and the truth run (see ``forward.ForwardSolver``).

    ``truth_components`` (optional) = {'local': field, 'external': field}: the truth contributions from
    ``attribution.contributions`` of the TRUTH ('external' = external_sources + boundary + background), on the
    inference grid and output times (computed by the caller; this module never builds truth tags).  They add the
    objectives 'local' and 'external', both with the weights/window of 'total' (``options.weight`` x trapezoid dt,
    t >= t_exclude):
      local    = weighted MSE of  sum_{local emission j} theta_j g_j  against truth_components['local'];
      external = weighted MSE of  background + sum_{j in external/boundary params} theta_j g_j  against
                 truth_components['external'], where `background` is the static inflow background of the INFERENCE
                 model (attribution.base_parts).
    The configured initial field is not part of either component (it is the same known field in model and truth), so
    nothing else enters; total/local/external are therefore dimensionally ug^2/m^6 and never mix emission with inflow
    units (theta enters only through the fields g_j theta_j).  Also stored: the external fraction of the truth
    (attribution.external_fraction definition, window weights = options.weight or ones) and the mean squares of the
    true components (for ``attribution_scales``)."""
    if role not in ROLES:
        raise ValueError("role must be 'design', 'test' or 'validation'")
    q_true = np.asarray(q_true, dtype=float)
    n_src = len(inference_scenario.sources)
    inf_tab, truth_tab = inference_scenario.param_table(), truth_scenario.param_table()
    n_params = len(inf_tab)
    if q_true.shape != (len(truth_tab),):
        raise ValueError(f"q_true must have one entry per truth parameter ({len(truth_tab)})")
    t_em = [i for i, p in enumerate(truth_tab) if p.kind == "emission"]
    t_in = [i for i, p in enumerate(truth_tab) if p.kind == "inflow"]
    i_em = [i for i, p in enumerate(inf_tab) if p.kind == "emission"]
    i_in = [i for i, p in enumerate(inf_tab) if p.kind == "inflow"]
    if len(t_em) < len(i_em) or len(t_in) < len(i_in):
        raise ValueError("truth must have at least as many parameters of each kind (emission, inflow) as the "
                         "inference model")
    truth_index = np.array(t_em[:len(i_em)] + t_in[:len(i_in)], dtype=int)  # inference param -> truth param
    if any(truth_tab[ti].name != m.name for ti, m in zip(truth_index, inf_tab)):
        raise ValueError("the first truth parameters of each kind must be the inference parameters (same name and "
                         "kind), in order")
    if any(source_signature(t) != source_signature(m)
           for t, m in zip(truth_scenario.sources[:n_src], inference_scenario.sources)):
        raise ValueError("the first truth sources must be the inference sources (same shape and group), in order")
    if n_noise < 1:
        raise ValueError("n_noise must be >= 1")
    if truth_components is not None and set(truth_components) != {"local", "external"}:
        raise ValueError("truth_components must have exactly the keys 'local' and 'external'")
    obs_times = _times(options, inference_scenario)
    resp = cache.get(inference_scenario, options.solver, forward=forward)
    truth = (truth_field if truth_field is not None
             else run_forward(forward, truth_scenario, q_true, options.solver).concentration)
    ref = truth if error_field is None else error_field
    _check_coords(ref, resp, "truth error field",
                  "; pass error_field on the inference grid and output times" if error_field is None else "")
    if truth_components is not None:
        for k, v in truth_components.items():
            _check_coords(v, resp, f"truth component {k!r}")
    # operator at all candidate sites (and eval sites), shapes (nt, ns, ...) -> (ns, nt, ...)
    nt = len(obs_times)

    def operator(c: Candidates) -> tuple[np.ndarray, np.ndarray]:
        b, A = resp.operator(c.sensor_layout(), obs_times)
        return (b.reshape(nt, len(c)).T.copy(), A.reshape(nt, len(c), n_params).transpose(1, 0, 2).copy())

    b_site, A_site = operator(candidates)
    truth_readings = np.asarray(sample(truth, candidates.sensor_layout(), obs_times).values, dtype=float).T.copy()
    b_eval = A_eval = eval_truth = None
    eval_groups: tuple[tuple[str, int, int], ...] | None = None
    ev = _eval_all(options)
    if ev is not None:
        b_eval, A_eval = operator(ev)
        eval_truth = np.asarray(sample(truth, ev.sensor_layout(), obs_times).values, dtype=float).T.copy()
        groups = _eval_groups(options)
        if groups is not None:
            stops = np.cumsum([len(c) for _, c in groups])
            eval_groups = tuple((n, int(stop - len(c)), int(stop)) for (n, c), stop in zip(groups, stops))
    # quadratic form of the weighted regional error (weights shared with metrics.field_mse)
    sel, dt, w, den = quadrature_weights(resp.times, resp.y.size, resp.x.size, options.weight, options.t_exclude)
    W = (dt[:, None, None] * w[None, :, :] / den).reshape(-1)
    d0 = (resp.base - np.asarray(ref.transpose("time", "y", "x").values, dtype=float))[sel].reshape(-1)
    g = resp.g[:, sel].reshape(n_params, -1)
    gW = g * W
    c0 = float(np.sum(W * d0 * d0))
    h = gW @ d0
    G = gW @ g.T
    objectives = {"total": QuadForm(c0, h, G)}
    component_ms: dict[str, float] | None = None
    fraction: _Fraction | None = None
    frac_true: float | None = None
    if truth_components is not None:
        arr = {k: np.asarray(v.transpose("time", "y", "x").values, dtype=float) for k, v in truth_components.items()}
        local_idx = [j for j in range(n_params) if resp.param_kinds[j] == "emission" and resp.param_groups[j] == "local"]
        ext_idx = [j for j in range(n_params) if j not in local_idx]
        background = base_parts(inference_scenario, options.solver, forward, cache)["background"]
        for key, idx, shift in (("local", local_idx, np.zeros_like(resp.base)), ("external", ext_idx, background)):
            d = (shift - arr[key])[sel].reshape(-1)
            hk, Gk = np.zeros(n_params), np.zeros((n_params, n_params))
            hk[idx] = gW[idx] @ d
            Gk[np.ix_(idx, idx)] = G[np.ix_(idx, idx)]
            objectives[key] = QuadForm(float(np.sum(W * d * d)), hk, Gk)
        component_ms = {k: float(np.sum(W * arr[k][sel].reshape(-1) ** 2)) for k in ("local", "external")}
        # external fraction: truth by attribution.external_fraction (total = the truth field), estimate as a linear form
        zero = np.zeros_like(resp.base)
        wmap = np.ones((resp.y.size, resp.x.size)) if options.weight is None else np.asarray(options.weight, float)
        contrib = {"local": resp.dataarray(arr["local"], "local"),
                   "external_sources": resp.dataarray(arr["external"], "external_sources"),
                   "boundary": resp.dataarray(zero, "boundary"), "background": resp.dataarray(zero, "background"),
                   "initial": resp.dataarray(np.asarray(ref.transpose("time", "y", "x").values, dtype=float)
                                             - arr["local"] - arr["external"], "initial")}
        frac_true = external_fraction(contrib, inference_scenario, options.t_exclude, FRACTION_EPS_REL, wmap)
        tot0 = np.einsum("tyx,yx->t", resp.base[sel], wmap)
        totg = np.einsum("jtyx,yx->jt", resp.g[:, sel], wmap)
        ext_mask = np.zeros(n_params)
        ext_mask[ext_idx] = 1.0
        fraction = _Fraction(tot0, totg, np.einsum("tyx,yx->t", background[sel], wmap),
                             totg * ext_mask[:, None], dt, float(wmap.sum()), den)
    rng = np.random.default_rng(noise_seed)
    noise = rng.standard_normal((n_noise, len(candidates), nt))
    n_truth_em = np.array([p.kind == "emission" for p in truth_tab])
    extra_em = n_truth_em.copy()
    extra_em[truth_index] = False
    return DesignCase(
        name=name, role=role, q_true=q_true, noise_seed=int(noise_seed),
        truth_fingerprint=_fingerprint(to_dict(truth_scenario), q_true.tolist()),
        inference_fingerprint=_fingerprint(to_dict(inference_scenario)), build_key=_build_key(options, obs_times),
        candidate_ids=candidates.ids, obs_times=obs_times, n_src=n_src, truth_readings=truth_readings,
        b_site=b_site, A_site=A_site, noise=noise, eval_truth=eval_truth, b_eval=b_eval, A_eval=A_eval,
        c0=c0, h=h, G=G, n_params=n_params, param_kinds=tuple(p.kind for p in inf_tab),
        param_names=tuple(p.name for p in inf_tab), objectives=objectives,
        unmodelled_mass=float(q_true[extra_em].sum() / q_true[n_truth_em].sum()),
        category=candidates.categories, eval_groups=eval_groups, component_ms=component_ms,
        fraction=fraction, external_fraction_true=frac_true, truth_index=truth_index)


# ------------------------------------------------------------------ fitting and scoring
def _check_layout(case: DesignCase, layout: Layout) -> None:
    n = len(case.candidate_ids)
    if len(layout) == 0 or len(set(layout)) != len(layout) or any(not 0 <= i < n for i in layout):
        raise ValueError("layout must be a non-empty tuple of unique valid candidate indices")


def attribution_scales(design_cases: list[DesignCase]) -> tuple[float, float]:
    """(s_local, s_external): mean over the DESIGN cases of the mean square (same weights/window as the objectives)
    of the TRUE local and external field, in ug^2/m^6.  They make 'attribution' = 0.5 (MSE_local / s_local +
    MSE_external / s_external) dimensionless, so the two components are weighed by how large the true field of each
    kind is and no emission (ug/s) or inflow (ug/m^3) amplitude is ever compared by raw magnitude.  Computed from
    design cases only (test and validation truths never enter); every case needs ``truth_components``."""
    _require_role(design_cases, "design", "attribution_scales")
    if any(c.component_ms is None for c in design_cases):
        raise ValueError("attribution_scales needs cases built with truth_components")
    ms = [cast(dict[str, float], c.component_ms) for c in design_cases]
    s_loc, s_ext = float(np.mean([m["local"] for m in ms])), float(np.mean([m["external"] for m in ms]))
    if not (s_loc > 0 and s_ext > 0):
        raise ValueError("a true component field is identically zero over the design cases: scale undefined")
    return s_loc, s_ext


def objective_value(case: DesignCase, theta: np.ndarray, objective: str, options: DesignOptions) -> float:
    """Value of a named objective at the parameter vector theta.  'total'/'local'/'external': weighted MSE
    (ug^2/m^6); 'attribution': 0.5 (E_local / s_local + E_external / s_external), needs options.attribution_scales."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if objective == "total":
        return case.field_error(theta)
    if not case.has_components:
        raise ValueError(f"objective {objective!r} needs a case built with truth_components (case {case.name})")
    th = np.asarray(theta, dtype=float)
    if objective != "attribution":
        return case.objectives[objective](th)
    if options.attribution_scales is None:
        raise ValueError("objective 'attribution' needs DesignOptions.attribution_scales (see attribution_scales())")
    s_loc, s_ext = options.attribution_scales
    return 0.5 * (case.objectives["local"](th) / s_loc + case.objectives["external"](th) / s_ext)


def _solve(case: DesignCase, layout: Layout, rep: int, options: DesignOptions) -> Any:
    _check_layout(case, layout)
    if not 0 <= rep < case.n_noise:
        raise ValueError(f"rep {rep} outside 0..{case.n_noise - 1}")
    if case.build_key != _build_key(options, case.obs_times):
        raise ValueError("DesignOptions (obs_times/t_exclude/weight/eval_sites) differ from those used in build_case")
    idx = list(layout)
    nt = len(case.obs_times)
    y = (case.truth_readings[idx] + options.noise_std * case.noise[rep, idx]).T.reshape(-1)  # time-major, sensor
    b = case.b_site[idx].T.reshape(-1)
    A = case.A_site[idx].transpose(1, 0, 2).reshape(nt * len(idx), case.n_params)
    return solve_linear(b, A, y, np.full(y.shape, options.noise_std), options.inference(), case.param_kinds)


def fit_layout(case: DesignCase, layout: Layout, rep: int, options: DesignOptions) -> dict[str, Any]:
    """Fit theta for one layout and noise repetition.  The fit sees only y = truth_readings[layout] + sigma*noise[rep,
    layout].  Returns q_hat (all parameters), mse_w, rmse, source_rel_err (relative L2 of the EMISSION parameters vs
    the matched truth emissions), inflow_abs_err ({name: error} in ug/m^3; empty without inflow bases) and
    inflow_rmse (RMS of those, nan without), unmodelled_source_mass (share of truth emission in unmodelled sources;
    0 if none), heldout_rmse (noiseless readings at eval_sites, estimate vs truth; nan without eval sites; with
    eval-site groups it pools all groups) and heldout_by_group ({name: rmse}), rank, effective_rank, condition,
    singular_values, q_std (linear-Gaussian posterior std), success, runtime_s.  With truth_components additionally
    mse_local, mse_external, mse_attribution (nan without options.attribution_scales) and external_fraction_true /
    external_fraction_est / external_fraction_error (None where undefined)."""
    t0 = _time.perf_counter()
    sol = _solve(case, layout, rep, options)
    mse = case.field_error(sol.q_hat)
    n = case.n_src
    held = float("nan")
    by_group: dict[str, float] = {}
    if case.A_eval is not None and case.b_eval is not None and case.eval_truth is not None:
        err = case.b_eval + case.A_eval @ sol.q_hat - case.eval_truth
        held = float(np.sqrt(np.mean(err ** 2)))
        for name, a, b in case.eval_groups or ():
            by_group[name] = float(np.sqrt(np.mean(err[a:b] ** 2)))
    d_inf = sol.q_hat[n:] - case.q_true[case.truth_index[n:]]
    out: dict[str, Any] = {
        "q_hat": sol.q_hat, "mse_w": mse, "rmse": float(np.sqrt(mse)),
        "source_rel_err": float(source_error(sol.q_hat[:n], case.q_true[:n])["relative_l2"]),
        "inflow_abs_err": {nm: float(d) for nm, d in zip(case.param_names[n:], d_inf)},
        "inflow_rmse": float(np.sqrt(np.mean(d_inf ** 2))) if d_inf.size else float("nan"),
        "unmodelled_source_mass": case.unmodelled_mass,
        "heldout_rmse": held, "heldout_by_group": by_group, "q_std": sol.q_std, "rank": sol.rank,
        "effective_rank": sol.effective_rank, "condition": sol.condition,
        "singular_values": sol.singular_values, "success": sol.success}
    if case.has_components and case.fraction is not None:
        out["mse_local"] = case.objectives["local"](sol.q_hat)
        out["mse_external"] = case.objectives["external"](sol.q_hat)
        out["mse_attribution"] = (objective_value(case, sol.q_hat, "attribution", options)
                                  if options.attribution_scales is not None else float("nan"))
        fe, ft = case.fraction(sol.q_hat), case.external_fraction_true
        out.update(external_fraction_true=ft, external_fraction_est=fe,
                   external_fraction_error=None if fe is None or ft is None else fe - ft)
    out["runtime_s"] = _time.perf_counter() - t0
    return out


def _require_role(cases: list[DesignCase], role: str, what: str) -> None:
    bad = [c.name for c in cases if c.role != role]
    if bad:
        raise ValueError(f"{what} accepts only role={role!r} cases (leakage guard); got {bad}")
    if not cases:
        raise ValueError("need at least one case")


def score_layout(layout: Layout, design_cases: list[DesignCase], options: DesignOptions,
                 objective: str = "total") -> dict[str, Any]:
    """Mean over cases and noise reps of the objective ('value'; per-case means in 'per_case', objective name in
    'objective').  For objective 'total' (default) 'mse_w' (= value) and 'rmse' = sqrt(mean MSE) are also returned.
    Refuses cases whose role is not 'design' (also for the objectives built on truth components)."""
    _require_role(design_cases, "design", "score_layout")
    return _score(layout, design_cases, options, objective)


def _score(layout: Layout, cases: list[DesignCase], options: DesignOptions, objective: str = "total"
           ) -> dict[str, Any]:
    per_case = {c.name: float(np.mean([objective_value(c, _solve(c, layout, r, options).q_hat, objective, options)
                                       for r in range(c.n_noise)])) for c in cases}
    mean = float(np.mean(list(per_case.values())))
    out: dict[str, Any] = {"objective": objective, "value": mean, "per_case": per_case}
    if objective == "total":
        out.update(mse_w=mean, rmse=float(np.sqrt(mean)))
    return out


@dataclass(frozen=True)
class GreedyTrace:
    layout: Layout  # sorted
    order: tuple[int, ...]  # sites in the order they were added
    scores: tuple[float, ...]  # mean design objective after each addition
    objective: str = "total"


def greedy_trace(candidates: Candidates, budget: int, design_cases: list[DesignCase],
                 constraints: Constraints, options: DesignOptions, objective: str = "total") -> GreedyTrace:
    """"Greedy optimized" selection: at each step add the feasible site giving the lowest mean design objective (over
    design cases and noise reps) of the enlarged layout.  A heuristic, not the optimum.  Ties (relative 1e-12) go to
    the lowest index.  Only role='design' cases are accepted.  InfeasibleError if the budget cannot be reached."""
    _require_role(design_cases, "design", "greedy selection")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    check_feasible(candidates, budget, constraints)
    chosen: list[int] = []
    scores: list[float] = []
    for _ in range(budget):
        feas = feasible_sites(candidates, constraints, tuple(chosen))
        if feas.size == 0:
            raise InfeasibleError("greedy construction got stuck under min_separation")
        sc = np.array([_score(tuple(sorted(chosen + [int(s)])), design_cases, options, objective)["value"]
                       for s in feas])
        best = int(np.flatnonzero(sc <= sc.min() * (1 + 1e-12))[0])  # feas ascending -> lowest index
        chosen.append(int(feas[best]))
        scores.append(float(sc[best]))
    return GreedyTrace(tuple(sorted(chosen)), tuple(chosen), tuple(scores), objective)


def select_layout(candidates: Candidates, budget: int, design_cases: list[DesignCase],
                  constraints: Constraints, options: DesignOptions, method: str = "greedy",
                  objective: str = "total") -> Layout:
    """The 'greedy optimized' layout (see ``greedy_trace`` for the score trace).  Only method='greedy' exists."""
    if method != "greedy":
        raise ValueError("method must be 'greedy'")
    return greedy_trace(candidates, budget, design_cases, constraints, options, objective).layout


def assert_disjoint(design_cases: list[DesignCase], test_cases: list[DesignCase],
                    validation_cases: list[DesignCase] | None = None) -> None:
    """Raise ValueError if any two of the design, test and (optional) validation case sets share a noise seed, a
    name or a truth fingerprint."""
    sets = {"design": design_cases, "test": test_cases, **({} if validation_cases is None
                                                           else {"validation": validation_cases})}
    names = list(sets)
    for what, get in (("noise seed", lambda c: c.noise_seed), ("name", lambda c: c.name),
                      ("truth fingerprint", lambda c: c.truth_fingerprint)):
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared = {get(c) for c in sets[a]} & {get(c) for c in sets[b]}
                if shared:
                    raise ValueError(f"{a} and {b} cases share {what}: {sorted(shared, key=str)}")


# ------------------------------------------------------------------ evaluation
COLUMNS = ["method", "layout_id", "budget", "case", "rep", "sensors", "mse_w", "rmse", "source_rel_err",
           "heldout_rmse", "unmodelled_source_mass", "rank", "effective_rank", "condition", "runtime_s"]
COMPONENT_COLUMNS = ["mse_local", "mse_external", "mse_attribution", "external_fraction_true",
                     "external_fraction_est", "external_fraction_error"]


def evaluate_layouts(layouts: dict[str, list[Layout]], test_cases: list[DesignCase], options: DesignOptions,
                     n_noise: int | None = None, allow_design: bool = False) -> pd.DataFrame:
    """Long-form table: one row per (method, layout, case, rep).  layout_id = '<method>-<k>'.  Test cases bring their
    own fresh noise (built with a seed disjoint from the design cases').  n_noise limits the reps used.
    Accepts only role='test' cases (role='design' too with allow_design=True; never 'validation')."""
    if not allow_design:
        _require_role(test_cases, "test", "evaluate_layouts")
    elif any(c.role == "validation" for c in test_cases):
        raise ValueError("evaluate_layouts never accepts role='validation' cases; use evaluate_validation")
    return _evaluate_rows(layouts, test_cases, options, n_noise)


def evaluate_validation(layouts: dict[str, list[Layout]], validation_cases: list[DesignCase],
                        options: DesignOptions, n_noise: int | None = None) -> pd.DataFrame:
    """Same table as ``evaluate_layouts`` for role='validation' cases only (used to choose alpha, never for layouts
    or reported test results)."""
    _require_role(validation_cases, "validation", "evaluate_validation")
    return _evaluate_rows(layouts, validation_cases, options, n_noise)


def _evaluate_rows(layouts: dict[str, list[Layout]], cases: list[DesignCase], options: DesignOptions,
                   n_noise: int | None) -> pd.DataFrame:
    """Columns = COLUMNS plus, only where relevant (so legacy tables are unchanged): heldout_<group>_rmse per eval
    group; COMPONENT_COLUMNS if any case has truth components; inflow_rmse if any model has inflow bases;
    n_reference_sites and category_counts if any candidate category differs from 'interior'."""
    group_names = list(dict.fromkeys(n for c in cases for n, _, _ in (c.eval_groups or ())))
    columns = list(COLUMNS) + [f"heldout_{n}_rmse" for n in group_names]
    with_comp = any(c.has_components for c in cases)
    with_inflow = any(c.n_params > c.n_src for c in cases)
    with_cat = any(set(c.category) - {DEFAULT_CATEGORY} for c in cases)
    columns += list(COMPONENT_COLUMNS) if with_comp else []
    columns += ["inflow_rmse"] if with_inflow else []
    columns += ["n_reference_sites", "category_counts"] if with_cat else []
    nan = float("nan")
    rows: list[dict[str, Any]] = []
    for method, lays in layouts.items():
        for k, lay in enumerate(lays):
            for c in cases:
                reps = c.n_noise if n_noise is None else n_noise
                if reps > c.n_noise:
                    raise ValueError(f"case {c.name} has only {c.n_noise} noise reps")
                sensors = ",".join(c.candidate_ids[i] for i in lay)
                cats = Counter(c.category[i] for i in lay) if c.category else Counter()
                for r in range(reps):
                    f = fit_layout(c, lay, r, options)
                    row = {"method": method, "layout_id": f"{method}-{k}", "budget": len(lay), "case": c.name,
                           "rep": r, "sensors": sensors, **{m: f[m] for m in COLUMNS[6:]}}
                    for n in group_names:
                        row[f"heldout_{n}_rmse"] = f["heldout_by_group"].get(n, nan)
                    for m in COMPONENT_COLUMNS if with_comp else ():
                        v = f.get(m)
                        row[m] = nan if v is None else v
                    if with_inflow:
                        row["inflow_rmse"] = f["inflow_rmse"]
                    if with_cat:
                        row["n_reference_sites"] = cats.get("reference", 0)
                        row["category_counts"] = ";".join(f"{k2}={cats[k2]}" for k2 in sorted(cats))
                    rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def paired_comparison(df: pd.DataFrame, a: str, b: str, metric: str = "mse_w") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Differences metric(a) - metric(b) (negative: a better, for error metrics) paired on (budget, case, rep).
    If a method has several layouts per budget (e.g. random), each layout is paired with the other side's rows, so
    there is one difference per (budget, case, rep, layout).  Returns (diffs, summary); summary per budget has
    mean_diff, sd_diff (sample sd, ddof=1; nan for n=1), n, frac_a_better.  Only these descriptive statistics."""
    keys = ["budget", "case", "rep"]
    df = cast(Any, df)  # pandas stubs are too strict for this indexing
    left: Any = df[df["method"] == a][keys + ["layout_id", metric]]
    right: Any = df[df["method"] == b][keys + ["layout_id", metric]]
    if left.empty or right.empty:
        raise ValueError(f"no rows for method {a!r} or {b!r}")
    m = left.merge(right, on=keys, suffixes=("_a", "_b"))
    m["diff"] = m[f"{metric}_a"] - m[f"{metric}_b"]
    g = m.groupby("budget")["diff"]
    summary = pd.DataFrame({"mean_diff": g.mean(), "sd_diff": g.std(ddof=1), "n": g.size(),
                            "frac_a_better": g.apply(lambda d: float(np.mean(d < 0)))}).reset_index()
    return m, summary


def random_distribution(df: pd.DataFrame, method: str, metric: str = "mse_w") -> pd.DataFrame:
    """Per budget: distribution over the layouts of `method` of the layout score (mean over cases and reps):
    n_layouts, mean, sd, min, q25, median, q75, max."""
    d = cast(Any, df)[df["method"] == method]
    if d.empty:
        raise ValueError(f"no rows for method {method!r}")
    per_layout = d.groupby(["budget", "layout_id"])[metric].mean().reset_index()
    g = per_layout.groupby("budget")[metric]
    return pd.DataFrame({"n_layouts": g.size(), "mean": g.mean(), "sd": g.std(ddof=1), "min": g.min(),
                         "q25": g.quantile(0.25), "median": g.median(), "q75": g.quantile(0.75),
                         "max": g.max()}).reset_index()

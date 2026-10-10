"""Sensor-placement machinery: candidates, constraints, layouts, cached design cases, fast fits, greedy selection.

Speed.  The forward model is affine in q, so for every case the unit responses are computed ONCE (``ResponseCache``)
and only arrays are kept in a ``DesignCase``: truth readings and the operator (b, A) at every candidate site, a
standard-normal common-random-number noise array, and the quadratic form of the weighted regional field error.  A
fit of any layout is then a tiny bounded least-squares problem (``inverse.solve_linear``, the very routine used by
``fit_sources``) and its field error MSE_w costs O(n_src^2): with d0 = base - truth,
``E(q) = c0 + 2 q.h + q'Gq`` (exact, same weights as ``metrics.field_mse``).

Leakage.  The fit sees only the noisy readings at the layout's sites (y = truth_readings + sigma * noise).  Selection
(``select_layout`` / ``greedy_trace``) accepts only cases with ``role == 'design'``; ``evaluate_layouts`` accepts
only ``role == 'test'`` cases (override with allow_design=True for diagnostics).  ``assert_disjoint`` checks that
design and test cases share no noise seed, name or truth fingerprint.

Baked at build time: obs_times, t_exclude, weight and eval_sites of the ``DesignOptions`` passed to ``build_case``
(fit-time options must carry the same values; this is checked).  noise_std, alpha, prior, q_scale are applied at fit
time.  Row order of the operator is time-major then sensor, as in ``inverse``.

Feasibility.  Choosing `budget` sites with pairwise distance >= min_separation is a maximum-independent-set problem
(NP-hard).  ``InfeasibleError`` is raised exactly when budget > number of allowed sites; otherwise detection is
heuristic: a deterministic index-order greedy construction plus ``N_FEASIBILITY_ORDERS`` seeded random orders.
"""
from __future__ import annotations

import hashlib
import json
import time as _time
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr

from .config import Domain, Scenario, SolverOptions, to_dict
from .inverse import InferenceOptions, ResponseCache, run_forward, solve_linear
from .forward import ForwardSolver, simulate
from .metrics import quadrature_weights, source_error
from .observations import SensorLayout, candidate_hull, sample

Layout = tuple[int, ...]  # sorted, unique candidate indices
N_FEASIBILITY_ORDERS = 200
MAX_RANDOM_ATTEMPTS = 200


class InfeasibleError(ValueError):
    """No layout satisfies the budget and constraints (exact if budget > #allowed, heuristic otherwise)."""


# ------------------------------------------------------------------ candidates and constraints
@dataclass(frozen=True, eq=False)
class Candidates:
    ids: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        x, y = np.asarray(self.x, dtype=float), np.asarray(self.y, dtype=float)
        if not (len(self.ids) == x.size == y.size) or x.ndim != 1 or y.ndim != 1:
            raise ValueError("ids, x, y must be 1-D with equal lengths")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("candidate ids must be unique")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("candidate coordinates must be finite")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def __len__(self) -> int:
        return len(self.ids)

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
    """allowed: bool mask over candidates (None = all); min_separation: minimum pairwise distance [m] (>= comparison)."""
    allowed: np.ndarray | None = None
    min_separation: float = 0.0

    def __post_init__(self) -> None:
        if not self.min_separation >= 0:
            raise ValueError("min_separation must be >= 0")


def _allowed(candidates: Candidates, constraints: Constraints) -> np.ndarray:
    if constraints.allowed is None:
        return np.ones(len(candidates), dtype=bool)
    a = np.asarray(constraints.allowed, dtype=bool)
    if a.shape != (len(candidates),):
        raise ValueError(f"allowed must have shape ({len(candidates)},), got {a.shape}")
    return a


def feasible_sites(candidates: Candidates, constraints: Constraints, chosen: Layout | tuple[int, ...] = ()
                   ) -> np.ndarray:
    """Indices (ascending) of allowed, not-yet-chosen sites at distance >= min_separation from every chosen site."""
    ok = _allowed(candidates, constraints).copy()
    for c in chosen:
        ok[c] = False
        if constraints.min_separation > 0:
            d = np.hypot(candidates.x - candidates.x[c], candidates.y - candidates.y[c])
            ok &= d >= constraints.min_separation * (1 - 1e-12)
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
    n_allowed = int(_allowed(candidates, constraints).sum())
    if budget > n_allowed:
        raise InfeasibleError(f"budget {budget} exceeds the {n_allowed} allowed sites")
    if constraints.min_separation == 0:
        return
    rng = np.random.default_rng(0)
    orders = [np.arange(len(candidates))] + [rng.permutation(len(candidates)) for _ in range(N_FEASIBILITY_ORDERS)]
    if all(_construct(candidates, budget, constraints, o) is None for o in orders):
        raise InfeasibleError(f"no layout of {budget} sites with min_separation {constraints.min_separation} found "
                              f"({len(orders)} construction orders tried; heuristic)")


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
    """noise_std [ug/m^3] (sigma); alpha/prior/q_scale/solver as in InferenceOptions; obs_times default
    600, 1200, ..., T (t=0 excluded); weight (ny, nx) for MSE_w; eval_sites for held-out RMSE."""
    noise_std: float
    obs_times: tuple[float, ...] | None = None
    alpha: float = 0.0
    prior: tuple[float, ...] | None = None
    q_scale: tuple[float, ...] | None = None
    t_exclude: float = 0.0
    weight: np.ndarray | None = None
    eval_sites: Candidates | None = None
    solver: SolverOptions = field(default_factory=SolverOptions)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.noise_std) and self.noise_std >= 0):
            raise ValueError("noise_std must be finite and >= 0")

    def inference(self) -> InferenceOptions:
        return InferenceOptions("linear", self.q_scale, self.alpha, self.prior, (), self.solver)


def _fingerprint(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=list).encode()).hexdigest()[:16]


def _times(options: DesignOptions, scenario: Scenario) -> tuple[float, ...]:
    if options.obs_times is not None:
        return tuple(float(t) for t in options.obs_times)
    T = scenario.times.duration
    return tuple(float(t) for t in np.arange(600.0, T + 1e-9, 600.0))


def _build_key(options: DesignOptions, obs_times: tuple[float, ...]) -> str:
    w = None if options.weight is None else hashlib.sha256(np.ascontiguousarray(options.weight).tobytes()).hexdigest()
    e = None if options.eval_sites is None else (options.eval_sites.ids, options.eval_sites.x.tolist(),
                                                 options.eval_sites.y.tolist())
    return _fingerprint(obs_times, options.t_exclude, w, e)


@dataclass(frozen=True, eq=False)
class DesignCase:
    """One truth scenario + inference model, reduced to arrays (see module docstring).  No truth fields."""
    name: str
    role: str  # 'design' | 'test'
    q_true: np.ndarray  # follows the TRUTH scenario's sources
    noise_seed: int
    truth_fingerprint: str  # truth scenario + q_true
    inference_fingerprint: str
    build_key: str
    candidate_ids: tuple[str, ...]
    obs_times: tuple[float, ...]
    n_src: int  # sources of the inference model
    truth_readings: np.ndarray  # (n_site, nt) noiseless
    b_site: np.ndarray  # (n_site, nt)
    A_site: np.ndarray  # (n_site, nt, n_src)
    noise: np.ndarray  # (n_noise, n_site, nt) standard normal
    eval_truth: np.ndarray | None  # (n_eval, nt)
    b_eval: np.ndarray | None
    A_eval: np.ndarray | None
    c0: float
    h: np.ndarray  # (n_src,)
    G: np.ndarray  # (n_src, n_src)

    @property
    def n_noise(self) -> int:
        return self.noise.shape[0]

    def field_error(self, q: np.ndarray) -> float:
        """Exact MSE_w of the field base + sum q_j g_j against the case truth (O(n_src^2))."""
        q = np.asarray(q, dtype=float)
        return max(float(self.c0 + 2.0 * q @ self.h + q @ self.G @ q), 0.0)


def build_case(name: str, role: str, truth_scenario: Scenario, q_true: np.ndarray, inference_scenario: Scenario,
               candidates: Candidates, options: DesignOptions, noise_seed: int, n_noise: int,
               cache: ResponseCache, error_field: xr.DataArray | None = None,
               truth_field: xr.DataArray | None = None, forward: ForwardSolver = simulate) -> DesignCase:
    """Run one truth simulation and precompute everything fits need.

    truth_scenario may differ from inference_scenario (wind, resolution, extra sources); q_true has one entry per
    TRUTH source and must be at least as long as the inference source list (the first n_src are matched with the
    inference sources; extra truth sources are 'unmodelled').  Sensors read the NATIVE truth field.  The regional
    error compares the inference model with `error_field` (default: the native truth), which must be on the
    inference model's (time, y, x) coordinates exactly; for a refined truth the caller passes a block-averaged field.
    Different output times raise ValueError (no time interpolation).  ``truth_field`` may pass an already computed
    ``simulate(truth_scenario, q_true, options.solver).concentration`` (saves the solve; the caller guarantees it).  ``forward`` replaces ``simulate`` for both the inference responses
    and the truth run (see ``forward.ForwardSolver``)."""
    if role not in ("design", "test"):
        raise ValueError("role must be 'design' or 'test'")
    q_true = np.asarray(q_true, dtype=float)
    n_src = len(inference_scenario.sources)
    if q_true.shape != (len(truth_scenario.sources),):
        raise ValueError(f"q_true must have one entry per truth source ({len(truth_scenario.sources)})")
    if len(truth_scenario.sources) < n_src:
        raise ValueError("truth must have at least as many sources as the inference model")
    if n_noise < 1:
        raise ValueError("n_noise must be >= 1")
    obs_times = _times(options, inference_scenario)
    resp = cache.get(inference_scenario, options.solver, forward=forward)
    truth = (truth_field if truth_field is not None
             else run_forward(forward, truth_scenario, q_true, options.solver).concentration)
    ref = truth if error_field is None else error_field
    for dim, coord in (("time", resp.times), ("y", resp.y), ("x", resp.x)):
        v = np.asarray(ref[dim].values, dtype=float)
        if v.shape != coord.shape or not np.allclose(v, coord, rtol=0, atol=1e-9 * max(1.0, np.abs(coord).max())):
            raise ValueError(f"truth error field and inference responses differ in coordinate {dim!r}"
                             + ("; pass error_field on the inference grid and output times" if error_field is None
                                else ""))
    # operator at all candidate sites (and eval sites), shapes (nt, ns, ...) -> (ns, nt, ...)
    nt = len(obs_times)

    def operator(c: Candidates) -> tuple[np.ndarray, np.ndarray]:
        b, A = resp.operator(c.sensor_layout(), obs_times)
        return (b.reshape(nt, len(c)).T.copy(), A.reshape(nt, len(c), n_src).transpose(1, 0, 2).copy())

    b_site, A_site = operator(candidates)
    truth_readings = np.asarray(sample(truth, candidates.sensor_layout(), obs_times).values, dtype=float).T.copy()
    b_eval = A_eval = eval_truth = None
    if options.eval_sites is not None:
        b_eval, A_eval = operator(options.eval_sites)
        eval_truth = np.asarray(sample(truth, options.eval_sites.sensor_layout(), obs_times).values,
                                dtype=float).T.copy()
    # quadratic form of the weighted regional error (weights shared with metrics.field_mse)
    sel, dt, w, den = quadrature_weights(resp.times, resp.y.size, resp.x.size, options.weight, options.t_exclude)
    W = (dt[:, None, None] * w[None, :, :] / den).reshape(-1)
    d0 = (resp.base - np.asarray(ref.transpose("time", "y", "x").values, dtype=float))[sel].reshape(-1)
    g = resp.g[:, sel].reshape(n_src, -1)
    c0 = float(np.sum(W * d0 * d0))
    h = (g * W) @ d0
    G = (g * W) @ g.T
    rng = np.random.default_rng(noise_seed)
    noise = rng.standard_normal((n_noise, len(candidates), nt))
    return DesignCase(
        name=name, role=role, q_true=q_true, noise_seed=int(noise_seed),
        truth_fingerprint=_fingerprint(to_dict(truth_scenario), q_true.tolist()),
        inference_fingerprint=_fingerprint(to_dict(inference_scenario)), build_key=_build_key(options, obs_times),
        candidate_ids=candidates.ids, obs_times=obs_times, n_src=n_src, truth_readings=truth_readings,
        b_site=b_site, A_site=A_site, noise=noise, eval_truth=eval_truth, b_eval=b_eval, A_eval=A_eval,
        c0=c0, h=h, G=G)


# ------------------------------------------------------------------ fitting and scoring
def _check_layout(case: DesignCase, layout: Layout) -> None:
    n = len(case.candidate_ids)
    if len(layout) == 0 or len(set(layout)) != len(layout) or any(not 0 <= i < n for i in layout):
        raise ValueError("layout must be a non-empty tuple of unique valid candidate indices")


def fit_layout(case: DesignCase, layout: Layout, rep: int, options: DesignOptions) -> dict[str, Any]:
    """Fit q for one layout and noise repetition.  The fit sees only y = truth_readings[layout] + sigma*noise[rep,
    layout].  Returns q_hat, mse_w, rmse, source_rel_err (relative L2 vs the first n_src truth amplitudes),
    unmodelled_source_mass (share of truth emission in extra sources; 0 if none), heldout_rmse (noiseless
    readings at eval_sites, estimate vs truth; nan without eval sites), rank, effective_rank, condition,
    singular_values, q_std (linear-Gaussian posterior std), success, runtime_s."""
    t0 = _time.perf_counter()
    _check_layout(case, layout)
    if not 0 <= rep < case.n_noise:
        raise ValueError(f"rep {rep} outside 0..{case.n_noise - 1}")
    if case.build_key != _build_key(options, case.obs_times):
        raise ValueError("DesignOptions (obs_times/t_exclude/weight/eval_sites) differ from those used in build_case")
    idx = list(layout)
    nt = len(case.obs_times)
    y = (case.truth_readings[idx] + options.noise_std * case.noise[rep, idx]).T.reshape(-1)  # time-major, sensor
    b = case.b_site[idx].T.reshape(-1)
    A = case.A_site[idx].transpose(1, 0, 2).reshape(nt * len(idx), case.n_src)
    sol = solve_linear(b, A, y, np.full(y.shape, options.noise_std), options.inference())
    mse = case.field_error(sol.q_hat)
    q_match = case.q_true[:case.n_src]
    held = float("nan")
    if case.A_eval is not None and case.b_eval is not None and case.eval_truth is not None:
        est = case.b_eval + case.A_eval @ sol.q_hat
        held = float(np.sqrt(np.mean((est - case.eval_truth) ** 2)))
    return {
        "q_hat": sol.q_hat, "mse_w": mse, "rmse": float(np.sqrt(mse)),
        "source_rel_err": float(source_error(sol.q_hat, q_match)["relative_l2"]),
        "unmodelled_source_mass": float(case.q_true[case.n_src:].sum() / case.q_true.sum()),
        "heldout_rmse": held, "q_std": sol.q_std, "rank": sol.rank, "effective_rank": sol.effective_rank, "condition": sol.condition,
        "singular_values": sol.singular_values, "success": sol.success, "runtime_s": _time.perf_counter() - t0}


def _require_role(cases: list[DesignCase], role: str, what: str) -> None:
    bad = [c.name for c in cases if c.role != role]
    if bad:
        raise ValueError(f"{what} accepts only role={role!r} cases (leakage guard); got {bad}")
    if not cases:
        raise ValueError("need at least one case")


def score_layout(layout: Layout, design_cases: list[DesignCase], options: DesignOptions) -> dict[str, Any]:
    """Mean over cases and noise reps of MSE_w ('mse_w'), 'rmse' = sqrt(mean MSE), and per-case mean MSE_w.
    Refuses cases whose role is not 'design'."""
    _require_role(design_cases, "design", "score_layout")
    return _score(layout, design_cases, options)


def _score(layout: Layout, cases: list[DesignCase], options: DesignOptions) -> dict[str, Any]:
    per_case = {c.name: float(np.mean([fit_layout(c, layout, r, options)["mse_w"] for r in range(c.n_noise)]))
                for c in cases}
    mean = float(np.mean(list(per_case.values())))
    return {"mse_w": mean, "rmse": float(np.sqrt(mean)), "per_case": per_case}


@dataclass(frozen=True)
class GreedyTrace:
    layout: Layout  # sorted
    order: tuple[int, ...]  # sites in the order they were added
    scores: tuple[float, ...]  # mean design MSE_w after each addition


def greedy_trace(candidates: Candidates, budget: int, design_cases: list[DesignCase],
                 constraints: Constraints, options: DesignOptions) -> GreedyTrace:
    """"Greedy optimized" selection: at each step add the feasible site giving the lowest mean design MSE_w (over
    design cases and noise reps) of the enlarged layout.  Ties (relative 1e-12) go to the lowest index.  Only
    role='design' cases are accepted.  InfeasibleError if the budget cannot be reached."""
    _require_role(design_cases, "design", "greedy selection")
    check_feasible(candidates, budget, constraints)
    chosen: list[int] = []
    scores: list[float] = []
    for _ in range(budget):
        feas = feasible_sites(candidates, constraints, tuple(chosen))
        if feas.size == 0:
            raise InfeasibleError("greedy construction got stuck under min_separation")
        sc = np.array([_score(tuple(sorted(chosen + [int(s)])), design_cases, options)["mse_w"] for s in feas])
        best = int(np.flatnonzero(sc <= sc.min() * (1 + 1e-12))[0])  # feas ascending -> lowest index
        chosen.append(int(feas[best]))
        scores.append(float(sc[best]))
    return GreedyTrace(tuple(sorted(chosen)), tuple(chosen), tuple(scores))


def select_layout(candidates: Candidates, budget: int, design_cases: list[DesignCase],
                  constraints: Constraints, options: DesignOptions, method: str = "greedy") -> Layout:
    """The 'greedy optimized' layout (see ``greedy_trace`` for the score trace).  Only method='greedy' exists."""
    if method != "greedy":
        raise ValueError("method must be 'greedy'")
    return greedy_trace(candidates, budget, design_cases, constraints, options).layout


def assert_disjoint(design_cases: list[DesignCase], test_cases: list[DesignCase]) -> None:
    """Raise ValueError if design and test cases share a noise seed, a name or a truth fingerprint."""
    for what, get in (("noise seed", lambda c: c.noise_seed), ("name", lambda c: c.name),
                      ("truth fingerprint", lambda c: c.truth_fingerprint)):
        shared = {get(c) for c in design_cases} & {get(c) for c in test_cases}
        if shared:
            raise ValueError(f"design and test cases share {what}: {sorted(shared, key=str)}")


# ------------------------------------------------------------------ evaluation
COLUMNS = ["method", "layout_id", "budget", "case", "rep", "sensors", "mse_w", "rmse", "source_rel_err",
           "heldout_rmse", "unmodelled_source_mass", "rank", "effective_rank", "condition", "runtime_s"]


def evaluate_layouts(layouts: dict[str, list[Layout]], test_cases: list[DesignCase], options: DesignOptions,
                     n_noise: int | None = None, allow_design: bool = False) -> pd.DataFrame:
    """Long-form table: one row per (method, layout, case, rep).  layout_id = '<method>-<k>'.  Test cases bring their
    own fresh noise (built with a seed disjoint from the design cases').  n_noise limits the reps used."""
    if not allow_design:
        _require_role(test_cases, "test", "evaluate_layouts")
    rows: list[dict[str, Any]] = []
    for method, lays in layouts.items():
        for k, lay in enumerate(lays):
            for c in test_cases:
                reps = c.n_noise if n_noise is None else n_noise
                if reps > c.n_noise:
                    raise ValueError(f"case {c.name} has only {c.n_noise} noise reps")
                sensors = ",".join(c.candidate_ids[i] for i in lay)
                for r in range(reps):
                    f = fit_layout(c, lay, r, options)
                    rows.append({"method": method, "layout_id": f"{method}-{k}", "budget": len(lay), "case": c.name,
                                 "rep": r, "sensors": sensors, **{m: f[m] for m in COLUMNS[6:]}})
    return pd.DataFrame(rows, columns=COLUMNS)


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

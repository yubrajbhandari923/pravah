"""EXP-019 toy: how wind error limits source inference, and three inference strategies (SYNTHETIC; Phase-1 Domain).

Strategies (all share one code path, ``fit_ensemble``): A known wind (ensemble of one = the truth wind, oracle),
B nominal wind (ensemble of one), C uncertain wind (``strategy_c``): ensemble of M winds from a DECLARED prior grid,
per-member regularised NNLS (``inverse.solve_linear``), posterior member weight ~ exp(-chi2_m/2) (known sigma, uniform
prior over members), prediction = weight-averaged field/amplitudes.

Truth side vs inference side.  The wind of the TRUTH is perturbed (``rotate_wind``); the inference only ever sees an
``Ensemble`` (member wind labels + their sensor operators), the observation vector ``y`` and a layout.  ``strategy_c``
takes exactly those and nothing else (guarded by tests/test_windunc.py).

Everything is affine in the amplitudes, so the field error of any estimate is a quadratic form in the amplitudes
whose matrices are inner products of cached response fields (``Gram``): no field is built per fit.  The forward model
has zero initial state and zero background here, so the base field is identically zero (asserted in ``Study``).

Interval.  90% interval of the field at held-out sensors: Gaussian, mean = weighted member prediction, variance =
weighted member spread + weighted linear-Gaussian posterior (noise-covariance) variance of each member; the q >= 0
bound is ignored (as in ``inverse.FitResult.q_std``).  The spread-only variant is stored too.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ... import design as D
from ... import population
from ...attribution import emission_error, identifiability_report
from ...config import Scenario, WindSegment, scenario_from_dict
from ...geo.regionio import GridSpec
from ...inverse import SCALE_BY_KIND, FitResult, InferenceOptions, LinearSolution, ResponseCache, Responses, solve_linear
from ...metrics import quadrature_weights

Key = tuple[float, float]  # (rotation of the (u, v) vector in degrees, counter-clockwise; speed factor)
Z90 = float(stats.norm.ppf(0.95))
STRATEGIES = ("A", "B", "C")


# ------------------------------------------------------------------------------------------------ wind helpers
def key(deg: float, factor: float = 1.0) -> Key:
    return (round(float(deg), 6), round(float(factor), 6))


def rotate_wind(u: float, v: float, deg: float, factor: float = 1.0) -> tuple[float, float]:
    """(u, v) rotated counter-clockwise by ``deg`` at speed x ``factor``."""
    th = math.radians(deg)
    return (factor * (u * math.cos(th) - v * math.sin(th)), factor * (u * math.sin(th) + v * math.cos(th)))


def vector_error(deg: float, factor: float = 1.0) -> float:
    """|du|/|u| of a wind rotated by ``deg`` and scaled by ``factor``: sqrt(1 + f^2 - 2 f cos(deg)); for f = 1 it is
    2 sin(deg/2) (so 5/10/15/30 deg = 0.087/0.174/0.261/0.518)."""
    return math.sqrt(max(1.0 + factor**2 - 2.0 * factor * math.cos(math.radians(deg)), 0.0))


def wind_scenario(base: Scenario, k: Key) -> Scenario:
    seg = tuple(WindSegment(w.t_start, *rotate_wind(w.u, w.v, k[0], k[1])) for w in base.transport.wind)
    return dataclasses.replace(base, transport=dataclasses.replace(base.transport, wind=seg))


def member_weights(chi2: np.ndarray) -> np.ndarray:
    """exp(-chi2/2) normalised (uniform prior over members), computed stably."""
    z = -0.5 * (np.asarray(chi2, dtype=float) - np.min(chi2))
    w = np.exp(z)
    return w / w.sum()


# ------------------------------------------------------------------------------------------------ guards
@dataclass(frozen=True)
class Case:
    """One truth scenario draw (amplitudes) with its own standard-normal noise array (shared by every wind error and
    every strategy: the paired design)."""
    name: str
    role: str  # design | validation | test
    seed: int
    noise_seed: int
    theta: np.ndarray  # (n_params,) emissions ug/s then inflow ug/m^3
    noise: np.ndarray  # (n_candidates, nt)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(np.round(self.theta, 9).tobytes()).hexdigest()[:16]


def draw_cases(role: str, n: int, seed: int, noise_seed0: int, base_theta: Sequence[float] | np.ndarray,
               amp_range: tuple[float, float], n_cand: int, nt: int) -> list[Case]:
    out = []
    for k in range(n):
        rng = np.random.default_rng([seed, k])
        theta = np.asarray(base_theta, dtype=float) * rng.uniform(amp_range[0], amp_range[1], len(base_theta))
        out.append(Case(f"{role}-{k}", role, seed * 1000 + k, noise_seed0 + k, theta,
                        np.random.default_rng(noise_seed0 + k).standard_normal((n_cand, nt))))
    return out


def assert_disjoint_cases(*groups: Sequence[Case]) -> None:
    """Raise if any two case sets share a name, seed, noise seed or truth fingerprint."""
    for what in ("name", "seed", "noise_seed", "fingerprint"):
        sets = [{getattr(c, what) for c in g} for g in groups]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                if sets[i] & sets[j]:
                    raise ValueError(f"case sets {i} and {j} share {what}: {sorted(sets[i] & sets[j], key=str)}")


def config_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=list).encode()).hexdigest()[:16]


def write_freeze(path: Path, frozen: dict[str, Any]) -> str:
    h = config_hash(frozen)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"hash": h, "frozen": frozen}, indent=1, default=list))
    return h


def assert_frozen(path: Path, frozen: dict[str, Any]) -> str:
    """The frozen choices (grid, prior, alpha, layouts, seeds) re-derived from the state actually used must hash to the
    value written BEFORE the test."""
    rec = json.loads(path.read_text())
    h = config_hash(frozen)
    if h != rec["hash"]:
        raise AssertionError(f"frozen choices changed after the freeze: {rec['hash']} != {h}")
    return h


# ------------------------------------------------------------------------------------------------ bases and operators
class Bank:
    """Cached linear response bases per wind key (cache key of ``ResponseCache`` includes the wind)."""

    def __init__(self, nominal: Scenario, directory: Path | str | None) -> None:
        self.nominal = nominal
        self.cache = ResponseCache(directory)
        self.n = nominal.n_params
        self.solve_seconds = 0.0

    def resp(self, k: Key) -> Responses:
        m0 = self.cache.misses
        t0 = time.perf_counter()
        r = self.cache.get(wind_scenario(self.nominal, k))
        if self.cache.misses > m0:
            self.solve_seconds += time.perf_counter() - t0
        return r


def site_ops(resp: Responses, sites: Any, times: Any) -> tuple[np.ndarray, np.ndarray]:
    """(b (ns, nt), A (ns, nt, n)) at the sensor sites."""
    b, A = resp.operator(sites, times)
    nt, ns = len(times), len(sites.ids)
    return b.reshape(nt, ns).T.copy(), A.reshape(nt, ns, -1).transpose(1, 0, 2).copy()


@dataclass(frozen=True, eq=False)
class Ensemble:
    """Winds a strategy may consider: member labels and their operators at all candidate sites.  Nothing else."""
    keys: tuple[Key, ...]
    b: np.ndarray  # (M, ns, nt)
    A: np.ndarray  # (M, ns, nt, n)
    kinds: tuple[str, ...]
    names: tuple[str, ...]
    units: tuple[str, ...]


def make_ensemble(bank: Bank, keys: Sequence[Key], sites: Any, times: Sequence[float]) -> Ensemble:
    ops = [site_ops(bank.resp(k), sites, times) for k in keys]
    r = bank.resp(keys[0])
    tab = bank.nominal.param_table()
    return Ensemble(tuple(keys), np.stack([o[0] for o in ops]), np.stack([o[1] for o in ops]), r.param_kinds,
                    tuple(p.name for p in tab), tuple(p.units for p in tab))


def prior_keys(directions: Sequence[float], factors: Sequence[float]) -> list[Key]:
    return [key(d, f) for d in directions for f in factors]


# ------------------------------------------------------------------------------------------------ fitting
@dataclass(frozen=True)
class FitOptions:
    sigma: float
    alpha: float


@dataclass
class EnsembleFit:
    weights: np.ndarray  # (M,)
    thetas: np.ndarray  # (M, n)
    chi2: np.ndarray  # (M,) data chi2 of each member's fit (sensors used in the fit)
    sols: list[LinearSolution]
    cov: np.ndarray  # (M, n, n) linear-Gaussian posterior covariance of theta (parameter units), bound ignored
    n_obs: int
    runtime_s: float

    @property
    def theta(self) -> np.ndarray:
        return self.weights @ self.thetas


def fit_ensemble(y: np.ndarray, idx: Sequence[int], ens: Ensemble, opts: FitOptions) -> EnsembleFit:
    """Fit each member to the readings ``y`` (time-major, then sensor, at the sensors ``idx``); weights ~ exp(-chi2/2)."""
    t0 = time.perf_counter()
    ix = list(idx)
    M, n = len(ens.keys), ens.A.shape[-1]
    nt = ens.b.shape[2]
    sig = np.full(nt * len(ix), opts.sigma)
    o = InferenceOptions("linear", None, opts.alpha)
    sols, thetas, chi2, cov = [], [], [], []
    for m in range(M):
        b = ens.b[m][ix].T.reshape(-1)
        A = ens.A[m][ix].transpose(1, 0, 2).reshape(nt * len(ix), n)
        s = solve_linear(b, A, y, sig, o, ens.kinds)
        sc = s.q_scale
        H = A / opts.sigma * sc[None, :]
        P = np.linalg.pinv(H.T @ H + opts.alpha * np.eye(n))
        sols.append(s)
        thetas.append(s.q_hat)
        chi2.append(s.weighted_residual**2)
        cov.append(sc[:, None] * P * sc[None, :])
    c = np.array(chi2)
    return EnsembleFit(member_weights(c), np.array(thetas), c, sols, np.array(cov), len(sig), time.perf_counter() - t0)


def strategy_c(y: np.ndarray, idx: Sequence[int], ens: Ensemble, opts: FitOptions) -> EnsembleFit:
    """Strategy C.  Its ONLY inputs are the observations, the layout, the declared-prior ensemble and the options: the
    true wind is not an argument and cannot be reached from them (test_windunc.py)."""
    return fit_ensemble(y, idx, ens, opts)


def predict(fit: EnsembleFit, ens: Ensemble, idx: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, total variance (spread + posterior) and spread-only variance of the readings at sites ``idx`` (ns, nt)."""
    ix = list(idx)
    pred = np.stack([ens.b[m][ix] + ens.A[m][ix] @ fit.thetas[m] for m in range(len(ens.keys))])
    mu = np.tensordot(fit.weights, pred, axes=1)
    spread = np.tensordot(fit.weights, (pred - mu) ** 2, axes=1)
    post = np.stack([np.einsum("stj,jk,stk->st", ens.A[m][ix], fit.cov[m], ens.A[m][ix]) for m in range(len(ens.keys))])
    return mu, spread + np.tensordot(fit.weights, post, axes=1), spread


def as_fit_result(sol: LinearSolution, ens: Ensemble, n_obs: int) -> FitResult:
    return FitResult(sol.q_hat, sol.success, sol.message, sol.weighted_residual, n_obs, sol.singular_values, sol.rank,
                     sol.condition, sol.effective_rank, sol.q_std, sol.reg_singular_values, sol.reg_condition, "linear",
                     sol.q_scale, 0.0, sol.prior, (), 0.0, ens.names, ens.kinds, ens.units, sol.column_norms,
                     sol.column_cosines)


# ------------------------------------------------------------------------------------------------ Gram forms
class Gram:
    """Inner products <g_a, g_b>_w of response fields over (time, y, x) for a set of winds: field error of any estimate
    is a quadratic form (see module docstring).  Weights: 'unif' (area), 'pop' (synthetic population counts); both with
    the trapezoid time weights of ``metrics.quadrature_weights`` (so the form equals ``metrics.field_mse``)."""

    def __init__(self, bank: Bank, keys: Sequence[Key], weight_maps: dict[str, np.ndarray | None], t_exclude: float = 0.0,
                 truth_bank: Bank | None = None, truth_keys: Sequence[Key] = ()):
        self.n = bank.n
        self.index = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        r0 = bank.resp(keys[0])
        ny, nx = r0.y.size, r0.x.size
        ws, sel = {}, None
        for name, wm in weight_maps.items():
            sel, dt, w, den = quadrature_weights(r0.times, ny, nx, wm, t_exclude)
            ws[name] = (dt[:, None, None] * w[None] / den).reshape(-1)
        Z = np.concatenate([bank.resp(k).g[:, sel].reshape(self.n, -1) for k in self.index])
        self.tindex = self.index  # truth fields live in the same bank unless a model-error control gives the truth its own
        if truth_bank is not None:
            tk = list(dict.fromkeys(truth_keys))
            self.tindex = {k: len(self.index) + i for i, k in enumerate(tk)}
            Z = np.concatenate([Z, *[truth_bank.resp(k).g[:, sel].reshape(self.n, -1) for k in tk]])
        self.G = {name: (Z * w) @ Z.T for name, w in ws.items()}
        self.I = Z @ ws["unif"]  # time-area integral of each response: linear functional for the external fraction

    def ix(self, keys: Sequence[Key]) -> np.ndarray:
        return np.concatenate([np.arange(self.index[k] * self.n, (self.index[k] + 1) * self.n) for k in keys])

    def ixt(self, keys: Sequence[Key]) -> np.ndarray:
        return np.concatenate([np.arange(self.tindex[k] * self.n, (self.tindex[k] + 1) * self.n) for k in keys])


class Forms:
    """Quadratic-form pieces for (estimate winds, truth wind): mse = e'Gee e - 2 e'Get t + t'Gtt t per component."""

    def __init__(self, gram: Gram, est_keys: Sequence[Key], truth_key: Key, kinds: Sequence[str], groups: Sequence[str]):
        n = gram.n
        ie, it = gram.ix(est_keys), gram.ixt([truth_key])
        local = np.array([g == "local" for g in groups])
        self.comp = {"total": np.ones(n, bool), "local": local, "external": ~local}
        self.n, self.M = n, len(est_keys)
        self.pieces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for w in gram.G:
            G = gram.G[w]
            for c, mask in self.comp.items():
                se = np.flatnonzero(np.tile(mask, self.M))
                st = np.flatnonzero(mask)
                self.pieces[(w, c)] = (G[np.ix_(ie[se], ie[se])], G[np.ix_(ie[se], it[st])], G[np.ix_(it[st], it[st])], se, st)
        self.Ie, self.It = gram.I[ie], gram.I[it]

    def mse(self, w: str, c: str, th_e: np.ndarray, th_t: np.ndarray) -> float:
        """th_e: (M*n,) member amplitudes already multiplied by their weights."""
        Gee, Get, Gtt, se, st = self.pieces[(w, c)]
        e, t = th_e[se], th_t[st]
        return max(float(e @ Gee @ e - 2.0 * e @ Get @ t + t @ Gtt @ t), 0.0)

    def ext_fraction_est(self, th_e: np.ndarray) -> float:
        tot = float(th_e @ self.Ie)
        return float(th_e[np.tile(self.comp["external"], self.M)] @ self.Ie[np.tile(self.comp["external"], self.M)] / tot) if tot > 0 else math.nan

    def ext_fraction_true(self, th_t: np.ndarray) -> float:
        tot = float(th_t @ self.It)
        return float(th_t[self.comp["external"]] @ self.It[self.comp["external"]] / tot) if tot > 0 else math.nan


# ------------------------------------------------------------------------------------------------ study setup
def build_scenario(cfg: dict[str, Any]) -> Scenario:
    sc = scenario_from_dict(cfg["scenario"])
    rot = cfg["wind_error"].get("control", {}).get("base_rotation_deg")
    if rot:  # EXP-023 control (d): every wind key is then relative to the rotated base wind
        sc = wind_scenario(sc, key(rot))
    if not sc.boundary.bases:
        raise ValueError("EXP-019 needs one inflow (external) basis: set scenario.boundary.bases")
    if sc.initial.kind != "zero" or sc.boundary.background != 0.0:
        raise ValueError("zero initial state and zero background are assumed (base field identically zero)")
    return sc


def conditions(grid: dict[str, Any]) -> dict[Key, list[str]]:
    """Unique truth wind errors (deg, speed factor) -> blocks they belong to ('direction', 'speed', 'joint')."""
    out: dict[Key, list[str]] = {}

    def add(d: float, p: float, blk: str) -> None:
        out.setdefault(key(d, 1.0 + p / 100.0), []).append(blk)

    for d in grid["direction_deg"]:
        add(d, 0.0, "direction")
    for p in grid["speed_pct"]:
        add(0.0, p, "speed")
    for d in grid["joint_direction_deg"]:
        for p in grid["joint_speed_pct"]:
            add(d, p, "joint")
    return out


def cond_id(k: Key) -> str:
    return f"d{k[0]:+06.1f}_s{(k[1] - 1) * 100:+06.1f}"


class Study:
    """Everything shared by validation and test: scenario, bank, candidate sites, held-out sites, population, prior."""

    def __init__(self, cfg: dict[str, Any], work: Path | str | None):
        we = cfg["wind_error"]
        self.cfg, self.we = cfg, we
        self.scn = build_scenario(cfg)
        self.tab = self.scn.param_table()
        self.bank = Bank(self.scn, work)
        self.control: dict[str, Any] = we.get("control", {})  # EXP-023 opt-in controls; empty = EXP-019 behaviour
        kf = self.control.get("truth_K_factor")
        # model-error control (c): the truth is generated with K x kf while every estimator keeps the nominal K
        self.truth_bank = self.bank if not kf or kf == 1.0 else Bank(dataclasses.replace(
            self.scn, transport=dataclasses.replace(self.scn.transport, K=self.scn.transport.K * kf)), work)
        self.alpha_by: dict[str, float] = {}  # per-strategy alpha override (control (a)); empty = one alpha
        T = self.scn.times.duration
        step = float(cfg["inverse"]["sensor_interval"])
        self.times = tuple(float(t) for t in np.arange(step, T + 1e-9, step))
        self.sigma = float(cfg["inverse"]["noise_std"])
        self.cands = D.candidate_grid(self.scn.domain, int(we["candidates_n"]))
        self.sites = self.cands.sensor_layout()
        rng = np.random.default_rng(int(we["heldout_seed"]))
        self.held = np.sort(rng.choice(len(self.cands), int(we["heldout_n"]), replace=False))
        allowed = np.ones(len(self.cands), bool)
        allowed[self.held] = False
        self.constraints = D.Constraints(allowed=allowed)
        d = self.scn.domain
        grid = GridSpec(0.0, 0.0, d.nx, d.ny, d.dx, d.dy, int(we["population"]["epsg"]))
        pop = we["population"]
        self.counts = population.FIXTURES[pop["fixture"]](grid, int(pop["total"]), int(pop["seed"]))
        self.base_theta = np.array(list(cfg["q"]) + [float(we["inflow_base"])])
        self.prior = prior_keys(we["prior"]["direction_deg"], we["prior"]["speed_factor"])
        self.nominal = key(0.0, 1.0)
        # ensembles that do not depend on any case or truth wind
        self.ens_prior = make_ensemble(self.bank, self.prior, self.sites, self.times)
        self.ens_nominal = make_ensemble(self.bank, [self.nominal], self.sites, self.times)
        if np.abs(self.bank.resp(self.nominal).base).max() > 0:
            raise ValueError("base field must be identically zero for the Gram forms")
        self.weight_maps: dict[str, np.ndarray | None] = {"unif": None, "pop": self.counts.astype(float)}
        self.groups = [p.group for p in self.tab]
        self.kinds = [p.kind for p in self.tab]
        self.alpha = float(we["alpha"]) if we.get("alpha") is not None else math.nan
        self.alpha_settings: dict[str, dict[str, float]] = {}
        self._truth_ops: dict[Key, tuple[np.ndarray, np.ndarray]] = {}
        self._ens_truth: dict[Key, Ensemble] = {}
        self._ens_known: dict[Key, Ensemble] = {}

    # --- truth side (never passed to strategy_c)
    def truth_ens(self, k: Key) -> Ensemble:
        if k not in self._ens_truth:
            self._ens_truth[k] = make_ensemble(self.truth_bank, [k], self.sites, self.times)
        return self._ens_truth[k]

    def known_ens(self, k: Key) -> Ensemble:
        """Strategy A's model: the true wind in the ESTIMATOR's model (= ``truth_ens`` unless the truth has model error)."""
        if self.truth_bank is self.bank:
            return self.truth_ens(k)
        if k not in self._ens_known:
            self._ens_known[k] = make_ensemble(self.bank, [k], self.sites, self.times)
        return self._ens_known[k]

    def assert_off_prior(self, truth_keys: Sequence[Key]) -> list[Key]:
        """Control (b): no truth wind may be a prior member.  Returns the offending keys (raises when the control requires none)."""
        bad = [k for k in truth_keys if k in set(self.prior)]
        if bad and self.control.get("off_prior"):
            raise ValueError(f"truth winds that are prior members: {bad}")
        return bad

    def readings(self, case: Case, k: Key) -> np.ndarray:
        e = self.truth_ens(k)
        return e.b[0] + e.A[0] @ case.theta

    def observe(self, case: Case, k: Key, idx: Sequence[int]) -> np.ndarray:
        """y at sites idx: truth readings + sigma * (the case's own noise array): same noise for every wind/strategy."""
        ix = list(idx)
        return (self.readings(case, k)[ix] + self.sigma * case.noise[ix]).T.reshape(-1)

    def gram(self, truth_keys: Sequence[Key]) -> Gram:
        return Gram(self.bank, list(dict.fromkeys([self.nominal, *self.prior, *truth_keys])), self.weight_maps,
                    truth_bank=None if self.truth_bank is self.bank else self.truth_bank, truth_keys=truth_keys)

    def frozen(self, truth_keys: Sequence[Key], layouts: dict[str, Any], seeds: dict[str, Any]) -> dict[str, Any]:
        fz = {"truth_wind_keys": [list(k) for k in truth_keys], "prior_keys": [list(k) for k in self.prior],
              "alpha": self.alpha, "sigma": self.sigma, "layouts": layouts, "seeds": seeds,
              "heldout": [int(i) for i in self.held]}
        if self.control:  # only controls extend the frozen record: the EXP-019 hash is unchanged
            fz["control"] = self.control
            fz["alpha_by_setting"] = self.alpha_settings
        return fz

    # --- evaluation
    def evaluate(self, case: Case, k: Key, idx: Sequence[int], gram: Gram, strategies: Sequence[str] = STRATEGIES,
                 forms_cache: dict[tuple[str, Key], Forms] | None = None) -> list[dict[str, Any]]:
        """Fit the layout ``idx`` under each strategy; one metrics row per strategy."""
        y = self.observe(case, k, idx)
        fc = forms_cache if forms_cache is not None else {}
        rows = []
        truth_read = self.readings(case, k)
        ens_by = {"A": self.known_ens(k), "B": self.ens_nominal, "C": self.ens_prior}
        for s in strategies:
            ens = ens_by[s]
            opts = FitOptions(self.sigma, self.alpha_by.get(s, self.alpha))
            fit = strategy_c(y, idx, ens, opts) if s == "C" else fit_ensemble(y, idx, ens, opts)
            if (s, k) not in fc:
                fc[(s, k)] = Forms(gram, ens.keys, k, self.kinds, self.groups)
            f = fc[(s, k)]
            rows.append(self._score(s, case, k, idx, ens, fit, f, y, truth_read))
        return rows

    def _score(self, s: str, case: Case, k: Key, idx: Sequence[int], ens: Ensemble, fit: EnsembleFit, f: Forms,
               y: np.ndarray, truth_read: np.ndarray) -> dict[str, Any]:
        th_e = (fit.weights[:, None] * fit.thetas).reshape(-1)
        th = fit.theta
        n_em = sum(1 for kd in self.kinds if kd == "emission")
        row: dict[str, Any] = {"strategy": s, "sensors": " ".join(self.cands.ids[i] for i in idx)}
        row["rmse_total"] = math.sqrt(f.mse("unif", "total", th_e, case.theta))
        row["rmse_pop"] = math.sqrt(f.mse("pop", "total", th_e, case.theta))
        row["rmse_local"] = math.sqrt(f.mse("unif", "local", th_e, case.theta))
        row["rmse_external"] = math.sqrt(f.mse("unif", "external", th_e, case.theta))
        fe, ft = f.ext_fraction_est(th_e), f.ext_fraction_true(case.theta)
        row["ext_frac_est"], row["ext_frac_true"], row["ext_frac_err"] = fe, ft, fe - ft
        h = list(self.held)
        mu, var, var_spread = predict(fit, ens, h)
        th_read = truth_read[h]
        row["rmse_heldout"] = float(np.sqrt(np.mean((mu - th_read) ** 2)))
        sd, sd0 = np.sqrt(var), np.sqrt(var_spread)
        row["cov90"] = float(np.mean(np.abs(mu - th_read) <= Z90 * sd))
        row["cov90_spread_only"] = float(np.mean(np.abs(mu - th_read) <= Z90 * sd0)) if s == "C" else math.nan
        row["width90"] = float(np.mean(2 * Z90 * sd))
        for j in range(n_em):
            row[f"relerr_{self.tab[j].name}"] = float((th[j] - case.theta[j]) / case.theta[j])
        row["bias_rel_mean"] = float(np.mean([(th[j] - case.theta[j]) / case.theta[j] for j in range(n_em)]))
        ee = emission_error(th, case.theta, self.scn)
        row["amp_relerr_l2"], row["local_relerr_l2"] = ee["all_emission"], ee["local"]
        row["inflow_abs_err"] = float(th[n_em] - case.theta[n_em])
        # chi2 of the prediction actually made (fit sensors), known sigma
        ix = list(idx)
        pred = np.tensordot(fit.weights, np.stack([ens.b[m][ix] + ens.A[m][ix] @ fit.thetas[m] for m in range(len(ens.keys))]), axes=1)
        chi2 = float(np.sum((y - pred.T.reshape(-1)) ** 2) / self.sigma**2)
        dof = fit.n_obs - len(th)
        row.update(chi2=chi2, dof=dof, chi2_per_dof=chi2 / dof, chi2_p=float(stats.chi2.sf(chi2, dof)),
                   chi2_min_member=float(fit.chi2.min()))
        top = int(np.argmax(fit.weights))
        rep = identifiability_report(as_fit_result(fit.sols[top], ens, fit.n_obs))
        row["flag_any"] = int(len(rep["flags"]) > 0)
        row["flags"] = ";".join(rep["flags"])
        row["effective_rank"] = rep["effective_rank"]
        row["w_max"] = float(fit.weights.max())
        row["n_eff_members"] = float(1.0 / np.sum(fit.weights**2))
        row["runtime_s"] = fit.runtime_s
        return row


# ------------------------------------------------------------------------------------------------ layouts
def design_layouts(st: Study, design_cases: list[Case], alpha: float, budgets: Sequence[int], n_random: int,
                   layout_seed: int, n_noise: int) -> dict[int, dict[str, Any]]:
    """Greedy-under-nominal-wind (design ensemble, ``design.greedy_trace``), spread and random layouts per budget.
    Greedy is nested, so one trace at the largest budget gives every smaller budget as a prefix."""
    opts = D.DesignOptions(noise_std=st.sigma, obs_times=st.times, alpha=alpha)
    resp = st.bank.resp(st.nominal)
    dcs = [D.build_case(c.name, "design", st.scn, c.theta, st.scn, st.cands, opts, c.noise_seed, n_noise, st.bank.cache,
                        truth_field=resp.field(c.theta)) for c in design_cases]
    trace = D.greedy_trace(st.cands, max(budgets), dcs, st.constraints, opts)
    out: dict[int, dict[str, Any]] = {}
    for b in budgets:
        out[b] = {"greedy": tuple(sorted(trace.order[:b])), "spread": D.spread_layout(st.cands, b, st.constraints),
                  "random": D.random_layouts(st.cands, b, n_random, st.constraints, layout_seed + b)}
    return out


def layout_table(layouts: dict[int, dict[str, Any]]) -> list[tuple[str, int, str, tuple[int, ...]]]:
    """(layout_id, budget, type, indices)."""
    rows = []
    for b, d in layouts.items():
        rows.append((f"greedy-{b}", b, "greedy", tuple(d["greedy"])))
        rows.append((f"spread-{b}", b, "spread", tuple(d["spread"])))
        rows += [(f"random-{b}-{i}", b, "random", tuple(lay)) for i, lay in enumerate(d["random"])]
    return rows


# ------------------------------------------------------------------------------------------------ statistics
PER_CASE_KEYS = ["cond", "strategy", "budget", "layout_type", "case"]


def per_case(raw: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Mean over the layouts of a type (20 random layouts -> one value) for each (condition, strategy, budget, case)."""
    return pd.DataFrame(raw.groupby(PER_CASE_KEYS, as_index=False)[list(metrics)].mean())


def paired_bootstrap(pc: pd.DataFrame, a: str, b: str, metric: str, by: Sequence[str], n_boot: int = 2000,
                     seed: int = 1, level: float = 0.95) -> pd.DataFrame:
    """Mean paired difference strategy a - strategy b over test cases with a percentile bootstrap interval (resampling
    CASES with replacement, the same resample for every row of a group)."""
    w = pc.pivot_table(index=[*by, "case"], columns="strategy", values=metric)
    d = (w[a] - w[b]).dropna().rename("diff").reset_index()
    rng = np.random.default_rng(seed)
    out = []
    for g, sub in d.groupby(list(by)):
        v = sub["diff"].to_numpy()
        bs = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
        lo, hi = np.quantile(bs, [(1 - level) / 2, 1 - (1 - level) / 2])
        g = g if isinstance(g, tuple) else (g,)
        out.append({**dict(zip(by, g)), "n_cases": len(v), "diff_mean": float(v.mean()), "ci_lo": float(lo),
                    "ci_hi": float(hi), "excludes_0": bool(lo > 0 or hi < 0), "contrast": f"{a}-{b}", "metric": metric,
                    "interval": f"percentile bootstrap {int(level * 100)}% over cases (n_boot={n_boot})"})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ identifiability demo
def wind_amplitude_confounding(st: Study, idx: Sequence[int], theta: np.ndarray, rotations: Sequence[float]
                               ) -> dict[str, Any]:
    """Noise-free numerical demonstration, for one layout: (i) singular values of the whitened, scaled nominal-wind
    operator; (ii) for each wind rotation phi the signal change y(phi) - y(0) = (A(phi) - A(0)) theta, its part that
    amplitude changes alone can absorb (projection on span A(0)) and the leftover, in units of the noise sigma
    (sqrt of the chi2 it would add); (iii) the amplitude shift that does the absorbing; (iv) singular values of the
    operator augmented with the wind-direction tangent column."""
    ix = list(idx)
    nt = len(st.times)
    n = len(theta)

    def op(k: Key) -> np.ndarray:
        e = st.truth_ens(k)
        return e.A[0][ix].transpose(1, 0, 2).reshape(nt * len(ix), n)

    sc = np.array([SCALE_BY_KIND[kd] for kd in st.kinds])
    A0 = op(st.nominal)
    H0 = A0 / st.sigma * sc
    out: dict[str, Any] = {"singular_values_nominal": np.linalg.svd(H0, compute_uv=False).tolist(), "rotations": []}
    y0 = A0 @ theta
    for phi in rotations:
        d = (op(key(phi)) @ theta - y0) / st.sigma
        coef, *_ = np.linalg.lstsq(H0, d, rcond=None)
        left = d - H0 @ coef
        out["rotations"].append({
            "deg": phi, "signal_change_sigma": float(np.linalg.norm(d)), "absorbed_by_amplitudes_sigma": float(np.linalg.norm(H0 @ coef)),
            "leftover_sigma": float(np.linalg.norm(left)), "implied_chi2_excess": float(left @ left),
            "dof": int(H0.shape[0] - n), "amplitude_shift_rel": (coef * sc / theta).tolist()})
    eps = 5.0
    tang = (op(key(eps)) - op(key(-eps))) @ theta / (2 * eps) / st.sigma  # d y / d phi per degree, in sigma units
    aug = np.column_stack([H0, tang])
    out["singular_values_augmented"] = np.linalg.svd(aug, compute_uv=False).tolist()
    proj = H0 @ np.linalg.lstsq(H0, tang, rcond=None)[0]
    out["tangent_norm_sigma_per_deg"] = float(np.linalg.norm(tang))
    out["tangent_sin_angle_to_span"] = float(np.linalg.norm(tang - proj) / np.linalg.norm(tang))
    return out

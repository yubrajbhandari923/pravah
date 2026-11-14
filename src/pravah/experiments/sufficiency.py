"""E08 sensor sufficiency: sampling schedules, fast exact scorer, per-scenario evaluation and breakpoint statistics.

Design: docs/research/sufficiency-design.md (committed before any E08 result). Reuses the E07 designs, frozen MAP estimator
and information separation (`cross_objective`); adds noise multipliers, sampling schedules (span W, readings n), per-regime
scoring, identifiability descriptors and the breakpoint k*(tau). Units: ug m^-3, ug s^-1, hours.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import binom, norm

from pravah.experiments import cross_objective as X

STRATEGIES = ("random", "coverage", "source_near", "dopt")
ARMS = {"w0.2": ("white", 0.2), "w1": ("white", 1.0), "w5": ("white", 5.0), "ar1": ("ar1", 1.0)}  # arm -> (kind, sigma mult)
SCHEDULES = {"S1": (1, 1), "S2": (6, 6), "S3": (24, 6), "S4": (24, 24), "S5": (168, 6), "S6": (168, 24), "S7": (168, 168)}
REF = ("w1", "S4")  # the E07 configuration
LOOKBACK = (1, 2, 3, 6, 12, 24, 48, 96, 168)
TAUS = (0.5, 0.3, 0.2, 0.1)
K_NONE = 64  # code for "> 32" (no tested k sufficient)


def schedule_mask(n_hours, W, n):
    """(n_hours,) bool: sampled calendar hours, h mod (W/n) == 0. Any W consecutive hours hold exactly n of them."""
    if W % n:
        raise ValueError("W must be a multiple of n")
    return np.arange(n_hours) % (W // n) == 0


class GramScorer:
    """Exact rewrite of cross_objective.score for many estimates: per issue time t the squared error summed over cells is
    a^T M_t a - 2 a^T b_t + c_t (M_t = sum F F^T, b_t = sum F r, c_t = sum r^2, r = truth - c_bg). Truth is read here only."""

    def __init__(self, F, truth_field, issue, a_true, a_ref, c_bg, groups=None):
        self.n_cell = int(np.prod(truth_field.shape[1:]))
        self.a_true, self.a_ref = np.asarray(a_true, float), a_ref
        self.groups = {} if groups is None else groups  # name -> bool mask over issue times
        K = len(F)
        self.mats = {}
        for name, hrs in (("fc", np.asarray(issue) + 1), ("an", np.asarray(issue))):
            Fi = F[:, hrs].reshape(K, len(hrs), -1)
            r = truth_field[hrs].reshape(len(hrs), -1) - c_bg
            self.mats[name] = (np.einsum("kti,lti->tkl", Fi, Fi), np.einsum("kti,ti->tk", Fi, r), (r ** 2).sum(1))

    def sse(self, a_hat, which="fc"):
        M, b, c = self.mats[which]
        return np.maximum(np.einsum("tk,tkl,tl->t", a_hat, M, a_hat) - 2 * (a_hat * b).sum(1) + c, 0.0)

    def __call__(self, u_hat):
        a = np.asarray(u_hat, float) * self.a_ref
        fc = self.sse(a)
        out = dict(forecast_rmse=float(np.sqrt(fc.mean() / self.n_cell)),
                   analysis_rmse=float(np.sqrt(self.sse(a, "an").mean() / self.n_cell)),
                   source_rmse=float(np.sqrt(np.mean((a - self.a_true) ** 2)) / self.a_ref))
        for g, m in self.groups.items():
            out[f"fc_rmse_{g}"] = float(np.sqrt(fc[m].mean() / self.n_cell)) if m.any() else np.nan
        return out


def issue_regimes(sc, issue):
    """Held-out wind regime of the 48 h test block containing each issue hour."""
    te = sc.split_hours("test")
    reg = np.asarray(sc.truth.wind_regimes["test"])
    return reg[(np.asarray(issue) - te[0]) // 48]


def time_to_information(A, issue, K):
    """Per issue time, the shortest hourly look-back in LOOKBACK with practical rank K (np.inf if none; look-backs
    longer than the calendar before the first issue time are skipped, only in short test calendars)."""
    out = np.full(len(issue), np.inf)
    for L in [L for L in LOOKBACK if L <= np.min(issue) + 1][::-1]:
        out[X.window_info(A, issue, L)[:, 0] >= 1] = L  # eigvalsh is ascending: column 0 is the smallest
    return out


def identifiability(A, issue, W, mask, K, regimes):
    ev = X.window_info(A, issue, W, mask)
    pr = (ev >= 1).sum(1)
    sr = (ev >= 1e-8 * np.maximum(ev[:, -1:], 1e-300)).sum(1)
    d = dict(lam_min_med=float(np.median(ev[:, 0])), frac_prank_lt_K=float((pr < K).mean()), prank_mean=float(pr.mean()),
             srank_min=int(sr.min()))
    for g in np.unique(regimes):
        d[f"frac_prank_lt_K_{g}"] = float((pr[regimes == g] < K).mean())
    return d


def make_designs(sc, G3, ks, sigma0, exclude_m=0):
    """{(mult, k, strategy, net): (sites or None, info)}. dopt is rebuilt per sigma multiplier; the others are sigma-free and
    stored under mult None."""
    T, seedp = len(sc.split_hours("train")), sc.cfg.seeds.streams(sc.replicate)["placement"]
    out = {}
    cand = X.exclude_near_sources(sc.site_xy, sc.admissible_sites(), sc.truth.source_xy, exclude_m) if exclude_m else None
    for mult in sorted({m for _, m in ARMS.values()}):
        p = X.problem(X.design_input(sc, G3, sigma0 * mult), cand=cand)
        todo = [("dopt", 0)] + ([] if mult != 1.0 else [("random", j) for j in range(X.N_RANDOM)] + [("coverage", 0), ("source_near", 0)])
        for k in ks:
            for name, net in todo:
                t = time.time()
                try:
                    S, info = X.make_design(name, p, k, T, seedp, net)
                    d = dict(status="ok", sites=" ".join(map(str, S)), design_s=time.time() - t,
                             **{c: info[c] for c in ("greedy_value", "refined_value", "n_swaps") if c in info})
                except ValueError as e:
                    S, d = None, dict(status=str(e))
                out[(mult if name == "dopt" else None, k, name, net)] = (S, d)
    return out


def evaluate(sc, F, G3, ks, R, schedules=SCHEDULES, arms=ARMS, exclude_m=0, full_window=False):
    """All E08 rows of one scenario. Returns (rows, designs_rows, timings). `random` rows are means over its 10 networks
    (defined only if all are; n_net_ok records how many were). exclude_m > 0: candidates within that distance of a true source
    are dropped (E07 control). full_window: estimator uses all readings from hour 0 (map_full) instead of the W-hour window;
    the schedule still sets which hours are read (identifiability descriptors keep the nominal W)."""
    K = sc.n_sources
    a_ref, c_bg, a_true = sc.cfg.sources.reference_rate_ug_s, sc.cfg.sources.background_ug_m3, sc.truth.strengths_ug_s
    issue, n_h = X.issue_times(sc), len(sc.truth.winds)
    regimes = issue_regimes(sc, issue)
    groups = {g: regimes == g for g in np.unique(regimes)}
    scorer = GramScorer(F, sc.truth.field, issue, a_true, a_ref, c_bg, groups)
    sigma0 = X.site_sigma(sc, False)
    tm = {}
    t0 = time.time()
    designs = make_designs(sc, G3, ks, sigma0, exclude_m)
    tm["design_wall_s"] = time.time() - t0
    t0 = time.time()
    base = dict(n_issue=len(issue), **{f"n_issue_{g}": int(m.sum()) for g, m in groups.items()})
    rows = [dict(arm="-", schedule="-", k=0, strategy="background_only", status="ok", **base, **scorer(np.zeros((len(issue), K))))]
    masks = {s: schedule_mask(n_h, W, n) for s, (W, n) in schedules.items()}
    for arm, (kind, mult) in arms.items():
        sigma = sigma0 * mult
        Z = [(X.readings(sc, sigma, kind, r, n_h) - c_bg) / sigma[:, None] for r in range(R)]  # common z across mults
        for k in ks:
            for strat in STRATEGIES:
                nets = range(X.N_RANDOM) if strat == "random" else [0]
                key = lambda j: (mult if strat == "dopt" else None, k, strat, j)
                ok = [j for j in nets if designs[key(j)][0] is not None]
                per = {s: [] for s in schedules}
                tti = []
                for j in (ok if len(ok) == len(nets) else []):
                    S = designs[key(j)][0]
                    A = a_ref * G3[S] / sigma[S][:, None, None]
                    if kind == "white":
                        tti.append(time_to_information(A, issue, K))
                    for s, (W, n) in schedules.items():
                        m = [scorer(X.estimate_series(A, z[S], issue, None if full_window else W, mask=masks[s])) for z in Z]
                        sc_ = {c: float(np.mean([x[c] for x in m])) for c in m[0]}
                        per[s].append({**sc_, **identifiability(A, issue, W, masks[s], K, regimes)})
                for s, (W, n) in schedules.items():
                    r = dict(arm=arm, schedule=s, W=W, n=n, cadence=W // n, k=k, B=k * n, sensor_hours=k * W, strategy=strat,
                             n_net_ok=len(ok), sigma_med=float(np.median(sigma)), **base)
                    if per[s]:
                        r.update(status="ok", **{c: float(np.mean([x[c] for x in per[s]])) for c in per[s][0]})
                        if tti:
                            t_ = np.concatenate(tti)
                            r.update(tti_med=float(np.median(t_)), frac_tti_le_168=float(np.isfinite(t_).mean()))
                    else:
                        r["status"] = "undefined: " + next(designs[key(j)][1]["status"] for j in nets if designs[key(j)][0] is None)
                    rows.append(r)
    tm["eval_wall_s"] = time.time() - t0
    drows = [dict(mult=m, k=k, strategy=s, net=j, **d) for (m, k, s, j), (_, d) in designs.items()]
    return rows, drows, tm


# ---------------------------------------------------------------- statistics (scenario = unit)
def df_ci(x, level=0.95):
    """Distribution-free CI for the median: (x_(j), x_(n+1-j)) with the largest j whose binomial coverage >= level.
    Returns (lo, hi, coverage); for small n that cannot reach `level`, (min, max) and its actual coverage."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    js = [j for j in range(1, n // 2 + 1) if 1 - 2 * binom.cdf(j - 1, n, .5) >= level]
    j = max(js) if js else 1
    return float(x[j - 1]), float(x[n - j]), float(1 - 2 * binom.cdf(j - 1, n, .5))


def wilson(s, n, level=0.95):
    if n == 0:
        return np.nan, np.nan
    z, p = norm.ppf(.5 + level / 2), s / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def breakpoint(ks, nrmse, tau):
    """Smallest tested k with nRMSE <= tau at it and at every larger tested k; K_NONE if none. NaN/inf = not sufficient."""
    o = np.argsort(ks)
    ks, ok = np.asarray(ks)[o], np.nan_to_num(np.asarray(nrmse, float)[o], nan=np.inf) <= tau
    kstar = K_NONE
    for i in range(len(ks) - 1, -1, -1):
        if not ok[i]:
            break
        kstar = int(ks[i])
    return kstar

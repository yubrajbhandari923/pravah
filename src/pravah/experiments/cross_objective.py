"""E07 cross-objective placement ablation: designs, frozen estimator, held-out scorer, paired statistics.

Design: docs/research/cross-objective-design.md (committed before any E07 result). Contract: tests/test_cross_objective.py.
Separation of information: designs see only a `DesignInput` (training-hour sensitivities, sigma, a_ref, known source
locations, candidates, placement seed); the estimator sees only the chosen sites' noisy readings up to the issue time and
their sensitivities; truth is read only by `score`. Units: ug m^-3, ug s^-1, m, hours.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls
from scipy.stats import binomtest

from pravah import benchmarks as B
from pravah import placement as P
from pravah.observations import bilinear_sample, noise_field
from pravah.sensitivity import anomaly_fields

STRATEGIES = ("uniform", "coverage", "source_near", "hotspot", "dopt", "aopt", "fieldvar", "diversity", "podqr")
N_RANDOM, N_PRIOR, WINDOW_H, WARMUP_H = 10, 32, 24, 6
ARMS = {"white": ("white", False), "ar1": ("ar1", False), "persite": ("white", True)}  # arm -> (noise kind, per-site sigma)
SWAP = {"dopt": P.dopt, "aopt": P.aopt, "fieldvar": P.fieldvar}


# ---------------------------------------------------------------- forward responses (known locations, known winds)
def source_field(cfg, xy, winds, n_h):
    """Full-grid anomaly per unit source at `xy` (ug m^-3 per ug s^-1), hours [0, n_h) after spin-up: one solve."""
    m = cfg.model
    S = B.source_basis(tuple(m.grid), tuple(m.domain_m), xy, cfg.sources.footprint_sigma_m) / m.mixing_depth_m
    return anomaly_fields(m, winds, S, n_h)


def site_responses(sc, F):
    """G3 (n_sites, hours, K): F sampled at the candidate sites."""
    dom = tuple(sc.cfg.model.domain_m)
    return np.stack([[bilinear_sample(F[k, h], sc.site_xy, dom) for h in range(F.shape[1])] for k in range(len(F))]).transpose(2, 1, 0)


def responses(sc):
    """F (K, hours, nx, ny): full-grid anomaly per unit source (ug m^-3 per ug s^-1) over the whole calendar, and
    G3 (n_sites, hours, K) at the candidate sites. Same solver, winds and spin-up as the truth; never uses strengths."""
    F = np.stack([source_field(sc.cfg, xy, sc.truth.winds, len(sc.truth.winds)) for xy in sc.truth.source_xy])
    return F, site_responses(sc, F)


def site_sigma(sc, per_site):
    """Protocol sigma: 0.05 x median training truth over candidate sites (scalar, broadcast) or at each site."""
    tr = np.asarray(sc.truth.site_truth)[:, sc.split_hours("train")]
    s = B.SIGMA_REL * (np.median(tr, axis=1) if per_site else np.full(len(tr), np.median(tr)))
    return s


# ---------------------------------------------------------------- designs
@dataclass(frozen=True)
class DesignInput:
    """Everything a design may know. No strengths, no truth field, no noise, no non-training hours."""
    xy: np.ndarray
    cand: np.ndarray
    min_sep: float
    a_ref: float
    sigma: np.ndarray          # (n_sites,)
    G3_train: np.ndarray       # (n_sites, T_train, K)
    source_xy: np.ndarray
    placement_seed: int
    prior_loguniform: tuple    # (lo, hi) x a_ref, the benchmark's generating distribution (not the realised strengths)


def design_input(sc, G3_all, sigma):
    tr = sc.split_hours("train")
    if tr[0] != 0:
        raise ValueError("training split must start at hour 0")
    return DesignInput(sc.site_xy.copy(), sc.admissible_sites(), sc.cfg.design.min_sensor_separation_m,
                       sc.cfg.sources.reference_rate_ug_s, np.asarray(sigma, float).copy(), G3_all[:, tr].copy(),
                       sc.truth.source_xy.copy(), sc.cfg.seeds.streams(sc.replicate)["placement"],
                       tuple(sc.cfg.sources.strength_loguniform))


def exclude_near_sources(xy, cand, source_xy, radius_m):
    """cand without the sites within radius_m (inclusive) of any source centre; order kept (gate-07 exclusion control)."""
    cand = np.asarray(cand, int)
    d = np.linalg.norm(np.asarray(xy)[cand][:, None] - np.asarray(source_xy)[None], axis=2).min(1)
    return cand[d > radius_m]


def problem(di: DesignInput, cand=None):
    """placement.Problem with the field-variance weight Q and the PRIOR-ENSEMBLE POD archive (gate-06 condition 1)."""
    G = di.G3_train
    n, T, K = G.shape
    lo, hi = di.prior_loguniform
    a = di.a_ref * np.exp(np.random.default_rng([di.placement_seed, 99]).uniform(np.log(lo), np.log(hi), (K, N_PRIOR)))
    Q = di.a_ref ** 2 * np.einsum("shk,shl->kl", G, G) / (n * T)
    return P.Problem(xy=di.xy, cand=di.cand if cand is None else cand, min_sep=di.min_sep, a_ref=di.a_ref, sigma=di.sigma,
                     G3=G, snapshots=(G @ a).reshape(n, T * N_PRIOR), source_xy=di.source_xy, Q=Q)


def pod_leading_modes(p, T):
    """r*: leading run of POD modes whose per-reading rms amplitude at their best candidate, accumulated over T readings
    (x sqrt(T), as D-opt accumulates T independent hours), is >= 1 noise sd. Fixed in the design note §3."""
    X = np.asarray(p.snapshots, float)
    U, s, _ = np.linalg.svd(X - X.mean(1, keepdims=True), full_matrices=False)
    amp = (s * np.abs(U[p.cand]) / p.sigma[p.cand][:, None]).max(0) * np.sqrt(T / X.shape[1])
    below = np.flatnonzero(amp < 1)
    return int(below[0]) if below.size else len(s)


def make_design(name, p, k, T, seed, net=0):
    """(sites, info). T = training hours; seed = placement seed (random only). ValueError = undefined design (recorded as
    a failure, never dropped)."""
    info = {}
    if name == "random":
        S = P._in_order(p, np.random.default_rng([seed, net]).permutation(p.cand), k, "random")
    elif name in ("uniform", "coverage", "source_near", "diversity"):
        S = P.STRATEGIES[name](p, k)
    elif name == "hotspot":
        S = P.hotspot_design(p, k)
    elif name in SWAP:
        g = P.greedy(p, k, SWAP[name])
        S, v, nsw = P.swap_refine(p, g, SWAP[name])
        info = dict(greedy_value=SWAP[name](p, g), refined_value=v, n_swaps=nsw, greedy_sites=" ".join(map(str, g)))
    elif name == "podqr":
        r = pod_leading_modes(p, T)
        info = dict(pod_r_lead=r)
        if k > r:
            raise ValueError(f"podqr undefined: k={k} > r*={r} leading above-noise modes")
        S = P.qr_select(p.pod(k), p, k)
    else:
        raise ValueError(f"unknown strategy {name!r}")
    assert len(set(S)) == k and set(S) <= set(p.cand.tolist()) and all(p.feasible(i, [j for j in S if j != i]) for i in S)
    return [int(i) for i in S], info


# ---------------------------------------------------------------- frozen estimator (sees readings up to issue time only)
def estimate_series(A, z, issue, window=WINDOW_H, nonneg=False):
    """MAP (prior N(0, I) in a_ref units; independent noise) from the `window` hours ending at each issue time.
    A: (k, hours, K) whitened a_ref-scaled sensitivities of the chosen sites; z: (k, hours) whitened anomalies
    (y - c_bg)/sigma. Rows after an issue time are never touched. Returns u_hat (n_issue, K) in a_ref units."""
    issue = np.asarray(issue, int)
    if issue.min() - window + 1 < 0:
        raise ValueError("window starts before hour 0")
    K = A.shape[2]
    Ph = np.einsum("shk,shl->hkl", A, A)
    bh = np.einsum("shk,sh->hk", A, z)
    cP = np.concatenate([np.zeros((1, K, K)), np.cumsum(Ph, 0)])
    cb = np.concatenate([np.zeros((1, K)), np.cumsum(bh, 0)])
    N = np.eye(K) + cP[issue + 1] - cP[issue + 1 - window]
    b = cb[issue + 1] - cb[issue + 1 - window]
    if not nonneg:
        return np.linalg.solve(N, b[..., None])[..., 0]
    out = np.empty_like(b)
    for t in range(len(issue)):
        L = np.linalg.cholesky(N[t])
        out[t] = nnls(L.T, np.linalg.solve(L, b[t]), maxiter=50 * K)[0]
    return out


def window_info(A, issue, window=WINDOW_H):
    """Eigenvalues (n_issue, K) of the window information A^T A (practical rank = count >= 1)."""
    Ph = np.einsum("shk,shl->hkl", A, A)
    cP = np.concatenate([np.zeros((1,) + Ph.shape[1:]), np.cumsum(Ph, 0)])
    return np.linalg.eigvalsh(cP[np.asarray(issue) + 1] - cP[np.asarray(issue) + 1 - window])


# ---------------------------------------------------------------- scorer (the only place truth is read)
def issue_times(sc):
    te = sc.split_hours("test")
    return np.arange(te[0] + WARMUP_H, te[-1])  # forecast t+1 stays inside the test segment


def pop_weight(grid, domain):
    rng = np.random.default_rng(6000)
    c = rng.uniform(2000, 8000, (3, 2))
    x = (np.arange(grid[0]) + .5) * domain[0] / grid[0]
    y = (np.arange(grid[1]) + .5) * domain[1] / grid[1]
    w = sum(np.exp(-((x[:, None] - a) ** 2 + (y[None] - b) ** 2) / (2 * 1500.0 ** 2)) for a, b in c)
    return w / w.sum()


def score(u_hat, F, truth_field, issue, a_true, a_ref, c_bg, w):
    """Metrics for one estimate series: forecast (primary), analysis, population-weighted forecast, source RMSE/a_ref."""
    a_hat = u_hat * a_ref
    fc = c_bg + np.einsum("tk,ktxy->txy", a_hat, F[:, issue + 1]) - truth_field[issue + 1]
    an = c_bg + np.einsum("tk,ktxy->txy", a_hat, F[:, issue]) - truth_field[issue]
    return dict(forecast_rmse=float(np.sqrt(np.mean(fc ** 2))), analysis_rmse=float(np.sqrt(np.mean(an ** 2))),
                popw_rmse=float(np.sqrt((fc ** 2 * w).sum((1, 2)).mean())),
                source_rmse=float(np.sqrt(np.mean((a_hat - a_true) ** 2)) / a_ref))


def readings(sc, sigma, kind, r, n_hours):
    """Noisy readings at ALL candidate sites (n_sites, hours) for draw r: shared by every design (paired draws)."""
    ns = sc.cfg.seeds.streams(sc.replicate)["noise"]
    e = noise_field(np.random.SeedSequence([ns, r]), sc.n_candidate_sites, n_hours, sigma, kind)
    return np.asarray(sc.truth.site_truth)[:, :n_hours] + e


# ---------------------------------------------------------------- paired statistics
def boot_median_ci(x, n=10000, seed=4000):
    x = np.asarray(x, float)
    i = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    m = np.median(x[i], axis=1)  # -inf (undefined designs) stay as order statistics
    return float(np.quantile(m, .025, method="lower")), float(np.quantile(m, .975, method="higher"))


def df_median_ci(x, cover=.95):
    """Distribution-free order-statistic CI for the median: (x(j), x(n+1-j)) with the largest j giving >= cover
    binomial coverage (gate-07 issue 5)."""
    from scipy.stats import binom
    x, n = np.sort(np.asarray(x, float)), len(x)
    j = max(i for i in range(1, n // 2 + 1) if 1 - 2 * binom.cdf(i - 1, n, .5) >= cover)
    return float(x[j - 1]), float(x[n - j])


def sign_test(x, tie=1e-12):
    x = np.asarray(x, float)
    pos, neg = int((x > tie).sum()), int((x < -tie).sum())
    p = binomtest(pos, pos + neg).pvalue if pos + neg else 1.0
    return pos, neg, len(x) - pos - neg, float(p)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate(np.minimum(1, (len(p) - np.arange(len(p))) * p[o]))
    out = np.empty_like(adj)
    out[o] = adj
    return out

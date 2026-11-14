"""E11 placement for causal forecasting: training-only forecast-variance objectives, the two estimator memories and the
multi-horizon held-out scorer. Design: docs/research/forecast-design-placement.md. Contract: tests/test_forecast_design.py.

The objectives use only a `ForecastInput` (E07 `DesignInput` + full-grid field weights of the TRAINING hours + responses
under a fixed design wind prior). Linear-Gaussian estimators have a data-independent posterior covariance, so the expected
forecast MSE V = tr(Q_{t+h} Sigma_t(S)) is computable without readings, noise, strengths or truth (design note §2).
Units: Q in (ug m^-3)^2 per a_ref^2; Sigma in a_ref^2; A whitened, a_ref-scaled.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from pravah import placement as P
from pravah.experiments import cross_objective as X

HORIZONS, WINDOW_H = (1, 6, 24), X.WINDOW_H
N_DIRS, BLOCK_H, PRIOR_SPEED = 8, 24, 3.0
FC = ("fc_grow", "fc_w24", "fc_grow_rob", "fc_w24_rob")
# Design priors as subsets of the per-direction blocks (64^2 addendum A1). Block d is FROM d * 45 deg: 0 = N, 4 = S, the
# test directions (S = steady southerly, N = the front of stagnation_then_front). noSN is the out-of-sample variant.
PRIORS = {"all8": tuple(range(N_DIRS)), "noSN": (1, 2, 3, 5, 6, 7)}
ROB = ("fc_grow_rob", "fc_w24_rob")


def design_winds(n_dirs=N_DIRS, block_h=BLOCK_H, speed=PRIOR_SPEED):
    """Design wind prior (§3.1): steady `speed` m/s FROM 0, 45, ... deg for block_h hours each (make_winds convention)."""
    th = np.radians(np.repeat(np.arange(n_dirs) * 360.0 / n_dirs, block_h))
    return np.stack([-speed * np.sin(th), -speed * np.cos(th)], 1)


def prior_block(cfg, xy, d, spinup_h):
    """Unit response (BLOCK_H, nx, ny) of the source at xy to design direction d alone: an independent steady 24 h block
    after its own `spinup_h` steady spin-up (the spin-up replays the block's wind). Does not modify cfg."""
    c = copy.deepcopy(cfg)
    c.model.spinup_h = spinup_h
    return X.source_field(c, xy, design_winds()[d * BLOCK_H:(d + 1) * BLOCK_H], BLOCK_H)


def prior_fields(blocks, dirs):
    """Design-prior responses (K, len(dirs) * BLOCK_H, ...) from per-direction blocks (K, N_DIRS, BLOCK_H, ...)."""
    b = np.asarray(blocks)[:, list(dirs)]
    return b.reshape((b.shape[0], -1) + b.shape[3:])


def loguniform_m2(lo, hi, K):
    """E[a a^T] (a_ref^2) of K iid LogUniform(lo, hi) strengths: s^2 I + abar^2 11^T (the declared strength prior)."""
    L = np.log(hi / lo)
    m1, m2 = (hi - lo) / L, (hi ** 2 - lo ** 2) / (2 * L)
    return (m2 - m1 ** 2) * np.eye(K) + m1 ** 2 * np.ones((K, K))


def field_weights(F, a_ref):
    """Q (hours, K, K) = a_ref^2 F_h^T F_h / N from unit responses F (K, hours, nx, ny)."""
    Fr = a_ref * F.reshape(F.shape[0], F.shape[1], -1)
    return np.einsum("khn,lhn->hkl", Fr, Fr) / Fr.shape[2]


@dataclass(frozen=True)
class ForecastInput:
    """Everything an E11 forecast design may know: E07's DesignInput plus training-hour field weights and the responses
    under the design wind prior (known locations, a fixed prior calendar). No strengths, truth, noise or non-training hour."""
    di: X.DesignInput
    Q_train: np.ndarray        # (T_train, K, K)
    G3_rob: np.ndarray         # (n_sites, n_dirs * block_h, K) per unit source
    Q_rob: np.ndarray          # (n_dirs * block_h, K, K)


def forecast_input(di, F, G3_rob, F_rob):
    """ForecastInput from E07's DesignInput, the unit responses F (K, hours, nx, ny) of which only the training hours
    are read, and the design-prior responses (G3_rob at sites, F_rob full grid)."""
    T = di.G3_train.shape[1]
    return ForecastInput(di, field_weights(F[:, :T], di.a_ref), np.asarray(G3_rob, float).copy(), field_weights(F_rob, di.a_ref))


def _cum(A3):
    """Per-site cumulative hourly information (n_sites, hours + 1, K, K)."""
    I = np.einsum("shk,shl->shkl", A3, A3)
    return np.concatenate([np.zeros((len(I), 1) + I.shape[2:]), np.cumsum(I, 1)], 1)


def _tr_inv(M, Q):
    """sum over the leading axis of tr(M^-1 Q)."""
    return float(np.trace(np.linalg.solve(M, Q), axis1=-2, axis2=-1).sum())


def expected_mse(C, Q, issue, horizons=HORIZONS, window=None, last=None, m2=None):
    """mean_h mean_t tr(Q[t+h] Sigma_t): expected squared forecast error of the field (per cell) of the MAP with prior
    N(0, I) from the window ending at t (None = all hours 0..t). C: (hours + 1, K, K) cumulative information of a site set;
    only t + h <= last (default: the last hour of Q) is used. Returns the mean over horizons of the per-horizon means.
    m2: the strengths' second moment E[a a^T] (a_ref^2) if not I; the MAP error -Sigma a + Sigma A'e then has expected
    weighted square tr(Q Sigma (m2 + M) Sigma), M = window information (gate-10/11 issue 9). m2=None is exactly tr(Q Sigma)."""
    issue, last = np.asarray(issue, int), len(Q) - 1 if last is None else last
    K, out = C.shape[-1], []
    for h in horizons:
        t = issue[issue + h <= last]
        if not len(t):
            raise ValueError(f"no issue time with t + {h} inside the calendar")
        lo = np.zeros_like(t) if window is None else t + 1 - window
        if lo.min() < 0:
            raise ValueError("window starts before hour 0")
        M = C[t + 1] - C[lo]
        if m2 is None:
            out.append(_tr_inv(np.eye(K) + M, Q[t + h]) / len(t))
        else:
            S = np.linalg.inv(np.eye(K) + M)
            out.append(float(np.einsum("tkl,tlm,tmn,tnk->", Q[t + h], S, np.asarray(m2) + M, S)) / len(t))
    return float(np.mean(out))


class Objective:
    """The four E11 objectives (design note §3) as larger-is-better functions value(p, S), for placement.greedy/swap."""

    def __init__(self, p: P.Problem, fi: ForecastInput, horizons=HORIZONS):
        T, K = fi.Q_train.shape[0], fi.Q_train.shape[1]
        if p.G3.shape[1] != T:
            raise ValueError("Problem G3 must cover exactly the training hours")
        self.K, self.T, self.h = K, T, tuple(horizons)
        self.C = _cum(p.A3)                                            # training, per site
        sig = np.broadcast_to(np.asarray(p.sigma, float), (len(p.G3),))
        self.Crob = _cum(fi.G3_rob * (p.a_ref / sig)[:, None, None])   # design prior, per site
        nb = fi.Q_rob.shape[0] // BLOCK_H
        self.b0, self.b1 = np.arange(nb) * BLOCK_H, (np.arange(nb) + 1) * BLOCK_H
        self.Qb = fi.Q_rob.reshape(nb, BLOCK_H, K, K).mean(1)
        Q = fi.Q_train
        self.Qg = np.mean([Q[h:].mean(0) for h in self.h], 0)          # J_grow weight: mean_h mean_{t<T-h} Q[t+h]
        # J_w24 as sum_t tr(W_t Sigma_t): W_t = sum_h [23 <= t < T-h] Q[t+h] / (|H| n_h)
        self.t24 = np.arange(WINDOW_H - 1, T - 1)
        W = np.zeros((len(self.t24), K, K))
        for h in self.h:
            ok = self.t24 + h <= T - 1
            if not ok.any():
                raise ValueError(f"training too short for horizon {h}")
            W[ok] += Q[self.t24[ok] + h] / (len(self.h) * ok.sum())
        self.W = W

    def _CS(self, C, S):
        return C[list(S)].sum(0)

    def grow(self, p, S):
        return -_tr_inv(np.eye(self.K) + self._CS(self.C, S)[-1], self.Qg)

    def w24(self, p, S):
        CS, t = self._CS(self.C, S), self.t24
        return -_tr_inv(np.eye(self.K) + CS[t + 1] - CS[t + 1 - WINDOW_H], self.W)

    def grow_rob(self, p, S):
        CR = self._CS(self.Crob, S)
        M = np.eye(self.K) + self._CS(self.C, S)[-1] + CR[self.b1] - CR[self.b0]
        return -_tr_inv(M, self.Qb) / len(self.Qb)

    def w24_rob(self, p, S):
        CR = self._CS(self.Crob, S)
        return -_tr_inv(np.eye(self.K) + CR[self.b1] - CR[self.b0], self.Qb) / len(self.Qb)

    def value(self, name):
        return {"fc_grow": self.grow, "fc_w24": self.w24, "fc_grow_rob": self.grow_rob, "fc_w24_rob": self.w24_rob}[name]


def make_fc_design(name, p, obj: Objective, k, starts=None):
    """Greedy + 1-swap on the named objective (as E07's dopt); with `starts` ({label: sites}), also 1-swap from each start
    and keep the best objective value (64^2 addendum A4; ties keep the earlier start, greedy first). Returns (sites, info):
    info['start'] names the winning start, info['greedy_sites'] / 'greedy_refined_value' keep the single-start result."""
    v = obj.value(name)
    g = P.greedy(p, k, v)
    S, val, nsw = P.swap_refine(p, g, v)
    info = dict(greedy_value=v(p, g), refined_value=val, n_swaps=nsw, start="greedy", greedy_refined_value=val,
                greedy_sites=" ".join(map(str, S)))
    for lab, S0 in (starts or {}).items():
        S1, v1, n1 = P.swap_refine(p, S0, v)
        info[f"refined_value_{lab}"] = v1
        if v1 > info["refined_value"] + 1e-12 * abs(info["refined_value"]):
            S, info["refined_value"], info["n_swaps"], info["start"] = S1, v1, n1, lab
    assert len(set(S)) == k and set(S) <= set(p.cand.tolist())
    return [int(i) for i in S], info


# ---------------------------------------------------------------- estimators and scorer
def estimate(A, z, issue, window=None):
    """MAP of a (a_ref units, prior N(0, I)) at each issue time from the `window` hours ending there (None = growing window,
    all hours 0..t). A (k, hours, K) whitened sensitivities; z (k, hours) or (k, hours, c) whitened anomalies. Reads no row
    after an issue time. window = 24 equals cross_objective.estimate_series."""
    issue = np.asarray(issue, int)
    K = A.shape[2]
    cP = np.concatenate([np.zeros((1, K, K)), np.cumsum(np.einsum("shk,shl->hkl", A, A), 0)])
    cb = np.cumsum(np.einsum("shk,sh...->hk...", A, z), 0)
    cb = np.concatenate([np.zeros((1,) + cb.shape[1:]), cb])
    lo = np.zeros_like(issue) if window is None else issue + 1 - window
    if lo.min() < 0:
        raise ValueError("window starts before hour 0")
    b = cb[issue + 1] - cb[lo]
    u = np.linalg.solve(np.eye(K) + cP[issue + 1] - cP[lo], b if b.ndim == 3 else b[..., None])
    return u if b.ndim == 3 else u[..., 0]


def score(u_hat, Fr, tf, issue, a_true_ref, last, horizons=HORIZONS):
    """Held-out metrics of one estimate series (the only place truth enters). u_hat (n_issue, K) a_ref units; Fr (K, hours, N)
    a_ref-scaled unit responses; tf (hours, N) truth anomaly; forecasts only to hours <= last (test end)."""
    out = {}
    for h in horizons:
        ok = issue + h <= last
        e = np.einsum("tk,ktn->tn", u_hat[ok], Fr[:, issue[ok] + h]) - tf[issue[ok] + h]
        out[f"forecast_rmse_h{h}"] = float(np.sqrt(np.mean(e ** 2)))
        out[f"n_issue_h{h}"] = int(ok.sum())
    e = np.einsum("tk,ktn->tn", u_hat, Fr[:, issue]) - tf[issue]
    out["analysis_rmse"] = float(np.sqrt(np.mean(e ** 2)))
    out["source_rmse"] = float(np.sqrt(np.mean((u_hat - a_true_ref) ** 2)))
    return out


# ---------------------------------------------------------------- paired statistics (design note §7)
def paired_summary(rho, margin=0.05):
    """Median of the paired relative differences rho (> 0 favours the design), distribution-free 95 % CI, sign test, verdict."""
    rho = np.asarray(rho, float)
    lo, hi = X.df_median_ci(rho)
    pos, neg, tie, p = X.sign_test(rho)
    verdict = "better" if lo > 0 else "non-inferior" if lo > -margin else "worse" if hi < 0 else "unresolved"
    return dict(n=len(rho), median=float(np.median(rho)), ci_lo=lo, ci_hi=hi, wins=pos, losses=neg, ties=tie, sign_p=p,
                verdict=verdict)


def cluster_summary(rho, cluster, margin=0.05):
    """paired_summary of the per-cluster means of rho (64^2 addendum A2: cluster = (K, seed), whose veering/mixed
    scenarios share sources, strengths, noise streams and test winds)."""
    rho = np.asarray(rho, float)
    _, g = np.unique(np.asarray(cluster).reshape(len(rho), -1), axis=0, return_inverse=True)
    return paired_summary(np.bincount(g.ravel(), rho) / np.bincount(g.ravel()), margin)

"""E15 robust (scenario-distribution) placement: per-scenario data-free forecast losses over a DECLARED ensemble of 24 h
wind blocks x strength priors, aggregated by mean / CVaR / worst case. Design: docs/research/robust-placement-design.md.
Contract: tests/test_robust_placement.py.

Only design-prior blocks (solved from declared winds, never from the scenario calendar) and E07 training information enter.
Reuses forecast_design (E11) for blocks, field weights, strength prior and the multi-start greedy + 1-swap search.
"""
from __future__ import annotations

import copy

import numpy as np

from pravah import placement as P
from pravah.experiments import cross_objective as X
from pravah.experiments import forecast_design as FD

ALPHA = 0.8
SPEEDS = (1.5, 3.0, 5.0)
TEST_DIRS = (0, 4)  # N (front of stagnation_then_front) and S (steady southerly): the test split's own directions
# wind blocks: ("steady", d, speed) or ("veer", d, speed) = 12 h from d then 12 h from d + 1
ENSEMBLES = {
    "noSN6": [("steady", d, s) for d in (1, 2, 3, 5, 6, 7) for s in SPEEDS] + [("veer", d, 3.0) for d in (1, 2, 5, 6)],
    "train3": [("steady", d, s) for d in (2, 6, 7) for s in SPEEDS] + [("veer", 6, 3.0)],
}
EXTRA = [("steady", 0, 3.0), ("steady", 4, 3.0)]  # only for the E11 all8 comparator (in-sample)
CRITERIA = ("mean", "cvar", "worst")
MEMORIES = ("w24", "full")


def block_winds(kind, d, speed):
    """(BLOCK_H, 2) hourly wind of one declared block (make_winds convention, FROM d * 45 deg)."""
    w = FD.design_winds(speed=speed)
    if kind == "steady":
        return w[d * FD.BLOCK_H:(d + 1) * FD.BLOCK_H].copy()
    if kind == "veer":
        h = FD.BLOCK_H // 2
        e = (d + 1) % FD.N_DIRS
        return np.concatenate([w[d * FD.BLOCK_H:d * FD.BLOCK_H + h], w[e * FD.BLOCK_H:e * FD.BLOCK_H + h]])
    raise ValueError(f"unknown block kind {kind!r}")


def block_field(cfg, xy, block, spinup_h):
    """Unit response (BLOCK_H, nx, ny) of the source at xy to one declared block after `spinup_h` h of spin-up (which replays
    the block's first hours: steady d for a veer). Does not modify cfg; equals FD.prior_block for steady 3 m/s blocks."""
    c = copy.deepcopy(cfg)
    c.model.spinup_h = spinup_h
    return X.source_field(c, xy, block_winds(*block), FD.BLOCK_H)


def cvar(L, alpha=ALPHA, axis=-1):
    """Equal-weight discrete CVaR_alpha (upper tail) along `axis`: mean of the worst (1 - alpha) fraction, with a fractional
    weight on the boundary scenario (Rockafellar-Uryasev). alpha = 0 is the mean; alpha -> 1 the max."""
    L = -np.sort(-np.asarray(L, float), axis=axis)
    n = L.shape[axis]
    m = (1 - alpha) * n
    if not 0 < m <= n:
        raise ValueError("alpha must lie in [0, 1)")
    j = int(np.floor(m + 1e-12))
    w = np.zeros(n)
    w[:j] = 1
    if j < n:
        w[j] = m - j
    return np.tensordot(np.moveaxis(L, axis, -1), w, axes=([-1], [0])) / m


AGG = {"mean": lambda L: float(np.mean(L)), "cvar": lambda L: float(cvar(L)), "worst": lambda L: float(np.max(L))}


class Ensemble:
    """Declared scenarios: site responses G (n_sites, n_s * BLOCK_H, K) per unit source, block field weights Q (n_s, K, K)
    and strength second moments m2 (n_s, K, K). Built from design-prior blocks only."""

    def __init__(self, G, Q, m2, labels):
        self.G, self.Q, self.m2, self.labels = np.asarray(G, float), np.asarray(Q, float), np.asarray(m2, float), list(labels)
        assert len(self.Q) == len(self.m2) == len(self.labels) == self.G.shape[1] // FD.BLOCK_H

    @classmethod
    def build(cls, blocks, site_fn, a_ref, m2_declared):
        """blocks: [(label, (K, BLOCK_H, nx, ny))] unit responses; site_fn maps (K, hours, nx, ny) -> (n_sites, hours, K).
        Each wind block appears twice: strength prior N(0, I) and the declared m2."""
        F = np.concatenate([b for _, b in blocks], 1)
        G = site_fn(F)
        Qb = FD.field_weights(F, a_ref).reshape(len(blocks), FD.BLOCK_H, F.shape[0], F.shape[0]).mean(1)
        K = F.shape[0]
        eye = np.eye(K)
        return cls(np.concatenate([G, G], 1), np.concatenate([Qb, Qb]),
                   np.concatenate([np.broadcast_to(eye, Qb.shape), np.broadcast_to(m2_declared, Qb.shape)]),
                   [(*lab, "unit") for lab, _ in blocks] + [(*lab, "m2") for lab, _ in blocks])


class RobustObjective:
    """Per-scenario normalised losses L_s(S) (design note §2) and their aggregates as larger-is-better values."""

    def __init__(self, p: P.Problem, ens: Ensemble, memory):
        if memory not in MEMORIES:
            raise ValueError(memory)
        sig = np.broadcast_to(np.asarray(p.sigma, float), (len(p.G3),))
        A = ens.G * (p.a_ref / sig)[:, None, None]
        n_s, K = len(ens.Q), ens.Q.shape[1]
        A = A.reshape(len(A), n_s, FD.BLOCK_H, K)
        self.I = np.einsum("sbhk,sbhl->sbkl", A, A)                 # (n_sites, n_s, K, K) block information per site
        self.Itr = p.F if memory == "full" else np.zeros((len(A), K, K))  # training information (E07 problem)
        self.Q, self.m2, self.K = ens.Q, ens.m2, K
        self.L0 = np.einsum("skl,slk->s", ens.Q, ens.m2)

    def losses(self, S):
        S = list(S)
        M = self.I[S].sum(0) + self.Itr[S].sum(0)
        Sig = np.linalg.inv(np.eye(self.K) + M)
        return np.einsum("skl,slm,smn,snk->s", self.Q, Sig, self.m2 + M, Sig) / self.L0

    def value(self, criterion):
        agg = AGG[criterion]
        return lambda p, S: -agg(self.losses(S))



def make_robust_design(p, obj: RobustObjective, criterion, k, starts=None):
    """E11's search (greedy + 1-swap, plus 1-swap from each start; best value kept) on a robust criterion."""
    return FD.make_fc_design(criterion, p, obj, k, starts)

"""Gaussian linear-Bayes surrogate for dynamic sensor design (pure NumPy).

Model: y = A_S theta + noise, noise iid with std ``sigma``; prior theta ~ N(theta0, P), P = diag(prior_var).
Posterior covariance: Sigma(S) = (A_S^T A_S / sigma^2 + P^{-1})^{-1}.
Surrogate risk: J(S) = tr(M Sigma(S) M^T) with M = W^{1/2} G, W the population-time weights (see
``population_weights``). The caller forms M; this module only sees the matrix.

Caveat: J is a SURROGATE. It ignores the nonnegativity of emissions, so it is the risk of an unconstrained
linear-Bayes estimator, not of the constrained estimator the inverse actually uses. Use it to rank layouts,
then check the disagreement on the real estimator (DYN-05).

Computation: the Woodbury identity gives tr(Sigma Q) = tr(P Q) - tr(K^{-1} B_S P Q P B_S^T) with
K = I + B_S P B_S^T, B = A/sigma and Q = M^T M. All candidate-pair blocks are precomputed once, so a subset
costs an r x r solve, r = |S| * n_obs. Exact; no n_theta x n_theta inverse per subset.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


def _trapezoid_weights(times: ArrayLike) -> FloatArray:
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size == 0 or np.any(np.diff(t) <= 0):
        raise ValueError("times must be 1-D and strictly increasing")
    if t.size == 1:
        return np.ones(1)
    half = np.diff(t) / 2.0
    w = np.zeros(t.size)
    w[:-1] += half
    w[1:] += half
    return w


def population_weights(counts: ArrayLike, times: ArrayLike | None = None) -> FloatArray:
    """Normalised weights: trapezoid time weight x population count, summing to 1.

    ``counts`` has time on axis 0 (shape (n_t, ...)) or is 1-D with one entry per time when ``times`` is given.
    Without ``times`` the time weight is uniform. Zero counts get exactly zero weight.
    """
    c = np.asarray(counts, dtype=float)
    if c.ndim == 0 or not np.all(np.isfinite(c)) or np.any(c < 0):
        raise ValueError("counts must be finite and non-negative, with a leading time axis")
    if not np.any(c > 0):
        raise ValueError("all population counts are zero")
    tw = np.ones(c.shape[0]) if times is None else _trapezoid_weights(times)
    if tw.shape[0] != c.shape[0]:
        raise ValueError("times length must match counts axis 0")
    w = c * tw.reshape((-1,) + (1,) * (c.ndim - 1))
    return w / w.sum()


@dataclass(frozen=True)
class ExhaustiveResult:
    best_subset: tuple[int, ...]
    best_risk: float
    median_risk: float
    worst_risk: float
    n_evaluated: int
    risks_sorted: FloatArray | None = None


@dataclass(frozen=True)
class DesignProblem:
    """A_cand (n_cand, n_obs, n_theta), M (n_out, n_theta) with W^{1/2} applied, prior_var (n_theta,), sigma > 0."""

    A_cand: FloatArray
    M: FloatArray
    prior_var: FloatArray
    sigma: float
    # Precomputed candidate-pair blocks; filled in __post_init__.
    _n_obs: int = field(init=False, repr=False, compare=False)
    _gram_p: FloatArray = field(init=False, repr=False, compare=False)  # B P B^T (N x N)
    _cross_q: FloatArray = field(init=False, repr=False, compare=False)  # B P Q P B^T (N x N)
    _cross_i: FloatArray = field(init=False, repr=False, compare=False)  # B P P B^T (information, Q = I)
    _trace_pq: float = field(init=False, repr=False, compare=False)
    _trace_p: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        A = np.asarray(self.A_cand, dtype=float)
        M = np.asarray(self.M, dtype=float)
        pv = np.asarray(self.prior_var, dtype=float)
        if A.ndim != 3 or M.ndim != 2 or pv.ndim != 1 or not A.shape[2] == M.shape[1] == pv.shape[0]:
            raise ValueError("expected A_cand (n_cand, n_obs, n_theta), M (n_out, n_theta), prior_var (n_theta,)")
        if A.shape[0] == 0 or A.shape[1] == 0:
            raise ValueError("A_cand must have at least one candidate and one observation per sensor")
        if not (np.all(pv > 0) and np.all(np.isfinite(pv))) or not (self.sigma > 0):
            raise ValueError("prior_var and sigma must be positive")
        n_cand, n_obs, n_theta = A.shape
        b = A.reshape(n_cand * n_obs, n_theta) / float(self.sigma)  # B = A / sigma, one row per candidate observation
        bp = b * pv  # B P
        q = M.T @ M
        object.__setattr__(self, "A_cand", A)
        object.__setattr__(self, "M", M)
        object.__setattr__(self, "prior_var", pv)
        object.__setattr__(self, "_n_obs", n_obs)
        object.__setattr__(self, "_gram_p", bp @ b.T)
        object.__setattr__(self, "_cross_q", bp @ q @ bp.T)
        object.__setattr__(self, "_cross_i", bp @ bp.T)
        object.__setattr__(self, "_trace_pq", float(np.sum(pv * np.diag(q))))
        object.__setattr__(self, "_trace_p", float(np.sum(pv)))

    @property
    def n_cand(self) -> int:
        return int(self.A_cand.shape[0])

    def _check_subsets(self, subsets: ArrayLike) -> IntArray:
        s = np.asarray(subsets, dtype=np.intp)
        if s.ndim != 2:
            raise ValueError("subsets must be 2-D (n_subsets, k)")
        if s.size and (s.min() < 0 or s.max() >= self.n_cand):
            raise ValueError("sensor index out of range")
        if s.shape[1] > 1 and np.any(np.diff(np.sort(s, axis=1), axis=1) == 0):
            raise ValueError("subset contains a repeated sensor")
        return s

    def _rows(self, subsets: IntArray, gram: FloatArray, cross: FloatArray, trace_term: float) -> FloatArray:
        if subsets.shape[1] == 0:
            return np.full(subsets.shape[0], trace_term)
        rows = (subsets[:, :, None] * self._n_obs + np.arange(self._n_obs)).reshape(subsets.shape[0], -1)
        ix = rows[:, :, None], rows[:, None, :]
        k_mat = np.eye(rows.shape[1]) + gram[ix]
        solved = np.linalg.solve(k_mat, cross[ix])
        return trace_term - np.einsum("bii->b", solved)

    def risk(self, subset: Sequence[int]) -> float:
        """Surrogate J(S) = tr(M Sigma_post(S) M^T)."""
        return float(self.risk_batch(np.asarray(subset, dtype=np.intp).reshape(1, -1))[0])

    def risk_batch(self, subsets: ArrayLike, chunk: int = 4096) -> FloatArray:
        """Surrogate risk for each row of ``subsets`` (n_subsets, k); evaluated in blocks of ``chunk`` rows."""
        if chunk < 1:
            raise ValueError("chunk must be >= 1")
        s = self._check_subsets(subsets)
        out = np.empty(s.shape[0])
        for lo in range(0, s.shape[0], chunk):
            out[lo : lo + chunk] = self._rows(s[lo : lo + chunk], self._gram_p, self._cross_q, self._trace_pq)
        return out

    def information_risk(self, subset: Sequence[int]) -> float:
        """A-optimal criterion tr(Sigma_post(S)) (M = identity), for source identification."""
        s = self._check_subsets(np.asarray(subset, dtype=np.intp).reshape(1, -1))
        return float(self._rows(s, self._gram_p, self._cross_i, self._trace_p)[0])

    def _check_k(self, k: int) -> None:
        if not 1 <= k <= self.n_cand:
            raise ValueError(f"k must be in [1, {self.n_cand}]")

    def exhaustive(self, k: int, chunk: int = 65536, return_all: bool = False) -> ExhaustiveResult:
        """Global minimum over all k-subsets (itertools.combinations, evaluated in blocks of ``chunk``).

        Stores one float per subset to report median and worst case; O(C(n_cand, k)) memory.
        """
        self._check_k(k)
        best_set: tuple[int, ...] = ()
        best_val = np.inf
        parts: list[FloatArray] = []
        it = itertools.combinations(range(self.n_cand), k)
        while True:
            block = list(itertools.islice(it, chunk))
            if not block:
                break
            vals = self.risk_batch(np.asarray(block, dtype=np.intp), chunk=chunk)
            parts.append(vals)
            j = int(np.argmin(vals))
            if vals[j] < best_val:
                best_val = float(vals[j])
                best_set = block[j]
        all_vals = np.concatenate(parts)
        return ExhaustiveResult(
            best_subset=tuple(int(x) for x in best_set),
            best_risk=best_val,
            median_risk=float(np.median(all_vals)),
            worst_risk=float(all_vals.max()),
            n_evaluated=int(all_vals.size),
            risks_sorted=np.sort(all_vals) if return_all else None,
        )

    def greedy(self, k: int) -> tuple[tuple[int, ...], float]:
        """Forward greedy: repeatedly add the sensor that most reduces the surrogate risk."""
        self._check_k(k)
        chosen: list[int] = []
        for _ in range(k):
            rest = [c for c in range(self.n_cand) if c not in chosen]
            trial = np.array([chosen + [c] for c in rest], dtype=np.intp).reshape(len(rest), len(chosen) + 1)
            chosen.append(rest[int(np.argmin(self.risk_batch(trial)))])
        subset = tuple(sorted(chosen))
        return subset, self.risk(subset)

    def swap_refine(self, subset: Sequence[int], max_iter: int = 100) -> tuple[tuple[int, ...], float]:
        """Best-improvement single swaps. Risk never increases; stops at a local minimum or after max_iter."""
        s = tuple(sorted(int(x) for x in subset))
        cur = self.risk(s)
        for _ in range(max_iter):
            outside = [c for c in range(self.n_cand) if c not in s]
            if not s or not outside:
                break
            neigh = np.array(
                [[x for x in s if x != o] + [i] for o in s for i in outside], dtype=np.intp
            ).reshape(len(s) * len(outside), len(s))
            vals = self.risk_batch(neigh)
            j = int(np.argmin(vals))
            if vals[j] >= cur:
                break
            cur = float(vals[j])
            s = tuple(sorted(int(x) for x in neigh[j]))
        return s, cur

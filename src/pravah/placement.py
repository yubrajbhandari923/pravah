"""Sensor placement strategies and criteria (E06) behind one deterministic interface.

Definitions: docs/research/placement-objectives.md. Contract: tests/test_placement.py.
Every design returns k distinct global site ids taken from `Problem.cand` (the exclusion mask is applied by whoever
builds `cand`) whose pairwise distances are >= `Problem.min_sep`. Ties: the earliest id in `cand` order wins, because a
candidate replaces the incumbent only if it beats it by more than TIE_TOL. Criteria are written so that larger is better,
except `coherence` (smaller is better). Units: G3 is per unit source (ug m^-3 per ug s^-1); a_ref and sigma make it
dimensionless (rows of A = a_ref * G / sigma are signal in noise-sd units for a reference-sized source change).
"""
import itertools
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

TIE_TOL = 1e-12


@dataclass
class Problem:
    xy: np.ndarray                  # (n_sites, 2) m, every site (ids index this)
    cand: np.ndarray                # candidate ids after the exclusion mask, in tie-break order
    min_sep: float                  # m
    a_ref: float = 1.0              # prior sd / reference source rate, ug s^-1
    sigma: object = 1.0             # noise sd, scalar or (n_sites,)
    G3: np.ndarray = None           # (n_sites, hours, K) training-window sensitivities
    snapshots: np.ndarray = None    # (n_sites, T) training-split field snapshots
    source_xy: np.ndarray = None    # (K, 2) known source locations
    Q: np.ndarray = None            # (K, K) field-weight matrix for `fieldvar` (prior-scaled field responses, see there)
    _pod: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.xy, self.cand = np.asarray(self.xy, float), np.asarray(self.cand, int)
        if len(set(self.cand.tolist())) != self.cand.size:
            raise ValueError("cand must be unique")

    def feasible(self, i, S):
        return i not in S and all(np.hypot(*(self.xy[i] - self.xy[j])) >= self.min_sep for j in S)

    def pod(self, r):
        """Cached pod_basis(snapshots, r)."""
        if r not in self._pod:
            self._pod[r] = pod_basis(self.snapshots, r)
        return self._pod[r]

    @cached_property
    def r_num(self):
        """Numerical rank of the centred snapshots (singular values > 1e-10 x the largest)."""
        X = np.asarray(self.snapshots, float)
        s = np.linalg.svd(X - X.mean(1, keepdims=True), compute_uv=False)
        return int((s > 1e-10 * s[0]).sum()) if s[0] > 0 else 0

    @cached_property
    def r_eff(self):
        """Number of POD modes that some candidate would see above noise: max_{i in cand} s_j |U_ij| / (sqrt(T) sigma_i) >= 1
        (rms over the training hours of mode j's contribution to that site's reading, in noise sd units)."""
        X = np.asarray(self.snapshots, float)
        U, s, _ = np.linalg.svd(X - X.mean(1, keepdims=True), full_matrices=False)
        sig = np.broadcast_to(np.asarray(self.sigma, float), (len(X),))[self.cand]
        amp = (s * np.abs(U[self.cand]) / sig[:, None]).max(0) / np.sqrt(X.shape[1])
        return int((amp >= 1).sum())

    @cached_property
    def A3(self):
        """(n_sites, hours, K) noise-whitened, a_ref-scaled sensitivities."""
        sig = np.broadcast_to(np.asarray(self.sigma, float), (len(self.G3),))
        if not (sig > 0).all():
            raise ValueError("sigma must be positive")
        return np.asarray(self.G3, float) * (self.a_ref / sig)[:, None, None]

    @cached_property
    def F(self):
        """Per-site information A_i^T A_i, (n_sites, K, K). Sites add information (independent noise across sites and hours)."""
        return np.einsum("shk,shl->skl", self.A3, self.A3)

    @cached_property
    def snr(self):
        """Largest singular value of A_i: >= 1 iff the site alone has practical rank >= 1."""
        return np.array([np.linalg.norm(a, 2) for a in self.A3])


# ---- criteria (functions of a site set) -----------------------------------------------------------------------------

def _post(p, S):
    K = p.G3.shape[2]
    return np.eye(K) + (p.F[list(S)].sum(0) if len(S) else 0)


def dopt(p, S):
    """Bayesian D: logdet(I + A_S^T A_S) = 2 x mutual information (nats) between sources (prior N(0, a_ref^2 I)) and data."""
    return float(np.linalg.slogdet(_post(p, S))[1])


def aopt(p, S):
    """Bayesian A: 1 - tr((I + A_S^T A_S)^-1) / K, mean fractional posterior-variance reduction (0 = prior only)."""
    return float(1 - np.trace(np.linalg.inv(_post(p, S))) / p.G3.shape[2])


def eopt(p, S):
    """Bayesian E: log lambda_min(I + A_S^T A_S), the worst-informed source direction (evaluation only)."""
    return float(np.log(np.linalg.eigvalsh(_post(p, S))[0]))


def fieldvar(p, S):
    """Posterior field-variance proxy, negated so larger is better: -tr(M(S)^-1 Q). With Q = mean over field points and hours
    of a_ref^2 g g^T (g = per-unit-source response at the point) this is minus the mean posterior variance of the field
    (prior units: the prior gives -tr(Q)). E07; a Q-weighted A-optimality."""
    return float(-np.trace(np.linalg.solve(_post(p, S), p.Q)))


def coherence(p, S):
    """max |cos| between the footprint vectors vec(G3[i]) of distinct sites in S (1 if a footprint is zero)."""
    f = p.G3[list(S)].reshape(len(S), -1)
    n = np.linalg.norm(f, axis=1)
    if len(S) < 2:
        return 0.0
    if (n == 0).any():
        return 1.0
    C = np.abs((f / n[:, None]) @ (f / n[:, None]).T)
    return float((C - np.eye(len(S)) * C.diagonal()).max())


def pod_basis(snapshots, r):
    """First r left singular vectors of the time-centred snapshot matrix (n_sites, T)."""
    X = np.asarray(snapshots, float)
    if not np.isfinite(X).all():
        raise ValueError("snapshots must be finite")
    U, s, _ = np.linalg.svd(X - X.mean(1, keepdims=True), full_matrices=False)
    if r > (s > s[0] * 1e-12).sum():
        raise ValueError(f"only {(s > s[0] * 1e-12).sum()} nonzero POD modes, asked for {r}")
    return U[:, :r]


def podqr_value(p, S):
    """log det(Psi_S^T Psi_S) with r = min(|S|, r_num) POD modes (the quantity QR pivoting greedily maximises when
    |S| <= r_num; -inf if singular, nan if there is no mode). Noise-unaware: a purely algorithmic objective."""
    r = min(len(S), p.r_num)
    if r == 0:
        return np.nan
    Psi = p.pod(r)[list(S)]
    sgn, v = np.linalg.slogdet(Psi.T @ Psi)
    return float(v) if sgn > 0 else -np.inf


# ---- designs --------------------------------------------------------------------------------------------------------

def greedy(p, k, value):
    """Add, k times, the feasible candidate maximising value(S + [i]) (first in cand order on ties)."""
    S = []
    for _ in range(k):
        best, bi = -np.inf, None
        for i in p.cand:
            if p.feasible(i, S):
                v = value(p, S + [int(i)])
                if v > best + TIE_TOL:
                    best, bi = v, int(i)
        if bi is None:
            raise ValueError(f"cannot place {k} sites with min_sep={p.min_sep}")
        S.append(bi)
    return S


def swap_refine(p, S, value, max_pass=50, tol=1e-9):
    """Best-improvement 1-swap local search from S (replace one site by a feasible candidate) while value improves by > tol.
    Returns (S, value, n_swaps). A local optimum only; with binding spacing greedy alone has no guarantee (gate-06 F3)."""
    S = [int(i) for i in S]
    v, n = value(p, S), 0
    for _ in range(max_pass):
        best, move = v + tol, None
        for a in range(len(S)):
            rest = S[:a] + S[a + 1:]
            for i in p.cand:
                if int(i) != S[a] and p.feasible(int(i), rest):
                    w = value(p, rest[:a] + [int(i)] + rest[a:])
                    if w > best:
                        best, move = w, (a, int(i))
        if move is None:
            break
        S[move[0]], v, n = move[1], best, n + 1
    return S, v, n


def _in_order(p, order, k, what):
    S = []
    for i in order:
        if p.feasible(int(i), S):
            S.append(int(i))
        if len(S) == k:
            return S
    raise ValueError(f"cannot place {k} sites ({what}) with min_sep={p.min_sep}")


def random_design(p, k, seed=0):
    """Seeded permutation of cand, first feasible k (uses no field or source information)."""
    return _in_order(p, np.random.default_rng(seed).permutation(p.cand), k, "random")


def uniform_design(p, k):
    """Targets at the cell centres of a near-square nx x ny grid (nx = ceil(sqrt k)) over the candidates' bounding box,
    row by row; each takes the nearest feasible candidate. No field or source information."""
    nx = int(np.ceil(np.sqrt(k)))
    ny = int(np.ceil(k / nx))
    lo, hi = p.xy[p.cand].min(0), p.xy[p.cand].max(0)
    tx, ty = lo[0] + (np.arange(nx) + .5) * (hi[0] - lo[0]) / nx, lo[1] + (np.arange(ny) + .5) * (hi[1] - lo[1]) / ny
    S = []
    for t in [(x, y) for y in ty for x in tx][:k]:
        d = np.hypot(*(p.xy[p.cand] - t).T)
        S.append(next((int(p.cand[j]) for j in np.argsort(d, kind="stable") if p.feasible(int(p.cand[j]), S)), None))
        if S[-1] is None:
            raise ValueError(f"cannot place {k} sites (uniform) with min_sep={p.min_sep}")
    return S


def coverage_design(p, k):
    """Farthest-point sampling: start farthest from the candidates' centroid, then maximise the distance to the design."""
    c = p.xy[p.cand]
    S = [int(p.cand[np.argmax(np.hypot(*(c - c.mean(0)).T))])]
    while len(S) < k:
        d = np.min([np.hypot(*(c - p.xy[j]).T) for j in S], 0)
        d[[not p.feasible(int(i), S) for i in p.cand]] = -1
        if d.max() < 0:
            raise ValueError(f"cannot place {k} sites (coverage) with min_sep={p.min_sep}")
        S.append(int(p.cand[np.argmax(d)]))
    return S


def hotspot_design(p, k):
    """Top-k feasible candidates by training-mean anomaly under equal expected strengths, sum_{h,k} G3[i, h, k] (the ranking
    does not depend on the common prior mean). Uses known locations (through G3) and the prior mean only."""
    score = np.asarray(p.G3, float)[p.cand].sum((1, 2))
    return _in_order(p, p.cand[np.argsort(-score, kind="stable")], k, "hotspot")


def source_near_design(p, k):
    """Round robin over the KNOWN source locations: each takes its nearest feasible candidate (the exclusion mask already
    removes sites within one footprint sigma of a source)."""
    if p.source_xy is None or not len(p.source_xy):
        raise ValueError("source_near needs source_xy")
    S = []
    for s in itertools.islice(itertools.cycle(np.asarray(p.source_xy, float)), 10 * k * len(p.source_xy)):
        d = np.hypot(*(p.xy[p.cand] - s).T)
        nxt = next((int(p.cand[j]) for j in np.argsort(d, kind="stable") if p.feasible(int(p.cand[j]), S)), None)
        if nxt is not None:
            S.append(nxt)
        if len(S) == k:
            return S
    raise ValueError(f"cannot place {k} sites (source_near) with min_sep={p.min_sep}")


def diversity_design(p, k, tau=1.0):
    """Footprint diversity: maximise lexicographically (snr_i >= tau, -max_{j in S} |cos(f_i, f_j)|, snr_i).
    So: never prefer a sub-noise site while an above-noise one is feasible; among those, add the footprint least coherent with
    the design; break remaining ties by signal. The first site is the strongest one."""
    f = p.G3.reshape(len(p.G3), -1)
    n = np.linalg.norm(f, axis=1)
    u = np.divide(f, n[:, None], out=np.zeros_like(f), where=n[:, None] > 0)
    S = []
    for _ in range(k):
        best, bi = None, None
        for i in p.cand:
            if not p.feasible(int(i), S):
                continue
            mc = max((abs(u[i] @ u[j]) for j in S), default=0.0) if n[i] > 0 else 1.0
            key = (p.snr[i] >= tau, -mc, p.snr[i])
            if best is None or _lex_gt(key, best):
                best, bi = key, int(i)
        if bi is None:
            raise ValueError(f"cannot place {k} sites (diversity) with min_sep={p.min_sep}")
        S.append(bi)
    return S


def _lex_gt(a, b):
    for x, y in zip(a, b):
        if x > y + TIE_TOL:
            return True
        if x < y - TIE_TOL:
            return False
    return False


def qr_select(Psi, p, k):
    """Column-pivoted QR (Businger-Golub) on Psi[cand]^T restricted to feasible pivots: each step takes the feasible
    candidate with the largest residual norm, then projects it out. Without a binding min_sep this is the pivot order of
    scipy.linalg.qr(Psi[cand].T, pivoting=True) (up to exact ties / rounding)."""
    R = np.array(Psi[p.cand].T, float)  # (r, n_cand)
    S = []
    for _ in range(k):
        nrm = np.linalg.norm(R, axis=0)
        best, bj = -np.inf, None
        for j, i in enumerate(p.cand):
            if p.feasible(int(i), S) and nrm[j] > best + TIE_TOL:
                best, bj = nrm[j], j
        if bj is None:
            raise ValueError(f"cannot place {k} sites (podqr) with min_sep={p.min_sep}")
        S.append(int(p.cand[bj]))
        if best > 0:
            q = R[:, bj] / best
            for _ in range(2):  # re-orthogonalise once (classical Gram-Schmidt twice)
                R -= np.outer(q, q @ R)
    return S


def podqr_design(p, k, noise_aware=True):
    """POD/QR (Manohar et al. 2018 style, as in PySensors' QR optimizer): r = k modes of the TRAINING snapshots, QR pivots.
    Undefined (ValueError) when k > r_eff (noise_aware) or k > r_num: the extra modes would be below noise / numerical dust."""
    r, what = (p.r_eff, "r_eff above-noise") if noise_aware else (p.r_num, "r_num numerical")
    if k > r:
        raise ValueError(f"podqr undefined: k={k} > {what} POD modes ({r})")
    return qr_select(p.pod(k), p, k)


STRATEGIES = {
    "uniform": uniform_design, "random": random_design, "coverage": coverage_design, "source_near": source_near_design,
    "dopt": lambda p, k: greedy(p, k, dopt), "aopt": lambda p, k: greedy(p, k, aopt),
    "diversity": diversity_design, "podqr": podqr_design,
    "podqr_num": lambda p, k: podqr_design(p, k, noise_aware=False),  # exploratory: ignores the noise level
}


def design(name, p, k, seed=0):
    """Common interface: k distinct feasible global site ids. `seed` only affects 'random'."""
    S = random_design(p, k, seed) if name == "random" else STRATEGIES[name](p, k)
    assert len(set(S)) == k and all(p.feasible(i, [j for j in S if j != i]) for i in S) and set(S) <= set(p.cand.tolist())
    return S


def brute_force(p, k, value, maximize=True, ids=None):
    """Exhaustive optimum of value over feasible k-subsets of ids (default cand): (best value, best set, n_feasible).
    First subset in lexicographic cand order wins ties. Only for tiny candidate sets."""
    ids = [int(i) for i in (p.cand if ids is None else ids)]
    far = {(a, b): np.hypot(*(p.xy[a] - p.xy[b])) >= p.min_sep for a, b in itertools.combinations(ids, 2)}
    sgn, best, bS, n = (1 if maximize else -1), -np.inf, None, 0
    for S in itertools.combinations(ids, k):
        if all(far[ab] for ab in itertools.combinations(S, 2)):
            n += 1
            v = sgn * value(p, list(S))
            if v > best + TIE_TOL:
                best, bS = v, list(S)
    return (sgn * best if bS is not None else np.nan), bS, n


# ---- scenario adapters (training split only) ------------------------------------------------------------------------

def training_snapshots(sc):
    """Noise-free field at every candidate site over the TRAIN split hours only, (n_sites, T)."""
    return np.asarray(sc.truth.site_truth)[:, sc.split_hours("train")]


def problem_from_scenario(sc, G3, cand=None):
    """Problem for a benchmark Scenario: admissible candidates, preregistered sigma and a_ref, known source locations,
    training snapshots. G3 (n_sites, hours, K) must come from training-window hours (e.g. scenario_response_matrix(sc, h)
    with h <= number of train hours; that matrix starts at hour 0 = first train hour)."""
    if G3.shape[1] > len(sc.split_hours("train")) or sc.split_hours("train")[0] != 0:
        raise ValueError("G3 must cover training hours only")
    return Problem(xy=sc.site_xy, cand=sc.admissible_sites() if cand is None else cand,
                   min_sep=sc.cfg.design.min_sensor_separation_m, a_ref=sc.cfg.sources.reference_rate_ug_s,
                   sigma=sc.noise_sigma, G3=G3, snapshots=training_snapshots(sc), source_xy=sc.truth.source_xy)

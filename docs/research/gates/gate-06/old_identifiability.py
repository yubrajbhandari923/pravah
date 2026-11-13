"""Practical rank, null-direction counterexamples and a greedy Fisher design (E04). Contract: tests/test_identifiability.py.
Synthetic passive tracer only; G is per unit source (ug m^-3 per ug s^-1), a_ref is the reference strength (ug s^-1)."""
import numpy as np


def practical_rank(G, sigma, a_ref):
    """#{singular values s of G/sigma with s*a_ref >= 1}: modes a reference-sized source change moves by >= 1 noise sigma."""
    G = np.asarray(G, float)
    if G.ndim != 2 or not (sigma > 0 and a_ref > 0):
        raise ValueError("G must be 2-D and sigma, a_ref positive")
    return int((np.linalg.svd(G / sigma, compute_uv=False) * a_ref >= 1).sum())


def null_counterexample(G, sigma, a, rel=0.5):
    """Move a along the weakest right singular vector of G/sigma (nonnegativity kept, step <= rel*||a||)."""
    G, a = np.asarray(G, float), np.asarray(a, float)
    Gw = G / sigma
    _, s, Vt = np.linalg.svd(Gw)
    v = Vt[-1]
    s_min = float(s[-1]) if s.size == a.size else 0.0  # fewer rows than columns: exact null space
    room = lambda d: min((a[k] / -d[k] for k in range(a.size) if d[k] < 0), default=np.inf)
    best = max((sg for sg in (1.0, -1.0)), key=lambda sg: room(sg * v))
    t = min(rel * np.linalg.norm(a), 0.999 * room(best * v))
    a2 = a + t * best * v
    return dict(a2=a2, rel_diff=float(np.linalg.norm(a2 - a) / np.linalg.norm(a)),
                data_diff_sigma=float(np.sqrt(np.mean((Gw @ (a2 - a)) ** 2))), s_min=s_min)


def greedy_design(G3, xy, n, sigma, a_ref, min_sep):
    """Greedily add the site maximising logdet(I + a_ref^2 Gw^T Gw) (Bayesian D-opt, prior sd a_ref); G3 is (sites, hours, K)."""
    G3, xy = np.asarray(G3, float), np.asarray(xy, float)
    K, chosen = G3.shape[2], []
    M = np.eye(K)
    for _ in range(n):
        best, bi = -np.inf, None
        for i in range(len(xy)):
            if i in chosen or any(np.hypot(*(xy[i] - xy[j])) < min_sep for j in chosen):
                continue
            Gi = G3[i] * a_ref / sigma
            v = np.linalg.slogdet(M + Gi.T @ Gi)[1]
            if v > best + 1e-12:
                best, bi = v, i
        if bi is None:
            raise ValueError(f"cannot place {n} sites with min_sep={min_sep}")
        chosen.append(bi)
        Gi = G3[bi] * a_ref / sigma
        M = M + Gi.T @ Gi
    return chosen

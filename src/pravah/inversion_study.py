"""Scoring helpers for the E03 static-inverse study (T-015). Contract: tests/test_inversion_study.py."""
import numpy as np

from .inverse import rank_info


def site_sigma(sigma0, n_sites, seed):
    """Per-site noise std = sigma0 * log-uniform[1/3, 3]; independent of any truth."""
    if not sigma0 > 0 or n_sites < 1:
        raise ValueError("sigma0 must be > 0 and n_sites >= 1")
    return sigma0 * np.exp(np.random.default_rng(seed).uniform(-np.log(3), np.log(3), n_sites))


def source_scores(a_hat, a, a_ref=100.0):
    a_hat, a = np.asarray(a_hat, float), np.asarray(a, float)
    if a_hat.shape != a.shape or a.ndim != 1 or not (np.isfinite(a_hat).all() and np.isfinite(a).all()) or not a.any():
        raise ValueError("a_hat and a must be finite, same-shape 1-D, and a not all zero")
    e = a_hat - a
    return dict(source_rmse_ref=float(np.sqrt(np.mean(e**2)) / a_ref),
                source_relative_rmse=float(np.linalg.norm(e) / np.linalg.norm(a)),
                parameter_bias=float(e.mean() / a.mean()))


def crb_fraction(G, sigma, a):
    """Fraction of sources with local unconstrained Cramer-Rao sd <= 0.5 a_k; 0.0 if G is rank deficient."""
    G, sigma, a = np.asarray(G, float), np.asarray(sigma, float), np.asarray(a, float)
    if sigma.shape != (G.shape[0],) or not (sigma > 0).all() or a.shape != (G.shape[1],):
        raise ValueError("sigma must be positive (n,), a must be (K,)")
    WG = G / sigma[:, None]
    if rank_info(WG)["rank"] < G.shape[1]:
        return 0.0
    sd = np.sqrt(np.diag(np.linalg.inv(WG.T @ WG)))
    return float(np.mean(sd <= 0.5 * a))


def failure_mode(rank, K, a_hat):
    if rank < K:
        return "non_identifiable"
    return "estimated" if np.any(np.asarray(a_hat) > 0) else "all_zero_estimate"

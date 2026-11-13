"""Sampling operator and observation noise (docs/research/truth-design.md §5). numpy only; holds no truth.

Units: positions m, concentration ug m^-3. Noise arrays are drawn once per scenario for ALL candidate sites and hours,
so any design that indexes the array sees identical noise at a shared site (same-seed fairness).
"""
import numpy as np

KINDS = ("white", "ar1")


def bilinear_sample(c, xy, domain):
    """Sample c[nx, ny] (cell centres ((i+.5)dx, (j+.5)dy)) at xy (n, 2) metres.

    Exact at cell centres, bilinear between, constant extension in the half-cell wall band.
    ValueError for points outside [0, Lx] x [0, Ly].
    """
    c = np.asarray(c, float)
    xy = np.atleast_2d(np.asarray(xy, float))
    nx, ny = c.shape
    L = np.array(domain, float)
    if xy.shape[1] != 2 or (xy < 0).any() or (xy > L).any():
        raise ValueError("observation point outside the domain")
    g = np.clip(xy / (L / (nx, ny)) - 0.5, 0, (nx - 1, ny - 1))  # fractional cell-centre index
    i0, j0 = np.minimum(g[:, 0].astype(int), max(nx - 2, 0)), np.minimum(g[:, 1].astype(int), max(ny - 2, 0))
    i1, j1 = np.minimum(i0 + 1, nx - 1), np.minimum(j0 + 1, ny - 1)
    fx, fy = g[:, 0] - i0, g[:, 1] - j0
    return ((1 - fx) * (1 - fy) * c[i0, j0] + fx * (1 - fy) * c[i1, j0]
            + (1 - fx) * fy * c[i0, j1] + fx * fy * c[i1, j1])


def noise_field(seed, n_sites, n_hours, sigma, kind="white", rho=0.8):
    """Noise (n_sites, n_hours) with marginal std sigma (scalar, or (n_sites,) for heteroscedastic sites).

    white: child 0 of SeedSequence(seed). ar1: child 1, stationary, e_t = rho e_{t-1} + sqrt(1-rho^2) z_t, same marginal
    sigma; sites are independent. Independent of truth by construction.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    sig = np.asarray(sigma, float)
    if sig.ndim == 0:
        sig = np.full(n_sites, float(sig))
    if sig.shape != (n_sites,) or (sig < 0).any() or not np.isfinite(sig).all():
        raise ValueError("sigma must be a finite nonnegative scalar or per-site vector")
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1)")
    # rebuild so a passed SeedSequence's spawn counter is never consumed: repeated calls stay identical
    ss = np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key) if isinstance(seed, np.random.SeedSequence) \
        else np.random.SeedSequence(seed)
    z = np.random.default_rng(ss.spawn(2)[KINDS.index(kind)]).standard_normal((n_sites, n_hours))
    if kind == "ar1":
        e = np.empty_like(z)
        e[:, 0] = z[:, 0]
        for t in range(1, n_hours):
            e[:, t] = rho * e[:, t - 1] + np.sqrt(1 - rho**2) * z[:, t]
        z = e
    return sig[:, None] * z

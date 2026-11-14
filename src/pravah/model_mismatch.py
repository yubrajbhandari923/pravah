"""E13 non-wind model mismatch (design: docs/research/model-mismatch-design.md).

The inverse/forecast model is FIXED (nominal E07 unit responses: nominal centres, 400 m footprint, constant emission,
known background 5, white noise). Each spec changes only the WORLD: the truth field (background, source shape/location,
an omitted source, diurnal emission) and/or the readings (AR(1), biases, drift, humidity-like gain, outages).
`truth_solves` lists the extra truth-side unit solves (never shown to the estimator); `world` assembles the truth and
readings; `map_masked` is the estimator (NaN readings are masked, never imputed). Units: ug m^-3, a_ref units for a.
"""
from __future__ import annotations

import numpy as np

from . import benchmarks as B
from .sensitivity import anomaly_fields

E13_SEED, BG_PRIOR_SD = 13000, 1.0          # bg offset prior N(0, 1^2) ug m^-3 (protocol §4 background variant)
CODE = dict(bgdrift=1, shift=2, missing=3, diurnal=4, sitebias=5, drift=6, drop=7)
OUTAGE_MEAN_H, HUMID_PEAK_H = 12.0, 4.0

# spec name -> parts ((family, magnitude), ...); design note §3
SPECS = {
    "oracle": (),
    "bgoff0.25": (("bgoff", .25),), "bgoff1": (("bgoff", 1.),),
    "bgdrift0.25": (("bgdrift", .25),), "bgdrift1": (("bgdrift", 1.),),
    "shift400": (("shift", 400.),), "shift800": (("shift", 800.),),
    "wide800": (("wide", 800.),),
    "missing1": (("missing", 1.),),
    "diurnal": (("diurnal", .5),),
    "ar1": (("ar1", .8),),
    "sitebias0.25": (("sitebias", .25),), "sitebias1": (("sitebias", 1.),),
    "common1": (("common", 1.),),
    "drift1": (("drift", 1.),),
    "humid0.3": (("humid", .3),),
    "drop0.2": (("drop", .2),), "drop0.5": (("drop", .5),),
    "combined": (("shift", 400.), ("bgdrift", .25), ("ar1", .8), ("sitebias", .25), ("drop", .2)),
}
# 64^2 subset (design §10): shift400 and wide800 dropped; combined uses shift800
SWEEP_SPECS = {n: p for n, p in SPECS.items() if n not in ("shift400", "wide800")}
SWEEP_SPECS["combined"] = (("shift", 800.),) + SPECS["combined"][1:]
SOLVE_FAMILIES = ("shift", "wide", "missing", "diurnal")


def _rng(family, seed, K):
    return np.random.default_rng([E13_SEED, CODE[family], seed, K])


# ---------------------------------------------------------------- truth-side sources (estimator never sees these)
def shifted_centres(cfg, seed, K, sxy, d):
    """True centres = nominal + d in a random direction per source (angles shared by all d), clipped 3 sigma inside."""
    ang = _rng("shift", seed, K).uniform(0, 2 * np.pi, K)
    c = np.asarray(sxy, float) + d * np.stack([np.cos(ang), np.sin(ang)], 1)
    m = 3 * cfg.sources.footprint_sigma_m
    return np.clip(c, m, np.asarray(cfg.model.domain_m) - m)


def extra_source(cfg, seed, K, sxy):
    """(centre, strength ug/s) of one omitted source on the lattice, >= min separation from every nominal centre."""
    rng, s = _rng("missing", seed, K), cfg.sources
    (Lx, Ly), (na, nb) = cfg.model.domain_m, s.lattice
    for p in rng.permutation(na * nb):
        a, b = divmod(int(p), nb)
        c = np.array([(a + .5) * Lx / na, (b + .5) * Ly / nb])
        if np.linalg.norm(np.asarray(sxy) - c, axis=1).min() >= s.min_separation_m:
            lo, hi = s.strength_loguniform
            return c, float(s.reference_rate_ug_s * np.exp(rng.uniform(np.log(lo), np.log(hi))))
    raise ValueError("no lattice cell for the omitted source")


def diurnal_amp(seed, K, T, amp=.5):
    """(K, T) protocol §4 factor g_k(h) = 1 + amp sin(2 pi (h - t_k)/24), t_k ~ U(0, 24)."""
    tk = _rng("diurnal", seed, K).uniform(0, 24, K)
    return 1 + amp * np.sin(2 * np.pi * (np.arange(T)[None] - tk[:, None]) / 24)


def truth_solves(cfg, seed, K, sxy, specs):
    """{key: (centre, footprint sigma, amp or None)} of the truth-side unit solves the specs ({name: parts}) need."""
    out = {}
    for parts in specs.values():
        for fam, mag in parts:
            if fam == "shift":
                for k, c in enumerate(shifted_centres(cfg, seed, K, sxy, mag)):
                    out[(f"shift{mag:g}", k)] = (c, cfg.sources.footprint_sigma_m, None)
            elif fam == "wide":
                for k, c in enumerate(sxy):
                    out[(f"wide{mag:g}", k)] = (np.asarray(c, float), mag, None)
            elif fam == "missing":
                out[("missing", 0)] = (extra_source(cfg, seed, K, sxy)[0], cfg.sources.footprint_sigma_m, None)
            elif fam == "diurnal":
                n = len(B.scenario_drivers(cfg, seed, K)[3])
                for k, c in enumerate(sxy):
                    out[(f"diurnal{mag:g}", k)] = (np.asarray(c, float), cfg.sources.footprint_sigma_m,
                                                    diurnal_amp(seed, K, n, mag)[k])
    return out


def unit_field(cfg, centre, sigma_fp, amp, winds):
    """Full-grid anomaly (hours, nx, ny) per unit strength of one truth-side source."""
    m = cfg.model
    S = B.source_basis(tuple(m.grid), tuple(m.domain_m), centre, sigma_fp) / m.mixing_depth_m
    return anomaly_fields(m, winds, S, len(winds), amp=amp)


# ---------------------------------------------------------------- worlds (truth + readings), scorer/reading side only
def _bg_drift(seed, K, T, A):
    t0 = _rng("bgdrift", seed, K).uniform(0, 24)
    h = np.arange(T)
    return A * (np.sin(2 * np.pi * (h - t0) / 24) + h / T)


def outage_mask(seed, K, n_sites, T, f):
    """(n_sites, T) bool observed. Per-site two-state Markov chain, mean outage OUTAGE_MEAN_H, stationary missing f."""
    rng = _rng("drop", seed, K)
    p_off_on = 1 / OUTAGE_MEAN_H
    p_on_off = f / (1 - f) * p_off_on
    u = rng.random((n_sites, T))
    off = np.empty((n_sites, T), bool)
    off[:, 0] = u[:, 0] < f
    for t in range(1, T):
        off[:, t] = np.where(off[:, t - 1], u[:, t] >= p_off_on, u[:, t] < p_on_off)
    return ~off


def world(sc, name, tfields, G_site, sigma, r, specs=None):
    """(tf (T, N) truth anomaly w.r.t. the NOMINAL background, Y (n_sites, T) readings with NaN = missing).

    tfields: {key: (T, nx, ny) truth-side unit field}; G_site: {key: (n_sites, T)} the same sampled at the sites.
    Reads sc.truth (never writes it). Noise draw r uses the E07 streams (white or ar1 kind), so worlds are paired."""
    from .experiments.cross_objective import readings
    parts = dict((specs or SPECS)[name])
    cfg, T = sc.cfg, len(sc.truth.winds)
    cbg, K = cfg.sources.background_ug_m3, sc.n_sources
    N = int(np.prod(cfg.model.grid))
    a = sc.truth.strengths_ug_s
    if not parts:
        return np.asarray(sc.truth.field, float).reshape(T, N) - cbg, readings(sc, sigma, "white", r, T)  # = E07
    pf = [f"{f}{m:g}" for f, m in parts.items() if f in ("shift", "wide", "diurnal")]
    if len(pf) > 1:
        raise ValueError("at most one plume-changing family per spec")
    if pf:  # replace the nominal plume by the truth-side one (same strengths)
        tf = sum(a[k] * tfields[(pf[0], k)].reshape(T, N) for k in range(K))
        st = cbg + sum(a[k] * G_site[(pf[0], k)] for k in range(K))
    else:
        tf = np.asarray(sc.truth.field, float).reshape(T, N) - cbg
        st = np.asarray(sc.truth.site_truth, float)[:, :T]
    if "missing" in parts:
        ax = extra_source(cfg, sc.replicate, K, sc.truth.source_xy)[1]
        tf = tf + ax * tfields[("missing", 0)].reshape(T, N)
        st = st + ax * G_site[("missing", 0)]
    bg = np.zeros(T)
    if "bgoff" in parts:
        bg = bg + parts["bgoff"]
    if "bgdrift" in parts:
        bg = bg + _bg_drift(sc.replicate, K, T, parts["bgdrift"])
    tf, st = tf + bg[:, None], st + bg[None]
    e = readings(sc, sigma, "ar1" if "ar1" in parts else "white", r, T) - np.asarray(sc.truth.site_truth)[:, :T]
    y = st * (1 + parts["humid"] * (.5 + .5 * np.sin(2 * np.pi * (np.arange(T) - HUMID_PEAK_H) / 24)))[None] \
        if "humid" in parts else st.copy()
    y = y + e
    n = len(y)
    if "sitebias" in parts:
        y = y + parts["sitebias"] * _rng("sitebias", sc.replicate, K).standard_normal(n)[:, None]
    if "common" in parts:
        y = y + parts["common"]
    if "drift" in parts:
        y = y + parts["drift"] * _rng("drift", sc.replicate, K).uniform(.5, 1.5, n)[:, None] * np.arange(T)[None] / T
    if "drop" in parts:
        y = np.where(outage_mask(sc.replicate, K, n, T, parts["drop"]), y, np.nan)
    return tf, y


# ---------------------------------------------------------------- estimator (masked MAP, optional background column)
def map_masked(A, z, issue, window=None, bg_col=None):
    """MAP and posterior covariance at each issue time, prior N(0, I) on every coefficient (a_ref units; the bg
    coefficient in units of BG_PRIOR_SD). A (k, T, K) whitened sensitivities, z (k, T, R) whitened anomalies with NaN =
    missing (dropped from the normal equations, never imputed). bg_col: (k,) whitened response to a +BG_PRIOR_SD offset
    (appended as an extra coefficient). Reads no row after an issue time. Returns u (n_issue, K', R), C (n_issue, K', K')."""
    obs = np.isfinite(z)
    if bg_col is not None:
        A = np.concatenate([A, np.broadcast_to(np.asarray(bg_col, float)[:, None, None], A.shape[:2] + (1,))], 2)
    m = obs.all(2)
    if not (obs == m[..., None]).all():
        raise ValueError("missingness must be the same for every draw")
    A = A * m[..., None]
    z = np.where(obs, z, 0.0)
    K = A.shape[2]
    cP = np.concatenate([np.zeros((1, K, K)), np.cumsum(np.einsum("shk,shl->hkl", A, A), 0)])
    cb = np.concatenate([np.zeros((1, K) + z.shape[2:]), np.cumsum(np.einsum("shk,shr->hkr", A, z), 0)])
    issue = np.asarray(issue, int)
    lo = np.zeros_like(issue) if window is None else issue + 1 - window
    if lo.min() < 0:
        raise ValueError("window starts before hour 0")
    Nt = np.eye(K) + cP[issue + 1] - cP[lo]
    return np.linalg.solve(Nt, cb[issue + 1] - cb[lo]), np.linalg.inv(Nt)

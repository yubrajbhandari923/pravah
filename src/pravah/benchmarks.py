"""Synthetic benchmark: complete hidden truth plus a separate, leak-checked inference API (T-010).

Implements docs/research/truth-design.md (DRAFT, perfect-model truth only; IMPERFECT variants are not generated here).
`Scenario` owns everything, including the hidden full field, strengths and all-candidate noisy readings. Inverse methods
get ONLY `Scenario.view(...)`, a frozen InferenceView built from copies: no truth, no strengths, no unselected sites,
no data from other splits. Units: m, s, ug m^-3, a_k in ug s^-1, winds m s^-1 (ux = -s sin th, uy = -s cos th).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .experiment_config import ExperimentConfig, check_no_leak, truth_id
from .forward import simulate
from .observations import bilinear_sample, noise_field

SPLITS = ("train", "validation", "test")
BLOCK_H = 48
SIGMA_REL = 0.05
WIND_BASE = 5000  # truth-design.md §6; Seeds has no wind stream yet


# ---------------------------------------------------------------- sources
def source_basis(grid, domain, centre, sigma, truncate=3.0):
    """phi[nx, ny] [m^-2]: Gaussian truncated at `truncate` sigma, discrete sum(phi dx dy) = 1 (clipped at walls)."""
    (nx, ny), (Lx, Ly) = grid, domain
    dx, dy = Lx / nx, Ly / ny
    x, y = (np.arange(nx) + .5) * dx, (np.arange(ny) + .5) * dy
    r2 = (x[:, None] - centre[0]) ** 2 + (y[None] - centre[1]) ** 2
    phi = np.where(r2 <= (truncate * sigma) ** 2, np.exp(-r2 / (2 * sigma**2)), 0.0)
    if phi.sum() == 0:
        raise ValueError("source footprint covers no cell centre")
    return phi / (phi.sum() * dx * dy)


def draw_sources(cfg: ExperimentConfig, replicate: int, n_sources: int, allowed=None):
    """(lattice cells [(a, b)], strengths ug s^-1). Stream SeedSequence([truth_seed, K]); greedy min-separation draw.
    allowed: optional bool (na, nb) lattice mask (E17 population mask); None = every cell (bit-identical to before)."""
    s, (Lx, Ly), (na, nb) = cfg.sources, cfg.model.domain_m, cfg.sources.lattice
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seeds.streams(replicate)["truth"], n_sources]))
    cells, xy = [], []
    for p in rng.permutation(na * nb):
        a, b = divmod(int(p), nb)
        if allowed is not None and not allowed[a, b]:
            continue
        c = np.array([(a + .5) * Lx / na, (b + .5) * Ly / nb])
        if all(np.hypot(*(c - q)) >= s.min_separation_m for q in xy):
            cells.append((a, b)); xy.append(c)
            if len(cells) == n_sources:
                break
    if len(cells) < n_sources:
        raise ValueError(f"could not place {n_sources} sources with separation {s.min_separation_m} m")
    lo, hi = s.strength_loguniform
    return cells, s.reference_rate_ug_s * np.exp(rng.uniform(np.log(lo), np.log(hi), n_sources))


# ---------------------------------------------------------------- winds
def make_winds(regimes, n_hours, seed):
    """Hourly (ux, uy) [n_hours, 2] and the regime of each 48 h block (truth-design.md §2)."""
    rng = np.random.default_rng(seed)
    out, used = [], []
    for b in range(-(-n_hours // BLOCK_H)):
        reg = regimes[b % len(regimes)]
        used.append(reg)
        h = np.arange(BLOCK_H)
        if reg == "constant_westerly":  # null control (gate-05): exactly 3 m/s from 270 deg, no per-block jitter
            out.append(np.tile([3.0, 0.0], (BLOCK_H, 1)))
            continue
        if reg == "steady_westerly":
            sp, th = np.full(BLOCK_H, 3.0), np.full(BLOCK_H, 270.0)
        elif reg == "veering":
            sp, th = np.full(BLOCK_H, 3.0), 270.0 + 45.0 * np.minimum(h, 12) / 12
        elif reg == "weak_variable":
            sp, th = np.repeat(rng.uniform(.5, 1.5, BLOCK_H // 6), 6), 270.0 + 180.0 * ((h // 6) % 2)
        elif reg == "steady_southerly":
            sp, th = np.full(BLOCK_H, 4.0), np.full(BLOCK_H, 180.0)
        elif reg == "stagnation_then_front":
            sp, th = np.where(h < 24, rng.uniform(.2, .5), 3.0), np.where(h < 24, 270.0, 0.0)
        elif reg == "diurnal_rotation":
            sp, th = np.full(BLOCK_H, 2.0), rng.uniform(0, 360) + 360.0 * h / 24
        else:
            raise ValueError(f"unknown wind regime {reg!r}")
        sp = np.clip(sp * rng.uniform(.8, 1.2), .2, 5.0)
        th = np.radians(th + rng.normal(0, 10))
        out.append(np.stack([-sp * np.sin(th), -sp * np.cos(th)], 1))
    return np.concatenate(out)[:n_hours], used


# ---------------------------------------------------------------- containers
@dataclass(frozen=True)
class Truth:
    """Hidden truth. Never passed to inverse methods."""
    field: np.ndarray            # [hour, i, j] ug m^-3, hour 0 = first hour after spin-up
    source_xy: np.ndarray        # (K, 2) m
    source_cells: list           # lattice (a, b)
    strengths_ug_s: np.ndarray   # (K,)
    winds: np.ndarray            # (hours, 2) m s^-1, all splits concatenated
    wind_regimes: dict           # split -> regime of each 48 h block
    site_truth: np.ndarray       # (n_candidates, hours), noise-free, from bilinear_sample at the site


@dataclass(frozen=True)
class InferenceView:
    """Everything an inverse method may see for one split and one chosen network (read-only copies)."""
    split: str
    purpose: str
    kind: str
    site_ids: np.ndarray
    site_xy: np.ndarray
    hours: np.ndarray
    readings: np.ndarray         # (n_sites, n_hours) noisy, ug m^-3
    winds: np.ndarray            # nominal wind for these hours
    source_xy: np.ndarray        # locations known to the estimator (protocol); strengths are NOT
    noise_sigma: np.ndarray      # instrument std at the chosen sites
    model: MappingProxyType      # nominal model parameters (read-only mapping)


def _ro(a):
    a = np.array(a, copy=True)
    a.flags.writeable = False
    return a


class Scenario:
    """One (replicate, K) benchmark scenario: `truth` (hidden), `view()` (inference API)."""

    def __init__(self, cfg, replicate, n_sources, truth, site_ij, site_xy, noise_sigma, noise, lattice_n):
        self.cfg, self.replicate, self.n_sources = cfg, replicate, n_sources
        self.truth, self.site_ij, self.site_xy = truth, site_ij, site_xy
        self.noise_sigma, self._noise, self.lattice_n = noise_sigma, noise, lattice_n
        self.truth_id = truth_id(cfg, replicate, n_sources)
        self.n_candidate_sites = len(site_xy)

    def admissible_sites(self):
        """Ids not on the outer ring of the lattice and farther than 1 sigma from every true source centre."""
        n = self.lattice_n
        i, j = divmod(np.arange(self.n_candidate_sites), n)
        ring = (i == 0) | (i == n - 1) | (j == 0) | (j == n - 1)
        d = np.linalg.norm(self.site_xy[:, None] - self.truth.source_xy[None], axis=2).min(1)
        return np.flatnonzero(~ring & (d > self.cfg.sources.footprint_sigma_m))

    def split_hours(self, split):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        lo, hi = getattr(self.cfg.split, split)
        return np.arange((lo - 1) * 24, hi * 24)

    def view(self, split, site_ids, purpose, kind="white") -> InferenceView:
        """Inference API. Raises LeakError if the split is not allowed for `purpose`; ValueError for bad sites."""
        hours = self.split_hours(split)
        days = sorted({int(h) // 24 + 1 for h in hours})
        check_no_leak(self.cfg.split, purpose, days, self.truth.wind_regimes[split], self.cfg.winds)
        ids = np.asarray(site_ids, int).ravel()
        if ids.size == 0 or len(set(ids.tolist())) != ids.size:
            raise ValueError("site_ids must be non-empty and unique")
        if not np.isin(ids, self.admissible_sites()).all():
            raise ValueError("site_ids contain inadmissible sites")
        xy = self.site_xy[ids]
        dist = np.linalg.norm(xy[:, None] - xy[None], axis=2) + np.eye(len(ids)) * 1e18
        if (dist < self.cfg.design.min_sensor_separation_m).any():
            raise ValueError("sites closer than the minimum sensor separation")
        if kind not in self._noise:
            raise ValueError(f"kind must be one of {tuple(self._noise)}")
        m = self.cfg.model
        return InferenceView(
            split, purpose, kind, _ro(ids), _ro(xy), _ro(hours),
            _ro(self.truth.site_truth[ids][:, hours] + self._noise[kind][ids][:, hours]),
            _ro(self.truth.winds[hours]), _ro(self.truth.source_xy), _ro(self.noise_sigma[ids]),
            MappingProxyType(dict(domain_m=tuple(m.domain_m), grid=tuple(m.grid), dt_s=m.dt_s, K_m2_s=m.diffusivity_m2_s,
                 lambda_s=m.removal_per_s, mixing_depth_m=m.mixing_depth_m,
                 c_bg=self.cfg.sources.background_ug_m3, footprint_sigma_m=self.cfg.sources.footprint_sigma_m)))


def scenario_drivers(cfg: ExperimentConfig, replicate: int, n_sources: int):
    """Cheap, solve-free inputs of a scenario: (cells, strengths, source_xy, winds (all splits), regimes, hours per split)."""
    dom = tuple(cfg.model.domain_m)
    cells, a = draw_sources(cfg, replicate, n_sources)
    na, nb = cfg.sources.lattice
    sxy = np.array([[(p + .5) * dom[0] / na, (q + .5) * dom[1] / nb] for p, q in cells])
    hrs = {s: len(range(getattr(cfg.split, s)[0] - 1, getattr(cfg.split, s)[1])) * 24 for s in SPLITS}
    winds, regimes = [], {}
    for k, s in enumerate(SPLITS):
        w, regimes[s] = make_winds(getattr(cfg.winds, "heldout" if s == "test" else "train"), hrs[s],
                                   WIND_BASE + 100 * k + replicate)
        winds.append(w)
    return cells, a, sxy, np.concatenate(winds), regimes, hrs


def generate_scenario(cfg: ExperimentConfig, replicate: int, n_sources: int, *, lattice_n: int = 16,
                      heteroscedastic: bool = False, truth_field: np.ndarray | None = None, drivers=None) -> Scenario:
    """Run the verified forward solver for spin-up + the whole calendar and sample all candidate sites.

    Spin-up winds replay the first spin-up hours of the train stream (a [decision]: the spec names no spin-up wind).
    sigma = 0.05 x median train-period truth at the candidate sites, one scalar per scenario (or per site if
    heteroscedastic). Noise is drawn from SeedSequence(noise seed), independent of truth. Cost: ~1 solve per 300 s.
    `truth_field`: the post-spin-up field of an earlier identical call (e.g. a disk cache); skips the solve.
    `drivers`: optional scenario_drivers-shaped tuple replacing the synthetic one (E17: ERA5 winds, masked sources).
    """
    cfg.validate()
    m = cfg.model
    grid, dom = tuple(m.grid), tuple(m.domain_m)
    if grid[0] % lattice_n or grid[1] % lattice_n:
        raise ValueError("lattice_n must divide the grid")
    cells, a, sxy, winds, regimes, hrs = drivers or scenario_drivers(cfg, replicate, n_sources)
    if truth_field is not None:
        out = np.asarray(truth_field, float)
        if out.shape != (len(winds),) + grid:
            raise ValueError(f"truth_field shape {out.shape} != {(len(winds),) + grid}")
    else:
        S = sum(ak * source_basis(grid, dom, c, cfg.sources.footprint_sigma_m) for ak, c in zip(a, sxy)) / m.mixing_depth_m
        spin = np.resize(winds[:max(m.spinup_h, 1)], (m.spinup_h, 2))
        # Solve the anomaly d = c - c_bg with c_bg = 0 (exact: the equation is affine and the inflow Dirichlet value is
        # c_bg). With c_bg = 5 FiPy's LU residual tolerance silently drops sources of this size (RUN evidence in task notes).
        d, steps, bg = np.zeros(grid), 3600 // m.dt_s, cfg.sources.background_ug_m3
        out = np.empty((len(winds),) + grid)
        for h, u in enumerate(np.concatenate([spin, winds])):
            d = simulate(nx=grid[0], ny=grid[1], Lx=dom[0], Ly=dom[1], dt=m.dt_s, n_steps=steps, u=tuple(u),
                         K=m.diffusivity_m2_s, lam=m.removal_per_s, c_bg=0.0, c0=d, bc="study", scheme=m.advection_scheme,
                         source=S)
            if h >= m.spinup_h:
                out[h - m.spinup_h] = bg + d
    dx, dy = dom[0] / grid[0], dom[1] / grid[1]
    sg = (grid[0] // lattice_n, grid[1] // lattice_n)
    ii, jj = np.divmod(np.arange(lattice_n**2), lattice_n)
    ij = np.stack([sg[0] // 2 + sg[0] * ii, sg[1] // 2 + sg[1] * jj], 1)
    xy = (ij + .5) * (dx, dy)
    site_truth = np.stack([bilinear_sample(f, xy, dom) for f in out], 1)
    train = slice(0, hrs["train"])
    sig = SIGMA_REL * (np.median(site_truth[:, train], axis=1) if heteroscedastic else np.median(site_truth[:, train]))
    ns = cfg.seeds.streams(replicate)["noise"]
    noise = {k: noise_field(ns, len(xy), len(winds), sig, k) for k in ("white", "ar1")}
    t = Truth(out, sxy, cells, a, winds, regimes, site_truth)
    return Scenario(cfg, replicate, n_sources, t, ij, xy, np.broadcast_to(sig, (len(xy),)).copy(), noise, lattice_n)

"""DYN-08 (EXP-036), REDUCED design: dynamic population-weighted sensor design at Belauri scale.  Real geography (padded 500 m grid), real
WorldPop weights, ERA5-derived UNIFORM wind histories; emissions, episodes and noise are SYNTHETIC.  Not empirical validation.  Full text,
hypotheses, stop rules and measured cost: docs/dyn08_plan.md.  Registered values: configs/dyn08.json (frozen before any layout is designed).

    python -m pravah.studies.belauri.dynamic_belauri --stage dry-run                    # no solves: design, counts, seeds, cost model
    python -m pravah.studies.belauri.dynamic_belauri --stage smoke  --out DIR           # tiny end-to-end run on the 2000 m padded level (local)
    python -m pravah.studies.belauri.dynamic_belauri --stage time   --out DIR           # measured per-solve / per-fit costs (local, one core)
    python -m pravah.studies.belauri.dynamic_belauri --stage design --out DIR [--workers N]   # surrogate layouts on the 6 design winds -> layouts.json
    python -m pravah.studies.belauri.dynamic_belauri --stage tune   --out DIR [--workers N]   # validation episodes, SPREAD layout only -> frozen_hparams.json
    python -m pravah.studies.belauri.dynamic_belauri --stage test   --out DIR [--workers N]   # test episodes, frozen layouts + hparams (one shot)
    python -m pravah.studies.belauri.dynamic_belauri --stage modelerr --out DIR [--workers N] # model-error conditions on the test winds
    python -m pravah.studies.belauri.dynamic_belauri --stage all --out DIR [--workers N]      # design -> tune -> gate -> test -> modelerr in one job

Pipeline logic is ported from the toy DYN-02/03/04 studies (``pravah.studies.toy.dynamic_study`` / ``dynamic_design_study`` /
``dynamic_history``), whose module-level toy globals (grid, sources, REF, 6 winds) cannot be parameterised in place; the generic parts
(``pravah.dynamic``, ``pravah.dynamic_design``, ``SumProblem``, ``search``) are imported.  Deviations from the toy are listed in the plan.
The forward solver is untouched.  Never run a stage on the cluster without the user's reservation approval (AGENTS.md).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls

from ... import dynamic as dyn
from ...config import InflowBasis, Scenario, WindSegment
from ...era5 import load_era5_wind
from ...era5_episodes import SPLITS, clip_speed, make_windows, select_windows, speed_median, strata_table, stratum
from ...forward import simulate
from ...geo.prepare import load_region
from ...geo.scenarios import LOCAL_ORIGIN_TOL, belauri_scenario
from ...inverse import ResponseCache, Responses
from ...io import write_json
from ...population import mask_to_study, regrid_counts

ROOT = Path(__file__).resolve().parents[4]
NAMES = ("road", "builtup", "external", "bg_wsw", "bg_ene")
NP = len(NAMES)
BASELINES = ("random", "spread", "highpop", "static_opt", "src_opt")
FAMILY_ORDER = ("random", "spread", "highpop", "static_opt", "dyn_opt", "src_opt", "lowpop")  # lowpop = bad-layout falsification control, not a baseline
HIST_MODELS = ("CT", "CAUSAL")
# Seed ranges ALREADY SPENT or declared by other records (AGENTS.md, docs/dyn08_readiness.md 2.8, .research records); DYN-08 seeds must avoid all of them.
USED_SEEDS = ((101, 101), (202, 202), (303, 305), (404, 404), (1101, 1108), (3500, 3500), (5000, 5999), (6000, 6999), (7000, 7999), (8000, 8999),
              (9000, 9499), (12345, 12345), (18403, 18403), (52501, 52501))


# ------------------------------------------------------------------------------------------------ configuration
def load_cfg(path: str | Path = "configs/dyn08.json") -> dict[str, Any]:
    return json.loads((ROOT / path).read_text())


def smoke_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Tiny variant on the 2000 m padded level (17 x 17), 2/2/2 winds, 3 families, k = 2, one noise level, small hyperparameter grid."""
    c = json.loads(json.dumps(cfg))
    c["name"] = "dyn08_smoke"
    c["origin_tol_by_resolution_m"] = {"2000": 0.25}  # 2000 m cells smear the local rasters across the polygon edge (0.82 of road mass inside): smoke only
    c["mesh"] = {"resolution_m": 2000.0, "dt_s": 450.0, "truth_dt_s": 225.0}
    c["wind"]["n"] = {"design": 2, "validation": 2, "test": 2}
    c["families"] = ["constant", "activation", "pulse"]
    c["ks"], c["noise"], c["n_random"] = [2], [0.10], 3
    c["seeds_per_wind_family"] = {"validation": 1, "test": 1}
    c["seeds"] = {**c["seeds"], "validation": 49000, "test": 49100, "wind_selection": 49900, "design_layouts": 49950}
    c["hp_grid"] = {"alpha_m_exp": [-1.0, 3.0, 2.0], "alpha_2_exp": [-1.0, 3.0, 2.0], "beta_exp": [0.0, 4.0, 2.0], "note": "smoke"}
    c["design"] = {**c["design"], "n_extra_starts": 2}
    c["model_error"] = {"conditions": {"rot10": {"rot_deg": 10.0}}, "n_seeds": 1}
    return c


def _grid(spec: list[float]) -> tuple[float, ...]:
    a, b, s = spec
    return tuple(10.0 ** e for e in np.arange(a, b + 1e-9, s))


def seed_conflicts(cfg: dict[str, Any]) -> list[str]:
    """Registered seed blocks (each block spans 1000 values) that overlap any spent / declared range."""
    bad = []
    for name, s in cfg["seeds"].items():
        if name.startswith("dev") or not isinstance(s, int):
            continue
        for lo, hi in USED_SEEDS:
            if s <= hi and s + 999 >= lo:
                bad.append(f"{name} {s}-{s + 999} overlaps {lo}-{hi}")
    return bad


# ------------------------------------------------------------------------------------------------ the study object (geometry, responses, truth)
class Study:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        m, ph, b = cfg["mesh"], cfg["physics"], cfg["bins"]
        self.res, self.dt, self.dt_truth = float(m["resolution_m"]), float(m["dt_s"]), float(m["truth_dt_s"])
        self.T, self.out_dt = float(ph["T"]), float(ph["output_interval"])
        self.obs_step, self.t_excl = float(ph["obs_interval_s"]), float(ph["t_exclude"])
        self.nb, self.nf = int(b["n_inv"]), int(b["n_fine"])
        self.REF = np.array([cfg["ref"][n] for n in NAMES], dtype=float)
        self.region = load_region(ROOT / cfg["region"]["manifest"])
        self.cache = ResponseCache(os.environ.get("PRAVAH_RESPONSE_CACHE", str(ROOT / ".cache" / "dyn08_responses")))
        self._counts: np.ndarray | None = None
        self._sites: dict[str, Any] | None = None
        self._resp: dict[tuple, Responses] = {}

    # ---- geometry
    @property
    def counts(self) -> np.ndarray:
        """People per cell on the simulation grid, zero outside the study mask (fractional people kept), study total conserved."""
        if self._counts is None:
            p = ROOT / self.cfg["region"]["population_500m"]
            if not p.exists():
                raise FileNotFoundError(f"{p} missing: run scripts/dyn08_padded_population.py (local, offline)")
            raw = np.load(p)
            if self.res != 500.0:
                raw, _ = regrid_counts(raw, self.region.level(500.0).grid, self.region.level(self.res).grid)
            sf = np.asarray(self.region.level(self.res).arrays["study_fraction"], dtype=float)
            self._counts = mask_to_study(raw, sf, float((raw * sf).sum()))
        return self._counts

    @property
    def sites(self) -> dict[str, Any]:
        """The 54 interior candidate sites of the geodesign table (1500 m lattice inside the polygon; UTM positions kept) mapped to cells of this grid.
        Sites sharing a cell (only on coarse grids) are dropped, first one kept; ids/xy/cells are returned in table order."""
        if self._sites is None:
            from .geodesign import build_candidate_table, load_geography
            cc = json.loads((ROOT / self.cfg["region"]["candidate_cfg"]).read_text())
            cc["region"]["manifest"] = str(ROOT / cc["region"]["manifest"])
            tab = build_candidate_table(load_geography(cc), cc)
            tab = tab[tab["category"] == "interior"].reset_index(drop=True)
            lx, ly = self.region.georef.utm_to_local(np.asarray(tab["utm_x"], dtype=float), np.asarray(tab["utm_y"], dtype=float))
            ix, iy = np.floor(lx / self.res).astype(int), np.floor(ly / self.res).astype(int)
            g = self.region.level(self.res).grid
            if ix.min() < 0 or iy.min() < 0 or ix.max() >= g.nx or iy.max() >= g.ny:
                raise ValueError("a candidate site falls outside the simulation grid")
            seen: set[tuple[int, int]] = set()
            keep = []
            for i, c in enumerate(zip(iy.tolist(), ix.tolist())):
                if c not in seen:
                    seen.add(c)
                    keep.append(i)
            self._sites = {"ids": [tab["id"][i] for i in keep], "xy": np.column_stack([lx, ly])[keep], "iy": iy[keep], "ix": ix[keep],
                           "dropped_shared_cell": len(tab) - len(keep), "n_table": len(tab)}
        return self._sites

    # ---- scenario, responses
    def scenario(self, segs: list[list[float]], dt: float | None = None, res: float | None = None) -> Scenario:
        ph, ref = self.cfg["physics"], self.cfg["ref"]
        winds = tuple(WindSegment(float(t), float(u), float(v)) for t, u, v in segs)
        bases = (InflowBasis("bg_wsw", ("west", "south"), ((0.0, 1.0),)), InflowBasis("bg_ene", ("east", "north"), ((0.0, 1.0),)))
        case = belauri_scenario(self.region, self.res if res is None else res, winds, groups=("road", "builtup", "external"),
                                duration_s=self.T, dt_s=self.dt if dt is None else dt, output_interval_s=self.out_dt, K=float(ph["K"]),
                                lam=float(ph["lam"]), H=float(ph["H"]), amplitudes={k: float(v) for k, v in ref.items()}, bases=bases,
                                external_profile=((0.0, 1.0),),
                                origin_tol=float(self.cfg.get("origin_tol_by_resolution_m", {}).get(f"{self.res if res is None else res:g}", LOCAL_ORIGIN_TOL)))  # profiles are replaced by dynamic.expand / truth_scenario
        return case.scenario

    def responses(self, segs: list[list[float]]) -> Responses:
        key = tuple(map(tuple, segs))
        if key not in self._resp:
            if len(self._resp) >= 4:
                self._resp.pop(next(iter(self._resp)))
            self._resp[key] = self.cache.get(dyn.expand(self.scenario(segs), self.nb))
        return self._resp[key]

    def obs_index(self, R: Responses) -> np.ndarray:
        t = np.asarray(R.times)
        return np.flatnonzero((t > self.t_excl + 1e-9) & (np.abs(t / self.obs_step - np.round(t / self.obs_step)) < 1e-9))

    # ---- reference scale, identifiability, histories
    def reference_scale(self, R: Responses) -> float:
        """S0: mean over observation times and the candidate cells of the reference history (every parameter at REF, constant).  Independent of any
        truth and layout.  (Toy: domain mean / n_bins; at Belauri the 34 km padded domain is mostly zero, so the candidate-site mean is used.)"""
        s = self.sites
        f = np.tensordot(np.repeat(self.REF, self.nb), R.g, axes=1)
        return float(f[self.obs_index(R)][:, s["iy"], s["ix"]].mean())

    def live_mask(self, R: Responses, s0: float) -> np.ndarray:
        """(NP, n_fine) bool: parameter p in inverse bin b is live if its response at >= 1 candidate site reaches 0.5 x the 10 % noise sd (0.05 S0)
        at REF amplitude.  Dead pairs (e.g. an inflow basis whose faces are outflow in that bin) are never used in a truth or scored."""
        s, oi = self.sites, self.obs_index(R)
        peak = R.g[:, oi][:, :, s["iy"], s["ix"]].max(axis=(1, 2)).reshape(NP, self.nb) * self.REF[:, None]
        return np.repeat(peak > 0.05 * s0, self.nf // self.nb, axis=1)

    def make_history(self, family: str, rng: np.random.Generator, live: np.ndarray) -> tuple[np.ndarray, bool]:
        """Port of ``dynamic_study.make_history`` with amplitudes in units of REF (rows 0-2 ug/s, rows 3-4 ug/m^3); boundary events only in live windows."""
        nf, REF = self.nf, self.REF
        h = np.zeros((NP, nf))
        h[:3] = rng.uniform(0.6, 1.6, 3)[:, None] * REF[:3, None]
        bamp = rng.uniform(0.5, 1.5) * REF[3]
        k = int(rng.integers(3, 9))
        if family == "activation":
            h[0, :k] = 0.0
        elif family == "shutdown":
            h[1, k:] = 0.0
        elif family == "ramp":
            h[2] = h[2, 0] * np.linspace(1.0, 3.0, nf)
        elif family == "opposing":
            h[0, k:] *= 3.0
            h[1, k:] *= 0.3
        elif family in ("pulse", "influx_end", "compensating"):
            order = [3, 4] if rng.random() < 0.5 else [4, 3]
            for b in order:
                lv = live[b]
                if family == "pulse":
                    w = int(rng.integers(2, 4))
                    starts = [i for i in range(nf - w + 1) if lv[i:i + w].all()]
                    if starts:
                        i = int(rng.choice(starts))
                        h[b, i:i + w] = bamp
                        return h, True
                elif family == "influx_end":
                    ks = [kk for kk in range(2, 10) if lv[:kk].sum() >= 2 and lv[0]]
                    if ks:
                        kk = int(rng.choice(ks))
                        h[b, :kk] = bamp
                        return h, True
                else:
                    ks = [kk for kk in range(2, 9) if lv[kk:].sum() >= 2]
                    if ks:
                        kk = int(rng.choice(ks))
                        h[0, kk:] *= 0.3
                        h[b, kk:] = bamp
                        return h, True
            return h, False
        elif family != "constant":
            raise ValueError(family)
        return h, True

    # ---- truth
    def truth_field(self, segs: list[list[float]], hist: np.ndarray, rng: np.random.Generator, perturb: dict | None = None) -> np.ndarray:
        """Truth at dt/2 with change times jittered by up to +-35 % of a fine bin.  ``perturb`` (model error; the INVERSE stays nominal):
        rot_deg / speed_scale rotate / scale every wind segment, K_scale scales K, truth_res_m (250) simulates on the finer padded level and
        block-averages 2 x 2 onto the inverse grid (mesh mismatch; needs res 500)."""
        import dataclasses
        perturb = perturb or {}
        edges = dyn.bin_edges(self.T, self.nf)
        edges[1:-1] += rng.uniform(-0.35, 0.35, self.nf - 1) * (self.T / self.nf)
        tres = float(perturb.get("truth_res_m", self.res))
        sc = dyn.truth_scenario(self.scenario(segs, dt=self.dt_truth, res=tres), hist, edges)
        if "rot_deg" in perturb or "speed_scale" in perturb or "K_scale" in perturb:
            th, sp = np.deg2rad(perturb.get("rot_deg", 0.0)), perturb.get("speed_scale", 1.0)
            c, sn = np.cos(th), np.sin(th)
            ws = tuple(WindSegment(w.t_start, sp * (c * w.u - sn * w.v), sp * (sn * w.u + c * w.v)) for w in sc.transport.wind)
            sc = dataclasses.replace(sc, transport=dataclasses.replace(sc.transport, wind=ws, K=sc.transport.K * perturb.get("K_scale", 1.0)))
        out = np.asarray(simulate(sc, np.ones(NP)).concentration.values)
        if tres != self.res:
            f = int(round(self.res / tres))
            if abs(f * tres - self.res) > 1e-9 or f != 2:
                raise ValueError("mesh-mismatch truth supports a 2 x 2 block average only")
            nt, ny, nx = out.shape
            out = out.reshape(nt, ny // 2, 2, nx // 2, 2).mean(axis=(2, 4))
        return out

    # ---- scoring
    def time_weights(self, t: np.ndarray) -> np.ndarray:
        w = np.zeros(len(t))
        w[1:] += np.diff(t) / 2
        w[:-1] += np.diff(t) / 2
        return w / (t[-1] - t[0])

    def jpop(self, R: Responses, theta: np.ndarray, tf: np.ndarray) -> tuple[float, float]:
        err = np.tensordot(theta.reshape(-1), R.g, axes=1) - tf
        wt = self.time_weights(np.asarray(R.times))
        c = self.counts
        pop = (wt[:, None, None] * err ** 2 * (c / c.sum())[None]).sum()
        return float(np.sqrt(pop)), float(np.sqrt((wt[:, None, None] * err ** 2).mean(axis=(1, 2)).sum()))

    def projection_floor(self, R: Responses, tf: np.ndarray) -> float:
        """Smallest attainable population-time-weighted RMSE on the 6-bin nonnegative basis (NNLS of the true field); the J_pop floor of the gates."""
        t = np.asarray(R.times)
        w = np.sqrt(np.outer(self.time_weights(t), self.counts.reshape(-1) / self.counts.sum())).reshape(-1)
        m = w > 0
        cols = (R.g.reshape(R.g.shape[0], -1)[:, m] * w[m][None, :] * np.repeat(self.REF, self.nb)[:, None]).T
        x, _ = nnls(cols, tf.reshape(-1)[m] * w[m], maxiter=5000)
        return float(np.linalg.norm(cols @ x - tf.reshape(-1)[m] * w[m]))

    # ---- estimators
    def fit(self, model: str, A, y, sig, hp: tuple[float, float], prior, obs_t=None) -> np.ndarray | None:
        a, b = hp
        try:
            if model == "M0":
                return dyn.fit_history(A, y, sig, NP, self.nb, "M0", alpha=a, ref=self.REF).theta
            if model in ("CT", "CAUSAL"):
                return self._history_estimate(model, A, y, sig, obs_t, a, b)
            return dyn.fit_history(A, y, sig, NP, self.nb, "M1" if model == "M1" else "M2", alpha=a, beta=b if model == "M2" else 0.0,
                                   theta0=prior, ref=self.REF).theta
        except dyn.NotConverged:
            return None

    def _history_estimate(self, name: str, A, y, sig, obs_t, a: float, b: float, a0: float = 1.0) -> np.ndarray:
        """Port of ``dynamic_history.estimate`` (CT: bin-b observations and columns only; CAUSAL: M2 on data up to the end of bin b, bin-b estimate kept;
        both with the causal static prior).  Hyperparameters are BORROWED from M1 (CT) and M2 (CAUSAL), not tuned: exploratory."""
        e = dyn.bin_edges(self.T, self.nb)
        bi = np.clip(np.searchsorted(e, np.asarray(obs_t) - 1e-6, side="left") - 1, 0, self.nb - 1)
        A3 = A.reshape(len(y), NP, self.nb)
        th = np.zeros((NP, self.nb))
        for k in range(self.nb):
            mp_ = bi <= k
            if not mp_.any():
                continue
            pk = dyn.fit_history(A3[mp_][:, :, :k + 1].reshape(mp_.sum(), -1), y[mp_], sig[mp_], NP, k + 1, "M0", alpha=a0, ref=self.REF).theta
            if name == "CT":
                m = bi == k
                if m.any():
                    th[:, k] = dyn.fit_history(A3[m][:, :, k], y[m], sig[m], NP, 1, "M1", alpha=a, theta0=pk[:, k:k + 1], ref=self.REF).theta[:, 0]
            else:
                th[:, k] = dyn.fit_history(A3[mp_][:, :, :k + 1].reshape(mp_.sum(), -1), y[mp_], sig[mp_], NP, k + 1, "M2", alpha=a, beta=b,
                                           theta0=pk, ref=self.REF).theta[:, k]
        return th


_STUDIES: dict[str, Study] = {}


def get_study(cfg: dict[str, Any]) -> Study:
    key = json.dumps(cfg, sort_keys=True)
    if key not in _STUDIES:
        _STUDIES[key] = Study(cfg)
    return _STUDIES[key]


# ------------------------------------------------------------------------------------------------ winds and episodes
def wind_sets(cfg: dict[str, Any]) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    """ERA5 windows per split (design / validation / test) by the registered rule; returns ({split: [wind]}, report).  wind = {name, segs, meta}."""
    w = cfg["wind"]
    era = load_era5_wind(str(ROOT / cfg["region"]["era5"]))
    lon, lat = cfg["region"]["centroid_lonlat"]
    allw = make_windows(era, lon, lat, int(w["hours"]))
    med = speed_median(allw)
    out: dict[str, list[dict]] = {}
    clipped = {}
    for split in SPLITS:
        sel = select_windows(allw, split, int(w["n"][split]), int(cfg["seeds"]["wind_selection"]), med, float(w["turn_deg"]))
        out[split] = []
        clipped[split] = 0
        for x in sel:
            u, v, nc = clip_speed(x.u, x.v, float(w["vmax_m_s"]))
            clipped[split] += nc
            out[split].append({"name": x.start, "segs": [[3600.0 * k, float(u[k]), float(v[k])] for k in range(len(u))],
                               "meta": {"stratum": stratum(x, med, float(w["turn_deg"])), "mean_speed": x.mean_speed, "from_deg": x.from_deg,
                                        "turn_deg": x.turn_deg, "max_speed_unclipped": float(np.hypot(x.u, x.v).max())}})
    rep = {"n_windows_record": len(allw), "speed_median_m_s": med, "strata_table": strata_table(allw, med, float(w["turn_deg"])),
           "clipped_hours": clipped, "vmax_m_s": w["vmax_m_s"]}
    return out, rep


def episodes(cfg: dict[str, Any], stage: str, winds: list[dict], perturb: dict | None = None, n_seeds: int | None = None) -> list[dict]:
    """One episode per (wind, family, seed index); seeds consecutive from the stage's registered base (never shared across stages)."""
    n = int(cfg["seeds_per_wind_family"][stage]) if n_seeds is None else n_seeds
    base = int(cfg["seeds"][stage])
    eps = []
    for i, (w, f, j) in enumerate(itertools.product(winds, cfg["families"], range(n))):
        eps.append({"id": i, "wind": w["name"], "segs": w["segs"], "family": f, "seed": base + i, "stage": stage, "perturb": perturb})
    return eps


# ------------------------------------------------------------------------------------------------ layouts (design stage)
def spread_layout(S: Study, k: int) -> tuple[int, ...]:
    """Farthest-point: start at the candidate nearest the centroid of the candidates, then the one farthest from the chosen set (ties: lowest index)."""
    xy = S.sites["xy"]
    chosen = [int(np.argmin(((xy - xy.mean(axis=0)) ** 2).sum(1)))]
    while len(chosen) < k:
        d = np.min(np.linalg.norm(xy[:, None] - xy[chosen][None], axis=2), axis=1)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return tuple(sorted(chosen))


def highpop_layout(S: Study, k: int) -> tuple[int, ...]:
    """Top-k candidates by people in their cell with pairwise distance >= highpop_min_sep_m (ties: lowest index)."""
    s, sep = S.sites, float(S.cfg["highpop_min_sep_m"])
    pop = S.counts[s["iy"], s["ix"]]
    chosen: list[int] = []
    for i in np.argsort(-pop, kind="stable"):
        if all(np.linalg.norm(s["xy"][i] - s["xy"][j]) >= sep for j in chosen):
            chosen.append(int(i))
        if len(chosen) == k:
            return tuple(sorted(chosen))
    raise ValueError("not enough separated candidates for the high-population layout")


def lowpop_layout(S: Study, k: int) -> tuple[int, ...]:
    """Falsification control: the k least populated candidates (pairwise distance >= highpop_min_sep_m; ties: lowest index).  Must lose to ``spread``."""
    s, sep = S.sites, float(S.cfg["highpop_min_sep_m"])
    pop = S.counts[s["iy"], s["ix"]]
    chosen: list[int] = []
    for i in np.argsort(pop, kind="stable"):
        if all(np.linalg.norm(s["xy"][i] - s["xy"][j]) >= sep for j in chosen):
            chosen.append(int(i))
        if len(chosen) == k:
            return tuple(sorted(chosen))
    raise ValueError("not enough separated candidates for the low-population control layout")


def random_layout(S: Study, k: int, r: int) -> tuple[int, ...]:
    n = len(S.sites["ids"])
    return tuple(sorted(int(i) for i in np.random.default_rng([int(S.cfg["seeds"]["design_layouts"]), k, r]).choice(n, k, replace=False)))


def wind_operator(S: Study, segs: list[list[float]], noise: float, mode: str, prior_var: float):
    """Per-wind Gaussian linear-Bayes surrogate (ported from ``dynamic_design_study.wind_operators``): x = theta/REF, A over the candidate sites at the
    observation times, M^T M = population-time weighted Gram matrix (Gram trick).  mode: dyn (30 params) | static (bins collapsed) | src (identifiable columns, M = I)."""
    from ...dynamic_design import DesignProblem
    R, s = S.responses(segs), S.sites
    g, oi = np.asarray(R.g), S.obs_index(R)
    ref = np.repeat(S.REF, S.nb)
    A = np.moveaxis(g[:, oi][:, :, s["iy"], s["ix"]], (0, 2), (2, 0)).reshape(len(s["ids"]), -1, g.shape[0]) * ref
    sigma = noise * S.reference_scale(R)
    w = np.sqrt(np.outer(S.time_weights(np.asarray(R.times)), S.counts.reshape(-1) / S.counts.sum()).reshape(-1))
    gw = g.reshape(g.shape[0], -1) * w[None] * ref[:, None]
    Q = gw @ gw.T
    if mode == "static":
        T = np.kron(np.eye(NP), np.ones((S.nb, 1)))
        A, Q = A @ T, T.T @ Q @ T
    elif mode == "src":
        nrm = np.sqrt((A ** 2).sum(axis=(0, 1)))
        keep = nrm > 1e-6 * nrm.max()
        A, Q = A[:, :, keep], np.eye(int(keep.sum()))
    elif mode != "dyn":
        raise ValueError(mode)
    lam, V = np.linalg.eigh((Q + Q.T) / 2)
    M = (V * np.sqrt(np.maximum(lam, 0.0))).T
    return DesignProblem(np.asarray(A, dtype=float), M, np.full(A.shape[2], float(prior_var)), sigma)


def _prebuild_one(args: tuple[dict, list[list[float]]]) -> int:
    cfg, segs = args
    get_study(cfg).responses(segs)
    return 0


def prebuild(cfg: dict[str, Any], winds: list[dict], workers: int) -> None:
    """Fill the response cache (one independent set per wind) before any pool of episodes starts."""
    jobs = [(cfg, w["segs"]) for w in winds]
    if workers <= 1 or len(jobs) == 1:
        list(map(_prebuild_one, jobs))
        return
    _single_thread()
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(_prebuild_one, jobs))


def _single_thread() -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"


def design(cfg: dict[str, Any], design_winds: list[dict]) -> dict:
    """Surrogate layouts for every (noise, k): robust over the design winds (mean of the per-wind surrogate risks)."""
    from ..toy import dynamic_design_study as dds
    SumProblem, search = dds.SumProblem, dds.search
    dds.EXH_MAX, dds.N_EXTRA_STARTS = int(cfg["design"]["exh_max"]), int(cfg["design"]["n_extra_starts"])
    S = get_study(cfg)
    pv = float(cfg["design"]["prior_var"])
    doc: dict = {"candidates": S.sites["ids"], "dropped_shared_cell": S.sites["dropped_shared_cell"], "prior_var": pv, "winds": [w["name"] for w in design_winds],
                 "layouts": {}, "search": {}, "timings_s": {}}
    for noise in cfg["noise"]:
        idx = {k: {"spread": spread_layout(S, k), "highpop": highpop_layout(S, k)} for k in cfg["ks"]}
        lay = {k: {"random": [list(random_layout(S, k, r)) for r in range(int(cfg["n_random"]))], "spread": list(idx[k]["spread"]), "highpop": list(idx[k]["highpop"]), "lowpop": list(lowpop_layout(S, k))}
               for k in cfg["ks"]}
        srch: dict = {}
        for fam, mode, kind in (("static_opt", "static", "risk"), ("src_opt", "src", "info"), ("dyn_opt", "dyn", "risk")):
            t0 = time.time()
            prob = SumProblem([wind_operator(S, w["segs"], noise, mode, pv) for w in design_winds], kind)
            doc["timings_s"][f"{noise}|{fam}|build"] = time.time() - t0
            for k in cfg["ks"]:
                res = search(prob, k, tuple(idx[k].values()))
                srch[f"{fam}|{k}"] = res
                lay[k][fam] = list(res["subset"])
                idx[k][fam] = tuple(res["subset"])
                doc["timings_s"][f"{noise}|{fam}|{k}"] = res["seconds"]
            if fam == "dyn_opt":
                for k in cfg["ks"]:
                    lay[k]["surrogate_risk"] = {"random": [prob.risk(tuple(r)) for r in lay[k]["random"]], **{f: prob.risk(tuple(lay[k][f])) for f in FAMILY_ORDER if f != "random"}}
            del prob
        doc["layouts"][str(noise)] = {str(k): v for k, v in lay.items()}
        doc["search"][str(noise)] = srch
    doc["hash"] = _hash({"layouts": doc["layouts"], "candidates": doc["candidates"]})
    return doc


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def layouts_of(doc: dict, noise: float, k: int) -> dict[str, tuple[int, ...]]:
    """name -> candidate-index tuple; random replicates are 'random#0'.."""
    d = doc["layouts"][str(noise)][str(k)]
    out = {f: tuple(d[f]) for f in FAMILY_ORDER if f not in ("random",)}
    out.update({f"random#{r}": tuple(v) for r, v in enumerate(d["random"])})
    return out


# ------------------------------------------------------------------------------------------------ episode evaluation
def run_episode(cfg: dict[str, Any], ep: dict, doc: dict, hp: dict, models: tuple[str, ...], only: tuple[str, ...] | None = None,
                hist_layouts: tuple[str, ...] = ()) -> list[dict]:
    """One row per (k, noise, layout, model, candidate); truth, floor and noise draw are shared.  hp[f"{k}|{noise}"][model] = list of [alpha, beta]
    candidates (exactly one, FROZEN, outside tuning; hp[...]['M0'] must hold exactly one entry: the static prior mean of M1 / M2).  ``only`` restricts the layout names (tuning: ['spread']).  CT/CAUSAL are evaluated only on ``hist_layouts``."""
    S = get_study(cfg)
    s = S.sites
    R = S.responses(ep["segs"])
    s0 = S.reference_scale(R)
    live = S.live_mask(R, s0)
    rng = np.random.default_rng([ep["seed"], 0])
    hist, valid = S.make_history(ep["family"], rng, live)
    tf = S.truth_field(ep["segs"], hist, np.random.default_rng([ep["seed"], 1]), ep.get("perturb"))
    t, oi = np.asarray(R.times), S.obs_index(R)
    obs_t = t[oi]
    binned = hist.reshape(NP, S.nb, S.nf // S.nb).mean(axis=2)
    base = {"episode": ep["id"], "wind": ep["wind"], "family": ep["family"], "valid": valid, "stage": ep["stage"], "cond": ep.get("cond", "matched")}
    nan = float("nan")
    rows = [{**base, "k": 0, "layout": "-", "noise": 0.0, "model": "floor", "alpha": nan, "beta": nan, "jpop": S.projection_floor(R, tf), "area": nan,
             "src_local": nan, "src_bnd": nan, "ident_frac": nan, "s0": s0, "live_frac": float(live.mean())}]
    for k, (ni, frac) in itertools.product(cfg["ks"], enumerate(cfg["noise"])):
        cfg_hp = hp[f"{k}|{frac}"]
        sd = frac * s0
        lays = layouts_of(doc, frac, k)
        for name, cand in lays.items():
            if only is not None and name.split("#")[0] not in only:
                continue
            iy, ix = s["iy"][list(cand)], s["ix"][list(cand)]
            y = tf[oi][:, iy, ix] + np.random.default_rng([ep["seed"], 2, ni, k]).standard_normal((len(oi), len(iy))) * sd
            A = np.moveaxis(R.g[:, oi][:, :, iy, ix], 0, -1).reshape(-1, R.g.shape[0])
            y = y.reshape(-1)
            sig = np.full(y.size, sd)
            ot = np.repeat(obs_t, len(iy))
            sens = dyn.sensitivity(A * np.repeat(S.REF, S.nb), sig, NP, S.nb)
            ident = sens > 1e-6 * sens.max()
            a0, b0 = cfg_hp["M0"][0]
            prior = S.fit("M0", A, y, sig, (a0, b0), None)
            ms = tuple(models) + (HIST_MODELS if name.split("#")[0] in hist_layouts else ())
            cands = [(m_, c_) for m_ in ms for c_ in (cfg_hp["M2" if m_ == "CAUSAL" else ("M1" if m_ == "CT" else m_)] if m_ != "M0" else cfg_hp["M0"])]
            for model, (a, b) in cands:
                if model == "M0":
                    th = prior if (a, b) == (a0, b0) else S.fit("M0", A, y, sig, (a, b), None)
                else:
                    th = S.fit(model, A, y, sig, (a, b), prior, ot)
                row = {**base, "k": k, "layout": name, "noise": frac, "model": model, "alpha": a, "beta": b, "s0": s0, "live_frac": float(live.mean()),
                       "ident_frac": float(ident.mean())}
                if th is None or prior is None:
                    rows.append({**row, "jpop": float("inf"), "area": nan, "src_local": nan, "src_bnd": nan})
                    continue
                jp, ar = S.jpop(R, th, tf)
                e = (th - binned) / S.REF[:, None]
                rows.append({**row, "jpop": jp, "area": ar,
                             "src_local": float(np.sqrt(np.mean(e[:3][ident[:3]] ** 2))) if ident[:3].any() else nan,
                             "src_bnd": float(np.sqrt(np.mean(e[3:][ident[3:]] ** 2))) if ident[3:].any() else nan})
    return rows


def _task(args: tuple) -> list[dict]:
    return run_episode(*args)


def pool_rows(cfg: dict, eps: list[dict], doc: dict, hp: dict, models: tuple[str, ...], workers: int, only=None, hist_layouts=()) -> list[dict]:
    jobs = [(cfg, ep, doc, hp, models, only, hist_layouts) for ep in eps]
    if workers <= 1:
        return [r for j in jobs for r in _task(j)]
    _single_thread()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        return [r for rows in ex.map(_task, jobs) for r in rows]


# ------------------------------------------------------------------------------------------------ tuning (validation, spread layout only)
def tune(cfg: dict, doc: dict, val_eps: list[dict], workers: int) -> dict:
    """Pass 1: M0 alpha; pass 2: M1 alpha, M2 (alpha, beta) with M0 fixed as the static prior.  Spread layout only, per (k, noise); frozen afterwards.
    Non-converged candidates score +inf; selections on a grid edge are flagged."""
    am, a2 = _grid(cfg["hp_grid"]["alpha_m_exp"]), _grid(cfg["hp_grid"]["alpha_2_exp"])
    betas = (0.0, *_grid(cfg["hp_grid"]["beta_exp"]))
    prebuild(cfg, [{"segs": e["segs"], "name": e["wind"]} for e in {e["wind"]: e for e in val_eps}.values()], workers)
    keys = [f"{k}|{fr}" for k in cfg["ks"] for fr in cfg["noise"]]

    def table(rows: list[dict]) -> dict:
        sc: dict = {}
        for r in rows:
            if r["model"] != "floor":
                sc.setdefault((r["k"], r["noise"], r["model"]), {}).setdefault((r["alpha"], r["beta"]), []).append(r["jpop"])
        return {kk: {c: float(np.mean(v)) for c, v in d.items()} for kk, d in sc.items()}

    s1 = table([r for r in pool_rows(cfg, val_eps, doc, {k: {"M0": [(a, 0.0) for a in am]} for k in keys}, ("M0",), workers, only=("spread",))])
    m0 = {k: min(s1[(int(k.split("|")[0]), float(k.split("|")[1]), "M0")], key=lambda c, d=s1[(int(k.split("|")[0]), float(k.split("|")[1]), "M0")]: d[c]) for k in keys}
    grids = {k: {"M0": [list(m0[k])], "M1": [(a, 0.0) for a in am], "M2": list(itertools.product(a2, betas))} for k in keys}
    s2 = table(pool_rows(cfg, val_eps, doc, grids, ("M1", "M2"), workers, only=("spread",)))
    frozen: dict = {k: {"M0": [list(m0[k])]} for k in keys}
    edge = []
    for (k_, fr, m), d in s2.items():
        if m == "M0":
            continue
        key = f"{k_}|{fr}"
        best = min(d, key=lambda c, d=d: d[c])
        frozen[key][m] = [list(best)]
        al = am if m == "M1" else a2
        if best[0] in (al[0], al[-1]) or (m == "M2" and best[1] in (betas[1], betas[-1])):
            edge.append(f"{key}:{m}:{best}")
    for k in keys:
        if m0[k][0] in (am[0], am[-1]):
            edge.append(f"{k}:M0:{m0[k]}")
    return {"frozen": frozen, "edge_selections": edge, "n_validation_episodes": len(val_eps)}


# ------------------------------------------------------------------------------------------------ summary
def boot_ci(d: np.ndarray, level: float, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(np.percentile(m, 100 * level / 2)), float(np.percentile(m, 100 * (1 - level / 2)))


def vec_random_half(rows: list[dict], valid: set, k: int, nz: float, half: int) -> dict[int, float]:
    """Episode -> mean M2 J_pop over random replicates 0-4 (half 0) or 5-9 (half 1): the null control of the contrasts."""
    acc: dict[int, list[float]] = {}
    for r in rows:
        if r["k"] == k and r["noise"] == nz and r["model"] == "M2" and r["episode"] in valid and r["layout"].startswith("random#") and (int(r["layout"][7:]) >= 5) == bool(half):
            acc.setdefault(r["episode"], []).append(r["jpop"])
    return {e: float(np.mean(v)) if all(np.isfinite(v)) else float("nan") for e, v in acc.items()}


def summarise(rows: list[dict], cfg: dict) -> dict:
    """Paired contrasts per (k, noise): dyn_opt vs each baseline (M2, jpop; random = mean of replicates per episode), M2 vs M1 (spread, dyn_opt),
    CAUSAL vs CT (spread, dyn_opt), with the registered effect criterion: Bonferroni bootstrap CI excludes 0 AND >= min_rel_gain AND sign
    persists leaving each wind out; wind-cluster bootstrap also reported (as F-054)."""
    valid = {r["episode"] for r in rows if r["valid"]}
    wind_of = {r["episode"]: r["wind"] for r in rows}
    winds = sorted(set(wind_of.values()))
    level = 0.05 / (len(cfg["ks"]) * len(cfg["noise"]) * len(BASELINES))
    gain = float(cfg["min_rel_gain"])

    def vec(k, nz, layout, model):
        acc: dict[int, list[float]] = {}
        for r in rows:
            if r["k"] == k and r["noise"] == nz and r["model"] == model and r["episode"] in valid and r["layout"].split("#")[0] == layout:
                acc.setdefault(r["episode"], []).append(r["jpop"])
        return {e: float(np.mean(v)) if all(np.isfinite(v)) else float("nan") for e, v in acc.items()}

    def contrast(va, vb, lvl):
        eps = sorted(set(va) & set(vb))
        a, b = np.array([va[e] for e in eps]), np.array([vb[e] for e in eps])
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            return None
        eps_ok = [e for e, o in zip(eps, ok) if o]
        d, a, b = (a - b)[ok], a[ok], b[ok]
        lo, hi = boot_ci(d, lvl)
        rel = float(d.mean() / b.mean()) if b.mean() else float("nan")
        w = np.array([wind_of[e] for e in eps_ok])
        leave = {x: float(d[w != x].mean()) for x in winds if (w != x).any()}
        wm = np.array([d[w == x].mean() for x in winds if (w == x).any()])
        wb = wm[np.random.default_rng(1).integers(0, len(wm), (10000, len(wm)))].mean(axis=1)
        return {"n": int(ok.sum()), "mean_diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "rel_change": rel, "frac_a_better": float((d < 0).mean()),
                "wind_cluster_ci": [float(np.percentile(wb, 100 * lvl / 2)), float(np.percentile(wb, 100 * (1 - lvl / 2)))],
                "mean_a": float(a.mean()), "mean_b": float(b.mean()), "leave_one_wind_out": leave,
                "effect_criterion_met": bool(hi < 0 and rel <= -gain and all(v < 0 for v in leave.values()))}

    out = []
    for k, nz in itertools.product(cfg["ks"], cfg["noise"]):
        for b in BASELINES:
            c = contrast(vec(k, nz, "dyn_opt", "M2"), vec(k, nz, b, "M2"), level)
            if c:
                out.append({"k": k, "noise": nz, "contrast": f"dyn_opt-{b} (M2)", **c})
        for lay in ("spread", "dyn_opt"):
            for a, b2 in (("M2", "M1"), ("CAUSAL", "CT")):
                c = contrast(vec(k, nz, lay, a), vec(k, nz, lay, b2), 0.05 / (len(cfg["ks"]) * len(cfg["noise"])))
                if c:
                    out.append({"k": k, "noise": nz, "layout": lay, "contrast": f"{a}-{b2}", **c})
    controls = []
    for k, nz in itertools.product(cfg["ks"], cfg["noise"]):
        bad = contrast(vec(k, nz, "lowpop", "M2"), vec(k, nz, "spread", "M2"), level)
        if bad:
            controls.append({"k": k, "noise": nz, "control": "lowpop-spread (M2): must be > 0", "passed": bool(bad["mean_diff"] > 0), **bad})
        nul = contrast(vec_random_half(rows, valid, k, nz, 0), vec_random_half(rows, valid, k, nz, 1), level)
        if nul:
            controls.append({"k": k, "noise": nz, "control": "random replicates 0-4 vs 5-9 (M2): registered effect criterion must NOT be met",
                             "passed": not nul["effect_criterion_met"], **nul})
    floor = [r["jpop"] for r in rows if r["model"] == "floor" and r["valid"]]
    return {"summary": out, "controls": controls, "controls_all_passed": all(c["passed"] for c in controls) if controls else None, "mean_projection_floor": float(np.mean(floor)) if floor else float("nan"), "n_invalid_episodes": len({r["episode"] for r in rows if not r["valid"]}),
            "n_nonconverged_fits": int(sum(1 for r in rows if not np.isfinite(r["jpop"]))), "bonferroni_level": level, "min_rel_gain": gain}


def summarise_model_error(rows: list[dict], cfg: dict) -> dict:
    """Mean M2 J_pop per (cond, k, noise, layout family) relative to the matched condition on the same episodes; M2 vs M1 relative change per condition."""
    valid = {r["episode"] for r in rows if r["valid"]}
    out = []
    for cond in sorted({r["cond"] for r in rows}):
        for k, nz, lay in itertools.product(cfg["ks"], cfg["noise"], FAMILY_ORDER):
            sel = lambda c, m: [r["jpop"] for r in rows if r["cond"] == c and r["k"] == k and r["noise"] == nz and r["model"] == m and r["layout"].split("#")[0] == lay and r["episode"] in valid]  # noqa: E731
            a2, a1, m2 = sel(cond, "M2"), sel(cond, "M1"), sel("matched", "M2")
            if a2 and m2:
                out.append({"cond": cond, "k": k, "noise": nz, "layout": lay, "M2_mean": float(np.mean(a2)), "M2_ratio_vs_matched": float(np.mean(a2) / np.mean(m2)),
                            "M2_minus_M1_rel": float((np.mean(a2) - np.mean(a1)) / np.mean(a1)) if a1 else float("nan")})
    return {"model_error": out}


def model_error(cfg: dict, winds: dict, doc: dict, fz: dict, workers: int, test_rows: list[dict] | None = None) -> dict:
    """Model-error conditions (truth perturbed, inverse nominal, layouts and hyperparameters frozen) on a SUBSET of the test episodes: the first seed of
    every (wind, family).  The matched condition is the test stage's own rows for the same episodes (re-run only if ``test_rows`` is None)."""
    n_per = int(cfg["seeds_per_wind_family"]["test"])
    sub = [e for e in episodes(cfg, "test", winds["test"]) if e["id"] % n_per == 0]
    prebuild(cfg, winds["test"], workers)
    ids = {e["id"] for e in sub}
    rows = [r for r in (test_rows or []) if r["episode"] in ids and r["model"] in ("floor", "M1", "M2")]
    if test_rows is None:
        rows = pool_rows(cfg, [{**e, "cond": "matched"} for e in sub], doc, fz["frozen"], ("M1", "M2"), workers)
    for name, pert in cfg["model_error"]["conditions"].items():
        rows += pool_rows(cfg, [{**e, "perturb": pert, "cond": name} for e in sub], doc, fz["frozen"], ("M1", "M2"), workers)
    return summarise_model_error(rows, cfg)


# ------------------------------------------------------------------------------------------------ cost model
def cost_model(cfg: dict) -> dict[str, Any]:
    """Core-hours = COUNTS x measured constants (cfg['cost']); every constant must have been measured (docs/dyn08_plan.md section 3).  A fitted row costs
    fit_<model>_ms + row_overhead_ms (the J_pop evaluation); an episode costs one truth solve + episode_fixed_s + its rows."""
    c = cfg["cost"]
    need = ["solve_500_dt50_s", "response_set_31_solves_s", "fit_M2_ms", "fit_M1_ms", "fit_M0_ms", "fit_history_ms", "design_search_s_per_noise", "row_overhead_ms", "tune_cell_episode_s",
            "episode_fixed_s"]
    if any(c.get(n) is None for n in need):
        return {"error": f"cost constants not measured: {[n for n in need if c.get(n) is None]}"}
    nw = cfg["wind"]["n"]
    nf, ks, nz = len(cfg["families"]), len(cfg["ks"]), len(cfg["noise"])
    n_val = nw["validation"] * nf * cfg["seeds_per_wind_family"]["validation"]
    n_test = nw["test"] * nf * cfg["seeds_per_wind_family"]["test"]
    conds = cfg["model_error"]["conditions"]
    n_me = nw["test"] * nf  # first seed of each (wind, family): a subset of the test episodes
    am = len(_grid(cfg["hp_grid"]["alpha_m_exp"]))
    n2 = len(_grid(cfg["hp_grid"]["alpha_2_exp"])) * (1 + len(_grid(cfg["hp_grid"]["beta_exp"])))
    n_layouts = len(BASELINES) - 1 + 1 + 1 + cfg["n_random"]  # spread, highpop, static, src + dyn + lowpop control + random replicates
    h, ms, ro = 3600.0, 1e-3, c["row_overhead_ms"]
    prior = c["fit_M0_ms"] + ro
    parts = {
        "response sets (all winds)": sum(nw.values()) * c["response_set_31_solves_s"] / h,
        "design search": nz * c["design_search_s_per_noise"] / h,
        "tune: truth solves": n_val * c["solve_500_dt50_s"] / h,
        "tune: fits (full grid, spread layout)": n_val * (c["episode_fixed_s"] + ks * nz * c["tune_cell_episode_s"]) / h,
        "test: truth solves": n_test * c["solve_500_dt50_s"] / h,
        "test: fits (M0, M1, M2 on 6 designed + 10 random layouts; CT, CAUSAL on 2)": n_test * (c["episode_fixed_s"] + ks * nz * (n_layouts * (c["fit_M0_ms"] + c["fit_M1_ms"] + c["fit_M2_ms"] + 3 * ro)
                                                                         + 2 * (c["fit_history_ms"] + 2 * ro)) * ms) / h,
    }
    truth_s = {k: (c.get("solve_250_dt50_s") if v.get("truth_res_m") == 250.0 else c["solve_500_dt50_s"]) for k, v in conds.items()}
    if any(v is None for v in truth_s.values()):
        return {"error": "solve_250_dt50_s not measured"}
    parts["model error: truth solves"] = n_me * sum(truth_s.values()) / h
    parts["model error: fits (M0, M1, M2)"] = n_me * len(conds) * (c["episode_fixed_s"] + ks * nz * n_layouts * (c["fit_M0_ms"] + c["fit_M1_ms"] + c["fit_M2_ms"] + 3 * ro) * ms) / h
    tot = sum(parts.values()) * c["cluster_speed_factor"]
    return {"parts_core_h": {k: v * c["cluster_speed_factor"] for k, v in parts.items()}, "total_core_h": tot, "contingency": c["contingency"],
            "total_with_contingency": tot * c["contingency"], "n_episodes": {"validation": n_val, "test": n_test, "model_error_per_condition": n_me},
            "n_truth_solves_500": n_val + n_test + n_me * sum(1 for v in conds.values() if v.get("truth_res_m") != 250.0),
            "n_truth_solves_250": n_me * sum(1 for v in conds.values() if v.get("truth_res_m") == 250.0), "n_response_sets": sum(nw.values())}


# ------------------------------------------------------------------------------------------------ stages
def stage_dry_run(cfg: dict) -> None:
    winds, rep = wind_sets(cfg)
    print(f"DYN-08 reduced design '{cfg['name']}': mesh {cfg['mesh']}, T {cfg['physics']['T']} s, bins {cfg['bins']}, ks {cfg['ks']}, noise {cfg['noise']}")
    print("seed conflicts with spent/declared ranges:", seed_conflicts(cfg) or "none")
    print(f"ERA5 windows in record: {rep['n_windows_record']}, speed median {rep['speed_median_m_s']:.3f} m/s, clipped hours {rep['clipped_hours']}")
    for sp, tab in rep["strata_table"].items():
        print(f"  {sp:10s} strata (windows available): {dict(sorted(tab.items()))}")
    for sp, ws in winds.items():
        print(f"  {sp}: " + "; ".join(f"{w['name']} {w['meta']['stratum']} v={w['meta']['mean_speed']:.2f} turn={w['meta']['turn_deg']:.0f}" for w in ws))
    nf = len(cfg["families"])
    nv = len(winds["validation"]) * nf * cfg["seeds_per_wind_family"]["validation"]
    nt = len(winds["test"]) * nf * cfg["seeds_per_wind_family"]["test"]
    print(f"episodes: validation {nv} (seeds {cfg['seeds']['validation']}..{cfg['seeds']['validation'] + nv - 1}), test {nt} (seeds {cfg['seeds']['test']}..{cfg['seeds']['test'] + nt - 1})")
    n_solves = sum(len(v) for v in winds.values()) * (NP * cfg["bins"]["n_inv"] + 1)
    print(f"response-set solves: {n_solves} ({sum(len(v) for v in winds.values())} winds x {NP * cfg['bins']['n_inv'] + 1}); truth solves: {nv + nt} + model error")
    cm = cost_model(cfg)
    print(json.dumps(cm, indent=1))


def linearity_check(S: Study, segs: list[list[float]], R: Responses) -> float:
    """Pilot gate 1: the response set reproduces a direct solve of the same piecewise-constant history on the inverse mesh and the exact bin edges
    (max |linear - direct| / max |direct|)."""
    rng = np.random.default_rng(5)
    hist = rng.uniform(0.2, 1.5, (NP, S.nb)) * S.REF[:, None]
    sc = dyn.truth_scenario(S.scenario(segs), hist, dyn.bin_edges(S.T, S.nb))
    direct = np.asarray(simulate(sc, np.ones(NP)).concentration.values)
    lin = np.asarray(R.field(hist.reshape(-1)).values) - np.asarray(R.base)
    return float(np.abs(lin - direct).max() / max(float(np.abs(direct).max()), 1e-300))


def stage_time(cfg: dict, out: Path) -> dict:
    """Measured single-core costs: one response set (31 solves, 8 ERA5 segments), truth solves at dt/2 (500 m and 250 m), M0/M1/M2/CT/CAUSAL fits at k = 3, 5,
    and the surrogate search on two winds.  Local, one process; nothing is written except timing.json."""
    _single_thread()
    S = get_study(cfg)
    winds, _ = wind_sets(cfg)
    w = winds["design"][0]
    t0 = time.perf_counter()
    R = S.responses(w["segs"])
    t_resp = time.perf_counter() - t0
    rng = np.random.default_rng(1)
    s0 = S.reference_scale(R)
    hist, _ = S.make_history("constant", rng, S.live_mask(R, s0))
    meas: dict[str, Any] = {"wind": w["name"], "response_set_31_solves_s": t_resp, "n_obs_times": int(len(S.obs_index(R))), "s0": s0}
    t0 = time.perf_counter()
    tf = S.truth_field(w["segs"], hist, np.random.default_rng(2))
    meas["solve_500_dt50_s"] = time.perf_counter() - t0
    # a single response-set solve at dt 100 (the 31-solve set already includes the base solve), measured separately
    sc = S.scenario(w["segs"])
    t0 = time.perf_counter()
    simulate(sc, np.zeros(len(sc.param_table())))
    meas["solve_500_dt100_s"] = time.perf_counter() - t0
    if S.res == 500.0:
        t0 = time.perf_counter()
        S.truth_field(w["segs"], hist, np.random.default_rng(2), {"truth_res_m": 250.0})
        meas["solve_250_dt50_s"] = time.perf_counter() - t0
    meas["linearity_rel_diff"] = linearity_check(S, w["segs"], R)  # registered pilot gate 1 (< 1e-6)
    # fits
    s = S.sites
    for k in cfg["ks"]:
        cand = spread_layout(S, k)
        iy, ix = s["iy"][list(cand)], s["ix"][list(cand)]
        oi = S.obs_index(R)
        sd = 0.10 * s0
        y = (tf[oi][:, iy, ix] + np.random.default_rng(3).standard_normal((len(oi), k)) * sd).reshape(-1)
        A = np.moveaxis(R.g[:, oi][:, :, iy, ix], 0, -1).reshape(-1, R.g.shape[0])
        sig = np.full(y.size, sd)
        ot = np.repeat(np.asarray(R.times)[oi], k)
        for model, hp in (("M0", (1.0, 0.0)), ("M1", (10.0, 0.0)), ("M2", (10.0, 100.0)), ("CT", (10.0, 100.0)), ("CAUSAL", (10.0, 100.0))):
            prior = S.fit("M0", A, y, sig, (1.0, 0.0), None)
            t0 = time.perf_counter()
            reps, th = 5, None
            for _ in range(reps):
                th = S.fit(model, A, y, sig, hp, prior, ot)
            meas[f"fit_{model}_k{k}_ms"] = (time.perf_counter() - t0) / reps * 1e3
            meas[f"fit_{model}_k{k}_converged"] = th is not None
    two = [wd for wd in winds["design"][:2]]
    t0 = time.perf_counter()
    S.responses(two[1]["segs"])
    meas["response_set_31_solves_s_2nd"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    d = design({**cfg, "noise": [0.10]}, two)
    meas["design_search_2winds_1noise_s"] = time.perf_counter() - t0
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "timing.json", meas)
    print("TIMING_JSON " + json.dumps(meas), flush=True)
    return meas


def _write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _load_frozen(path: Path) -> dict:
    fz = json.loads(path.read_text())
    if fz["hash"] != _hash(fz["frozen"]):
        raise RuntimeError("freeze guard: hyperparameter file does not match its hash")
    return fz


def _load_layouts(path: Path) -> dict:
    doc = json.loads(path.read_text())
    if doc["hash"] != _hash({"layouts": doc["layouts"], "candidates": doc["candidates"]}):
        raise RuntimeError("freeze guard: layouts file does not match its hash")
    return doc


def stage_smoke(cfg: dict, out: Path) -> None:
    """End-to-end on the 2000 m padded level: design -> tune -> test -> model error -> summaries, in a few minutes on one core."""
    c = smoke_cfg(cfg)
    out.mkdir(parents=True, exist_ok=True)
    winds, rep = wind_sets(c)
    S = get_study(c)
    print(f"smoke: {len(S.sites['ids'])} candidate cells of {S.sites['n_table']} sites ({S.sites['dropped_shared_cell']} dropped, shared cell); "
          f"population study total {S.counts.sum():.1f}")
    doc = design(c, winds["design"])
    write_json(out / "layouts.json", doc)
    val = episodes(c, "validation", winds["validation"])
    tu = tune(c, doc, val, 1)
    fz = {"hash": _hash(tu["frozen"]), **tu}
    write_json(out / "frozen_hparams.json", fz)
    test = episodes(c, "test", winds["test"])
    rows = pool_rows(c, test, doc, tu["frozen"], ("M0", "M1", "M2"), 1, hist_layouts=("spread", "dyn_opt"))
    _write_rows(out / "episodes.csv", rows)
    res = summarise(rows, c)
    res.update(model_error(c, winds, doc, fz, 1, rows))
    write_json(out / "summary.json", {"frozen_hash": fz["hash"], "layouts_hash": doc["hash"], "n_test_episodes": len(test), **res,
                                      "note": "SMOKE on the 2000 m padded level: plumbing check only, no scientific content"})
    print(f"smoke ok: {len(rows)} rows, nonconverged {res['n_nonconverged_fits']}, invalid episodes {res['n_invalid_episodes']}, layouts {doc['hash']}, hparams {fz['hash']}")
    for r in res["summary"][:4]:
        print(" ", r["k"], r["noise"], r["contrast"], f"{r['mean_diff']:+.3g} rel {r['rel_change']:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("dry-run", "smoke", "time", "design", "tune", "test", "modelerr", "all"), required=True)
    ap.add_argument("--config", default="configs/dyn08.json")
    ap.add_argument("--out", default=os.environ.get("RESEARCH_RUN_DIR", "results/dyn08"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--layouts", default=None, help="layouts.json (tune/test/modelerr); default OUT/layouts.json")
    ap.add_argument("--freeze", default=None, help="frozen_hparams.json (test/modelerr); default OUT/frozen_hparams.json")
    a = ap.parse_args()
    cfg, out = load_cfg(a.config), Path(a.out)
    if a.stage == "dry-run":
        return stage_dry_run(cfg)
    if a.stage == "smoke":
        return stage_smoke(cfg, out)
    if a.stage == "time":
        stage_time(cfg, out)
        return
    if seed_conflicts(cfg):
        raise SystemExit(f"seed conflict: {seed_conflicts(cfg)}")
    out.mkdir(parents=True, exist_ok=True)
    winds, rep = wind_sets(cfg)
    write_json(out / "winds.json", {"report": rep, "winds": {k: [{"name": w["name"], "meta": w["meta"]} for w in v] for k, v in winds.items()}})
    layouts_path, freeze_path = Path(a.layouts or out / "layouts.json"), Path(a.freeze or out / "frozen_hparams.json")
    if a.stage in ("design", "all"):
        prebuild(cfg, winds["design"], a.workers)
        doc = design(cfg, winds["design"])
        write_json(layouts_path, doc)
        print("LAYOUTS_JSON " + json.dumps({"hash": doc["hash"], "layouts": doc["layouts"], "surrogate": {n: {k: v["surrogate_risk"] for k, v in d.items()} for n, d in doc["layouts"].items()}}), flush=True)
    if a.stage == "design":
        return
    doc = _load_layouts(layouts_path)
    if a.stage in ("tune", "all"):
        r = tune(cfg, doc, episodes(cfg, "validation", winds["validation"]), a.workers)
        write_json(freeze_path, {"hash": _hash(r["frozen"]), **r})
        print("FROZEN_JSON " + json.dumps({"hash": _hash(r["frozen"]), "frozen": r["frozen"], "edge_selections": r["edge_selections"]}), flush=True)
        frac = len(r["edge_selections"]) / (len(r["frozen"]) * 3)
        if a.stage == "all" and frac > 0.25:  # registered gate (docs/dyn08_plan.md): extend the grid and re-register; never open the test seeds
            raise SystemExit(f"GATE: {frac:.0%} of the selected hyperparameters are on a grid edge (> 25 %): stopping before the test stage")
    if a.stage == "tune":
        return
    fz = _load_frozen(freeze_path)
    rows: list[dict] | None = None
    if a.stage in ("test", "all"):
        prebuild(cfg, winds["test"], a.workers)
        eps = episodes(cfg, "test", winds["test"])
        rows = pool_rows(cfg, eps, doc, fz["frozen"], ("M0", "M1", "M2"), a.workers, hist_layouts=("spread", "dyn_opt"))
        _write_rows(out / "episodes.csv", rows)
        res = summarise(rows, cfg)
        write_json(out / "summary.json", {"frozen_hash": fz["hash"], "layouts_hash": doc["hash"], "n_episodes": len(eps), **res})
        print("SUMMARY_JSON " + json.dumps({"frozen_hash": fz["hash"], "layouts_hash": doc["hash"], **res}), flush=True)
    if a.stage == "test":
        return
    me = model_error(cfg, winds, doc, fz, a.workers, rows)
    write_json(out / "modelerr_summary.json", {"frozen_hash": fz["hash"], "layouts_hash": doc["hash"], **me})
    print("MODELERR_JSON " + json.dumps(me), flush=True)


if __name__ == "__main__":
    main()

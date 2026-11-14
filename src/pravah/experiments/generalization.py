"""E16 generalization: one-at-a-time physical/source sensitivity cells, per-scenario evaluation under both estimator
memories, and the predefined ranking-stability statistics. Design: docs/research/generalization-ablations.md (committed
before any E16 result). Contract: tests/test_generalization.py.

Every cell changes the WORLD (truth and estimator share the changed physics); nothing here is a model mismatch. Designs,
the map24 estimator, readings and CIs are E07's (`cross_objective`); map_grow is E11's `forecast_design.estimate`; the
scorer and k* are E08's (`sufficiency`). Units: ug m^-3, ug s^-1, m, s (config), hours (calendar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from pravah.experiments import cross_objective as X
from pravah.experiments import forecast_design as FD
from pravah.experiments import sufficiency as S

# cell -> {"section.field": value}; design note §3. Values in the config's units (m^2 s^-1, s^-1, s, m, ug m^-3).
VANLEER = {"model.advection_scheme": "vanleer", "model.dt_s": 60}  # 32^2: max |u| 4.6 m/s (seed 4) -> Courant 0.88 (deviation 1)
CELLS = {
    "nominal": {},
    "diff_hi": {"model.diffusivity_m2_s": 1000.0},
    "diff_lo": dict(VANLEER),
    "loss_hi": {"model.removal_per_s": 1 / (2 * 3600)},
    "src_wide": {"sources.footprint_sigma_m": 800.0},
    "src_sparse": {"sources.min_separation_m": 3000.0},
    "bg_hi": {"sources.background_ug_m3": 20.0},
    "diff_lo_x_loss_hi": {**VANLEER, "model.removal_per_s": 1 / (2 * 3600)},  # the one declared interaction
}
# admissible range and unit of every field an axis may touch (physical plausibility + solver validity)
RANGES = {
    "model.diffusivity_m2_s": (1.0, 5000.0, "m^2 s^-1"),
    "model.removal_per_s": (0.0, 1 / 3600, "s^-1"),           # e-folding no shorter than 1 h
    "model.dt_s": (1, 3600, "s"),
    "model.advection_scheme": (("upwind", "vanleer"), None, "-"),
    "sources.footprint_sigma_m": (100.0, 2000.0, "m"),
    "sources.min_separation_m": (0.0, 5000.0, "m"),
    "sources.background_ug_m3": (0.0, 100.0, "ug m^-3"),
}
STRATEGIES = ("random", "coverage", "source_near", "dopt")
ESTIMATORS = {"map24": X.WINDOW_H, "map_grow": None}
KS, TAU, K_REF = (2, 4, 8, 16), 0.2, 8
# conclusion -> (estimator, A, B): rho = (RMSE_B - RMSE_A) / RMSE_B > 0 favours A (expected sign + at nominal)
CONTRASTS = {
    "C1_sn_vs_dopt_map24": ("map24", "source_near", "dopt"),
    "C2_dopt_vs_random_map24": ("map24", "dopt", "random"),
    "C3_dopt_vs_sn_grow": ("map_grow", "dopt", "source_near"),
}
MEMORY = {"C4_memory_dopt": "dopt", "C4_memory_sn": "source_near"}  # M = (map24 - map_grow) / map24


def validate_cell(overrides):
    """Raise ValueError if any override is not an axis field or lies outside its admissible range."""
    for key, v in overrides.items():
        if key not in RANGES:
            raise ValueError(f"{key!r} is not a declared sensitivity axis")
        lo, hi, _ = RANGES[key]
        if isinstance(lo, tuple):
            if v not in lo:
                raise ValueError(f"{key}={v!r} not in {lo}")
        elif not lo <= v <= hi:
            raise ValueError(f"{key}={v} outside [{lo}, {hi}] {RANGES[key][2]}")


def apply_cell(cfg, overrides):
    """Set the overrides on cfg (in place), validate, return cfg."""
    validate_cell(overrides)
    for key, v in overrides.items():
        sec, attr = key.split(".")
        setattr(getattr(cfg, sec), attr, type(getattr(getattr(cfg, sec), attr))(v))
    cfg.validate()
    return cfg


def dimensionless(cfg, U=3.0):
    """Nominal-wind dimensionless groups; K_eff from the A2 modified-equation estimate (upwind: K + U dx/2 + U^2 dt/2;
    vanleer: K, a lower bound since it over-diffuses puffs only a few cells wide at 32^2)."""
    m, s = cfg.model, cfg.sources
    L, dx = m.domain_m[0], m.domain_m[0] / m.grid[0]
    keff = m.diffusivity_m2_s + (U * dx / 2 + U ** 2 * m.dt_s / 2 if m.advection_scheme == "upwind" else 0.0)
    return dict(K_eff_m2_s=keff, Pe=U * L / keff, Pe_phys=U * L / m.diffusivity_m2_s, Da=m.removal_per_s * L / U,
                footprint_over_L=s.footprint_sigma_m / L, spacing_over_L=s.min_separation_m / L, grid_Pe=U * dx / m.diffusivity_m2_s,
                courant=U * m.dt_s / dx)


# ---------------------------------------------------------------- per-scenario evaluation
def evaluate(sc, F, G3, ks=KS, R=5):
    """Rows (one per k x strategy x estimator; random = mean over its N_RANDOM networks) for one scenario. Designs see
    training-hour responses only; estimators see readings up to each issue time; truth enters only the scorer."""
    K = sc.n_sources
    a_ref, c_bg, a_true = sc.cfg.sources.reference_rate_ug_s, sc.cfg.sources.background_ug_m3, sc.truth.strengths_ug_s
    issue, n_h = X.issue_times(sc), len(sc.truth.winds)
    reg = S.issue_regimes(sc, issue)
    scorer = S.GramScorer(F, sc.truth.field, issue, a_true, a_ref, c_bg, {g: reg == g for g in np.unique(reg)})
    sigma = X.site_sigma(sc, False)
    p = X.problem(X.design_input(sc, G3, sigma))
    T, seedp = len(sc.split_hours("train")), sc.cfg.seeds.streams(sc.replicate)["placement"]
    Z = np.stack([(X.readings(sc, sigma, "white", r, n_h) - c_bg) / sigma[:, None] for r in range(R)], -1)  # paired draws
    rows = [dict(k=0, strategy="background_only", estimator="-", status="ok", sigma=float(sigma[0]),
                 **scorer(np.zeros((len(issue), K))))]
    for k in ks:
        for strat in STRATEGIES:
            per, status = {e: [] for e in ESTIMATORS}, "ok"
            for j in range(X.N_RANDOM) if strat == "random" else [0]:
                try:
                    sites, _ = X.make_design(strat, p, k, T, seedp, j)
                except ValueError as ex:  # undefined design: a failure, never dropped
                    status = str(ex)
                    break
                A = a_ref * G3[sites] / sigma[sites][:, None, None]
                for e, W in ESTIMATORS.items():
                    u = FD.estimate(A, Z[sites], issue, W)  # (n_issue, K, R)
                    m = [scorer(u[..., r]) for r in range(R)]
                    per[e].append({c: float(np.mean([x[c] for x in m])) for c in m[0]})
            for e in ESTIMATORS:
                r = dict(k=k, strategy=strat, estimator=e, status=status, sigma=float(sigma[0]))
                if status == "ok":
                    r.update({c: float(np.mean([x[c] for x in per[e]])) for c in per[e][0]})
                rows.append(r)
    return rows


# ---------------------------------------------------------------- statistics (scenario = (K, seed) is the unit)
def with_nrmse(df):
    """Adds nforecast = forecast RMSE / background-only RMSE of the same (cell, K, seed); undefined -> inf."""
    bg = df[df.strategy == "background_only"].set_index(["cell", "K", "seed"]).forecast_rmse
    d = df[df.strategy != "background_only"].copy()
    d["nforecast"] = d.forecast_rmse.where(d.status == "ok", np.inf).to_numpy() / bg.reindex(
        pd.MultiIndex.from_frame(d[["cell", "K", "seed"]])).to_numpy()
    return d


def _pivot(d, est, k):
    q = d[(d.estimator == est) & (d.k == k)]
    return q.pivot_table(index=["cell", "K", "seed"], columns="strategy", values="nforecast", dropna=False)


def contrast_rows(d, k=K_REF):
    """Per (conclusion, cell, K, seed): rho. Undefined designs give -inf for A (worst)."""
    out = []
    for name, (est, a, b) in CONTRASTS.items():
        pv = _pivot(d, est, k)
        out.append(((pv[b] - pv[a]) / pv[b]).rename("rho").reset_index().assign(conclusion=name))
    p24, pg = _pivot(d, "map24", k), _pivot(d, "map_grow", k)
    for name, s in MEMORY.items():
        out.append(((p24[s] - pg[s]) / p24[s]).rename("rho").reset_index().assign(conclusion=name))
    r = pd.concat(out, ignore_index=True)
    r["rho"] = r.rho.replace(np.nan, -np.inf)
    return r


def verdict(lo, hi, med):
    if lo > 0:
        return "holds"
    if hi < 0:
        return "reversed"
    return "weakened" if med > 0 else "flipped (indicative)"


def verdict_table(rho):
    """Design note §5 steps 1-3 for every (conclusion, cell)."""
    out = []
    for (c, cell), g in rho.groupby(["conclusion", "cell"]):
        g = g.set_index(["K", "seed"]).rho
        nom = rho[(rho.conclusion == c) & (rho.cell == "nominal")].set_index(["K", "seed"]).rho
        lo, hi = X.df_median_ci(g.to_numpy())
        pos, neg, tie, p = X.sign_test(g.to_numpy())
        dlt = (g - nom.reindex(g.index)).dropna().to_numpy()
        dlo, dhi = X.df_median_ci(dlt) if len(dlt) >= 2 else (np.nan, np.nan)
        out.append(dict(conclusion=c, cell=cell, n=len(g), median=float(np.median(g)), ci_lo=lo, ci_hi=hi, wins=pos,
                        losses=neg, ties=tie, sign_p=p, verdict=verdict(lo, hi, np.median(g)),
                        delta_vs_nominal=float(np.median(dlt)) if len(dlt) else np.nan, delta_ci_lo=dlo, delta_ci_hi=dhi))
    return pd.DataFrame(out)


def kendall_table(d, k=K_REF):
    """§5 step 4: per scenario, Kendall tau of the 4-strategy ranking (cell vs nominal) and whether the top strategy stays."""
    out = []
    for est in ESTIMATORS:
        pv = _pivot(d, est, k)[list(STRATEGIES)]
        nom = pv.xs("nominal", level="cell")
        for (cell, K, seed), v in pv.iterrows():
            n = nom.loc[(K, seed)]
            out.append(dict(estimator=est, cell=cell, K=K, seed=seed, tau=float(kendalltau(v.to_numpy(), n.to_numpy())[0]),
                            top=v.idxmin(), top_nominal=n.idxmin(), same_top=bool(v.idxmin() == n.idxmin())))
    t = pd.DataFrame(out)
    s = t.groupby(["estimator", "cell"]).agg(n=("tau", "size"), median_tau=("tau", "median"), mean_tau=("tau", "mean"),
                                              frac_same_top=("same_top", "mean")).reset_index()
    s["top_counts"] = [" ".join(f"{a}:{b}" for a, b in t[(t.estimator == e) & (t.cell == c)].top.value_counts().items())
                       for e, c in zip(s.estimator, s.cell)]
    return t, s


def kstar_table(d, tau=TAU):
    """C5: k*(tau) per (cell, K, seed, strategy) under map24 (E08 REF) and the §5 verdict per cell."""
    q = d[d.estimator == "map24"]
    ks = q.groupby(["cell", "K", "seed", "strategy"]).apply(lambda g: S.breakpoint(g.k.to_numpy(), g.nforecast.to_numpy(), tau),
                                                             include_groups=False).rename("kstar").reset_index()
    ks["le_K"] = ks.kstar <= ks.K
    out = []
    for cell, g in ks.groupby("cell"):
        fr = g.groupby("strategy").le_K.mean()
        med = g.groupby(["strategy", "K"]).kstar.median()
        rand_gt = (g[g.strategy == "random"].groupby("K").kstar.median() > np.array(sorted(g.K.unique()))).all()
        ok = fr.get("source_near", 0) >= .8 and fr.get("dopt", 0) >= .8 and rand_gt
        out.append(dict(cell=cell, n=g[["K", "seed"]].drop_duplicates().shape[0], **{f"frac_le_K_{s}": float(fr.get(s, np.nan)) for s in STRATEGIES},
                        **{f"med_kstar_{s}_K{K}": float(v) for (s, K), v in med.items()}, verdict="holds" if ok else "conditional"))
    return ks, pd.DataFrame(out)


def robustness(vt, kv):
    """Overall pilot label per conclusion (§5): robust / conditional / weakened in some regimes."""
    out = []
    for c, g in vt.groupby("conclusion"):
        v = g[g.cell != "nominal"].verdict
        lab = "conditional" if v.isin(["reversed", "flipped (indicative)"]).any() else "robust (pilot)" if (v == "holds").all() \
            else "weakened in some regimes"
        out.append(dict(conclusion=c, label=lab, nominal=g[g.cell == "nominal"].verdict.iloc[0],
                        cells_not_holding=" ".join(g[(g.cell != "nominal") & (g.verdict != "holds")].cell)))
    v = kv[kv.cell != "nominal"].verdict
    out.append(dict(conclusion="C5_kstar_le_K", label="robust (pilot)" if (v == "holds").all() else "conditional",
                    nominal=kv[kv.cell == "nominal"].verdict.iloc[0], cells_not_holding=" ".join(kv[(kv.cell != "nominal") & (kv.verdict != "holds")].cell)))
    return pd.DataFrame(out)

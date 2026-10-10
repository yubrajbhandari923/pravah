"""Stage F2b: robustness of the FIXED geographic sensor layouts.  Real geography; synthetic emissions and meteorology.

The layouts of a ``geodesign`` run (layouts.csv, alpha_selected.json) were selected on DESIGN scenarios only.  Here they are
re-used VERBATIM (nothing is re-selected, no alpha is re-chosen) and evaluated on the deterministically regenerated TEST
ensemble (same seeds, same noise arrays, same sigma) under separate conditions, one at a time:

  matched               the baseline (rows identical to the main run's selected-alpha test rows: the pairing key)
  unseen_wind           truth wind outside the design/validation draw (faster, direction offset from the sector centres);
                        the estimator is told the NOMINAL wind of the same case (the wind drawn before the replacement)
  unseen_wind_declared  the same unseen truth wind, but correctly declared (pure distribution shift, wind known)
  hidden_wind_error     truth wind = declared wind with speed x (1 + e) and a rotation; the estimator gets the declared wind
  boundary_time_profile truth inflow (bg) onset / ramp differs from the profile the estimator assumes
  omitted_facility      truth has an extra LOCAL facility-cluster source the inverse model does not have
  omitted_external      truth has an extra EXTERNAL Gaussian source the inverse model does not have
  refined_truth         truth simulated on a 2x finer grid with a smaller time step, the regional error computed on the
                        block-average of the fine field, sensors read the fine field (inverse-crime contrast)

Refinement stability (optional, expensive): layouts re-selected on the finer model (design cases on the 250 m grid) are compared
with the 500 m layouts (Jaccard, nearest-site distance) and on the same fine test truth.  The numerical-diffusion estimate of
the baseline scheme is reported separately; the physical K is not tuned.

Mismatch rows need not worsen in every draw: the outputs are raw rows, paired mismatch-minus-matched differences, per-case win
counts and the method contrasts under each condition.  Identifiability flags depend on the operator and sigma only, never on
the data, so they cannot detect a wrong model (``flags_vs_error.csv`` shows the numbers).  Everything is a provisional
simulation design, not field-validated.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

from . import design as D
from .belauri import draw_map
from .config import Boundary, GaussianSource, RasterSource, Scenario, StudyMask, Times, WindSegment
from .design import DesignCase
from .experiments import _peak_mb, block_average, perturb_wind
from .forward import FIPY
from .geodesign import (ALPHA_ZERO, INTERIOR, LABEL, PROVISIONAL, SD_NOTE, SELECTED, SETTING_A, SETTING_B, STANDARD,
                        STRONG, CaseDraw, Ensemble, LayoutRec, Progress, Setup, _say, _stamp, config_hash, draw_cases,
                        fit_row, summarise)
from .geodesign import paired_tables as method_contrasts
from .geo.scenarios import numerical_diffusion_estimate, toward_vector
from .inverse import ResponseCache, run_forward
from .io import make_run_dir, provenance, write_csv, write_json
from .sources import check_origin_tags

CONDITIONS = ("matched", "unseen_wind", "unseen_wind_declared", "hidden_wind_error", "boundary_time_profile",
              "omitted_facility", "omitted_external", "refined_truth")
KEYS = ["setting", "regime", "noise_fraction", "alpha_label", "method", "layout_id", "budget", "case", "rep"]
PAIR_METRICS = ("rmse", "rmse_local", "rmse_external", "attr_err", "ext_frac_abs_err", "source_rel_err",
                "inflow_abs_err_bg_abs")
NUMDIFF_NOTE = ("The physical K is not tuned.  The baseline scheme (exponential convection + backward Euler) adds numerical "
                "spreading estimated by geo.scenarios.numerical_diffusion_estimate (K_eff/K = 1 + excess, wind along a grid "
                "axis, a scalar-speed approximation for oblique winds); it is reported separately from K and is part of "
                "every forward model, truth and inference alike in the matched experiments.")

DEFAULTS: dict[str, Any] = {
    "layouts_from": None,
    "regimes": [STANDARD],
    "alpha_labels": [SELECTED],
    "conditions": list(CONDITIONS),
    "unseen_wind": {"speed_range": [2.6, 3.5], "sector_offset_deg": 22.5, "seed": 909},
    "hidden_wind_error": {"speed_error": 0.2, "rotation_deg": 15.0},
    "boundary_time_profile": {"bg_profile": [[0.0, 0.0], [14400.0, 0.5], [21600.0, 1.0]], "external_profile": None},
    "omitted_source": {"facility_ug_s": 5.0e5, "external_ug_s": 5.0e5, "external_sigma_m": 1000.0,
                       "external_position_frac": [0.12, 0.88], "external_profile": None},
    "refined_truth": {"factor": 2, "dt_factor": 0.5},
    "refinement": {"enabled": False, "reselect": True, "n_design_fine": 3, "n_test_fine": 3, "budgets": [3, 5],
                   "settings": [SETTING_A, SETTING_B], "resolution_m": 250.0, "dt_s": 50.0},
    "numerical_diffusion": {"speeds_m_s": [0.8, 1.5, 2.5, 3.5]},
    "figures": True,
}
NON_DESIGN_KEYS = ("name", "notes", "experiment", "robust")


def robust_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """cfg['robust'] merged over DEFAULTS (one level deep per sub-dict)."""
    r = copy.deepcopy(DEFAULTS)
    for k, v in cfg.get("robust", {}).items():
        r[k] = {**r[k], **v} if isinstance(r.get(k), dict) and isinstance(v, dict) else v
    bad = [c for c in r["conditions"] if c not in CONDITIONS]
    if bad:
        raise ValueError(f"unknown conditions {bad}; choose from {CONDITIONS}")
    if "matched" not in r["conditions"]:
        raise ValueError("the matched condition is the pairing baseline and cannot be dropped")
    return r


# ---------------------------------------------------------------------------------------------- fixed design
@dataclass(frozen=True)
class FixedDesign:
    """Layouts and frozen alphas of a geodesign run directory, exactly as written there."""
    run_dir: Path
    layouts: list[LayoutRec]
    alphas: dict[float, float]
    design_sigma: dict[float, float]  # noise fraction -> sigma used for the greedy selection (ug/m^3)
    files_sha256: dict[str, str]
    config: dict[str, Any]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _norm_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    c = json.loads(json.dumps(cfg, default=str))
    for k in NON_DESIGN_KEYS:
        c.pop(k, None)
    return c


def check_same_study(cfg: dict[str, Any], run_cfg: dict[str, Any]) -> None:
    """The robustness config must describe the same study as the run that produced the layouts (everything except
    name / notes / experiment / robust is compared); otherwise the regenerated test ensemble would not be the main one."""
    a, b = _norm_cfg(cfg), _norm_cfg(run_cfg)
    bad = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if bad:
        raise ValueError(f"config differs from the layouts' run config in {bad}: the matched baseline would not be the main run's")


def load_fixed_design(run_dir: Path | str, st: Setup, regimes: list[str] | None = None) -> FixedDesign:
    """Read layouts.csv + alpha_selected.json (+ config.json) of a geodesign run.  Site IDs are mapped to candidate indices
    of ``st`` (unknown IDs raise); layout_id is rebuilt and must equal the stored one.  Nothing is selected here."""
    rd = Path(run_dir)
    for n in ("layouts.csv", "alpha_selected.json", "config.json"):
        if not (rd / n).is_file():
            raise FileNotFoundError(f"{rd / n} is missing: layouts_from must be a finished geodesign run directory")
    check_same_study(st.cfg, json.loads((rd / "config.json").read_text()))
    alpha_payload = json.loads((rd / "alpha_selected.json").read_text())
    alphas = {f: float(alpha_payload["alpha_by_noise_fraction"][f"{f:g}"]) for f in st.fractions}
    table = pd.read_csv(rd / "layouts.csv")
    index = {cid: i for i, cid in enumerate(st.cands.ids)}
    recs: list[LayoutRec] = []
    sigma: dict[float, float] = {}

    def opt(v: Any) -> Any:
        return None if pd.isna(v) else v
    for r in table.to_dict("records"):
        if regimes is not None and r["regime"] not in regimes:
            continue
        ids = str(r["sites"]).split(",")
        unknown = [i for i in ids if i not in index]
        if unknown:
            raise ValueError(f"layout {r['layout_id']}: sites {unknown} are not candidates of this study")
        order = tuple(index[i] for i in ids)
        greedy = str(r["method"]).startswith("greedy")
        f = opt(r["noise_fraction"])
        obj = None if str(r["objective"]).startswith("none") else str(r["objective"])
        rec = LayoutRec(str(r["setting"]), str(r["regime"]), str(r["method"]), int(r["budget"]), int(r["k"]),
                        tuple(sorted(order)), None if f is None else float(f), obj, opt(r["alpha"]),
                        opt(r["design_sigma_ug_m3"]), None if opt(r["layout_seed"]) is None else int(r["layout_seed"]),
                        order if greedy else ())
        if rec.layout_id != r["layout_id"]:
            raise ValueError(f"layout id mismatch: rebuilt {rec.layout_id} vs stored {r['layout_id']}")
        if len(rec.layout) != int(r["n_sites"]) or len(set(order)) != len(order):
            raise ValueError(f"layout {r['layout_id']}: site count differs from n_sites")
        if greedy and rec.design_sigma is not None and rec.noise_fraction is not None:
            sigma[rec.noise_fraction] = float(rec.design_sigma)
        recs.append(rec)
    if not recs:
        raise ValueError("no layouts after filtering")
    files = {n: _sha(rd / n) for n in ("layouts.csv", "alpha_selected.json", "config.json")}
    return FixedDesign(rd, recs, alphas, sigma, files, json.loads((rd / "config.json").read_text()))


# ---------------------------------------------------------------------------------------------- truth variants
def unseen_winds(cfg: dict[str, Any], rc: dict[str, Any], draws: list[CaseDraw]) -> list[tuple[tuple[float, float, float], ...]]:
    """Per case the unseen truth wind: each segment keeps its start time; the direction is the nearest design sector centre
    plus or minus ``sector_offset_deg`` (random sign), the speed U(speed_range).  The nominal draws sit within
    +- sector_half_width_deg of the centres and U(0.8, 2.5) m/s, so with offset 22.5 deg > half width 15 deg and speeds
    above the design range the truth wind lies outside the design / validation draw.  Deterministic in rc['unseen_wind']."""
    u = rc["unseen_wind"]
    lo_s, hi_s = (float(v) for v in u["speed_range"])
    n = int(cfg["ensemble"]["n_sectors"])
    width = 360.0 / n
    rng = np.random.default_rng(int(u["seed"]))
    out = []
    for d in draws:
        segs = []
        for t, f, _s in d.winds:
            centre = round(f / width) * width
            sign = 1.0 if rng.random() < 0.5 else -1.0
            segs.append((float(t), float((centre + sign * float(u["sector_offset_deg"])) % 360.0),
                         float(rng.uniform(lo_s, hi_s))))
        out.append(tuple(segs))
    return out


def upsample_scenario(sc: Scenario, factor: int, dt_factor: float) -> Scenario:
    """The same physics on a grid ``factor`` times finer: raster source weights and the study mask are replicated onto the
    finer cells (the same emission FIELD, so only the numerics differ), Gaussians are re-normalised on the fine grid by the
    solver, the time step is scaled by ``dt_factor``, the output grid and times are unchanged."""
    if factor < 1 or int(factor) != factor or not dt_factor > 0:
        raise ValueError("factor must be a positive integer and dt_factor positive")
    ones = np.ones((factor, factor))
    srcs = tuple(RasterSource.from_array(s.name, s.group, np.kron(np.asarray(s.weights, dtype=float), ones), s.profile)
                 if isinstance(s, RasterSource) else s for s in sc.sources)
    study = (StudyMask.from_array(np.kron(np.asarray(sc.study.weights, dtype=float), ones))
             if isinstance(sc.study, StudyMask) else sc.study)
    dom = dataclasses.replace(sc.domain, nx=sc.domain.nx * factor, ny=sc.domain.ny * factor)
    t = sc.times
    return dataclasses.replace(sc, domain=dom, sources=srcs, study=study, times=Times(t.duration, t.dt * dt_factor, t.output_interval))


def _split_truth(st: Setup, sc: Scenario, q: np.ndarray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """(total, local, external) truth fields of ``sc`` by two direct solves: total = F(q), local = F(q with the external
    emissions and the inflow set to 0); external = total - local.  Exact for the zero-background, zero-initial-field
    scenarios used here (the model is affine in q with base 0); anything else raises."""
    if sc.boundary.background != 0.0 or sc.initial.kind != "zero":
        raise ValueError("truth tags by two solves need background 0 and a zero initial field")
    tab = sc.param_table()
    q_loc = np.where([p.kind == "emission" and p.group == "local" for p in tab], q, 0.0)
    total = run_forward(FIPY, sc, q, st.solver).concentration
    local = run_forward(FIPY, sc, q_loc, st.solver).concentration
    return total, local, total - local


def _truth_wind_text(sc: Scenario) -> str:
    out = []
    for w in sc.transport.wind:
        sp = math.hypot(w.u, w.v)
        frm = math.degrees(math.atan2(-w.u, -w.v)) % 360.0
        out.append(f"t={w.t_start:g}s from {frm:.1f} deg at {sp:.2f} m/s")
    return ";".join(out)


def variant_case(st: Setup, kind: str, d: CaseDraw, unseen: tuple[tuple[float, float, float], ...], noise_seed: int,
                 rc: dict[str, Any]) -> tuple[DesignCase, dict[str, Any]]:
    """One test case under condition ``kind`` (not 'matched'): same draw, same noise seed (so the noise arrays equal the
    matched case's), the truth differs from the inference model as the condition says.  The inference model is always built
    from the NOMINAL draw except for 'unseen_wind_declared'.  Truth components (local / external) come from the truth itself;
    the truth parameters are matched to the inference parameters by kind block inside ``design.build_case``."""
    t0 = time.perf_counter()
    miss = st.cache.misses
    reps = st.reps
    if kind == "unseen_wind_declared":
        case, info = st.build(dataclasses.replace(d, winds=unseen), "test", noise_seed)
        info.update(condition=kind, truth_wind=info["wind"], nominal_wind=";".join(
            f"t={t:g}s from {f:.1f} deg at {s:.2f} m/s" for t, f, s in d.winds), max_courant_truth=info["max_courant"],
            unmodelled_emission_share=0.0)
        info["build_s"] = time.perf_counter() - t0
        return case, info
    bc = st.scenario(d)
    inf, theta = bc.scenario, bc.theta
    truth_sc, truth_q = inf, theta
    err: xr.DataArray | None = None
    if kind == "unseen_wind":
        wind = tuple(WindSegment(t, *toward_vector(f, s)) for t, f, s in unseen)
        truth_sc = dataclasses.replace(inf, transport=dataclasses.replace(inf.transport, wind=wind))
    elif kind == "hidden_wind_error":
        h = rc["hidden_wind_error"]
        truth_sc = perturb_wind(inf, float(h["speed_error"]), float(h["rotation_deg"]))
    elif kind == "boundary_time_profile":
        b = rc["boundary_time_profile"]
        bases = tuple(dataclasses.replace(x, profile=tuple((float(t), float(v)) for t, v in b["bg_profile"]))
                      for x in inf.boundary.bases)
        truth_sc = dataclasses.replace(inf, boundary=Boundary(inf.boundary.mode, inf.boundary.background, bases))
        if b.get("external_profile") is not None:
            ext = tuple((float(t), float(v)) for t, v in b["external_profile"])
            truth_sc = dataclasses.replace(truth_sc, sources=tuple(
                dataclasses.replace(s, profile=ext) if s.group == "external" else s for s in truth_sc.sources))
    elif kind == "omitted_facility":
        o = rc["omitted_source"]
        extra = dataclasses.replace(st.geo.region.raster_source(st.res, "facility"), name="facility")
        truth_sc, truth_q = _with_extra_source(inf, theta, extra, float(o["facility_ug_s"]))
    elif kind == "omitted_external":
        o = rc["omitted_source"]
        dom = inf.domain
        fx, fy = (float(v) for v in o["external_position_frac"])
        prof = o.get("external_profile") or st.cfg["model"]["external_profile"]
        extra = GaussianSource("extra_external", fx * dom.lx, fy * dom.ly, float(o["external_sigma_m"]),
                               tuple((float(t), float(v)) for t, v in prof), "external")
        truth_sc, truth_q = _with_extra_source(inf, theta, extra, float(o["external_ug_s"]))
    elif kind == "refined_truth":
        rt = rc["refined_truth"]
        truth_sc = upsample_scenario(inf, int(rt["factor"]), float(rt["dt_factor"]))
    else:
        raise ValueError(f"unknown condition {kind!r}")
    if kind in ("omitted_facility", "omitted_external"):
        check_origin_tags(truth_sc, tol=0.06)  # the extra source's origin tag must be physically right
    total, local, external = _split_truth(st, truth_sc, truth_q)
    if kind == "refined_truth":
        err, local, external = block_average(total, inf), block_average(local, inf), block_average(external, inf)
        # the caller guarantees truth_field = simulate(truth_scenario, ...); build_case only reads the truth scenario for the
        # parameter table and the source-signature check, and the fine raster sources are replicas (different array hash)
        # of the inference sources, so the inference scenario stands in for the truth scenario here
        case = D.build_case(d.name, "test", inf, truth_q, inf, st.cands, st.opts(0.0), noise_seed, reps, st.cache,
                            error_field=err, truth_field=total, truth_components={"local": local, "external": external})
    else:
        case = D.build_case(d.name, "test", truth_sc, truth_q, inf, st.cands, st.opts(0.0), noise_seed, reps, st.cache,
                            truth_field=total, truth_components={"local": local, "external": external})
    wmax = max(math.hypot(w.u, w.v) for w in truth_sc.transport.wind)
    info = {"case": d.name, "role": "test", "condition": kind, "noise_seed": noise_seed, "truth_wind": _truth_wind_text(truth_sc),
            "nominal_wind": ";".join(f"t={t:g}s from {f:.1f} deg at {s:.2f} m/s" for t, f, s in d.winds),
            "max_courant_truth": wmax * truth_sc.times.dt / truth_sc.domain.dx,
            "unmodelled_emission_share": case.unmodelled_mass, "external_fraction_true": case.external_fraction_true,
            "new_response_sets": st.cache.misses - miss, "build_s": time.perf_counter() - t0}
    return case, info


def _with_extra_source(inf: Scenario, theta: np.ndarray, extra: GaussianSource | RasterSource, amplitude: float
                       ) -> tuple[Scenario, np.ndarray]:
    """Truth scenario = inference sources + one extra emission source (appended AFTER them, so the kind blocks match);
    theta_truth = [inference emissions, extra amplitude, inference inflow amplitudes]."""
    n = len(inf.sources)
    sc = dataclasses.replace(inf, sources=inf.sources + (extra,))
    return sc, np.concatenate([theta[:n], [amplitude], theta[n:]])


def build_condition(st: Setup, kind: str, draws: list[CaseDraw], first_noise_seed: int, rc: dict[str, Any],
                    matched: Ensemble, unseen: list[tuple[tuple[float, float, float], ...]], progress: Progress = None
                    ) -> Ensemble:
    """The test ensemble under ``kind`` (same sigma as the matched ensemble: the noise level is defined by the MATCHED
    signal scale so that rows are paired).  'matched' returns the given ensemble."""
    if kind == "matched":
        return matched
    cases, infos = [], []
    for i, d in enumerate(draws):
        c, info = variant_case(st, kind, d, unseen[i], first_noise_seed + i, rc)
        cases.append(c)
        infos.append(info)
        _say(progress, f"{kind} case {d.name} built in {info['build_s']:.1f} s")
    return Ensemble("test", cases, infos, matched.scale, matched.sigma)


# ---------------------------------------------------------------------------------------------- evaluation
def evaluate_cases(st: Setup, layouts: list[LayoutRec], cases: list[DesignCase], sigma: dict[float, float],
                   alphas: dict[float, float], scales: tuple[float, float], labels: list[str],
                   progress: Progress = None, tag: str = "") -> pd.DataFrame:
    """Rows (layout x case x rep) per noise fraction and alpha label; same columns and arithmetic as
    ``geodesign.evaluate_test`` (which it mirrors for the selected-alpha rows) so that matched rows can be compared with
    the main run's test_raw.csv.  Identifiability flags only for the selected-alpha rows."""
    D._require_role(cases, "test", "evaluate_cases")
    units = tuple(p.units for p in st.scenario(draw_cases(st.cfg, 1, 0, "u")[0]).scenario.param_table())
    fixed = {SELECTED: None, ALPHA_ZERO: 0.0, STRONG: float(st.cfg["alpha"]["strong_prior"])}
    cats = cases[0].category
    rows: list[dict[str, Any]] = []
    for f in st.fractions:
        for label in labels:
            fv = fixed[label]
            a = alphas[f] if fv is None else float(fv)
            opts = st.opts(sigma[f], a, scales)
            for rec in layouts:
                if rec.noise_fraction not in (None, f):
                    continue
                for c in cases:
                    for rep in range(c.n_noise):
                        r = fit_row(c, rec, rep, opts, cats, label == SELECTED, units)
                        rows.append({"setting": rec.setting, "regime": rec.regime, "noise_fraction": f, "sigma_ug_m3": sigma[f],
                                     "alpha_label": label, "alpha": a, "method": rec.method, "layout_id": rec.layout_id,
                                     "budget": rec.budget, **r})
        _say(progress, f"{tag} evaluated noise fraction {f:g} ({len(rows)} rows)")
    return pd.DataFrame(rows)


def check_against_main(matched_raw: Any, run_dir: Path) -> dict[str, Any]:
    """Compare the regenerated matched rows with the main run's test_raw.csv on the shared keys (rmse, attr_err); raise if
    they differ beyond round-off (the pairing guarantee).  Returns the comparison record."""
    p = run_dir / "test_raw.csv"
    if not p.is_file():
        return {"compared": False, "reason": "test_raw.csv not found in layouts_from"}
    main = pd.read_csv(p, low_memory=False)
    cols = ["rmse", "attr_err", "rmse_local", "rmse_external"]
    m = matched_raw.merge(main[KEYS + cols], on=KEYS, suffixes=("", "_main"))
    if m.empty:
        raise ValueError("no matched rows could be paired with the main run's test_raw.csv")
    worst = 0.0
    for c in cols:
        a, b = m[c].to_numpy(float), m[f"{c}_main"].to_numpy(float)
        worst = max(worst, float(np.nanmax(np.abs(a - b) / (1e-12 + np.abs(b)))))
    if worst > 1e-6:
        raise ValueError(f"regenerated matched rows differ from the main run's test_raw.csv (max relative difference {worst:.3g})")
    return {"compared": True, "n_rows": int(len(m)), "n_main_rows_selected": int((main["alpha_label"] == SELECTED).sum()),
            "max_relative_difference": worst}


def check_design_scores(st: Setup, design: Ensemble, fixed: FixedDesign, alphas: dict[float, float],
                        scales: tuple[float, float]) -> dict[str, Any]:
    """The regenerated DESIGN ensemble must reproduce design_scores.csv of the main run (it is only used here for the
    attribution scales).  Scores a sample of the stored layouts (no selection) and raises on a mismatch."""
    p = fixed.run_dir / "design_scores.csv"
    if not p.is_file():
        return {"compared": False, "reason": "design_scores.csv not found"}
    tab = pd.read_csv(p)
    by_id = {r.layout_id: r for r in fixed.layouts}
    worst, n = 0.0, 0
    for r in tab.to_dict("records")[:: max(1, len(tab) // 12)]:
        rec = by_id.get(str(r["layout_id"]))
        if rec is None:
            continue
        f = float(r["noise_fraction"])
        opts = st.opts(design.sigma[f], alphas[f], scales)
        tot = D.score_layout(rec.layout, design.cases, opts, "total")["rmse"]
        att = D.score_layout(rec.layout, design.cases, opts, "attribution")["value"]
        for new, old in ((tot, r["design_rmse"]), (att, r["design_attribution"])):
            worst = max(worst, abs(new - old) / (1e-12 + abs(old)))
        n += 1
    if worst > 1e-6:
        raise ValueError(f"regenerated design ensemble does not reproduce design_scores.csv (max relative difference {worst:.3g})")
    return {"compared": True, "n_layouts": n, "max_relative_difference": worst}


# ---------------------------------------------------------------------------------------------- tables
def _prep(raw: Any) -> Any:
    d: Any = raw.copy()
    d["ext_frac_abs_err"] = d["external_fraction_error"].abs()
    d["inflow_abs_err_bg_abs"] = d["inflow_abs_err_bg"].abs()
    for c in [c for c in d.columns if c.startswith("err_")]:
        d[f"abs_{c}"] = d[c].abs()
    return d


def paired_rows(raw: Any) -> pd.DataFrame:
    """Per (condition, setting, regime, noise fraction, alpha label, layout, case, rep): metric under the condition, the
    matched metric on the SAME row key and their difference (condition - matched; positive = worse), for the total, local,
    external RMSE, the attribution error, the external-fraction error, the emission relative error and the bg inflow error,
    plus the per-parameter absolute errors (emission amplitudes in ug/s, inflow in ug/m^3: kinds never mixed).
    ``flags_same`` = identifiability flags equal to the matched row's."""
    d = _prep(raw)
    per_param = [c for c in d.columns if c.startswith("abs_err_")]
    mets = list(PAIR_METRICS) + per_param
    base: Any = d[d["condition"] == "matched"][KEYS + mets + ["id_flags"]]
    oth: Any = d[d["condition"] != "matched"]
    m = oth[["condition"] + KEYS + mets + ["id_flags"]].merge(base, on=KEYS, suffixes=("", "_matched"))
    for c in mets:
        m[f"diff_{c}"] = m[c] - m[f"{c}_matched"]
    m["flags_same"] = (m["id_flags"].fillna("") == m["id_flags_matched"].fillna(""))
    return m


def paired_summary(pr: Any) -> pd.DataFrame:
    """Aggregate ``paired_rows`` per (condition, setting, regime, noise fraction, alpha label, method, budget, metric):
    layouts of a method are averaged first (per case and rep), then mean / sd (labelled) of the differences, the number of
    pairs, the number of cases whose mean difference is positive (worse) and the fraction of pairs that are worse.
    A worsening in every draw is NOT required."""
    mets = [c[5:] for c in pr.columns if c.startswith("diff_")]
    grp = ["condition", "setting", "regime", "noise_fraction", "alpha_label", "method", "budget", "case", "rep"]
    cols = [f"diff_{m}" for m in mets] + mets + [f"{m}_matched" for m in mets]
    lm: Any = cast(Any, pr.groupby(grp)[cols]).mean().reset_index()
    rows: list[dict[str, Any]] = []
    for key, g in lm.groupby([c for c in grp if c not in ("case", "rep")]):
        for m in mets:
            diff = g[f"diff_{m}"]
            ok = diff.notna()
            if not ok.any():
                continue
            case_mean = diff[ok].groupby(g["case"][ok]).mean()
            rows.append({**dict(zip([c for c in grp if c not in ("case", "rep")], key)), "metric": m,
                         "mean_matched": float(g[f"{m}_matched"][ok].mean()), "mean_condition": float(g[m][ok].mean()),
                         "mean_diff": float(diff[ok].mean()), "sd_diff": float(diff[ok].std(ddof=1)) if ok.sum() > 1 else float("nan"),
                         "n_pairs": int(ok.sum()), "n_cases": int(len(case_mean)), "n_cases_worse": int((case_mean > 0).sum()),
                         "frac_pairs_worse": float((diff[ok] > 0).mean()), "sd_note": SD_NOTE})
    return pd.DataFrame(rows)


def contrasts_by_condition(raw: Any) -> pd.DataFrame:
    """The method contrasts of ``geodesign.paired_tables`` (greedy_total vs spread / random mean, greedy_attribution vs
    greedy_total and spread, with_reference vs interior_only, consecutive budgets) recomputed UNDER each condition."""
    parts = []
    for c, g in raw.groupby("condition", sort=False):
        t = method_contrasts(g.drop(columns="condition"))
        t.insert(0, "condition", c)
        parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def ranking_changes(contrasts: pd.DataFrame) -> pd.DataFrame:
    """For every contrast row of a mismatch condition: its mean difference, the matched one, and whether the sign (which
    method is better) is the same.  Descriptive; contrasts are averages over a handful of cases."""
    keys = ["comparison", "setting", "regime", "noise_fraction", "budget", "method_a", "method_b", "metric"]
    c: Any = contrasts[contrasts["comparison"] != "consecutive_budgets"]
    base: Any = c[c["condition"] == "matched"][keys + ["mean_diff", "a_wins_cases", "n_cases"]]
    oth: Any = c[c["condition"] != "matched"][["condition"] + keys + ["mean_diff", "a_wins_cases", "n_cases"]]
    m = oth.merge(base, on=keys, suffixes=("", "_matched"))
    m["same_sign_as_matched"] = np.sign(m["mean_diff"]) == np.sign(m["mean_diff_matched"])
    return m


def flags_vs_error(raw: Any) -> pd.DataFrame:
    """Per (condition, method, budget) for selected-alpha rows: the fraction of rows with any identifiability flag, the
    fraction whose flags equal the matched row's, the mean RMSE, and the matched mean RMSE.  Flags are a function of the
    operator and sigma only; they cannot see a wrong model, so the flagged fraction does not move when the RMSE does."""
    d = _prep(raw[raw["alpha_label"] == SELECTED])
    d["flagged"] = d["id_flags"].fillna("").astype(str).str.len() > 0
    rows = []
    base: Any = d[d["condition"] == "matched"]
    for (c, meth, b), g in d.groupby(["condition", "method", "budget"]):
        bm = base[(base["method"] == meth) & (base["budget"] == b)]
        mm = g.merge(bm[KEYS + ["id_flags"]], on=KEYS, suffixes=("", "_m"))
        rows.append({"condition": c, "method": meth, "budget": int(b), "n_rows": int(len(g)),
                     "frac_flagged": float(g["flagged"].mean()), "frac_flagged_matched": float(bm["flagged"].mean()),
                     "frac_flags_equal_matched": float((mm["id_flags"].fillna("") == mm["id_flags_m"].fillna("")).mean()) if len(mm) else float("nan"),
                     "mean_rmse": float(g["rmse"].mean()), "mean_rmse_matched": float(bm["rmse"].mean()),
                     "rmse_ratio_to_matched": float(g["rmse"].mean() / bm["rmse"].mean())})
    return pd.DataFrame(rows)


def per_condition_summary(raw: Any) -> pd.DataFrame:
    parts = []
    for c, g in raw.groupby("condition", sort=False):
        t = summarise(g.drop(columns="condition"))
        t.insert(0, "condition", c)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------------------------- numerical diffusion
def diffusion_table(cfg: dict[str, Any], rc: dict[str, Any], draws: dict[str, list[CaseDraw]], unseen_max: float | None,
                    fine: tuple[float, float] | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """K_eff/K of the baseline scheme at the design mesh (dx, dt) and at the refined mesh for the speeds of the configured
    ensembles (min / mean / max over the drawn wind segments), the configured reference speeds and the unseen-wind maximum.
    ``numerical_diffusion_estimate`` is an along-axis Fourier estimate (docs/specification.md)."""
    K = float(cfg["physics"]["K"])
    meshes = [("design", float(cfg["mesh"]["resolution_m"]), float(cfg["mesh"]["dt_s"]))]
    if fine is not None:
        meshes.append(("refined", fine[0], fine[1]))
    speeds_by_role = {r: [s for d in ds for _, _, s in d.winds] for r, ds in draws.items()}
    sp: dict[str, float] = {}
    for r, v in speeds_by_role.items():
        sp[f"{r}_min"], sp[f"{r}_mean"], sp[f"{r}_max"] = float(min(v)), float(np.mean(v)), float(max(v))
    for s in rc["numerical_diffusion"]["speeds_m_s"]:
        sp[f"reference_{float(s):g}"] = float(s)
    if unseen_max is not None:
        sp["unseen_wind_max"] = float(unseen_max)
    rows = []
    for mesh, dx, dt in meshes:
        for label, u in sp.items():
            ex = numerical_diffusion_estimate(u, dx, dt, K)
            rows.append({"mesh": mesh, "dx_m": dx, "dt_s": dt, "speed_label": label, "speed_m_s": u, "peclet": u * dx / K,
                         "courant": u * dt / dx, "excess_K_eff_over_K_minus_1": ex, "K_eff_over_K": 1.0 + ex,
                         "K_eff_m2_s": K * (1.0 + ex)})
    tab = pd.DataFrame(rows)
    d: Any = tab[tab["mesh"] == "design"]
    lo = min(sp[k] for k in sp if k.endswith("_min"))
    hi = max(sp[k] for k in sp if k.endswith("_max"))
    at = {u: 1.0 + numerical_diffusion_estimate(u, meshes[0][1], meshes[0][2], K) for u in (lo, hi)}
    summary = {"K_physical_m2_s": K, "tuned": False, "note": NUMDIFF_NOTE,
               "design_mesh": {"dx_m": meshes[0][1], "dt_s": meshes[0][2]},
               "drawn_speed_range_m_s": [lo, hi],
               "K_eff_over_K_at_design_mesh_over_drawn_speeds": {"at_min_speed": at[lo], "at_max_speed": at[hi]},
               "K_eff_over_K_at_design_mesh_all_labelled_speeds": [float(d["K_eff_over_K"].min()), float(d["K_eff_over_K"].max())],
               "speeds_m_s": sp}
    return tab, summary


# ---------------------------------------------------------------------------------------------- refinement stability
def _fine_setup(st: Setup, res: float, dt: float) -> Setup:
    """A copy of ``st`` on the finer mesh: SAME candidates, held-out points, observation times and response cache; mesh,
    time step and the study weights switch to the finer level.  (``Setup`` would rebuild the candidate lattice with a
    different hull, so the pool of sites could change.)"""
    fst = copy.copy(st)
    fst.res, fst.dt = float(res), float(dt)
    fst.weight = np.asarray(st.geo.region.level(res).arrays["study_fraction"], dtype=float)
    return fst


def layout_overlap(st: Setup, a: LayoutRec, b: LayoutRec) -> dict[str, Any]:
    """Jaccard index of the site sets and the nearest-site distances between two layouts (metres; mean over both
    directions, plus the maximum)."""
    sa, sb = set(a.layout), set(b.layout)
    xa, ya = st.cands.x[list(a.layout)], st.cands.y[list(a.layout)]
    xb, yb = st.cands.x[list(b.layout)], st.cands.y[list(b.layout)]
    dist = np.hypot(xa[:, None] - xb[None, :], ya[:, None] - yb[None, :])
    nn = np.concatenate([dist.min(axis=1), dist.min(axis=0)])
    cats = st.cands.categories
    return {"jaccard": len(sa & sb) / len(sa | sb), "n_shared_sites": len(sa & sb),
            "mean_nearest_site_distance_m": float(nn.mean()), "max_nearest_site_distance_m": float(nn.max()),
            "n_reference_500m": sum(cats[i] != INTERIOR for i in a.layout), "n_reference_250m": sum(cats[i] != INTERIOR for i in b.layout)}


@dataclass
class RefinementResult:
    layouts_fine: list[LayoutRec]
    overlap: pd.DataFrame
    raw: pd.DataFrame
    gap: pd.DataFrame
    fine_layouts_table: pd.DataFrame
    timings: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


def refinement_study(st: Setup, fixed: FixedDesign, rc: dict[str, Any], test_draws: list[CaseDraw], test_seed0: int,
                     matched: Ensemble, progress: Progress = None) -> RefinementResult:
    """Re-select greedy layouts on the finer model and compare with the fixed 500 m layouts.  Design cases on the fine grid
    (the first ``n_design_fine`` design draws, real fine rasters, fine inference model, design sigma / alpha of the main
    run), greedy_total and greedy_attribution at the budget subset, then Jaccard and nearest-site distances, then BOTH sets
    evaluated on the same fine test truth (first ``n_test_fine`` test draws, fine inference) at the matched-test sigma."""
    rf = rc["refinement"]
    cfg = st.cfg
    tim: dict[str, float] = {}
    fst = _fine_setup(st, float(rf["resolution_m"]), float(rf["dt_s"]))
    n_d = min(int(rf["n_design_fine"]), int(cfg["ensembles"]["design"]["n_cases"]))
    if not 1 <= n_d <= 6:
        raise ValueError("refinement.n_design_fine must be between 1 and 6")
    n_t = min(int(rf["n_test_fine"]), len(test_draws))
    ed = cfg["ensembles"]["design"]
    miss0, t0 = st.cache.misses, time.perf_counter()
    fine_design: list[DesignCase] = []
    for i, d in enumerate(draw_cases(cfg, int(ed["n_cases"]), int(ed["seed"]), "d")[:n_d]):
        c, info = fst.build(d, "design", int(ed["noise_seed"]) + i)
        fine_design.append(c)
        _say(progress, f"fine design case {d.name} built ({info['new_response_sets']} new response sets)")
    tim["fine_design_cases_s"] = time.perf_counter() - t0
    sets_design = st.cache.misses - miss0
    scales = D.attribution_scales(fine_design)
    budgets = [int(b) for b in rf["budgets"]]
    layouts_fine: list[LayoutRec] = []
    t0 = time.perf_counter()
    if rf["reselect"]:
        for setting in rf["settings"]:
            cons = st.constraints(setting, STANDARD)
            feas = [b for b in budgets if b not in D.infeasible_budgets(st.cands, cons, budgets)]
            if not feas:
                continue
            for f in st.fractions:
                opts = fst.opts(fixed.design_sigma[f], fixed.alphas[f], scales)
                for method, obj in (("greedy_total", "total"), ("greedy_attribution", "attribution")):
                    tr = D.greedy_trace(st.cands, max(feas), fine_design, cons, opts, obj)
                    for b in feas:
                        layouts_fine.append(LayoutRec(setting, STANDARD, method, b, 0, tuple(sorted(tr.order[:b])), f, obj,
                                                      fixed.alphas[f], fixed.design_sigma[f], None, tr.order[:b], tr.scores[b - 1]))
    tim["fine_selection_s"] = time.perf_counter() - t0
    fixed_by_key = {(r.setting, r.regime, r.method, r.budget, r.noise_fraction, r.k): r for r in fixed.layouts}
    ov = []
    for r in layouts_fine:
        a = fixed_by_key.get((r.setting, r.regime, r.method, r.budget, r.noise_fraction, 0))
        if a is not None:
            ov.append({"setting": r.setting, "method": r.method, "budget": r.budget, "noise_fraction": r.noise_fraction,
                       **layout_overlap(st, a, r), "sites_500m": ",".join(st.cands.ids[i] for i in a.layout),
                       "sites_250m": ",".join(st.cands.ids[i] for i in r.layout)})
    overlap = pd.DataFrame(ov)
    # both sets on the same fine test truth, fine inference model, matched-test sigma
    t0 = time.perf_counter()
    miss1 = st.cache.misses
    fine_test: list[DesignCase] = []
    for i, d in enumerate(test_draws[:n_t]):
        c, _info = fst.build(d, "test", test_seed0 + i)
        fine_test.append(c)
        _say(progress, f"fine test case {d.name} built")
    tim["fine_test_cases_s"] = time.perf_counter() - t0
    sets_test = st.cache.misses - miss1
    keep = {(s, STANDARD, b) for s in rf["settings"] for b in budgets}
    base_layouts = [r for r in fixed.layouts if (r.setting, r.regime, r.budget) in keep]
    t0 = time.perf_counter()
    raw_parts = []
    for model, lays in (("500m", base_layouts), ("250m", layouts_fine)):
        if not lays or not fine_test:
            continue
        df = evaluate_cases(fst, lays, fine_test, matched.sigma, fixed.alphas, scales, [SELECTED], progress, f"refinement {model}")
        df.insert(0, "design_model", model)
        raw_parts.append(df)
    raw = pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()
    tim["fine_evaluation_s"] = time.perf_counter() - t0
    gap = refinement_gap(raw)
    tab = pd.DataFrame([{"layout_id": r.layout_id, "setting": r.setting, "method": r.method, "budget": r.budget,
                         "noise_fraction": r.noise_fraction, "design_model": "250m", "objective": r.objective, "alpha": r.alpha,
                         "design_sigma_ug_m3": r.design_sigma, "sites": ",".join(st.cands.ids[i] for i in (r.order or r.layout)),
                         "n_design_cases": n_d, "label": LABEL, "provenance": PROVISIONAL} for r in layouts_fine])
    seconds_per_set = tim["fine_design_cases_s"] / max(sets_design, 1)
    info = {"n_design_fine": n_d, "n_test_fine": n_t, "budgets": budgets, "settings": rf["settings"],
            "fine_response_sets_design": sets_design, "fine_response_sets_test": sets_test,
            "seconds_per_fine_case_including_solves": {"design": tim["fine_design_cases_s"] / max(n_d, 1),
                                                       "test": tim["fine_test_cases_s"] / max(n_t, 1)},
            "seconds_per_fine_response_set_approx": seconds_per_set, "fine_mesh": {"resolution_m": fst.res, "dt_s": fst.dt},
            "attribution_scales_fine_design": list(scales),
            "note": "attr_err of the refinement tables uses the FINE design cases' attribution scales; it is not comparable "
                    "with attr_err of the 500 m tables"}
    return RefinementResult(layouts_fine, overlap, raw, gap, tab, tim, info)


def refinement_gap(raw: Any) -> pd.DataFrame:
    """250 m-selected minus 500 m-selected layouts on the same fine test truth, paired on (setting, noise fraction, method,
    budget, case, rep): mean, sd (labelled), n pairs, cases won by the 250 m layouts (negative = 250 m layouts better)."""
    if raw.empty or set(raw["design_model"]) != {"500m", "250m"}:
        return pd.DataFrame()
    keys = ["setting", "regime", "noise_fraction", "method", "budget", "case", "rep"]
    mets = ["rmse", "rmse_local", "rmse_external", "attr_err"]
    r: Any = raw[raw["method"].isin(["greedy_total", "greedy_attribution"])]
    a: Any = r[r["design_model"] == "250m"][keys + mets]
    b: Any = r[r["design_model"] == "500m"][keys + mets]
    m = a.merge(b, on=keys, suffixes=("_250", "_500"))
    rows = []
    for key, g in m.groupby(["setting", "noise_fraction", "method", "budget"]):
        for met in mets:
            diff = g[f"{met}_250"] - g[f"{met}_500"]
            cm = diff.groupby(g["case"]).mean()
            rows.append({"setting": key[0], "noise_fraction": key[1], "method": key[2], "budget": int(key[3]), "metric": met,
                         "mean_250m": float(g[f"{met}_250"].mean()), "mean_500m": float(g[f"{met}_500"].mean()),
                         "mean_diff_250m_minus_500m": float(diff.mean()),
                         "sd_diff": float(diff.std(ddof=1)) if len(diff) > 1 else float("nan"), "n_pairs": int(len(diff)),
                         "cases_won_by_250m_layouts": int((cm < 0).sum()), "n_cases": int(len(cm)), "sd_note": SD_NOTE})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- figures
def fig_rmse_by_condition(summ: pd.DataFrame, setting: str, f: float) -> Figure:
    s: Any = summ[(summ["setting"] == setting) & (summ["regime"] == STANDARD) & (summ["alpha_label"] == SELECTED)
                  & (summ["noise_fraction"] == f)]
    conds = [c for c in CONDITIONS if c in set(s["condition"])]
    ncol = min(4, max(len(conds), 1))
    nrow = math.ceil(max(len(conds), 1) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow), squeeze=False, layout="constrained", sharey=True)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, c in zip(axes.ravel(), conds):
        ax.set_visible(True)
        for m in ("random", "spread", "greedy_total", "greedy_attribution"):
            d = s[(s["condition"] == c) & (s["method"] == m)].sort_values("budget")
            if d.empty:
                continue
            ax.plot(d["budget"], d["mean_rmse"], marker="o", label=m)
            ax.fill_between(d["budget"], d["mean_rmse"] - d["sd_rmse"].fillna(0), d["mean_rmse"] + d["sd_rmse"].fillna(0), alpha=0.12)
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("sensors")
    axes[0, 0].set_ylabel("total regional RMSE (ug/m^3)")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"{setting}, noise fraction {f:g}: fixed layouts under each condition; mean +/- 1 s.d. (descriptive)", fontsize=8)
    return _stamp(fig)


def fig_paired(ps: pd.DataFrame, setting: str, f: float, metric: str = "rmse") -> Figure:
    s: Any = ps[(ps["setting"] == setting) & (ps["regime"] == STANDARD) & (ps["alpha_label"] == SELECTED)
                & (ps["noise_fraction"] == f) & (ps["metric"] == metric)]
    conds = [c for c in CONDITIONS if c in set(s["condition"])]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    w = 0.8 / max(len(conds), 1)
    for k, c in enumerate(conds):
        d = s[(s["condition"] == c) & (s["method"] == "greedy_total")].sort_values("budget")
        ax.errorbar(d["budget"] + (k - len(conds) / 2) * w * 0.5, d["mean_diff"], yerr=d["sd_diff"].fillna(0), marker="o",
                    capsize=2, label=c)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set(xlabel="sensors", ylabel=f"paired {metric} difference, condition - matched (ug/m^3)")
    ax.legend(fontsize=6)
    ax.set_title(f"greedy_total layouts, {setting}, noise fraction {f:g}: positive = worse than matched; bars +/- 1 s.d. of the pairs",
                 fontsize=8)
    return _stamp(fig)


def fig_refinement(st: Setup, fixed: FixedDesign, ref: RefinementResult, f: float) -> Figure:
    buds = sorted({r.budget for r in ref.layouts_fine})
    b = buds[-1]
    meths = ("greedy_total", "greedy_attribution")
    fig, axes = plt.subplots(1, len(meths), figsize=(5.2 * len(meths), 4.6), squeeze=False, layout="constrained")
    for ax, m in zip(axes[0], meths):
        draw_map(ax, st.geo.region, st.res)
        ax.scatter(st.cands.x / 1000.0, st.cands.y / 1000.0, s=5, c="0.8", zorder=3)
        a = [r for r in fixed.layouts if (r.setting, r.regime, r.method, r.budget, r.noise_fraction, r.k) == (SETTING_A, STANDARD, m, b, f, 0)]
        n = [r for r in ref.layouts_fine if (r.setting, r.method, r.budget, r.noise_fraction) == (SETTING_A, m, b, f)]
        for lay, col, mk, lab in ((a, "tab:blue", "o", "500 m design"), (n, "tab:red", "x", "250 m design")):
            if lay:
                i = list(lay[0].layout)
                ax.scatter(st.cands.x[i] / 1000.0, st.cands.y[i] / 1000.0, s=70, c=col, marker=mk, label=lab, zorder=5)
        ax.set_title(f"{m}, budget {b}", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Greedy sites chosen on the 500 m vs the 250 m design model (heuristics; few design cases)", fontsize=8)
    return _stamp(fig)


# ---------------------------------------------------------------------------------------------- the runner
@dataclass
class RobustResult:
    out_dir: Path
    files: list[str]
    timings: dict[str, float]
    raw: pd.DataFrame
    paired: pd.DataFrame
    figures: list[str] = field(default_factory=list)


def _save_fig(out: Path, fig: Figure, name: str, figs: list[str]) -> None:
    d = out / "figures"
    d.mkdir(exist_ok=True)
    fig.savefig(d / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    figs.append(name)


def run_geodesign_robust(cfg: dict[str, Any], out_dir: Path | str, overwrite: bool = False, progress: Progress = None,
                         layouts_from: Path | str | None = None, cache: ResponseCache | None = None) -> RobustResult:
    """Evaluate the fixed layouts of a geodesign run under the robustness conditions (module docstring).  ``layouts_from``
    defaults to cfg['robust']['layouts_from'].  Writes config.json, provenance.json, timings.json, test_raw.csv (all
    conditions, long form), summary.csv, paired_rows.csv, paired_mismatch.csv, contrasts.csv, ranking_changes.csv,
    flags_vs_error.csv, cases.csv, numerical_diffusion.{csv,json}, optionally refinement_*.csv, and figures/.
    Nothing is selected or tuned: the layouts and alphas are read from ``layouts_from`` and checked."""
    t_all = time.perf_counter()
    rc = robust_settings(cfg)
    src = layouts_from if layouts_from is not None else rc["layouts_from"]
    if src is None:
        raise ValueError("no layouts: pass layouts_from or set robust.layouts_from")
    out = make_run_dir(out_dir, overwrite)
    write_json(out / "config.json", cfg)
    tim: dict[str, float] = {}
    st = Setup(cfg, cache)
    fixed = load_fixed_design(Path(src), st, list(rc["regimes"]))
    _say(progress, f"{len(fixed.layouts)} fixed layouts read from {src}; nothing is re-selected")
    ens_cfg = cfg["ensembles"]
    # the DESIGN ensemble is regenerated only for the attribution scales (and checked against design_scores.csv)
    t0 = time.perf_counter()
    design = st.ensemble("design", progress)
    scales = D.attribution_scales(design.cases)
    design_check = check_design_scores(st, design, fixed, fixed.alphas, scales)
    tim["design_regeneration_s"] = time.perf_counter() - t0
    # the TEST ensemble: matched baseline = the main run's test cases
    t0 = time.perf_counter()
    et = ens_cfg["test"]
    draws = draw_cases(cfg, int(et["n_cases"]), int(et["seed"]), "t")
    seed0 = int(et["noise_seed"])
    matched = st.ensemble("test", progress)
    D.assert_disjoint(design.cases, matched.cases)
    tim["matched_cases_s"] = time.perf_counter() - t0
    unseen = unseen_winds(cfg, rc, draws)
    labels = list(rc["alpha_labels"])
    raws, infos = [], list(matched.info)
    for i in infos:
        i.update(condition="matched", truth_wind=i["wind"], nominal_wind=i["wind"], max_courant_truth=i["max_courant"],
                 unmodelled_emission_share=0.0)
    cond_ens: dict[str, Ensemble] = {}
    for kind in rc["conditions"]:
        t0 = time.perf_counter()
        _say(progress, f"condition {kind}")
        ens = build_condition(st, kind, draws, seed0, rc, matched, unseen, progress)
        tim[f"{kind}_cases_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        df = evaluate_cases(st, fixed.layouts, ens.cases, ens.sigma, fixed.alphas, scales, labels, progress, kind)
        df.insert(0, "condition", kind)
        raws.append(df)
        tim[f"{kind}_evaluation_s"] = time.perf_counter() - t0
        if kind != "matched":
            infos += ens.info
        cond_ens[kind] = ens
    raw = pd.concat(raws, ignore_index=True)
    main_check = check_against_main(raw[raw["condition"] == "matched"], Path(src))
    write_csv(out / "test_raw.csv", raw)
    write_csv(out / "cases.csv", pd.DataFrame(infos))
    summ = per_condition_summary(raw)
    pr = paired_rows(raw)
    ps = paired_summary(pr[pr["alpha_label"] == SELECTED]) if len(pr) else pd.DataFrame()
    con = contrasts_by_condition(raw[raw["alpha_label"] == SELECTED])
    rk = ranking_changes(con) if len(con) else pd.DataFrame()
    fl = flags_vs_error(raw)
    for name, tab in (("summary.csv", summ), ("paired_rows.csv", pr), ("paired_mismatch.csv", ps), ("contrasts.csv", con),
                      ("ranking_changes.csv", rk), ("flags_vs_error.csv", fl)):
        write_csv(out / name, tab)
    files = ["config.json", "test_raw.csv", "cases.csv", "summary.csv", "paired_rows.csv", "paired_mismatch.csv", "contrasts.csv",
             "ranking_changes.csv", "flags_vs_error.csv"]
    # numerical diffusion (separately from the physical K)
    rf = rc["refinement"]
    rt = rc["refined_truth"]
    umax = max(s for w in unseen for _, _, s in w) if unseen else None
    fine_mesh = (float(cfg["mesh"]["resolution_m"]) / int(rt["factor"]), float(cfg["mesh"]["dt_s"]) * float(rt["dt_factor"]))
    all_draws = {r: draw_cases(cfg, int(e["n_cases"]), int(e["seed"]), r[0]) for r, e in ens_cfg.items()}
    nd_tab, nd = diffusion_table(cfg, rc, all_draws, umax, fine_mesh)
    write_csv(out / "numerical_diffusion.csv", nd_tab)
    write_json(out / "numerical_diffusion.json", nd)
    files += ["numerical_diffusion.csv", "numerical_diffusion.json"]
    # refinement stability
    ref: RefinementResult | None = None
    if rf["enabled"]:
        t0 = time.perf_counter()
        ref = refinement_study(st, fixed, rc, draws, seed0, matched, progress)
        tim["refinement_s"] = time.perf_counter() - t0
        tim.update({f"refinement_{k}": v for k, v in ref.timings.items()})
        for name, tab in (("refinement_overlap.csv", ref.overlap), ("refinement_test_raw.csv", ref.raw),
                          ("refinement_gap.csv", ref.gap), ("refinement_layouts_250m.csv", ref.fine_layouts_table)):
            write_csv(out / name, tab)
        write_json(out / "refinement_info.json", ref.info)
        files += ["refinement_overlap.csv", "refinement_test_raw.csv", "refinement_gap.csv", "refinement_layouts_250m.csv",
                  "refinement_info.json"]
    figs: list[str] = []
    if rc["figures"]:
        f0 = st.fractions[0]
        for s in st.settings:
            _save_fig(out, fig_rmse_by_condition(summ, s, f0), f"rmse_by_condition_{s}.png", figs)
        _save_fig(out, fig_paired(ps, SETTING_A, f0), "paired_difference.png", figs)
        if ref is not None and ref.layouts_fine:
            _save_fig(out, fig_refinement(st, fixed, ref, f0), "refinement_sites.png", figs)
    tim["total_s"] = time.perf_counter() - t_all
    write_json(out / "timings.json", {"stages_s": tim, "total_s": tim["total_s"], "peak_memory_mb": _peak_mb(),
                                      "n_response_sets": st.cache.misses})
    seeds = {r: {"scenario_seed": e["seed"], "first_noise_seed": e["noise_seed"], "n_cases": e["n_cases"]} for r, e in ens_cfg.items()}
    seeds["unseen_wind"] = rc["unseen_wind"]["seed"]
    write_json(out / "provenance.json", provenance(cfg, seeds, {
        "experiment": "geodesign_robust", "label": LABEL, "status": PROVISIONAL, "layouts_from": str(src),
        "layouts_from_sha256": fixed.files_sha256, "config_hash": config_hash(cfg), "robust_settings": rc,
        "alpha_by_noise_fraction": {f"{f:g}": a for f, a in fixed.alphas.items()}, "n_fixed_layouts": len(fixed.layouts),
        "layouts_reused_verbatim": True, "matched_vs_main_test_raw": main_check, "design_regeneration_check": design_check,
        "noise_sigma_ug_m3": {f"{f:g}": matched.sigma[f] for f in st.fractions},
        "max_courant_by_condition": {c: float(max(i["max_courant_truth"] for i in e.info)) for c, e in cond_ens.items() if c != "matched"},
        "max_courant_matched": float(max(i["max_courant"] for i in matched.info)),
        "identifiability_note": "flags are a function of the operator and sigma only; they cannot detect a wrong model",
        "note": "emissions, inflow, winds and noise are SYNTHETIC; mismatch differences are descriptive and need not worsen every draw"}))
    files += ["timings.json", "provenance.json"]
    return RobustResult(out, files, tim, raw, ps, figs)

"""EXP-018: toy PRESCRIBED spatiotemporal winds (ridge deflection / delayed reversal / recirculation) and their effect on
the arrival of an upstream external plume at a study box.  Synthetic everything; not terrain-derived; no new solver.

    python scripts/prescribed_winds.py --config configs/prescribed_winds.json [--profile smoke|pilot|production] [--out DIR]

Per case (scenario x outer-domain size x reversal time t_r) two forward solves (external source only; local sources only);
by linearity of the model the total is their sum (checked by tests/test_winds.py).  Definitions (docs/winds.md):
* arrival: first (linearly interpolated) up-crossing of the external-only study-mean series through (a) a fraction of its
  own run peak, (b) an absolute threshold; ``return_arrival`` is the second up-crossing of (a).
* returned mass fraction: external mass that enters the box after the box mass has peaked and fallen (see returned_mass).
* mass closure: |M_end - (emitted - removed - net boundary loss)| / emitted, from the forward budget terms."""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from pravah import forward as F  # noqa: E402
from pravah import winds as W  # noqa: E402
from pravah.config import (Boundary, Domain, GaussianSource, Scenario, StudyRegion, Times, Transport,  # noqa: E402
                           WindSegment)
from pravah.sources import study_weights  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STAMP = "EXP-018 prescribed wind, not terrain-derived"
SIDES = ("west", "east", "south", "north")


# ------------------------------------------------------------------------------------------------ pure analysis
def upcrossings(t: np.ndarray, y: np.ndarray, thr: float) -> list[float]:
    """Linearly interpolated times where y rises through ``thr`` (a y[0] >= thr counts as a crossing at t[0])."""
    out = [float(t[0])] if y[0] >= thr else []
    for k in range(len(y) - 1):
        if y[k] < thr <= y[k + 1]:
            out.append(float(t[k] + (thr - y[k]) / (y[k + 1] - y[k]) * (t[k + 1] - t[k])))
    return out


def returned_mass(t: np.ndarray, m_box: np.ndarray, s_box: np.ndarray, lam: float, H: float, dA: float,
                  emitted: float, a_min: float = 1e-3, theta: float = 0.05) -> tuple[float, float]:
    """(returned mass fraction of ``emitted``, exit time t_x [s]; nan if the plume never left) for an EXTERNAL-ONLY run.
    m_box = H dA sum(c w) is the mass in the study box, s_box = sum(c w) (for the removal term).  The box has no source, so
    the net inward exchange in [t_k, t_k+1] is N_k = dm + lam H dA int s dt (trapezoid).  t_x = first time m_box has fallen
    below (1-theta) of its running maximum once that maximum exceeded a_min*emitted (the plume has entered AND started to
    leave).  returned = sum over intervals after t_x of max(N_k, 0) / emitted: mass flowing in again after the first exit.
    A plume that first misses the box and then is steered into it counts as first entry, not as return."""
    rm = np.maximum.accumulate(m_box)
    hit = np.nonzero((rm >= a_min * emitted) & (m_box < (1.0 - theta) * rm))[0]
    if len(hit) == 0:
        return 0.0, math.nan
    kx = int(hit[0])
    dt = np.diff(t)
    n_in = np.diff(m_box) + lam * H * dA * 0.5 * (s_box[1:] + s_box[:-1]) * dt
    return float(np.maximum(n_in[kx:], 0.0).sum() / emitted), float(t[kx])


def side_coefficients(scn: Scenario) -> list[dict[str, tuple[Any, np.ndarray]]]:
    """Per wind segment and side: (slice, coefficient) so that the net mass leaving through that side [ug/s] is
    coef . c[slice]; coef = H dA (un/h if un>0 [upwind export]; 2K/h^2 if un<0 [diffusive loss at the Dirichlet inflow face])."""
    d, tr = scn.domain, scn.transport
    out = []
    for w in tr.wind:
        un = F._side_normal_velocities(scn, w)  # pyright: ignore[reportPrivateUsage]
        geo = F._side_geometry(scn)  # pyright: ignore[reportPrivateUsage]
        out.append({s: (geo[s][0], tr.H * d.dx * d.dy * (np.where(un[s] > 0, un[s] / geo[s][1], 0.0)
                                                         + np.where(un[s] < 0, 2 * tr.K / geo[s][1] ** 2, 0.0)))
                    for s in SIDES})
    return out


def side_rates(scn: Scenario, conc: np.ndarray, times: np.ndarray) -> np.ndarray:
    """(n_times, 4) net leaving rate per side [ug/s] at the output frames (end-of-step c; wind of the preceding interval)."""
    coefs = side_coefficients(scn)
    out = np.zeros((len(times), 4))
    for k in range(1, len(times)):
        co = coefs[F._wind_at(scn, 0.5 * (times[k - 1] + times[k]))]  # pyright: ignore[reportPrivateUsage]
        out[k] = [float(co[s][1] @ conc[k][co[s][0]]) for s in SIDES]
    return out


def mass_terms(res: Any, scn: Scenario) -> dict[str, float]:
    b, d, tr = res.budget, scn.domain, scn.transport
    emitted, removed = float(b.emission.sum()), float(b.removal.sum())
    leaving = float((b.outflow - b.inflow).sum())
    remaining = float(tr.H * d.dx * d.dy * res.concentration.values[-1].sum())
    return {"emitted_ug": emitted, "removed_ug": removed, "leaving_ug": leaving, "remaining_ug": remaining,
            "closure_rel": abs(remaining - (emitted - removed - leaving)) / max(emitted, 1e-300),
            "step_residual_rel": float(res.diagnostics["max_rel_budget_residual"]),
            "min_conc": float(res.diagnostics["min_concentration"]), "max_conc": float(res.concentration.values.max())}


# ------------------------------------------------------------------------------------------------ cases
def case_geometry(cfg: dict[str, Any], size: float) -> tuple[Domain, tuple[float, float]]:
    dm = cfg["domain"]
    n = round(size * dm["base_side_m"] / dm["dx"])
    if abs(n * dm["dx"] - size * dm["base_side_m"]) > 1e-6 or (n - round(dm["study_side_m"] / dm["dx"])) % 2:
        raise ValueError(f"size {size}: study box does not align with the grid")
    return Domain(n * dm["dx"], n * dm["dx"], n, n), (n * dm["dx"] / 2, n * dm["dx"] / 2)


def build_wind(cfg: dict[str, Any], name: str, dom: Domain, c: tuple[float, float], t_r: float | None) -> tuple[WindSegment, ...]:
    u, v = cfg["transport"]["u"], cfg["transport"]["v"]
    def bump(key: str) -> tuple[tuple[float, float], float, float]:
        p = cfg["winds"][key]
        return ((c[0] + p["offset_m"][0], c[1] + p["offset_m"][1]), p["sigma_m"],
                p["peak_speed_m_s"] * p["sigma_m"] * math.sqrt(math.e))  # peak of |grad psi| = A/(sigma sqrt(e))
    if name == "steady":
        return W.steady_uniform(dom, u, v)
    if name == "reversal":
        return W.delayed_reversal(dom, u, v, float(t_r))  # type: ignore[arg-type]
    if name in ("ridge", "ridge_reversal"):
        ctr, sg, amp = bump("ridge")
        return W.ridge_deflection(dom, u, v, ctr, sg, amp, t_r if name == "ridge_reversal" else None)
    if name == "eddy":
        ctr, sg, amp = bump("eddy")
        return W.recirculation_eddy(dom, u, v, ctr, sg, amp)
    raise ValueError(f"unknown wind scenario {name!r}")


def build_scenario(cfg: dict[str, Any], name: str, size: float, t_r: float | None) -> tuple[Scenario, tuple[float, float]]:
    dom, c = case_geometry(cfg, size)
    s, tm, tr = cfg["sources"], cfg["times"], cfg["transport"]
    ex, lo = s["external"], s["local"]
    prof = ((0.0, 0.0), (ex["on_s"], 1.0), (ex["off_s"], 0.0))  # external puff: declared onset and end
    half = cfg["domain"]["study_side_m"] / 2
    scn = Scenario(domain=dom,
                   sources=(GaussianSource("ext", c[0] + ex["offset_m"][0], c[1] + ex["offset_m"][1], ex["sigma_m"], prof, "external"),
                            GaussianSource("loc", c[0] + lo["offset_m"][0], c[1] + lo["offset_m"][1], lo["sigma_m"], ((0.0, 1.0),), "local")),
                   transport=Transport(build_wind(cfg, name, dom, c, t_r), tr["K"], tr["lam"], tr["H"]),
                   boundary=Boundary("open"), times=Times(tm["duration"], tm["dt"], tm["output_interval"]),
                   study=StudyRegion(c[0] - half, c[1] - half, c[0] + half, c[1] + half))
    return scn, c


def run_case(cfg: dict[str, Any], name: str, size: float, t_r: float | None) -> dict[str, Any]:
    scn, _ = build_scenario(cfg, name, size, t_r)
    q_ext, q_loc = cfg["sources"]["external"]["rate_ug_s"], cfg["sources"]["local"]["rate_ug_s"]
    th = cfg["thresholds"]
    t0 = time.perf_counter()
    res = {"ext": F.simulate(scn, [q_ext, 0.0]), "loc": F.simulate(scn, [0.0, q_loc])}
    runtime = time.perf_counter() - t0
    w = study_weights(scn)
    assert w is not None
    d, tr = scn.domain, scn.transport
    dA, wsum = d.dx * d.dy, float(w.sum())
    t = np.array(scn.output_times)
    row: dict[str, Any] = {"wind": name, "size_factor": size, "t_r_s": t_r if t_r is not None else math.nan,
                           "nx": d.nx, "runtime_s": runtime}
    ts: dict[str, np.ndarray] = {}
    for key, r in res.items():
        c = r.concentration.values
        ts[key] = (c * w).sum(axis=(1, 2)) / wsum  # study-mean concentration [ug/m^3]
        for k, v in mass_terms(r, scn).items():
            row[f"{key}_{k}"] = v
    tot = ts["ext"] + ts["loc"]
    pk = float(ts["ext"].max())
    up = upcrossings(t, ts["ext"], th["arrival_fraction"] * pk) if pk > 0 else []
    ua = upcrossings(t, ts["ext"], th["arrival_abs_ug_m3"])
    m_box = tr.H * dA * (res["ext"].concentration.values * w).sum(axis=(1, 2))
    rf, tx = returned_mass(t, m_box, (res["ext"].concentration.values * w).sum(axis=(1, 2)), tr.lam, tr.H, dA,
                           row["ext_emitted_ug"], th["returned_a_min"], th["returned_theta"])
    expo = lambda y: float(np.trapezoid(y, t))  # noqa: E731  ug s / m^3
    row.update({"ext_peak_ug_m3": pk, "ext_peak_time_s": float(t[ts["ext"].argmax()]),
                "arrival_frac_s": up[0] if up else math.nan, "return_arrival_frac_s": up[1] if len(up) > 1 else math.nan,
                "arrival_abs_s": ua[0] if ua else math.nan, "exit_time_s": tx, "returned_mass_fraction": rf,
                "ext_exposure": expo(ts["ext"]), "loc_exposure": expo(ts["loc"]), "total_exposure": expo(tot),
                "total_peak_ug_m3": float(tot.max()), "ext_share_of_exposure": expo(ts["ext"]) / max(expo(tot), 1e-300),
                "ext_mass_in_box_end_frac": float(m_box[-1] / row["ext_emitted_ug"])})
    fl: dict[str, np.ndarray] = {key: side_rates(scn, r.concentration.values, t) for key, r in res.items()}
    for key in res:  # cross-check of the per-side post-processing against the budget (end-of-step c vs sub-step sums)
        side_int = np.trapezoid(fl[key].sum(axis=1), t)
        row[f"{key}_side_vs_budget_rel"] = abs(side_int - row[f"{key}_leaving_ug"]) / max(row[f"{key}_emitted_ug"], 1e-300)
        for i, s in enumerate(SIDES):
            row[f"{key}_leave_{s}_ug"] = float(np.trapezoid(fl[key][:, i], t))
    row["_ts"], row["_fl"], row["_t"] = ts, fl, t
    return row


# ------------------------------------------------------------------------------------------------ outputs
def case_list(cfg: dict[str, Any]) -> list[tuple[str, float, float | None]]:
    out = []
    for name in cfg["scenarios"]:
        trs: list[float | None] = list(cfg["t_r_s"]) if "reversal" in name else [None]
        out += [(name, s, tr) for s in cfg["size_factors"] for tr in trs]
    return out


def sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Study-mean sensitivity to the outer-domain size: each metric relative to the LARGEST size of the same (wind, t_r)."""
    rows = []
    for (wind, _), g in df.assign(_tr=df.t_r_s.fillna(-1)).groupby(["wind", "_tr"]):  # pyright: ignore[reportGeneralTypeIssues]
        ref = g.loc[g.size_factor.idxmax()]
        for _, r in g.iterrows():
            rows.append({"wind": wind, "t_r_s": r.t_r_s, "size_factor": r.size_factor,
                         **{f"{m}_rel_to_largest": (r[m] - ref[m]) / ref[m] if ref[m] else math.nan
                            for m in ("ext_peak_ug_m3", "ext_exposure", "total_exposure", "returned_mass_fraction")},
                         "returned_mass_fraction": r.returned_mass_fraction, "ext_exposure": r.ext_exposure,
                         "total_exposure": r.total_exposure})
    return pd.DataFrame(rows)


def stamp(fig: Any) -> None:
    fig.text(0.995, 0.995, STAMP, ha="right", va="top", fontsize=8, color="0.3", weight="bold")


def fig_schematic(cfg: dict[str, Any], out: Path) -> None:
    size = max(cfg["size_factors"])
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    half = cfg["domain"]["study_side_m"] / 2
    for ax, name in zip(axs, ("steady", "ridge", "eddy")):
        scn, c = build_scenario(cfg, name, size, None)
        d = scn.domain
        psi = scn.transport.wind[0].psi
        uf, vf = F.psi_face_fluxes(scn, psi) if psi is not None else (np.full((d.ny, d.nx + 1), cfg["transport"]["u"]),
                                                                   np.full((d.ny + 1, d.nx), cfg["transport"]["v"]))
        uc, vc = 0.5 * (uf[:, 1:] + uf[:, :-1]), 0.5 * (vf[1:] + vf[:-1])
        x, y = (np.arange(d.nx) + 0.5) * d.dx / 1e3, (np.arange(d.ny) + 0.5) * d.dy / 1e3
        ax.streamplot(x, y, uc, vc, color=np.hypot(uc, vc), cmap="viridis", density=1.4, linewidth=0.9)
        ax.add_patch(Rectangle(((c[0] - half) / 1e3, (c[1] - half) / 1e3), 2 * half / 1e3, 2 * half / 1e3, fill=False, ec="k", lw=1.8))
        for s, mk, col in zip(scn.sources, ("*", "o"), ("tab:red", "tab:blue")):
            ax.plot(s.x / 1e3, s.y / 1e3, mk, color=col, ms=11, mec="k", label=f"{s.group} source")  # type: ignore[union-attr]
        ax.set(title=f"{name} (t < t_r)", xlabel="x [km]", ylabel="y [km]", aspect="equal", xlim=(0, d.lx / 1e3), ylim=(0, d.ly / 1e3))
    axs[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Streamlines (colour: speed, m/s), largest outer domain; black box = study box; reversal flips all arrows at t_r",
                 fontsize=9, x=0.02, ha="left")
    stamp(fig)
    fig.savefig(out / "fig1_wind_schematic.png", dpi=130)
    plt.close(fig)


def fig_arrival_returned(df: pd.DataFrame, out: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    cols = dict(zip(sorted(df.size_factor.unique()), ("tab:blue", "tab:orange", "tab:green", "tab:red")))
    ls = {"reversal": "-", "ridge_reversal": "--"}
    for ax, m, lab in ((axs[0], "arrival_frac_s", "arrival time (first crossing of fraction of peak) [h]"),
                       (axs[1], "return_arrival_frac_s", "return arrival (second crossing; absent if none) [h]"),
                       (axs[2], "returned_mass_fraction", "returned external mass fraction [-]")):
        for wn, g in df[df.wind.isin(ls)].groupby("wind"):
            for s, gs in g.groupby("size_factor"):
                gs = gs.sort_values(by="t_r_s")  # pyright: ignore[reportCallIssue]
                ax.plot(gs.t_r_s / 3600, gs[m] / (3600 if m.endswith("_s") else 1), ls[str(wn)], marker="o", color=cols[s],
                        label=f"{wn}, {s:g}x")
        for wn, g in df[~df.wind.isin(ls)].groupby("wind"):  # no t_r: one value per size, drawn as a short marker at x = nan-free left edge
            for s, gs in g.groupby("size_factor"):
                ax.axhline(float(np.asarray(gs[m])[0]) / (3600 if m.endswith("_s") else 1), color=cols[s], lw=0.8, ls=":", alpha=0.6)
        ax.set(xlabel="reversal time t_r [h]", ylabel=lab)
    axs[2].legend(fontsize=7, ncol=2)
    axs[0].set_title("dotted: scenarios without reversal (steady, ridge, eddy), colour = outer-domain size", fontsize=8)
    stamp(fig)
    fig.savefig(out / "fig2_arrival_returned_vs_tr.png", dpi=130)
    plt.close(fig)


def fig_flux(rows: list[dict[str, Any]], cfg: dict[str, Any], out: Path) -> None:
    rev = [r for r in rows if "reversal" in r["wind"]]
    if not rev:
        return
    wn = "reversal" if any(r["wind"] == "reversal" for r in rev) else "ridge_reversal"
    trs = sorted(cfg["t_r_s"])
    sel = [r for r in rev if r["wind"] == wn and r["t_r_s"] == trs[len(trs) // 2]]
    sel = sorted(sel, key=lambda r: r["size_factor"])
    sel = [sel[0], sel[-1]] if len(sel) > 1 else sel
    fig, axs = plt.subplots(1, len(sel), figsize=(5.5 * len(sel), 4), constrained_layout=True, sharey=True, squeeze=False)
    for ax, r in zip(axs[0], sel):
        for i, s in enumerate(SIDES):
            ax.plot(r["_t"] / 3600, r["_fl"]["ext"][:, i] / 1e3, label=s)
        ax.axvline(r["t_r_s"] / 3600, color="k", ls=":")
        ax.set(title=f"{r['wind']}, t_r = {r['t_r_s'] / 3600:g} h, outer domain {r['size_factor']:g}x", xlabel="time [h]")
    axs[0][0].set_ylabel("external plume: net loss through side [mg/s]")
    axs[0][0].legend(title="side")
    stamp(fig)
    fig.savefig(out / "fig3_boundary_flux.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prescribed_winds.json")
    ap.add_argument("--profile", default="production", choices=("smoke", "pilot", "production"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = json.loads((ROOT / a.config).read_text())
    prod = json.loads(json.dumps(cfg))
    cfg.update(cfg.get("profiles", {}).get(a.profile, {}))
    out = Path(a.out) if a.out else ROOT / "results" / ("phase3-winds-toy" if a.profile == "production" else f"phase3-winds-toy-{a.profile}")
    out.mkdir(parents=True, exist_ok=True)
    cases = case_list(cfg)
    print(f"[{STAMP}] profile {a.profile}: {len(cases)} cases -> {out}", flush=True)
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for k, (name, size, tr) in enumerate(cases):
        r = run_case(cfg, name, size, tr)
        rows.append(r)
        print(f"[{time.perf_counter() - t0:6.0f}s] {k + 1}/{len(cases)} {name} size {size:g} t_r {tr} nx {r['nx']} "
              f"solve {r['runtime_s']:.1f}s closure {max(r['ext_closure_rel'], r['loc_closure_rel']):.1e} "
              f"arr {r['arrival_frac_s']:.0f} ret {r['returned_mass_fraction']:.3f}", flush=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    ck = cfg["checks"]
    clo = float(df[["ext_closure_rel", "loc_closure_rel", "ext_step_residual_rel", "loc_step_residual_rel"]].max().max())
    neg = float(max((-df[f"{k}_min_conc"] / df[f"{k}_max_conc"]).max() for k in ("ext", "loc")))
    checks = {"max_closure_rel": clo, "closure_limit": ck["closure_rel"], "closure_ok": clo < ck["closure_rel"],
              "max_negative_rel_to_max_conc": neg, "negative_limit": ck["negative_rel"], "negative_ok": neg < ck["negative_rel"],
              "max_side_vs_budget_rel": float(df[["ext_side_vs_budget_rel", "loc_side_vs_budget_rel"]].max().max())}
    df.to_csv(out / "cases_raw.csv", index=False)
    pd.concat([pd.DataFrame({"wind": r["wind"], "size_factor": r["size_factor"], "t_r_s": r["t_r_s"], "t_s": r["_t"],
                             "ext_study_mean_ug_m3": r["_ts"]["ext"], "loc_study_mean_ug_m3": r["_ts"]["loc"],
                             "total_study_mean_ug_m3": r["_ts"]["ext"] + r["_ts"]["loc"]}) for r in rows]).to_csv(out / "timeseries_raw.csv", index=False)
    pd.concat([pd.DataFrame({"wind": r["wind"], "size_factor": r["size_factor"], "t_r_s": r["t_r_s"], "t_s": r["_t"], "component": key,
                             **{f"leaving_{s}_ug_s": r["_fl"][key][:, i] for i, s in enumerate(SIDES)}})
               for r in rows for key in ("ext", "loc")]).to_csv(out / "boundary_flux_raw.csv", index=False)
    sens = sensitivity(df)
    sens.to_csv(out / "sensitivity_outer_domain.csv", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    (out / "provenance.json").write_text(json.dumps({"label": STAMP, "profile": a.profile, "config": cfg, "n_cases": len(cases),
                                                    "python": sys.version.split()[0], "platform": platform.platform(),
                                                    "wall_s": time.perf_counter() - t0}, indent=1, default=str))
    (out / "summary.md").write_text(f"# EXP-018 ({a.profile}) -- {STAMP}\n\nchecks: {json.dumps(checks)}\n\n"
                                    "## Study-mean sensitivity to outer-domain size (relative to the largest size)\n\n"
                                    + "```\n" + sens.round(4).to_string(index=False) + "\n```\n")
    fig_schematic(cfg, out)
    if len(df):
        fig_arrival_returned(df, out)
        fig_flux(rows, cfg, out)
    wall = time.perf_counter() - t0
    if a.profile == "pilot":  # extrapolate: production case count per size x mean pilot time per case at that size
        per = df.groupby("size_factor").runtime_s.mean()
        n_prod = pd.Series([s for _, s, _ in case_list(prod)]).value_counts()
        est = float(sum(n_prod[s] * per[s] for s in n_prod.index if s in per.index))
        missing = [s for s in n_prod.index if s not in per.index]
        print(f"PRODUCTION ESTIMATE: {est / 60:.1f} min of solves ({len(case_list(prod))} cases; sizes without pilot data: {missing})")
    print(f"done in {wall:.0f}s; checks {json.dumps(checks)}")
    if not (checks["closure_ok"] and checks["negative_ok"]):
        raise SystemExit("EXP-018 stop rule: a check failed")


if __name__ == "__main__":
    main()

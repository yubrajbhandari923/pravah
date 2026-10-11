"""EXP-021: converged re-run of the EXP-018 prescribed-wind study (external plume -> study box).  Synthetic; not terrain-derived.

    python scripts/prescribed_winds_converged.py --config configs/prescribed_winds_converged.json [--profile smoke|pilot|production] [--out DIR]

Fixes over EXP-018 (docs/winds.md section 5): fixed physical geometry (study box and source never move; the source is >= 6 cells and
>= 6 km from every boundary of the SMALLEST domain), outer half-widths from a declared set whose largest member is >= 2 x U x duration,
automatic enlargement (declared cap) until the two largest domains agree on the box exposure to < tol for EVERY scenario, a numerical
check (dx, dx/2, dx/2 with dt/2) on the base scenario, reversal times after the plume has cleared the box, and two returned-mass
definitions plus flags instead of a silent undercount.  External source only (no local solve)."""
from __future__ import annotations

import argparse
import importlib.util
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

from pravah import forward as F  # noqa: E402
from pravah import winds as W  # noqa: E402
from pravah.config import Boundary, Domain, GaussianSource, Scenario, StudyRegion, Times, Transport, WindSegment  # noqa: E402
from pravah.sources import study_weights  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("prescribed_winds", ROOT / "scripts/prescribed_winds.py")
pw = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(pw)  # type: ignore[union-attr]
STAMP = "EXP-021 prescribed wind, not terrain-derived"
SIDES = pw.SIDES


# ------------------------------------------------------------------------------------------------ pure analysis
def passage_returns(t: np.ndarray, m: np.ndarray, s: np.ndarray, lam: float, H: float, dA: float, emitted: float,
                    a_min: float, exit_frac: float, floor_frac: float) -> dict[str, Any]:
    """Second returned-mass definition (box-mass time series only) plus the exit time of the first passage.
    First passage ends at t_exit = first time m has fallen below ``exit_frac`` x its running maximum P1 (P1 >= a_min*emitted);
    if that never happens the plume has not exited (``not_yet_exited``; returned values are NaN, not 0).
    ratio = int_{t>=t_exit} max(m - floor_frac*P1, 0) dt / int_{t<=t_exit} m dt  (residence-time ratio of the later passage to the first);
    ret_mass_frac = ratio x (mass that entered in the first passage, sum of positive net inward exchange before t_exit) / emitted: a
    mass-equivalent that is exact when both passages have the same transit time (decay and dispersion of late returns are then
    counted as real loss).  ret_eul_after_exit = the EXP-018 net-flux sum started at t_exit instead of at 'm fell 5 % below its max'.
    end_inside: box still above the floor at the end of the run, so a return is cut off (undercount)."""
    rm = np.maximum.accumulate(m)
    hit = np.nonzero((rm >= a_min * emitted) & (m < exit_frac * rm))[0]
    nan = math.nan
    if len(hit) == 0:
        return {"t_exit_s": nan, "p1_ug": float(m.max()), "ret_residence_ratio": nan, "ret_mass_frac": nan,
                "ret_eul_after_exit": nan, "not_yet_exited": True, "end_inside": bool(m[-1] > floor_frac * m.max())}
    k = int(hit[0])
    p1 = float(rm[k])
    n_in = np.diff(m) + lam * H * dA * 0.5 * (s[1:] + s[:-1]) * np.diff(t)
    ratio = float(np.trapezoid(np.maximum(m[k:] - floor_frac * p1, 0.0), t[k:]) / np.trapezoid(m[:k + 1], t[:k + 1]))
    return {"t_exit_s": float(t[k]), "p1_ug": p1, "ret_residence_ratio": ratio,
            "ret_mass_frac": ratio * float(np.maximum(n_in[:k], 0.0).sum()) / emitted,
            "ret_eul_after_exit": float(np.maximum(n_in[k:], 0.0).sum() / emitted), "not_yet_exited": False,
            "end_inside": bool(m[-1] > floor_frac * p1)}


def convergence(df: pd.DataFrame, tol: float) -> tuple[float, pd.DataFrame]:
    """Per (scenario, t_r): relative exposure difference between the two LARGEST half-widths."""
    rows = []
    for (sc, _), g in df.assign(_k=df.t_r_s.fillna(-1)).groupby(["scenario", "_k"]):
        g = g.sort_values("half_width_m")
        a, b = g.iloc[-2], g.iloc[-1]
        rows.append({"scenario": sc, "t_r_s": b.t_r_s, "hw_second_m": a.half_width_m, "hw_ref_m": b.half_width_m,
                     "exposure_rel_diff": abs(a.ext_exposure - b.ext_exposure) / max(abs(b.ext_exposure), 1e-300)})
    out = pd.DataFrame(rows)
    return float(out.exposure_rel_diff.max()), out.assign(ok=out.exposure_rel_diff < tol)


def vs_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Each row against the largest half-width of the same (scenario, t_r): relative for exposure, absolute (fractions of emitted) otherwise."""
    absm = ["returned_eul", "ret_eul_after_exit", "ret_mass_frac"] + [f"leave_{s}_frac" for s in SIDES]
    rows = []
    for _, g in df.assign(_k=df.t_r_s.fillna(-1)).groupby(["scenario", "_k"]):
        ref = g.loc[g.half_width_m.idxmax()]
        for _, r in g.iterrows():
            rows.append({"scenario": r.scenario, "t_r_s": r.t_r_s, "half_width_m": r.half_width_m,
                         "ext_exposure_rel_to_ref": (r.ext_exposure - ref.ext_exposure) / max(abs(ref.ext_exposure), 1e-300),
                         **{f"{m}_minus_ref": r[m] - ref[m] for m in absm}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ geometry and winds
def geometry(cfg: dict[str, Any], hw: float, dx: float | None = None) -> tuple[Domain, tuple[float, float]]:
    dm = cfg["domain"]
    dx = dx or dm["dx"]
    yh = min(hw, dm["y_half_cap_m"])
    nx, ny, nb = round(2 * hw / dx), round(2 * yh / dx), round(dm["study_side_m"] / dx)
    if abs(nx * dx - 2 * hw) > 1e-6 or abs(ny * dx - 2 * yh) > 1e-6 or (nx - nb) % 2 or (ny - nb) % 2 or abs(nb * dx - dm["study_side_m"]) > 1e-6:
        raise ValueError(f"half-width {hw}, dx {dx}: grid or study box does not align")
    return Domain(nx * dx, ny * dx, nx, ny), (hw, yh)


def check_margins(cfg: dict[str, Any]) -> None:
    dm, off = cfg["domain"], cfg["source"]["offset_m"]
    hw = min(dm["half_widths_m"])
    yh = min(hw, dm["y_half_cap_m"])
    need = max(dm["min_margin_m"], dm["min_margin_cells"] * dm["dx"])
    m = {"west": hw + off[0], "east": hw - off[0], "south": yh + off[1], "north": yh - off[1]}
    bad = {k: v for k, v in m.items() if v < need}
    if bad:
        raise ValueError(f"source closer than {need} m to a boundary of the smallest domain: {bad}")
    v = cfg["winds"]["vortex"]
    d = math.hypot(*v["offset_m"])  # vortex centre must be outside the box by >= 2 sigma
    edge = max(abs(v["offset_m"][0]), abs(v["offset_m"][1])) - dm["study_side_m"] / 2
    if edge < 2 * v["sigma_m"]:
        raise ValueError(f"vortex centre (distance {d:.0f} m) is within 2 sigma of the study box")


def build_wind(cfg: dict[str, Any], name: str, dom: Domain, c: tuple[float, float], t_r: float | None) -> tuple[WindSegment, ...]:
    u, v = cfg["transport"]["u"], cfg["transport"]["v"]
    p = cfg["winds"].get(name, {})
    if name == "steady":
        return W.steady_uniform(dom, u, v)
    if name == "reversal":
        return W.delayed_reversal(dom, u, v, float(t_r))  # type: ignore[arg-type]
    if name in ("channel_step", "channel_bump"):
        return W.deflected_channel(dom, u, c[0] + p["x0_offset_m"], p["width_m"], p["offset_m"], name.split("_")[1])
    if name == "vortex":
        return W.recirculation_eddy(dom, u, v, (c[0] + p["offset_m"][0], c[1] + p["offset_m"][1]), p["sigma_m"],
                                    p["peak_speed_m_s"] * p["sigma_m"] * math.sqrt(math.e))
    raise ValueError(f"unknown scenario {name!r}")


def build_scenario(cfg: dict[str, Any], name: str, hw: float, t_r: float | None, dx: float | None = None,
                   dt: float | None = None) -> Scenario:
    dom, c = geometry(cfg, hw, dx)
    s, tm, tr = cfg["source"], cfg["times"], cfg["transport"]
    half = cfg["domain"]["study_side_m"] / 2
    return Scenario(domain=dom,
                    sources=(GaussianSource("ext", c[0] + s["offset_m"][0], c[1] + s["offset_m"][1], s["sigma_m"],
                                            ((0.0, 0.0), (s["on_s"], 1.0), (s["off_s"], 0.0)), "external"),),
                    transport=Transport(build_wind(cfg, name, dom, c, t_r), tr["K"], tr["lam"], tr["H"]),
                    boundary=Boundary("open"), times=Times(tm["duration"], dt or tm["dt"], tm["output_interval"]),
                    study=StudyRegion(c[0] - half, c[1] - half, c[0] + half, c[1] + half))


def run_case(cfg: dict[str, Any], name: str, hw: float, t_r: float | None, dx: float | None = None,
             dt: float | None = None) -> dict[str, Any]:
    scn = build_scenario(cfg, name, hw, t_r, dx, dt)
    rt = cfg["returned"]
    t0 = time.perf_counter()
    res = F.simulate(scn, [cfg["source"]["rate_ug_s"]])
    runtime = time.perf_counter() - t0
    w = study_weights(scn)
    assert w is not None
    d, tr = scn.domain, scn.transport
    dA, t = d.dx * d.dy, np.array(scn.output_times)
    conc = res.concentration.values
    s_box = (conc * w).sum(axis=(1, 2))
    m_box = tr.H * dA * s_box
    mt = pw.mass_terms(res, scn)
    emitted = mt["emitted_ug"]
    row: dict[str, Any] = {"scenario": name, "t_r_s": t_r if t_r is not None else math.nan, "half_width_m": hw, "dx_m": d.dx,
                           "dt_s": scn.times.dt, "nx": d.nx, "ny": d.ny, "runtime_s": runtime, "nsteps": len(t) and round(scn.times.duration / scn.times.dt),
                           **{k: v for k, v in mt.items()}}
    row["ext_exposure"] = float(np.trapezoid(s_box / float(w.sum()), t))  # ug s / m^3, study mean
    row["ext_peak_ug_m3"] = float((s_box / float(w.sum())).max())
    row["returned_eul"], row["t_exit_eul_s"] = pw.returned_mass(t, m_box, s_box, tr.lam, tr.H, dA, emitted, rt["a_min"], rt["eul_theta"])
    row.update(passage_returns(t, m_box, s_box, tr.lam, tr.H, dA, emitted, rt["a_min"], rt["exit_frac"], rt["floor_frac"]))
    mtr = float(np.interp(t_r, t, m_box)) / emitted if t_r is not None else math.nan
    row["box_frac_at_tr"] = mtr
    row["exited_before_tr"] = bool(t_r is None or (not row["not_yet_exited"] and row["t_exit_s"] <= t_r))
    row["tail_inside_at_tr"] = bool(t_r is not None and mtr * emitted > rt["clear_frac"] * row["p1_ug"])
    fl = pw.side_rates(scn, conc, t)
    row["side_vs_budget_rel"] = abs(np.trapezoid(fl.sum(axis=1), t) - mt["leaving_ug"]) / emitted
    for i, sd in enumerate(SIDES):
        row[f"leave_{sd}_ug"] = float(np.trapezoid(fl[:, i], t))
        row[f"leave_{sd}_frac"] = row[f"leave_{sd}_ug"] / emitted
    row["emitted_frac_removed"] = mt["removed_ug"] / emitted
    row["_t"], row["_m"], row["_emitted"] = t, m_box, emitted
    return row


# ------------------------------------------------------------------------------------------------ driver
def case_list(cfg: dict[str, Any]) -> list[tuple[str, float | None]]:
    return [(n, tr) for n in cfg["scenarios"] for tr in (cfg["t_r_s"] if n == "reversal" else [None])]


def plan_steps(cfg: dict[str, Any]) -> list[tuple[int, int]]:
    """(cells, steps) of every solve the declared (non-enlarged) plan makes, for the production estimate."""
    out = []
    steps = round(cfg["times"]["duration"] / cfg["times"]["dt"])
    for hw in cfg["domain"]["half_widths_m"]:
        d, _ = geometry(cfg, hw)
        out += [(d.nx * d.ny, steps)] * len(case_list(cfg))
    d, _ = geometry(cfg, cfg["numerics"]["half_width_m"])
    return out + [(d.nx * d.ny, steps), (4 * d.nx * d.ny, steps), (4 * d.nx * d.ny, 2 * steps)]


def stamp(fig: Any) -> None:
    fig.text(0.995, 0.005, STAMP, ha="right", va="bottom", fontsize=8, color="0.3", weight="bold")


def figures(df: pd.DataFrame, conv: pd.DataFrame, rows: list[dict[str, Any]], num: pd.DataFrame, cfg: dict[str, Any],
            tol: float, out: Path) -> None:
    lab = lambda r: r.scenario + ("" if math.isnan(r.t_r_s) else f" t_r={r.t_r_s / 3600:g}h")  # noqa: E731
    vr = vs_reference(df)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for (sc, k), g in vr.assign(_k=vr.t_r_s.fillna(-1)).groupby(["scenario", "_k"]):
        g = g[g.half_width_m < g.half_width_m.max()].sort_values("half_width_m")
        n = sc + ("" if k < 0 else f" t_r={k / 3600:g}h")
        axs[0].plot(g.half_width_m / 1e3, g.ext_exposure_rel_to_ref.abs() * 100, "o-", label=n)
        axs[1].plot(g.half_width_m / 1e3, g.ret_mass_frac_minus_ref.abs().where(g.ret_mass_frac_minus_ref.notna()), "o-", label=n)
    axs[0].axhline(tol * 100, color="k", ls=":")
    axs[0].set(xscale="log", yscale="log", xlabel="outer half-width [km]", ylabel="|exposure - reference| / reference [%]",
               title="Box exposure vs reference domain (dotted: stop tolerance)")
    axs[1].set(xscale="log", xlabel="outer half-width [km]", ylabel="|returned mass (def. 2) - reference| [frac. of emitted]",
               title="Returned mass vs reference domain")
    axs[0].legend(fontsize=7)
    stamp(fig)
    fig.savefig(out / "fig1_convergence.png", dpi=130)
    plt.close(fig)

    ref_hw = df.half_width_m.max()
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    for r in (r for r in rows if r["half_width_m"] == ref_hw):
        ln, = ax.plot(r["_t"] / 3600, r["_m"] / r["_emitted"], label=r["scenario"] + ("" if math.isnan(r["t_r_s"]) else f" t_r={r['t_r_s'] / 3600:g}h"))
        if not math.isnan(r["t_exit_s"]):
            ax.plot(r["t_exit_s"] / 3600, np.interp(r["t_exit_s"], r["_t"], r["_m"]) / r["_emitted"], "v", color=ln.get_color())
        if not math.isnan(r["t_r_s"]):
            ax.axvline(r["t_r_s"] / 3600, color=ln.get_color(), ls=":", lw=0.8)
    ax.set(xlabel="time [h]", ylabel="external mass in study box / emitted", title=f"Reference domain (half-width {ref_hw / 1e3:g} km); triangle: first-passage exit; dotted: t_r")
    ax.legend(fontsize=7)
    stamp(fig)
    fig.savefig(out / "fig2_box_mass_series.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    st = df[df.scenario == "steady"].sort_values("half_width_m")
    bot = np.zeros(len(st))
    for sd in SIDES:
        axs[0].bar(np.arange(len(st)), st[f"leave_{sd}_frac"], bottom=bot, label=sd)
        bot += st[f"leave_{sd}_frac"].to_numpy()
    axs[0].set_xticks(np.arange(len(st)), [f"{h / 1e3:g}" for h in st.half_width_m])
    axs[0].set(xlabel="outer half-width [km]", ylabel="mass lost through side / emitted (steady)", title="Boundary loss by side")
    axs[0].legend(fontsize=7)
    axs[1].plot(num.dx_m.astype(str) + "/" + num.dt_s.astype(str), num.ext_exposure / num.ext_exposure.iloc[-1], "o-")
    axs[1].set(xlabel="dx [m] / dt [s]", ylabel="box exposure / finest", title="Numerical check, base scenario, smallest domain")
    stamp(fig)
    fig.savefig(out / "fig3_side_loss_and_numerics.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prescribed_winds_converged.json")
    ap.add_argument("--profile", default="production", choices=("smoke", "pilot", "production"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = json.loads((ROOT / a.config).read_text())
    prod = json.loads(json.dumps(cfg))
    cfg.update(cfg.get("profiles", {}).get(a.profile, {}))
    out = Path(a.out) if a.out else ROOT / "results" / ("phase3-winds-converged" if a.profile == "production" else f"phase3-winds-converged-{a.profile}")
    out.mkdir(parents=True, exist_ok=True)
    check_margins(cfg)
    en, ck = cfg["enlarge"], cfg["checks"]
    adv = cfg["transport"]["u"] * cfg["times"]["duration"]  # plume advection distance U*T
    hws = sorted(cfg["domain"]["half_widths_m"])
    if hws[-1] < en["ref_min_advection_factor"] * adv:
        raise ValueError(f"largest half-width {hws[-1]} < {en['ref_min_advection_factor']} x U T = {en['ref_min_advection_factor'] * adv}")
    cases = case_list(cfg)
    print(f"[{STAMP}] profile {a.profile}: {len(cases)} cases x {len(hws)} domains, U*T = {adv / 1e3:g} km -> {out}", flush=True)
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    def run_domain(hw: float) -> None:
        for name, tr in cases:
            r = run_case(cfg, name, hw, tr)
            rows.append(r)
            print(f"[{time.perf_counter() - t0:6.0f}s] hw {hw / 1e3:g} km {name} t_r {tr} cells {r['nx'] * r['ny']} solve {r['runtime_s']:.1f}s "
                  f"closure {r['closure_rel']:.1e} exposure {r['ext_exposure']:.4g} exit {r['t_exit_s']:.0f}", flush=True)

    for hw in hws:
        run_domain(hw)
    enlarged: list[float] = []
    while True:
        worst, conv = convergence(pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]), en["tol"])
        nxt = math.ceil(hws[-1] * en["factor"] / en["round_m"]) * en["round_m"]
        if worst < en["tol"] or nxt > en["cap_m"]:
            break
        print(f"not converged (worst {worst:.3g} >= {en['tol']}): adding half-width {nxt / 1e3:g} km", flush=True)
        hws.append(nxt)
        enlarged.append(nxt)
        run_domain(nxt)
    converged = worst < en["tol"]
    nm = cfg["numerics"]
    num = pd.DataFrame([{k: v for k, v in run_case(cfg, nm["scenario"], nm["half_width_m"], None, dx, dt).items() if not k.startswith("_")}
                        for dx, dt in ((cfg["domain"]["dx"], cfg["times"]["dt"]), (cfg["domain"]["dx"] / 2, cfg["times"]["dt"]),
                                       (cfg["domain"]["dx"] / 2, cfg["times"]["dt"] / 2))])
    num["ext_exposure_rel_to_finest"] = (num.ext_exposure - num.ext_exposure.iloc[-1]) / num.ext_exposure.iloc[-1]
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    allc = pd.concat([df, num])
    clo = float(allc[["closure_rel", "step_residual_rel"]].max().max())
    neg = float((-allc.min_conc / allc.max_conc).max())
    exposure_trunc = float(df[(df.scenario == nm["scenario"]) & (df.half_width_m == nm["half_width_m"])].ext_exposure.iloc[0])
    ref_steady = float(df[(df.scenario == nm["scenario"]) & (df.half_width_m == df.half_width_m.max())].ext_exposure.iloc[0])
    checks = {"max_closure_rel": clo, "closure_limit": ck["closure_rel"], "closure_ok": clo < ck["closure_rel"],
              "max_negative_rel_to_max_conc": neg, "negative_limit": ck["negative_rel"], "negative_ok": neg < ck["negative_rel"],
              "max_side_vs_budget_rel": float(df.side_vs_budget_rel.max()),
              "advection_distance_m": adv, "reference_half_width_m": float(df.half_width_m.max()),
              "reference_over_2UT": float(df.half_width_m.max() / (2 * adv)) if adv else math.nan,
              "worst_exposure_rel_diff_two_largest": worst, "convergence_tol": en["tol"], "converged": bool(converged),
              "enlarged_half_widths_m": enlarged, "enlarge_cap_m": en["cap_m"],
              "numerical_exposure_rel_dx_vs_finest": float(num.ext_exposure_rel_to_finest.iloc[0]),
              "numerical_exposure_rel_dx2_vs_finest": float(num.ext_exposure_rel_to_finest.iloc[1]),
              "truncation_exposure_rel_smallest_vs_ref_base": (exposure_trunc - ref_steady) / ref_steady,
              "reversals_with_plume_not_cleared": [[r["half_width_m"], r["t_r_s"]] for r in rows if not r["exited_before_tr"]]}
    drop = lambda d: d.drop(columns=[c for c in d.columns if c.startswith("_")])  # noqa: E731
    drop(df).to_csv(out / "cases_raw.csv", index=False)
    num.to_csv(out / "numerical_check_raw.csv", index=False)
    pd.concat([pd.DataFrame({"scenario": r["scenario"], "t_r_s": r["t_r_s"], "half_width_m": r["half_width_m"], "t_s": r["_t"],
                             "box_mass_ug": r["_m"], "box_mass_frac_emitted": r["_m"] / r["_emitted"]}) for r in rows]).to_csv(out / "timeseries_raw.csv", index=False)
    vr = vs_reference(df)
    vr.to_csv(out / "vs_reference.csv", index=False)
    conv.to_csv(out / "convergence.csv", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    wall = time.perf_counter() - t0
    (out / "provenance.json").write_text(json.dumps({"label": STAMP, "profile": a.profile, "config": cfg, "n_cases": len(rows),
                                                    "python": sys.version.split()[0], "platform": platform.platform(), "wall_s": wall}, indent=1, default=str))
    (out / "summary.md").write_text(f"# EXP-021 ({a.profile}) -- {STAMP}\n\nchecks: {json.dumps(checks)}\n\n## Convergence (two largest domains)\n\n```\n"
                                    + conv.round(5).to_string(index=False) + "\n```\n\n## Numerical check (base scenario, smallest domain)\n\n```\n"
                                    + num[["dx_m", "dt_s", "ext_exposure", "ext_exposure_rel_to_finest", "ext_peak_ug_m3"]].to_string(index=False)
                                    + "\n```\n\n## Against the reference domain\n\n```\n" + vr.round(5).to_string(index=False) + "\n```\n")
    figures(df, conv, rows, num, cfg, en["tol"], out)
    if a.profile == "pilot":  # production estimate from the pilot's seconds per cell-step
        rate = float((df.runtime_s / (df.nx * df.ny * df.nsteps)).mean())
        est = rate * sum(c * s for c, s in plan_steps(prod))
        print(f"PRODUCTION ESTIMATE: {est / 60:.1f} min of solves ({len(plan_steps(prod))} solves, declared set only, no enlargement; "
              f"{rate * 1e6:.2f} us per cell-step from the pilot)")
    print(f"done in {wall:.0f}s; converged {converged}; checks {json.dumps({k: v for k, v in checks.items() if k != 'reversals_with_plume_not_cleared'})}")
    if not (checks["closure_ok"] and checks["negative_ok"]):
        raise SystemExit("EXP-021 stop rule: a check failed")


if __name__ == "__main__":
    main()

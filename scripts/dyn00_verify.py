"""DYN-00 (EXP-028): dynamical transport verification.  Run once as a registered run, not on every pytest.

    .research/bin/research run exec EXP-028 -- .venv/bin/python scripts/dyn00_verify.py

Tolerances are fixed here, before the first run: mass closure rel <= 1e-6, superposition rel <= 1e-8, analytic
mass histories (closed box) first-order in dt, nonnegativity min >= -1e-9 * max.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np

from pravah import dynamic as dyn
from pravah.config import (Boundary, Domain, GaussianSource, InflowBasis, RasterSource, Scenario, Times,
                           Transport, WindSegment)
from pravah.forward import _boundary_arrays, simulate
from pravah.inverse import compute_responses
from pravah.sources import source_fields

OUT = Path(os.environ.get("RESEARCH_RUN_DIR", "results/dyn00"))
ROWS: list[dict] = []
H = 3600.0


def rec(check: str, value: float, tol: float, ok: bool, note: str = "") -> None:
    ROWS.append({"check": check, "value": float(value), "tol": tol, "pass": bool(ok), "note": note})
    print(f"{'PASS' if ok else 'FAIL'} {check}: {value:.3e} (tol {tol:.1e}) {note}")


def box(n=16, wind=((0.0, 1.0, 0.3),), mode="open", bases=(), srcs=None, dur=2 * H, dt=60.0, K=None, lam=None):
    d = Domain(lx=8000.0, ly=6000.0, nx=n, ny=int(0.75 * n))
    srcs = srcs or (GaussianSource("a", 2000.0, 3000.0, 700.0), GaussianSource("b", 5000.0, 2500.0, 700.0))
    tr = Transport(wind=tuple(WindSegment(*w) for w in wind))
    if K is not None:
        tr = dataclasses.replace(tr, K=K)
    if lam is not None:
        tr = dataclasses.replace(tr, lam=lam)
    return Scenario(domain=d, sources=srcs, transport=tr, boundary=Boundary(mode=mode, bases=bases),
                    times=Times(duration=dur, dt=dt, output_interval=600.0))


def prof(s, p):
    return dataclasses.replace(s, profile=p)


def basis_face_reclassification() -> None:
    """Basis-face reclassification under wind reversal (replaces the vacuous west_no_import_after_reversal check
    per F-038/R1: that west profile was zero from 1800 s, before the 3600 s reversal, so the checked 'inflow' budget
    term was dominated by the east Dirichlet loss and tested nothing).

    Constant west-only and east-only bases (amplitude 1, profile 1) under wind (+1,0) then (-1,0) at 3600 s.
    Supply of a basis in a wind segment = sum over its sides of the cell-summed unit inflow array from
    forward._boundary_arrays. Fixed tolerances: supply == 0.0 exactly on outflow faces, > 0 on inflow faces;
    mass closure max_rel_budget_residual <= 1e-6 with both bases on.
    """
    west = InflowBasis("w_const", sides=("west",), profile=((0.0, 1.0),))
    east = InflowBasis("e_const", sides=("east",), profile=((0.0, 1.0),))
    scn = box(wind=((0.0, 1.0, 0.0), (H, -1.0, 0.0)), bases=(west, east))
    unit = [_boundary_arrays(scn, w)[3] for w in scn.transport.wind]
    w0, w1 = (float(unit[k]["west"].sum()) for k in (0, 1))
    e0, e1 = (float(unit[k]["east"].sum()) for k in (0, 1))
    rec("basis_west_supply_pre_reversal", w0, 0.0, w0 > 0.0, "must be >0 (west is inflow)")
    rec("basis_west_supply_post_reversal", w1, 0.0, w1 == 0.0, "must be exactly 0 (west is outflow)")
    rec("basis_east_supply_pre_reversal", e0, 0.0, e0 == 0.0, "must be exactly 0 (east is outflow)")
    rec("basis_east_supply_post_reversal", e1, 0.0, e1 > 0.0, "must be >0 (east becomes inflow)")
    res = simulate(scn, [5.0, 5.0, 1.0, 1.0])
    rec("mass_closure[basis_reclassification]", res.diagnostics["max_rel_budget_residual"], 1e-6,
        res.diagnostics["max_rel_budget_residual"] <= 1e-6, "both bases on")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # C1 zero forcing
    r = simulate(box(), np.zeros(2))
    rec("zero_forcing_max_abs", np.abs(r.concentration.values).max(), 0.0, np.abs(r.concentration.values).max() == 0.0)

    # C2/C3 mass closure and nonnegativity: activation+shutdown, simultaneous change, pulse+reversal, tangential, zero wind
    a_on = prof(GaussianSource("a", 2000.0, 3000.0, 700.0), ((0.0, 0.0), (1800.0, 1.0), (4800.0, 0.0)))
    b_ramp = prof(GaussianSource("b", 5000.0, 2500.0, 700.0), ((0.0, 2.0), (1800.0, 0.5), (4800.0, 3.0)))
    west = InflowBasis("w_pulse", sides=("west",), profile=((0.0, 1.0), (1800.0, 0.0)))
    allp = InflowBasis("all_pulse", profile=((0.0, 0.0), (900.0, 1.0), (2700.0, 0.0)))
    cases = {
        "activation_shutdown": (box(srcs=(a_on, b_ramp)), [5.0, 5.0]),
        "pulse_wind_reversal": (box(wind=((0.0, 1.0, 0.0), (3600.0, -1.0, 0.0)), bases=(west, allp)), [5.0, 5.0, 3.0, 3.0]),
        "rotation_changing_faces": (box(wind=((0.0, 1.0, 0.0), (2400.0, 0.0, 1.0), (4800.0, -1.0, -0.5)), bases=(allp,)), [5.0, 5.0, 3.0]),
        "tangential_wind": (box(wind=((0.0, 0.0, 1.0),), bases=(allp,)), [5.0, 5.0, 3.0]),
        "zero_wind": (box(wind=((0.0, 0.0, 0.0),), bases=(allp,)), [5.0, 5.0, 3.0]),
    }
    for name, (scn, q) in cases.items():
        res = simulate(scn, q)
        c = res.concentration.values
        rec(f"mass_closure[{name}]", res.diagnostics["max_rel_budget_residual"], 1e-6,
            res.diagnostics["max_rel_budget_residual"] <= 1e-6)
        rec(f"min_conc_rel[{name}]", c.min() / max(c.max(), 1e-300), -1e-9, c.min() >= -1e-9 * c.max(), "min/max")
    # basis-face reclassification under reversal (replaces the vacuous west_no_import_after_reversal, F-038/R1)
    basis_face_reclassification()

    # C4 superposition (linear scheme)
    s = box()
    f = lambda q: simulate(s, q).concentration.values  # noqa: E731
    rec("superposition", np.abs(f([2, 3]) - f([2, 0]) - f([0, 3])).max() / np.abs(f([2, 3])).max(), 1e-8,
        np.abs(f([2, 3]) - f([2, 0]) - f([0, 3])).max() / np.abs(f([2, 3])).max() <= 1e-8)

    # C5 closed-box analytic mass history under on/off emission: dM/dt = q s(t) - lam M; implicit Euler => O(dt) error
    lam = 1.0 / 7200.0
    q = 4.0
    T1, T2 = 1800.0, 4800.0
    errs = {}
    for dt in (240.0, 120.0, 60.0, 30.0):
        sc = box(wind=((0.0, 0.0, 0.0),), mode="closed", srcs=(prof(GaussianSource("a", 2000.0, 3000.0, 700.0),
                                                                    ((0.0, 0.0), (T1, 1.0), (T2, 0.0))),), dur=2 * H, dt=dt, lam=lam)
        res = simulate(sc, [q])
        tt = np.array(sc.output_times)
        M_num = sc.transport.H * sc.domain.dx * sc.domain.dy * res.concentration.values.sum(axis=(1, 2))
        M1 = q / lam * (1 - np.exp(-lam * np.clip(tt - T1, 0, T2 - T1)))
        M_ex = np.where(tt <= T2, M1, q / lam * (1 - np.exp(-lam * (T2 - T1))) * np.exp(-lam * (tt - T2)))
        errs[dt] = np.abs(M_num - M_ex).max() / M_ex.max()
    orders = [np.log2(errs[a] / errs[b]) for a, b in ((240.0, 120.0), (120.0, 60.0), (60.0, 30.0))]
    rec("closed_box_analytic_err_dt30", errs[30.0], 5e-3, errs[30.0] <= 5e-3, f"orders={np.round(orders, 2).tolist()}")
    rec("closed_box_temporal_order_last", orders[-1], 0.8, 0.8 <= orders[-1] <= 1.3, "expect ~1 (implicit Euler)")

    # C7 refinement of a sensor series with a time-varying source under reversing wind
    def series(n, dt):
        sc = box(n=n, wind=((0.0, 1.0, 0.2), (3600.0, -1.0, -0.2)), srcs=(a_on,), dt=dt)
        c = simulate(sc, [5.0]).concentration
        return np.array([float(c.sel(time=t).interp(x=4500.0, y=3000.0)) for t in (1800.0, 3600.0, 5400.0, 7200.0)])
    ref_t = series(16, 15.0)
    et = [np.abs(series(16, dt) - ref_t).max() / np.abs(ref_t).max() for dt in (240.0, 120.0, 60.0)]
    rec("dt_refinement_monotone", et[-1], et[0], et[0] > et[1] > et[2], f"errs={np.round(et, 5).tolist()}")
    ref_s = series(64, 60.0)
    es = [np.abs(series(n, 60.0) - ref_s).max() / np.abs(ref_s).max() for n in (16, 32)]
    rec("dx_refinement_monotone", es[-1], es[0], es[0] > es[1], f"errs(16,32 vs 64)={np.round(es, 5).tolist()}")

    # C8 static equivalence: all-ones bin history == legacy constant-profile run (also line source below)
    sc = box(bases=(InflowBasis("w", sides=("west",)),))
    nb = 4
    R = compute_responses(dyn.expand(sc, nb))
    amp = np.array([3.0, 2.0, 1.5])
    direct = simulate(sc, amp).concentration.values
    sup = (R.field(np.repeat(amp, nb)).values - R.base)
    rec("static_equivalence_rel", np.abs(sup - direct).max() / np.abs(direct).max(), 1e-6,
        np.abs(sup - direct).max() / np.abs(direct).max() <= 1e-6)
    # non-trivial per-bin history (n_params x nb, parameter-major): expanded-model superposition vs direct run
    hist = np.array([[0.0, 3.0, 1.0, 2.0], [1.0, 0.0, 2.0, 4.0], [0.5, 1.5, 0.0, 3.0]])
    edges = dyn.bin_edges(sc.times.duration, nb)
    sup_nt = R.field(hist.reshape(-1)).values - R.base
    direct_nt = simulate(dyn.truth_scenario(sc, hist, edges), np.ones(sc.n_params)).concentration.values
    err_nt = np.abs(sup_nt - direct_nt).max() / np.abs(direct_nt).max()
    rec("static_equivalence_nontrivial_rel", err_nt, 1e-6, err_nt <= 1e-6, "history [0,3,1,2] etc., piecewise profiles")

    # C9 footprint normalisation: point (Gaussian), line (road) and area rasters integrate to 1/m^2*m^2
    scn = box()
    ny, nx = scn.domain.ny, scn.domain.nx
    line = np.zeros((ny, nx)); line[ny // 2, 2:nx - 2] = 1.0
    area = np.zeros((ny, nx)); area[3:7, 5:11] = 1.0
    dA = scn.domain.dx * scn.domain.dy
    rs = [RasterSource.from_array("road", "local", line), RasterSource.from_array("town", "local", area)]
    sf = source_fields(dataclasses.replace(scn, sources=(scn.sources[0], *rs)))
    err = max(abs(float(f_.sum() * dA) - 1.0) for f_ in sf)
    rec("footprint_integral_err", err, 1e-12, err <= 1e-12)
    rline = simulate(dataclasses.replace(scn, sources=(rs[0],)), [5.0])
    rec("line_source_mass_closure", rline.diagnostics["max_rel_budget_residual"], 1e-6,
        rline.diagnostics["max_rel_budget_residual"] <= 1e-6)

    (OUT / "dyn00_checks.json").write_text(json.dumps(ROWS, indent=1))
    bad = [r["check"] for r in ROWS if not r["pass"]]
    print(f"\n{len(ROWS) - len(bad)}/{len(ROWS)} checks pass; failing: {bad}")
    try:
        import research  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
        research.log_metric("checks_passed", len(ROWS) - len(bad))
        research.log_metric("checks_total", len(ROWS))
        research.register_artifact(str(OUT / "dyn00_checks.json"), description="DYN-00 verification table")
    except Exception as e:  # noqa: BLE001 - metrics are optional outside a Research OS run
        print("research logging skipped:", e)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()

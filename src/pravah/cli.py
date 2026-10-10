"""Command line: ``python -m pravah {simulate,inverse,benchmark} --config CFG --out DIR [--overwrite]``."""
from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import io
from .config import FieldResult, Scenario, SolverOptions, load_config, scenario_from_dict, to_dict


def _setup(cfg_path: str, out: str, overwrite: bool) -> tuple[dict[str, Any], Path]:
    cfg = load_config(cfg_path)
    return cfg, io.make_run_dir(out, overwrite)


def _scenario(cfg: dict[str, Any]) -> tuple[Scenario, np.ndarray, SolverOptions]:
    return (scenario_from_dict(cfg["scenario"]), np.asarray(cfg["q"], dtype=float),
            SolverOptions(**cfg.get("solver", {})))


def _budget_summary(res: FieldResult) -> dict[str, Any]:
    out: dict[str, Any] = dict(res.diagnostics)
    if res.budget is not None:
        out.update({f"total_{k}": float(res.budget[k].sum()) for k in ("emission", "removal", "inflow", "outflow")})
    out["status"] = res.status
    out["runtime_s"] = res.runtime_s
    out["max_concentration"] = float(res.concentration.max())
    return out


def cmd_simulate(args: argparse.Namespace) -> int:
    from . import plotting
    from .forward import simulate
    cfg, out = _setup(args.config, args.out, args.overwrite)
    sc, q, solver = _scenario(cfg)
    res = simulate(sc, q, solver)
    io.save_field(out / "concentration.nc", res.concentration)
    if res.budget is not None:
        io.save_field(out / "budget.nc", res.budget)
    io.write_json(out / "metrics.json", _budget_summary(res))
    io.write_json(out / "provenance.json", io.provenance(cfg, {}, {"solver": to_dict(solver)}))
    plotting.plot_field_snapshots(res.concentration).savefig(out / "snapshots.png", dpi=120)
    if res.budget is not None:
        plotting.plot_mass_budget(res.budget).savefig(out / "mass_budget.png", dpi=120)
    print(f"simulate: {res.diagnostics['n_steps']} steps, {res.runtime_s:.1f} s, "
          f"max budget residual {res.diagnostics['max_rel_budget_residual']:.2e}, "
          f"{res.diagnostics['n_negative']} negative cells -> {out}")
    return 0


def cmd_inverse(args: argparse.Namespace) -> int:
    from . import plotting
    from .forward import simulate
    from .inverse import InferenceOptions, ResponseCache, fit_sources, reconstruct
    from .metrics import field_rmse, sensor_rmse, source_error
    from .observations import ObservationSpec, SensorLayout, observe
    cfg, out = _setup(args.config, args.out, args.overwrite)
    sc, q, solver = _scenario(cfg)
    inv = cfg["inverse"]
    truth = simulate(sc, q, solver)
    sens = inv["sensors"]
    layout = SensorLayout(tuple(f"S{i + 1}" for i in range(len(sens))), tuple(float(s[0]) for s in sens),
                          tuple(float(s[1]) for s in sens))
    n = int(sc.times.duration // inv["sensor_interval"])
    times = tuple(inv["sensor_interval"] * (i + 1) for i in range(n))
    obs = observe(truth.concentration, layout, ObservationSpec(times, inv["noise_std"], inv["seed"]))
    held = tuple(inv.get("heldout_ids", ()))
    cache, report, fits = ResponseCache(), {}, {}
    methods: list[Literal["linear", "direct"]] = ["linear", "direct"] if args.direct else ["linear"]
    for m in methods:
        t0 = time.perf_counter()
        fit = fit_sources(sc, obs, InferenceOptions(method=m, heldout_ids=held, solver=solver, alpha=inv.get("alpha", 0.0)), cache)
        rec = reconstruct(sc, fit, solver)
        fits[m] = fit
        used = layout.subset([i for i in layout.ids if i not in held])
        report[m] = {
            "q_true": q.tolist(), "q_hat": fit.q_hat.tolist(), "source_error": source_error(fit.q_hat, q),
            "success": fit.success, "message": fit.message, "weighted_residual": fit.weighted_residual,
            "n_obs": fit.n_obs, "singular_values": fit.singular_values, "rank": fit.rank,
            "effective_rank": fit.effective_rank, "condition": fit.condition, "q_std": fit.q_std,
            "regional_rmse": field_rmse(rec.concentration, truth.concentration),
            "heldout_sensor_rmse": (sensor_rmse(rec.concentration, truth.concentration, layout.subset(held), times)
                                    if held else None),
            "fit_sensor_rmse": sensor_rmse(rec.concentration, truth.concentration, used, times),
            "wall_s": time.perf_counter() - t0,
        }
        if m == "linear":
            plotting.plot_compare_fields(truth.concentration, rec.concentration).savefig(out / "truth_vs_fit.png", dpi=120)
    if "direct" in fits:
        report["direct_vs_linear_max_rel_diff"] = float(
            np.max(np.abs(fits["direct"].q_hat - fits["linear"].q_hat) / np.abs(fits["linear"].q_hat)))
    report["assumptions"] = ("Wind, K, lambda, H, source geometry, initial state and boundary forcing are known; only "
                             "the 3 emission amplitudes are fitted. Accurate fields do not imply correct attribution.")
    io.write_json(out / "inverse_metrics.json", report)
    io.write_json(out / "provenance.json", io.provenance(cfg, {"noise_seed": inv["seed"]}, {}))
    lin = report["linear"]
    print(f"inverse: q_hat={np.round(lin['q_hat'], 0).tolist()} true={q.tolist()} rank={lin['rank']} "
          f"cond={lin['condition']:.3g} regional_rmse={lin['regional_rmse']:.3g} -> {out}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .experiments import run_benchmark, run_mismatch
    cfg = load_config(args.config)
    out = Path(args.out)
    kind = cfg.get("experiment", "benchmark")
    if kind == "mismatch":
        run_mismatch(cfg, Path(cfg["mismatch"]["layouts_from"]), out, overwrite=args.overwrite, progress=print)
    else:
        run_benchmark(cfg, out, overwrite=args.overwrite, progress=print)
    print(f"{kind} written to {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pravah", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, helptext in (("simulate", cmd_simulate, "run the forward model"),
                               ("inverse", cmd_inverse, "synthetic source-recovery experiment"),
                               ("benchmark", cmd_benchmark, "sensor-placement benchmark (or mismatch, per config)")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--config", required=True, help="JSON config (see configs/)")
        p.add_argument("--out", required=True, help="output directory (refused if non-empty)")
        p.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty --out")
        if name == "inverse":
            p.add_argument("--direct", action="store_true", help="also run the direct least_squares route and compare")
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")  # CLI is headless; library code never sets the backend
    return int(args.fn(args))

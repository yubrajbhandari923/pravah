"""Command line: ``python -m pravah {simulate,inverse,benchmark} --config CFG --out DIR [--overwrite]``.

Forward solver selection (all three commands; default FiPy, results unchanged):
  --solver SPEC             solver for both roles: 'fipy' or 'module:callable' (a FACTORY called once with the kwargs
                            and returning a ForwardSolver, see pravah.forward.get_forward)
  --truth-solver SPEC       solver that generates the synthetic truth (overrides --solver for that role)
  --inference-solver SPEC   solver for response bases, fits and reconstructions (overrides --solver for that role)
  --solver-kwargs JSON      kwargs object for the factory; applies to --solver and to each role without its own
  --truth-solver-kwargs JSON, --inference-solver-kwargs JSON   role-specific kwargs
Each role's solver is resolved once here and passed down as an object; the BackendInfo of both is recorded in
provenance.json.  See docs/solver-interchangeability.md."""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import io
from .config import FieldResult, Scenario, SolverOptions, load_config, scenario_from_dict, to_dict
from .forward import ForwardSolver, forward_info, get_forward


def _kwargs(text: str | None, flag: str) -> dict[str, Any] | None:
    if text is None:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{flag} must be a JSON object: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"{flag} must be a JSON object, got {type(obj).__name__}")
    return obj


def _forwards(args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[ForwardSolver, ForwardSolver]:
    """(truth, inference) solvers: CLI flags win over cfg['backends'] (a role without any flag uses the config /
    FiPy).  --solver / --solver-kwargs are the shared defaults of both roles."""
    from .studies.toy.experiments import resolve_backends
    shared_kw = _kwargs(args.solver_kwargs, "--solver-kwargs")
    shared = get_forward(args.solver, shared_kw) if (args.solver is not None or shared_kw is not None) else None
    out: dict[str, ForwardSolver | None] = {}
    for role in ("truth", "inference"):
        spec = getattr(args, f"{role}_solver")
        kw = _kwargs(getattr(args, f"{role}_solver_kwargs"), f"--{role}-solver-kwargs")
        if spec is None and kw is None:
            out[role] = shared  # one object for both roles when only --solver is given
        else:
            out[role] = get_forward(spec if spec is not None else args.solver, kw if kw is not None else shared_kw)
    return resolve_backends(cfg, out["truth"], out["inference"])


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
    from .inverse import run_forward
    cfg, out = _setup(args.config, args.out, args.overwrite)
    sc, q, solver = _scenario(cfg)
    truth_fwd, _ = _forwards(args, cfg)
    res = run_forward(truth_fwd, sc, q, solver)
    io.save_field(out / "concentration.nc", res.concentration)
    if res.budget is not None:
        io.save_field(out / "budget.nc", res.budget)
    io.write_json(out / "metrics.json", _budget_summary(res))
    io.write_json(out / "provenance.json", io.provenance(cfg, {}, {"solver": to_dict(solver),
                                                                       "backend": forward_info(truth_fwd).to_dict()}))
    plotting.plot_field_snapshots(res.concentration).savefig(out / "snapshots.png", dpi=120)
    if res.budget is not None:
        plotting.plot_mass_budget(res.budget).savefig(out / "mass_budget.png", dpi=120)
    print(f"simulate: {res.diagnostics['n_steps']} steps, {res.runtime_s:.1f} s, "
          f"max budget residual {res.diagnostics['max_rel_budget_residual']:.2e}, "
          f"{res.diagnostics['n_negative']} negative cells -> {out}")
    return 0


def cmd_inverse(args: argparse.Namespace) -> int:
    from . import plotting
    from .inverse import InferenceOptions, ResponseCache, fit_sources, reconstruct, run_forward
    from .metrics import field_rmse, sensor_rmse, source_error
    from .observations import ObservationSpec, SensorLayout, observe
    cfg, out = _setup(args.config, args.out, args.overwrite)
    sc, q, solver = _scenario(cfg)
    inv = cfg["inverse"]
    truth_fwd, inf_fwd = _forwards(args, cfg)
    truth = run_forward(truth_fwd, sc, q, solver)
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
        fit = fit_sources(sc, obs, InferenceOptions(method=m, heldout_ids=held, solver=solver, alpha=inv.get("alpha", 0.0)), cache,
                          forward=inf_fwd)
        rec = reconstruct(sc, fit, solver, forward=inf_fwd)
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
    io.write_json(out / "provenance.json", io.provenance(cfg, {"noise_seed": inv["seed"]},
                                                  {"backends": {"truth": forward_info(truth_fwd).to_dict(),
                                                                "inference": forward_info(inf_fwd).to_dict()}}))
    lin = report["linear"]
    print(f"inverse: q_hat={np.round(lin['q_hat'], 0).tolist()} true={q.tolist()} rank={lin['rank']} "
          f"cond={lin['condition']:.3g} regional_rmse={lin['regional_rmse']:.3g} -> {out}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .studies.toy.experiments import run_alpha_selection, run_benchmark, run_mismatch
    cfg = load_config(args.config)
    out = Path(args.out)
    truth_fwd, inf_fwd = _forwards(args, cfg)
    kind = cfg.get("experiment", "benchmark")
    if kind == "transboundary":
        from .studies.toy.transboundary import run_transboundary
        run_transboundary(cfg, out, overwrite=args.overwrite, progress=print)
    elif kind == "geodesign":
        from .studies.belauri.geodesign import run_geodesign
        run_geodesign(cfg, out, overwrite=args.overwrite, progress=print)
    elif kind == "geodesign_robust":
        from .studies.belauri.geodesign_robust import run_geodesign_robust
        run_geodesign_robust(cfg, out, overwrite=args.overwrite, progress=print)
    elif kind == "belauri_forward":
        from .studies.belauri.belauri import run_belauri_forward
        run_belauri_forward(cfg, out, overwrite=args.overwrite, progress=print)
    elif kind == "alpha_selection":
        sel = run_alpha_selection(cfg, out, overwrite=args.overwrite, progress=print, truth_forward=truth_fwd,
                  inference_forward=inf_fwd)
        print(f"selected alpha = {sel.alpha:g} (criterion {sel.selected['selected_criterion']:.4g})")
    elif kind == "mismatch":
        run_mismatch(cfg, Path(cfg["mismatch"]["layouts_from"]), out, overwrite=args.overwrite, progress=print, truth_forward=truth_fwd,
                  inference_forward=inf_fwd)
    else:
        run_benchmark(cfg, out, overwrite=args.overwrite, progress=print, truth_forward=truth_fwd,
                  inference_forward=inf_fwd)
    print(f"{kind} written to {out}")
    return 0


def cmd_geo(args: argparse.Namespace) -> int:
    """geo fetch (bounded network, cached) / geo prepare (offline). GIS libraries are imported only here."""
    cfg = load_config(args.config)
    if args.geo_cmd == "fetch":
        from .geo.fetch import fetch_all
        recs = fetch_all(cfg, args.cache_dir)
        print(json.dumps({k: {f: v[f] for f in ("url", "id", "retrieved_utc", "bytes", "sha256", "license")} for k, v in recs.items()},
                         indent=2))
        return 0
    from .geo.prepare import prepare_region
    man = prepare_region(cfg, args.cache_dir, args.out)
    print(f"geo prepare: boundary_status={man['boundary_status']}, {len(man['processed'])} arrays, "
          f"{len(man['incomplete'])} incomplete item(s) -> {Path(args.out).parent / 'manifest.json'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pravah", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, helptext in (("simulate", cmd_simulate, "run the forward model"),
                               ("inverse", cmd_inverse, "synthetic source-recovery experiment"),
                               ("benchmark", cmd_benchmark, "sensor-placement benchmark (or alpha selection / mismatch, per config)")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--config", required=True, help="JSON config (see configs/)")
        p.add_argument("--out", required=True, help="output directory (refused if non-empty)")
        p.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty --out")
        p.add_argument("--solver", metavar="SPEC", help="forward solver for both roles: 'fipy' (default) or "
                       "'module:callable' (factory called once with --solver-kwargs)")
        p.add_argument("--truth-solver", metavar="SPEC", help="solver for the synthetic truth (overrides --solver)")
        p.add_argument("--inference-solver", metavar="SPEC", help="solver for responses/fits (overrides --solver)")
        p.add_argument("--solver-kwargs", metavar="JSON", help="JSON object of factory kwargs for --solver and for "
                       "each role without role-specific kwargs")
        p.add_argument("--truth-solver-kwargs", metavar="JSON", help="JSON kwargs for the truth solver only")
        p.add_argument("--inference-solver-kwargs", metavar="JSON", help="JSON kwargs for the inference solver only")
        if name == "inverse":
            p.add_argument("--direct", action="store_true", help="also run the direct least_squares route and compare")
        p.set_defaults(fn=fn)
    geo = sub.add_parser("geo", help="Belauri geography: fetch (network, cached) and prepare (offline); needs pravah[geo]")
    gsub = geo.add_subparsers(dest="geo_cmd", required=True)
    for gname, ghelp in (("fetch", "download boundaries, OSM and DEM into the cache (bounded, polite, no credentials)"),
                         ("prepare", "offline: cached raw files -> processed arrays + manifest.json")):
        gp = gsub.add_parser(gname, help=ghelp)
        gp.add_argument("--config", required=True, help="geo JSON config, e.g. configs/belauri_geo.json")
        gp.add_argument("--cache-dir", default="data/belauri/raw", help="raw download cache (git-ignored)")
        if gname == "prepare":
            gp.add_argument("--out", default="data/belauri/processed", help="processed arrays dir; manifest.json goes in its parent")
        gp.set_defaults(fn=cmd_geo)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")  # CLI is headless; library code never sets the backend
    return int(args.fn(args))

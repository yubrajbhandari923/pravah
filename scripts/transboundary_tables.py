"""Regenerate the production-results section of docs/transboundary.md from a transboundary results directory.

    python scripts/transboundary_tables.py [--results results/transboundary] [--doc docs/transboundary.md]

Offline and deterministic: reads only the CSV/JSON files of the run (no solver, no randomness) and replaces the text
between the markers  <!-- PRODUCTION_RESULTS_BEGIN -->  and  <!-- PRODUCTION_RESULTS_END -->  in the doc.  Files that a
run does not have (e.g. scale_sensitivity.csv from an older run) give a one-line "not available" note instead.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BEGIN, END = "<!-- PRODUCTION_RESULTS_BEGIN -->", "<!-- PRODUCTION_RESULTS_END -->"
ROOT = Path(__file__).resolve().parents[1]


def g(x: Any, nd: int = 3) -> str:
    """Compact number (3 significant digits); '-' for missing."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "-"
    return f"{float(x):.{nd}g}"


def pct(f: float) -> str:
    return f"{100 * float(f):g}%"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def read(res: Path, rel: str) -> pd.DataFrame | None:
    p = res / rel
    return pd.read_csv(p) if p.is_file() else None


def flags_of(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)


def has_flag(s: pd.Series, name: str) -> pd.Series:
    return flags_of(s).str.contains(name, regex=False)


# ------------------------------------------------------------------ sections
def header(res: Path) -> str:
    sel = json.loads((res / "recovery/alpha_selected.json").read_text())
    tim = json.loads((res / "timings.json").read_text()) if (res / "timings.json").is_file() else {}
    cfg = json.loads((res / "config.json").read_text())
    prov = json.loads((res / "provenance.json").read_text()) if (res / "provenance.json").is_file() else {}
    r = cfg["recovery"]
    bits = [f"`{res.name}`", f"{tim['total_s'] / 60:.1f} min wall" if "total_s" in tim else "wall time not recorded",
            f"validation {r['n_validation']} + test {r['n_test']} cases, {r['noise_reps']} noise reps",
            f"alpha selected on VALIDATION cases only: {sel['alpha_by_noise_fraction']}"]
    git = prov.get("git_revision")
    if git:
        bits.append(f"code revision {git[:10]}, dirty = {prov.get('git_dirty')}, diff hash {prov.get('git_diff_hash')}")
    return ("Results directory " + "; ".join(bits) + ". sd is a standard deviation across rows, not a confidence interval. "
            "Cases x repetitions are not independent samples.")


def recovery(res: Path) -> str:
    s = read(res, "recovery/summary.csv")
    assert s is not None
    sel = json.loads((res / "recovery/alpha_selected.json").read_text())["alpha_by_noise_fraction"]
    inflow = sorted(c[:-5] for c in s.columns if c.startswith("inflow_abs_err_ug_m3_") and c.endswith("_mean"))
    s = s[s["alpha_role"] == "selected"]
    hdr = ["layout", "noise (% of signal scale)", "selected alpha", "n rows", "total RMSE", "local RMSE", "external RMSE",
           "abs F_ext error", "local amp. rel. err", "external amp. rel. err"] + [f"{c[len('inflow_abs_err_ug_m3_'):]} inflow abs. err (ug/m3)" for c in inflow]
    rows = [[r.layout, pct(r.noise_fraction), g(sel[f"{r.noise_fraction:g}"]), str(r.n_rows), g(r.rmse_total_mean),
             g(r.rmse_local_mean), g(r.rmse_external_mean), g(r.f_ext_error_mean), g(r.emission_rel_err_local_mean),
             g(r.emission_rel_err_external_mean)] + [g(getattr(r, f"{c}_mean")) for c in inflow]
            for r in s.sort_values(["layout", "noise_fraction"]).itertuples()]
    return "**Recovery on the test ensemble (selected alpha; means over cases x reps; RMSE in ug/m3)**\n\n" + table(hdr, rows)


def prior(res: Path) -> str:
    s = read(res, "recovery/summary.csv")
    assert s is not None
    sel = json.loads((res / "recovery/alpha_selected.json").read_text())["alpha_by_noise_fraction"]
    cfg = json.loads((res / "config.json").read_text())
    rows = []
    for r in s.sort_values(["layout", "noise_fraction", "alpha_role"]).itertuples():
        a = {"none": 0.0, "strong": cfg["recovery"]["alpha_strong"], "selected": sel[f"{r.noise_fraction:g}"]}[r.alpha_role]
        rows.append([r.layout, pct(r.noise_fraction), f"{r.alpha_role} (alpha={g(a)})", g(r.rmse_total_mean),
                     g(r.emission_rel_err_local_mean), g(r.emission_rel_err_external_mean)])
    return ("**Prior sensitivity (test ensemble; alpha acts in scaled units, see 'Scales' in the text)**\n\n"
            + table(["layout", "noise", "prior", "total RMSE", "local amp. rel. err", "external amp. rel. err"], rows))


PAIR_METRICS = [("total RMSE", "rmse_total"), ("local RMSE", "rmse_local"), ("external RMSE", "rmse_external"),
                ("abs F_ext error", "f_ext_error"), ("inflow amplitude abs. error (ug/m3)", "inflow_abs_err_ug_m3_uniform")]


def equal_budget(res: Path) -> str:
    raw = read(res, "recovery/test_raw.csv")
    assert raw is not None
    cfg = json.loads((res / "config.json").read_text())
    a_name, b_name = "interior8", "interior+upwind"
    raw = raw[raw["alpha_role"] == "selected"].copy()
    raw["f_ext_error"] = pd.to_numeric(raw["f_ext_error"], errors="coerce").abs()
    keys = ["case", "rep", "noise_fraction"]
    a, b = raw[raw["layout"] == a_name].set_index(keys), raw[raw["layout"] == b_name].set_index(keys)
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    n_sens = f"{int(a['n_sensors'].iloc[0])} vs {int(b['n_sensors'].iloc[0])} sensors"
    rows = []
    for f, jf in j.groupby("noise_fraction"):
        for label, m in PAIR_METRICS:
            if f"{m}_a" not in jf.columns:
                continue
            d = (jf[f"{m}_b"] - jf[f"{m}_a"]).astype(float)  # interior+upwind minus interior8
            per_case = d.groupby(level="case").mean()
            n_cases = len(per_case)
            rows.append([pct(f), label, g(d.mean()), g(d.std(ddof=1)), f"{int((d > 0).sum())}/{len(d)}",
                         f"{int((per_case > 0).sum())}/{n_cases}"])
    ns = int(cfg["recovery"]["noise_reps"])
    return (f"**Equal-budget comparison ({n_sens}; {a_name} versus {b_name}; selected alpha)**\n\n"
            f"Each noise level has {len(d)} pairs = {n_cases} cases x {ns} noise reps: the pairs are not independent, so the "
            "per-case count (cases averaged over reps) is the more honest tally.  Difference = "
            f"{b_name} minus {a_name} (positive = the {a_name} layout is lower on that metric).  The noise is not literally "
            "shared between the 6- and 8-sensor layouts (the normals are drawn per sensor and reading) nor across the two noise "
            "levels (the same standard normals are reused, scaled by sigma).\n\n"
            + table(["noise", f"metric ({b_name} minus {a_name})", "mean diff", "sd of diff",
                     f"pairs where {a_name} is lower", f"cases where {a_name} is lower"], rows))


def scale_sensitivity(res: Path) -> str:
    ss = read(res, "recovery/scale_sensitivity.csv")
    title = "**Scale sensitivity (interior8, test ensemble, alpha as selected at the baseline scales, not reselected)**\n\n"
    if ss is None:
        return title + "`recovery/scale_sensitivity.csv` is not available in this results directory."
    order = {"baseline": 0, "inflow_x0.1": 1, "inflow_x10": 2, "emission_x0.1": 3, "emission_x10": 4}
    rows = []
    for (f, v), gp in sorted(ss.groupby(["noise_fraction", "variant"]), key=lambda kv: (kv[0][0], order.get(kv[0][1], 9))):
        rows.append([pct(f), v, g(gp["alpha"].iloc[0]), g(gp["rmse_total"].mean()), g(gp["emission_rel_err_local"].mean()),
                     g(gp["emission_rel_err_external"].mean()), g(gp["inflow_abs_err_ug_m3_uniform"].mean()),
                     g(gp["effective_rank"].mean()), f"{int(has_flag(gp['flags'], 'weakly_constrained').sum())}/{len(gp)}"])
    return (title + "Multipliers apply to the default per-kind scales (emission 1e5 ug/s, inflow 0.05 ug/m3).  With alpha > 0 "
            "the scale is the prior width, and the emission scale 1e5 equals the true L1 amplitude.  effective_rank "
            "(and so the weakly_constrained flag) depends on the scales by construction.\n\n"
            + table(["noise", "variant", "alpha", "total RMSE", "local amp. rel. err", "external amp. rel. err",
                     "inflow abs. err (ug/m3)", "mean effective rank", "rows weakly_constrained"], rows))


def equivalence(res: Path) -> str:
    eq = read(res, "domain_size/boundary_basis_equivalence.csv")
    title = "**Boundary input versus a localised plume (truth: E1 explicit in domain B; fitted models in domain A omit E1)**\n\n"
    if eq is None:
        return title + "not available."
    cfg = json.loads((res / "config.json").read_text())
    tl1, tl2 = cfg["sources"]["L1"]["q"], cfg["sources"]["L2"]["q"]
    has_onset = "basis_onset_s" in eq.columns
    rows = []
    eq = eq.assign(_on=eq["basis_onset_s"].fillna(-1.0) if has_onset else -1.0)
    for (m, on), gp in eq.groupby(["model", "_on"], sort=False):
        fl = sorted({x for s in flags_of(gp["flags"]) for x in s.split(";") if x}) if "flags" in gp else []
        if m == "A_E1_explicit_true_theta":
            rows.append([m, "-", g(gp["rmse_total_rel_to_peak"].mean()), "-", "-", "-"])
            continue
        e1 = "-" if not has_onset or on < 0 else (
            "matches E1 onset" if bool(gp["onset_matches_e1"].iloc[0]) else "differs from E1 onset")
        rows.append([m, f"{g(on, 5)} s ({e1})" if on >= 0 else ("no basis" if has_onset else "not recorded (older run)"), g(gp["rmse_total_rel_to_peak"].mean()),
                     f"{g(gp['theta_hat_L1'].mean())} / {g(gp['theta_hat_L2'].mean())}", ", ".join(fl) or "none",
                     f"{g(gp['effective_rank'].mean())}"])
    return (title + f"Truth amplitudes L1 = {g(tl1)}, L2 = {g(tl2)} ug/s.  RMSE / peak is the total-field RMSE over the study "
            "region relative to the peak of the true total.  The basis temporal profile is a known input: the two onsets measure "
            "the sensitivity to it.\n\n"
            + table(["model", "basis onset", "RMSE / peak", "mean L1 / L2 estimate (ug/s)", "flags (union over reps)",
                     "mean effective rank"], rows))


def domain_size(res: Path) -> str:
    d = read(res, "domain_size/field_difference.csv")
    if d is None:
        return "**Outer-domain size**\n\nnot available."
    rows = [[r.scenario, g(r.rmse_total_rel_to_peak), g(r.external_fraction_A), g(r.external_fraction_B), str(r.note)]
            for r in d.itertuples()]
    return ("**Outer-domain size (open boundary 5 km (A) vs 10 km (B) from the study region)**\n\n"
            + table(["scenario", "A vs B RMSE / peak", "external fraction A", "external fraction B", "note"], rows))


def special(res: Path) -> str:
    out = ["**Special cases**"]
    diag = json.loads((res / "special/diagnostics.json").read_text())
    conf = read(res, "special/confounded.csv")
    c = diag["confounded"]
    by = c.get("by_noise_fraction") or {"(last noise level only, older run)": c}
    rows, sent = [], []
    for key, d in by.items():
        f = float(key) if key.replace(".", "", 1).isdigit() else float(conf["noise_fraction"].max())
        gp = conf[(conf["noise_fraction"] == f) & (conf["alpha_role"] == "none")] if conf is not None else None
        ident = d["identifiability_alpha0"]
        names = d["param_names"]
        cos = d["column_cosines"][names.index("L1")][names.index("E1c")]
        sent.append(f"at {pct(f)} noise the flags are [{', '.join(ident['flags']) or 'none'}] (effective rank "
                    f"{ident['effective_rank']}) and the mean amplitude errors are {g(gp['emission_rel_err_local'].mean())} (local) "
                    f"and {g(gp['emission_rel_err_external'].mean())} (external)")
        rows.append([pct(f), g(abs(cos), 4), ", ".join(ident["flags"]) or "none", str(ident["effective_rank"]),
                     g(ident["condition"]), g(gp["rmse_total_rel_to_peak"].mean()), g(gp["emission_rel_err_local"].mean()),
                     g(gp["emission_rel_err_external"].mean()), g(gp["f_ext_error"].abs().mean())])
    out.append("Confounded (alpha = 0; diagnostics of the first noise draw of each noise level; amplitude errors are means over "
               "the repetitions):\n\n" + table(["noise", "abs cos(L1, E1c)", "flags", "effective rank", "condition", "total RMSE / peak",
                                                "local amp. rel. err", "external amp. rel. err", "abs F_ext error"], rows)
               + "\n\nIn words: " + "; ".join(sent) + ".  The confounded flag alone does not mean the split failed: compare the amplitude errors across noise levels.  "
               "abs cos is close to the 0.99 threshold at the production geometry (5 m/s, 3.8 km upwind, dedicated layout); "
               "this is a constructed example, and the flag is pairwise.")
    na = read(res, "special/no_arrival.csv")
    if na is not None:
        r = diag["no_arrival"]["external_to_local_study_mean_ratio"]
        rows = []
        for role in ("none", "selected", "strong"):
            gp = na[na["alpha_role"] == role]
            if len(gp):
                rows.append([f"{role} (alpha={g(gp['alpha'].iloc[0])})", g(gp["theta_hat_E1"].mean()), g(gp["rmse_total_rel_to_peak"].mean()),
                             g(gp["f_ext_true"].mean()), g(gp["f_ext_est"].mean()), g(gp["condition"].mean()),
                             ", ".join(sorted({x for s in flags_of(gp["flags"]) for x in s.split(";") if x}))])
        out.append(f"No arrival: external-to-local study-mean ratio {g(r)} (verified by the forward model; threshold "
                   f"{g(diag['no_arrival']['max_allowed_ratio'])}).  The E1 column is SMALL, not numerically zero; at alpha = 0 the "
                   "estimate is determined by noise amplified through a near-null direction.  True E1 = "
                   f"{g(na['theta_true_E1'].mean())} ug/s (mean).\n\n"
                   + table(["prior", "mean E1 estimate (ug/s)", "total RMSE / peak", "F_ext true", "F_ext est", "condition", "flags"], rows))
    wm = read(res, "special/wind_mismatch.csv")
    if wm is not None:
        rows = []
        for (case, role), gp in wm.groupby(["case", "alpha_role"]):
            rows.append([case, f"{role} (alpha={g(gp['alpha'].iloc[0])})", g(gp["rmse_total_rel_to_peak"].mean()), g(gp["f_ext_true"].mean()),
                         g(gp["f_ext_est"].mean()), g(gp["emission_rel_err_local"].mean()),
                         ", ".join(sorted({x for s in flags_of(gp["flags"]) for x in s.split(";") if x})) or "none"])
        out.append("Wind mismatch (truth wind rotated / scaled, inference declares the nominal wind):\n\n"
                   + table(["case", "prior", "total RMSE / peak", "F_ext true", "F_ext est", "local amp. rel. err", "flags (union)"], rows))
    om = read(res, "special/omitted_source.csv")
    if om is not None:
        rows = []
        for (case, role), gp in om.groupby(["case", "alpha_role"]):
            rows.append([case, role, g(gp["rmse_total_rel_to_peak"].mean()), g(gp["f_ext_true"].mean()), g(gp["f_ext_est"].mean())])
        out.append("Omitted external source (the model lacks the source that is in the truth):\n\n"
                   + table(["case", "prior", "total RMSE / peak", "F_ext true", "F_ext est"], rows))
    return "\n\n".join(out)


def build(res: Path) -> str:
    parts = [header(res), recovery(res), prior(res), equal_budget(res), scale_sensitivity(res), equivalence(res),
             domain_size(res), special(res)]
    return "\n\n".join(parts)


def replace_between(doc: Path, text: str) -> None:
    src = doc.read_text()
    pat = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if len(pat.findall(src)) != 1:
        raise SystemExit(f"{doc}: need exactly one {BEGIN} ... {END} pair")
    doc.write_text(pat.sub(lambda _: f"{BEGIN}\n{text}\n{END}", src))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "results/transboundary"))
    ap.add_argument("--doc", default=str(ROOT / "docs/transboundary.md"))
    ap.add_argument("--print", action="store_true", help="print the markdown instead of editing the doc")
    a = ap.parse_args()
    text = build(Path(a.results))
    if a.print:
        print(text)
    else:
        replace_between(Path(a.doc), text)


if __name__ == "__main__":
    main()

"""Render e02-checks.csv and e02-splits.csv into markdown fragments included by chapters/02-benchmark.qmd."""
import csv, pathlib
here = pathlib.Path(__file__).parent
checks = list(csv.DictReader((here / "e02-checks.csv").open()))
splits = list(csv.DictReader((here / "e02-splits.csv").open()))
def md(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    return "\n".join(out + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]) + "\n"
gates = [r for r in checks if r["passed"] != "info"]
(here / "e02-splits.md").write_text(md(splits, ["scenario", "truth_id", "schema_version", "config_hash", "git_sha", "truth_seed", "noise_seed", "split", "days", "n_hours", "wind_regimes", "fitting_allowed", "evaluation_allowed"]))
(here / "e02-gates.md").write_text(f"{sum(r['passed']=='True' for r in gates)} of {len(gates)} gated checks passed.\n\n" + md(gates, ["scenario", "check", "value", "passed"]))
(here / "e02-info.md").write_text(md([r for r in checks if r["passed"] == "info"], ["scenario", "check", "value"]))

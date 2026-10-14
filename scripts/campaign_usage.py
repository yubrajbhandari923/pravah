"""Cumulative CPU-hour ledger of the cluster campaign from `scripts/hpc_sacct.sh` output (billed = reservation CPUs x ElapsedRaw / 3600).

Usage: campaign_usage.py LOG [LOG ...] [--job ID:CPUS ...] [--out results/campaign_usage.csv] [--budget 512] [--selfcheck]
LOG lines are the parsable2 rows of hpc_sacct.sh (JobID|JobName|State|Elapsed|ElapsedRaw|AllocCPUS|TotalCPU|CPUTimeRAW|...); step rows are ignored.
Known jobs without a sacct row keep empty hours (unknown, not zero)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

KNOWN = {201966: 4, 201967: 16, 201971: 4, 201972: 16, 201973: 16, 201974: 16, 201975: 32, 201976: 32, 201977: 16,
         201981: 32, 201982: 16, 201983: 16, 201987: 1}  # Slurm id: reserved CPUs


def secs(t: str) -> float:
    """[D-][HH:]MM:SS[.fff] -> seconds."""
    d, _, rest = t.rpartition("-")
    p = [float(x) for x in rest.split(":")]
    while len(p) < 3:
        p.insert(0, 0.0)
    return (int(d) * 86400 if d else 0) + p[0] * 3600 + p[1] * 60 + p[2]


def parse(lines: list[str]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for ln in lines:
        c = ln.strip().split("|")
        if len(c) < 7 or not re.fullmatch(r"\d+", c[0].strip()):  # header, separators, steps (123.batch) and arrays are skipped
            continue
        out[int(c[0])] = {"name": c[1], "state": c[2], "elapsed_s": float(c[4]) if c[4] else secs(c[3]),
                          "alloc_cpus": int(c[5]), "total_cpu_s": secs(c[6]) if c[6] else float("nan")}
    return out


def ledger(known: dict[int, int], acct: dict[int, dict], budget: float) -> pd.DataFrame:
    rows = []
    for j in sorted(set(known) | set(acct)):
        a = acct.get(j)
        cpus = known.get(j, a["alloc_cpus"] if a else None)
        if a and cpus != a["alloc_cpus"]:
            print(f"warning: job {j}: table says {cpus} CPUs, sacct AllocCPUS {a['alloc_cpus']}; using sacct")
            cpus = a["alloc_cpus"]
        rows.append({"job": j, "name": a["name"] if a else "", "state": a["state"] if a else "", "cpus": cpus,
                     "elapsed_s": a["elapsed_s"] if a else None,
                     "billed_cpu_h": cpus * a["elapsed_s"] / 3600 if a else None,
                     "actual_cpu_h": a["total_cpu_s"] / 3600 if a else None})
    df = pd.DataFrame(rows)
    df["cum_billed_cpu_h"] = df["billed_cpu_h"].fillna(0).cumsum()
    df["cum_actual_cpu_h"] = df["actual_cpu_h"].fillna(0).cumsum()
    df["remaining_of_budget_cpu_h"] = budget - df["cum_billed_cpu_h"]
    return df


def selfcheck() -> None:
    log = ["JobID|JobName|State|Elapsed|ElapsedRaw|AllocCPUS|TotalCPU|CPUTimeRAW|MaxRSS|ReqMem",
           "201975|exp027|COMPLETED|00:50:00|3000|32|11:00:00|96000||64G", "201975.batch|batch|COMPLETED|00:50:00|3000|32|11:00:00|96000|1G|"]
    d = ledger(KNOWN, parse(log), 512)
    r = d[d.job == 201975].iloc[0]
    assert abs(r.billed_cpu_h - 32 * 3000 / 3600) < 1e-9 and abs(r.actual_cpu_h - 11) < 1e-9 and secs("1-02:03:04.5") == 93784.5
    assert pd.isna(d[d.job == 201966].iloc[0].billed_cpu_h) and len(d) == len(KNOWN)
    print("selfcheck ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--job", action="append", default=[], help="extra ID:CPUS")
    ap.add_argument("--out", default="results/campaign_usage.csv")
    ap.add_argument("--budget", type=float, default=512.0)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    known = KNOWN | {int(k): int(v) for k, v in (x.split(":") for x in a.job)}
    df = ledger(known, parse([ln for f in a.logs for ln in Path(f).read_text().splitlines()]), a.budget)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))
    print(f"billed {df.billed_cpu_h.sum():.2f} CPU-h, actual {df.actual_cpu_h.sum():.2f} CPU-h, budget {a.budget:g}; "
          f"{int(df.billed_cpu_h.isna().sum())} jobs without sacct data")


if __name__ == "__main__":
    main()

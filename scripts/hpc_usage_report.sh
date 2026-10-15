#!/usr/bin/env bash
# Sum billed and used CPU-hours of Research OS jobs (Slurm job names starting with EXP-) from sacct, per job and in total.
# Billed = AllocCPUS x ElapsedRaw / 3600 (what Slurm charges: the reservation); used = TotalCPU. Optionally prints the budget left.
# Run it on the cluster login node (or as a 1-CPU job), never from a workstation loop:   bash scripts/hpc_usage_report.sh [--since YYYY-MM-DD] [--budget H] [--prefix EXP-]
# Offline (parses a saved sacct dump with the format printed by --print-format, or runs --selftest):  bash scripts/hpc_usage_report.sh --input FILE | --selftest
set -euo pipefail
SINCE="2026-10-01"; BUDGET=""; PREFIX="EXP-"; INPUT=""
FMT="JobID,JobName%40,State,ElapsedRaw,AllocCPUS,TotalCPU"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --since) SINCE="$2"; shift 2;;
    --budget) BUDGET="$2"; shift 2;;
    --prefix) PREFIX="$2"; shift 2;;
    --input) INPUT="$2"; shift 2;;
    --print-format) echo "sacct -X --parsable2 --noheader --starttime SINCE --format=$FMT"; exit 0;;
    --selftest) INPUT="$(mktemp)"; trap 'rm -f "$INPUT"' EXIT
      printf '%s\n' '101|EXP-027-run|COMPLETED|2640|32|11:30:00' '102|EXP-033-x|TIMEOUT|3600|4|3:59:00' '103|bash|COMPLETED|100|1|00:01:00' '104|EXP-036|RUNNING|7200|8|00:00:00' > "$INPUT"
      OUT=$(bash "$0" --input "$INPUT" --budget 88); echo "$OUT"
      echo "$OUT" | grep -q "TOTAL billed_cpu_h 43.467" || { echo "selftest FAILED" >&2; exit 1; }
      echo "selftest ok"; exit 0;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
if [ -n "$INPUT" ]; then SRC=$(cat "$INPUT"); else
  SRC=$(sacct -X --parsable2 --noheader --starttime "$SINCE" --format="$FMT"); fi
printf '%s\n' "$SRC" | awk -F'|' -v pre="$PREFIX" -v budget="$BUDGET" '
function secs(t,  d,r,p,n,s) { d=0; r=t; if (index(t,"-")) { split(t,a,"-"); d=a[1]; r=a[2] }
  n=split(r,p,":"); s=0; for(i=1;i<=n;i++) s=s*60+p[i]; return d*86400+s }
index($2,pre)==1 { b=$5*$4/3600; u=secs($6)/3600; tb+=b; tu+=u; n++
  printf "%-10s %-40s %-10s cpus=%-3d billed=%8.2f used=%8.2f\n",$1,$2,$3,$5,b,u }
END { printf "TOTAL jobs %d\nTOTAL billed_cpu_h %.3f\nTOTAL used_cpu_h %.3f\n",n,tb,tu
  if (budget!="") printf "BUDGET %.1f remaining_billed %.3f\n",budget,budget-tb }'

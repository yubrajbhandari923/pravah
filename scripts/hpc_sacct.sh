#!/usr/bin/env bash
# Print Slurm accounting for a list of job ids (run as a 1-CPU, 5-minute cluster job; feeds scripts/campaign_usage.py).
# Usage: hpc_sacct.sh JOBID [JOBID ...]
set -euo pipefail
[ "$#" -ge 1 ] || { echo "usage: $0 JOBID [JOBID ...]" >&2; exit 2; }
IDS=$(IFS=,; echo "$*")
sacct -j "$IDS" -X --parsable2 --format=JobID%14,JobName,State,Elapsed,ElapsedRaw,AllocCPUS,TotalCPU,CPUTimeRAW,MaxRSS,ReqMem
echo "--- steps (MaxRSS) ---"
sacct -j "$IDS" --parsable2 --format=JobID%14,JobName,State,Elapsed,AllocCPUS,TotalCPU,CPUTimeRAW,MaxRSS,ReqMem

#!/bin/bash
# Generic CPU job. Usage (from the snapshot dir on the cluster; PRAVAH_HPC_ROOT from settings.local.yaml):
#   sbatch -J pv-<tag> -c <cpus> --mem=<mem> -t <time> --export=ALL,PRAVAH_HPC_ROOT=$R -o $R/logs/%x-%j.out scripts/hpc/run.sbatch <command...>
# Runs <command> inside the project uv env from this code snapshot; logs to $W/logs, run dir $W/runs/<jobid>.
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:30:00
set -euo pipefail
W=${PRAVAH_HPC_ROOT:?export PRAVAH_HPC_ROOT (see settings.local.yaml)}
export UV_CACHE_DIR=$W/cache/uv UV_PYTHON_INSTALL_DIR=$W/cache/python UV_PROJECT_ENVIRONMENT=$W/envs/main
export XDG_CACHE_HOME=$W/cache/xdg MPLCONFIGDIR=$W/cache/matplotlib PYTHONPYCACHEPREFIX=$W/cache/pycache
export TMPDIR=/tmp/pv-$SLURM_JOB_ID; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT  # node-local: multiprocessing sockets/semaphores misbehave on NFS
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # parallelism comes from worker processes, one per CPU
export RESEARCH_RUN_DIR=$W/runs/$SLURM_JOB_ID; mkdir -p "$RESEARCH_RUN_DIR"
echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK snapshot=$(basename "$PWD") start=$(date -Is)"
"$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import sys; print(sys.version)'
"$@"
echo "end=$(date -Is)"

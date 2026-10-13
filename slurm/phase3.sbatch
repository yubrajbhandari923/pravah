#!/bin/bash
# DEPRECATED (kept for a cluster without SSH access): the supported route is `research hpc submit` (docs/hpc-runbook.md, D-005).
# ONE job for all heavy Phase 3 work: phases run in sequence inside scripts/hpc_run.py, parallelism is Python processes.
# The code is CPU-only: GPUs are not used. Fill the <PLACEHOLDERS>; submit with:  sbatch slurm/phase3.sbatch
# (CPUS=128 sbatch --cpus-per-task=128 ... to use a whole node; the directive below is overridden by the command line.)
#SBATCH --job-name=<JOBNAME>
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=<MEM, e.g. 128G (about 0.5 GB per worker is plenty)>
# code is CPU-only: do NOT request GPUs.   (#SBATCH --gres=gpu:0)
#SBATCH --output=pravah-phase3-%j.out

set -euo pipefail

# --- site-specific: edit ---
# module load python/3.12
REPO="${REPO:-$HOME/pravah}"
source "$REPO/.venv/bin/activate"     # created by: bash slurm/setup_env.sh

export PRAVAH_WORKERS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

cd "$REPO"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} workers $PRAVAH_WORKERS"
python scripts/hpc_run.py --config configs/hpc_phase3.json

echo "Done. If the job ended early (time limit, failure): just resubmit, finished phases are skipped."
echo "Copy back: results/phase3-hpc-bundle.tar.gz and results/hpc_manifest.json (see docs/hpc-runbook.md)."

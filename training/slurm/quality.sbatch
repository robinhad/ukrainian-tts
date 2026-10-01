#!/usr/bin/env bash
#SBATCH --job-name=uktts-quality
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output=training/quality_runs/slurm-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PATH="$PWD/training/.venv/bin:$PATH"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN_DIR=${RUN_DIR:-training/quality_runs/v10}
exec python -m training.quality.supervise --output "$RUN_DIR/jobs/${SLURM_JOB_ID}" \
  --interval 5 -- "$@"

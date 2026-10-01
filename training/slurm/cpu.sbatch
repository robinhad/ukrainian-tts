#!/usr/bin/env bash
#SBATCH --job-name=uktts-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --output=training/quality_runs/slurm-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
PYTHON=${CPU_PYTHON:-"$PWD/training/.venv/bin/python"}
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
RUN_DIR=${RUN_DIR:-training/quality_runs/v10}
exec "$PYTHON" -m training.quality.supervise --output "$RUN_DIR/jobs/${SLURM_JOB_ID}" \
  --interval 60 -- "$@"

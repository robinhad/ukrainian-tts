#!/usr/bin/env bash
#SBATCH --job-name=uktts-v10-100k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=108G
#SBATCH --time=14-00:00:00
#SBATCH --output=training/quality_runs/slurm-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PATH="$PWD/training/.venv/bin:$PATH"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
exec python -m training.quality.supervise --output training/quality_runs/v10/training \
  --interval 60 -- bash training/scripts/run_quality_v10_training.sh

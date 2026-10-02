#!/usr/bin/env bash
#SBATCH --job-name=uktts-v12-process
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=108G
#SBATCH --time=2-00:00:00
#SBATCH --output=training/quality_runs/slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export PATH="$PWD/training/.venv/bin:$PWD/training/quality_runs/mfa_trim/tools/env/bin:$PATH"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -m training.quality.supervise --output "${PROCESS_MONITOR_DIR:-training/quality_runs/v12/processing}" \
  --interval 60 --minimum-available-gib 24 -- \
  python -m training.scripts.process_quality_v12 "$@"

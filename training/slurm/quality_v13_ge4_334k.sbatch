#!/usr/bin/env bash
# Submit after the successful 284K training and evaluation job.
#SBATCH --job-name=uktts-v13-ge4-to334k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=108G
#SBATCH --time=14-00:00:00
#SBATCH --output=training/quality_runs/slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export PATH="$PWD/training/.venv/bin:$PWD/training/quality_runs/mfa_trim/tools/env/bin:$PATH"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
exec python -m training.quality.supervise --output training/quality_runs/v13/preparation \
  --interval 60 --minimum-available-gib 24 -- \
  bash training/scripts/run_quality_v13_training.sh

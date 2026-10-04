#!/usr/bin/env bash
# Continue the completed 120K run for 100K additional optimizer updates.
#SBATCH --job-name=uktts-v12-to220k
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
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CONTINUE_FROM_STEPS=120000 CONTINUE_TO_STEPS=220000
export CONTINUE_MILESTONES="200k 220k"
exec python -m training.quality.supervise --output training/quality_runs/v12/continuation_220k/preparation \
  --interval 60 --minimum-available-gib 24 -- \
  bash training/scripts/continue_quality_v12_training.sh

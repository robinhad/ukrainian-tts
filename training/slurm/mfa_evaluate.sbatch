#!/usr/bin/env bash
#SBATCH --job-name=uktts-mfa-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=training/quality_runs/slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export PATH="$PWD/training/.venv/bin:$PWD/training/quality_runs/mfa_trim/tools/env/bin:$PATH"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -m pytest -q training/tests/test_mfa_trim.py
python -m training.scripts.render_mfa_trim
python -m training.quality evaluate \
  --panel training/quality_runs/v10/panels/processing.jsonl \
  --wav-dir training/quality_runs/mfa_trim/wav \
  --output training/quality_runs/mfa_trim/quality --label mfa_trim --resume

#!/usr/bin/env bash
#SBATCH --job-name=uktts-normalized
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=training/quality_runs/slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PATH="$PWD/training/.venv/bin:$PATH"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q training/tests/test_quality_processing.py training/tests/test_normalized_selector.py
python -m training.scripts.evaluate_normalized_original
python -m training.scripts.select_normalized_best \
  --normalized training/quality_runs/normalized_original/per_file.jsonl \
  --processed training/quality_runs/v10/search/clearervoice_sidon_deepfilternet3_wet1/quality/per_file.jsonl \
  --output training/quality_runs/normalized_original/selection --threshold 3.5 --copy-audio
python -m training.scripts.report_normalized_original
python training/scripts/plot_processing_sigmos.py \
  --input training/reports/quality_v10_processing.csv \
  --selection training/reports/quality_v10_best_of_original_and_cascade_selection.csv \
  --normalized-selection training/reports/quality_normalized_original_selection.csv \
  --output-prefix training/reports/quality_v10_processing_median_sigmos

#!/usr/bin/env bash
# Recover evaluation after a completed 100K run without repeating training.
#SBATCH --job-name=uktts-v12-eval100k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=108G
#SBATCH --time=1-00:00:00
#SBATCH --output=training/quality_runs/slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source training/activate.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TRAIN_SET=quality_v12_train VALID_SET=quality_v12_dev TEST_SETS=quality_v12_eval
export MANIFEST_DIR="$PWD/training/data/quality_v12/manifests"
export DUMP_DIR="$PWD/training/dump_quality_v12" EXP_DIR="$PWD/training/exp_quality_v12"
export TTS_EXP="$EXP_DIR/tts_jets_quality_v12_50k"
export QUALITY_ROOT="$PWD/training/quality_runs/v12/checkpoints"
export MILESTONES="75k 100k" UKTTS_QUALITY_PREVIOUS="$QUALITY_ROOT/50k_quality"
python -m training.scripts.audit_finetune_checkpoint \
  --checkpoint "$TTS_EXP/checkpoint.pth" --expected-steps 100000 \
  --expected-scheduler-epoch 100 --model-artifact "$TTS_EXP/milestones/100k.pth" \
  --output training/quality_runs/v12/checkpoint_100000.json
exec python -m training.quality.supervise \
  --output training/quality_runs/v12/evaluation_100k_recovery --interval 60 \
  --minimum-available-gib 24 -- bash training/scripts/evaluate_quality_v10_checkpoints.sh

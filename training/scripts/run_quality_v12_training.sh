#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}/.."
: "${SLURM_JOB_ID:?Use a SLURM GPU allocation}"
source "${ROOT}/activate.sh"
export QUALITY_RUN_DIR="${ROOT}/quality_runs/v12"
mkdir -p "$QUALITY_RUN_DIR"
bash "${ROOT}/scripts/prepare_quality_v12.sh"
export TRAIN_SET=quality_v12_train VALID_SET=quality_v12_dev TEST_SETS=quality_v12_eval
export MANIFEST_DIR="${ROOT}/data/quality_v12/manifests"
export DUMP_DIR="${ROOT}/dump_quality_v12" EXP_DIR="${ROOT}/exp_quality_v12"
export TTS_EXP="${EXP_DIR}/tts_jets_quality_v12_50k"
export QUALITY_ROOT="${QUALITY_RUN_DIR}/checkpoints"
export MILESTONES="25k 50k" STEPS=50000
export WANDB_METRICS=true WANDB_PROJECT=ukrainian-tts
export WANDB_NAME=quality-v12-mfa-best-ge3.5-50k
export UKTTS_CUDA_CACHE_INTERVAL=10
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# Inputs and prepared artifacts were verified above. Only this experiment's
# own checkpoint may resume; otherwise the run starts from scratch.
if [[ -e "$TTS_EXP/checkpoint.pth" ]]; then export RESUME=true; fi
unset INIT_CHECKPOINT
exec python -m training.quality.supervise --output "$QUALITY_RUN_DIR/training" \
  --interval 60 --minimum-available-gib 24 --activity-log "$TTS_EXP/train.log" \
  -- bash "${ROOT}/scripts/run_quality_v10_training.sh"

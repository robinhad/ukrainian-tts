#!/usr/bin/env bash
# Resume the complete 50K state through 100K total updates, including LR decay.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}/.."
: "${SLURM_JOB_ID:?Use a SLURM GPU allocation}"
source "${ROOT}/activate.sh"
bash "${ROOT}/scripts/prepare_quality_v12.sh"
export QUALITY_RUN_DIR="${ROOT}/quality_runs/v12"
export TRAIN_SET=quality_v12_train VALID_SET=quality_v12_dev TEST_SETS=quality_v12_eval
export MANIFEST_DIR="${ROOT}/data/quality_v12/manifests"
export DUMP_DIR="${ROOT}/dump_quality_v12" EXP_DIR="${ROOT}/exp_quality_v12"
# Keep the original experiment, TensorBoard history and W&B run identity.
export TTS_EXP="${EXP_DIR}/tts_jets_quality_v12_50k"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -m training.scripts.audit_finetune_checkpoint \
  --checkpoint "$TTS_EXP/checkpoint.pth" --expected-steps 50000 \
  --expected-scheduler-epoch 50 --model-artifact "$TTS_EXP/milestones/50k.pth" \
  --output "$QUALITY_RUN_DIR/continuation_50k_audit.json"
# Copy, rather than hard-link: ESPnet overwrites checkpoint.pth when saving.
mkdir -p "$QUALITY_RUN_DIR/continuation_100k"
cp --reflink=auto "$TTS_EXP/checkpoint.pth" "$QUALITY_RUN_DIR/continuation_100k/checkpoint_50k.pth"
cp "$TTS_EXP/config.yaml" "$QUALITY_RUN_DIR/continuation_100k/config_50k.yaml"
export QUALITY_ROOT="${QUALITY_RUN_DIR}/checkpoints"
export UKTTS_QUALITY_PREVIOUS="$QUALITY_ROOT/50k_quality"
export MILESTONES="75k 100k" STEPS=100000 RESUME=true
export WANDB_METRICS=true WANDB_PROJECT=ukrainian-tts
export WANDB_NAME=quality-v12-mfa-best-ge3.5-100k
export UKTTS_CUDA_CACHE_INTERVAL=10
unset INIT_CHECKPOINT
exec python -m training.quality.supervise --output "$QUALITY_RUN_DIR/continuation_100k/training" \
  --interval 60 --minimum-available-gib 24 --activity-log "$TTS_EXP/train.log" \
  -- bash "${ROOT}/scripts/run_quality_v10_training.sh"

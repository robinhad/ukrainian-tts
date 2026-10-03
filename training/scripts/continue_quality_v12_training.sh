#!/usr/bin/env bash
# Resume complete training state through a higher total, including LR decay.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}/.."
: "${SLURM_JOB_ID:?Use a SLURM GPU allocation}"
FROM_STEPS=${CONTINUE_FROM_STEPS:-50000}
TO_STEPS=${CONTINUE_TO_STEPS:-100000}
if [[ ! "$FROM_STEPS" =~ ^[1-9][0-9]*$ || ! "$TO_STEPS" =~ ^[1-9][0-9]*$ ]] || \
   (( FROM_STEPS % 1000 != 0 || TO_STEPS % 1000 != 0 || TO_STEPS <= FROM_STEPS )); then
  echo 'Continuation bounds must increase and be positive multiples of 1000.' >&2
  exit 2
fi
FROM_LABEL="$((FROM_STEPS / 1000))k"
TO_LABEL="$((TO_STEPS / 1000))k"
DEFAULT_MILESTONES="$TO_LABEL"
if (( FROM_STEPS == 50000 && TO_STEPS == 100000 )); then DEFAULT_MILESTONES="75k 100k"; fi
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
  --checkpoint "$TTS_EXP/checkpoint.pth" --expected-steps "$FROM_STEPS" \
  --expected-scheduler-epoch "$((FROM_STEPS / 1000))" --model-artifact "$TTS_EXP/milestones/$FROM_LABEL.pth" \
  --output "$QUALITY_RUN_DIR/continuation_${FROM_LABEL}_audit.json"
# Copy, rather than hard-link: ESPnet overwrites checkpoint.pth when saving.
mkdir -p "$QUALITY_RUN_DIR/continuation_$TO_LABEL"
cp --reflink=auto "$TTS_EXP/checkpoint.pth" "$QUALITY_RUN_DIR/continuation_$TO_LABEL/checkpoint_$FROM_LABEL.pth"
cp "$TTS_EXP/config.yaml" "$QUALITY_RUN_DIR/continuation_$TO_LABEL/config_$FROM_LABEL.yaml"
export QUALITY_ROOT="${QUALITY_RUN_DIR}/checkpoints"
export UKTTS_QUALITY_PREVIOUS="$QUALITY_ROOT/${FROM_LABEL}_quality"
export MILESTONES="${CONTINUE_MILESTONES:-$DEFAULT_MILESTONES}" STEPS="$TO_STEPS" RESUME=true
export ITERS_PER_EPOCH=1000
export WANDB_METRICS=true WANDB_PROJECT=ukrainian-tts
export WANDB_NAME="quality-v12-mfa-best-ge3.5-$TO_LABEL"
export UKTTS_CUDA_CACHE_INTERVAL=10
unset INIT_CHECKPOINT
exec python -m training.quality.supervise --output "$QUALITY_RUN_DIR/continuation_$TO_LABEL/training" \
  --interval 60 --minimum-available-gib 24 --activity-log "$TTS_EXP/train.log" \
  -- bash "${ROOT}/scripts/run_quality_v10_training.sh"

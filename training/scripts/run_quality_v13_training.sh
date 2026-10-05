#!/usr/bin/env bash
# Parse the full workflow before execution so later source edits cannot shift reads.
main() {
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}/.."
: "${SLURM_JOB_ID:?Use a SLURM GPU allocation}"
source "${ROOT}/activate.sh"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
SOURCE_EXP="$ROOT/exp_quality_v12/tts_jets_quality_v12_50k"
export QUALITY_RUN_DIR="$ROOT/quality_runs/v13"
export TRAIN_SET=quality_v12_train VALID_SET=quality_v12_dev TEST_SETS=quality_v12_eval
export MANIFEST_DIR="$ROOT/data/quality_v13/manifests"
export DUMP_DIR="$ROOT/dump_quality_v13" EXP_DIR="$ROOT/exp_quality_v13"
export TTS_EXP="$EXP_DIR/tts_jets_quality_v13_ge4_334k"
# Revalidate both the parent corpus and all prepared features before reusing them.
python -m training.scripts.audit_quality_v12_training --stage inputs
python -m training.scripts.audit_quality_v12_training --stage prepared
python -m training.scripts.audit_finetune_checkpoint \
  --checkpoint "$SOURCE_EXP/checkpoint.pth" --expected-steps 284000 \
  --expected-scheduler-epoch 284 --model-artifact "$SOURCE_EXP/milestones/284k.pth" \
  --output "$QUALITY_RUN_DIR/source_284k_audit.json"
python -m training.scripts.prepare_quality_v13
mkdir -p "$TTS_EXP/milestones"
if [[ ! -e "$TTS_EXP/checkpoint.pth" ]]; then
  # Copy the full optimizer/reporter/scheduler state; never hard-link a mutable checkpoint.
  cp --reflink=auto "$SOURCE_EXP/checkpoint.pth" "$TTS_EXP/checkpoint.pth.part"
  mv "$TTS_EXP/checkpoint.pth.part" "$TTS_EXP/checkpoint.pth"
  cp "$SOURCE_EXP/config.yaml" "$QUALITY_RUN_DIR/source_284k_config.yaml"
  cp --reflink=auto "$SOURCE_EXP/milestones/284k.pth" "$TTS_EXP/milestones/284k.pth"
fi
# Verify every resume, including retries after this stage has already advanced.
read -r CURRENT_STEPS CURRENT_EPOCH < <(python - "$TTS_EXP/checkpoint.pth" <<'PY'
import sys, torch
state = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
epoch = state['reporter']['epoch']
steps = state['reporter']['stats'][epoch]['train']['total_count']
if not 284000 <= steps <= 334000 or steps != epoch * 1000:
    raise ValueError('Unexpected filtered-stage checkpoint position')
print(int(steps), epoch)
PY
)
python -m training.scripts.audit_finetune_checkpoint \
  --checkpoint "$TTS_EXP/checkpoint.pth" --expected-steps "$CURRENT_STEPS" \
  --expected-scheduler-epoch "$CURRENT_EPOCH" --output "$QUALITY_RUN_DIR/resume_audit.json"
export QUALITY_ROOT="$QUALITY_RUN_DIR/checkpoints"
export UKTTS_QUALITY_PREVIOUS="$ROOT/quality_runs/v12/checkpoints/284k_quality"
export MILESTONES=334k STEPS=334000 RESUME=true ITERS_PER_EPOCH=1000
export WANDB_METRICS=true WANDB_PROJECT=ukrainian-tts WANDB_NAME=quality-v13-mfa-ge4-284k-to334k
export UKTTS_CUDA_CACHE_INTERVAL=10
unset INIT_CHECKPOINT
python -m training.quality.supervise --output "$QUALITY_RUN_DIR/training" \
  --interval 60 --minimum-available-gib 24 --activity-log "$TTS_EXP/train.log" \
  -- bash "$ROOT/scripts/run_quality_v10_training.sh"
python -m training.scripts.audit_finetune_checkpoint \
  --checkpoint "$TTS_EXP/checkpoint.pth" --expected-steps 334000 \
  --expected-scheduler-epoch 334 --model-artifact "$TTS_EXP/milestones/334k.pth" \
  --output "$QUALITY_RUN_DIR/final_334k_audit.json"
exit 0
}
main "$@"

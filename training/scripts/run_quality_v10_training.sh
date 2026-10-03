#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}/.."
: "${SLURM_JOB_ID:?Run training inside a SLURM GPU allocation}"
source "${ROOT}/activate.sh"
if [[ -n "${CALIBRATION_FILE:-}" ]]; then
  calibration_settings=$(python - "$CALIBRATION_FILE" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    values = json.load(stream)
bins, workers, tf32 = (values[k] for k in ('batch_bins', 'workers', 'use_tf32'))
cache_interval = values.get('cuda_cache_interval', 0)
if (type(bins) is not int or bins < 1 or type(workers) is not int or workers < 1
        or type(tf32) is not bool or type(cache_interval) is not int or cache_interval < 0):
    raise ValueError('Invalid measured training configuration')
print(bins, workers, str(tf32).lower(), cache_interval)
PY
  )
  read -r BATCH_BINS WORKERS USE_TF32 UKTTS_CUDA_CACHE_INTERVAL <<< "$calibration_settings"
fi
export UKTTS_CUDA_CACHE_INTERVAL=${UKTTS_CUDA_CACHE_INTERVAL:-10}
export UKTTS_ESPEAK_DATA_HASH_FILE="${ROOT}/quality_runs/v10/frontend/ESPEAK_NG_DATA_HASH"
export GPU_COUNT=1
export TRAIN_SET=${TRAIN_SET:-quality_v10_train} VALID_SET=${VALID_SET:-quality_v10_dev} TEST_SETS=${TEST_SETS:-quality_v10_eval}
export DATA_SETS="$TRAIN_SET $VALID_SET $TEST_SETS"
export MANIFEST_DIR=${MANIFEST_DIR:-"${ROOT}/data/quality_v10/manifests"}
export DUMP_DIR=${DUMP_DIR:-"${ROOT}/dump_quality_v10"} EXP_DIR=${EXP_DIR:-"${ROOT}/exp_quality_v10"}
TTS_EXP=${TTS_EXP:-"${EXP_DIR}/tts_jets_quality_v10_100k"}
STEPS=${STEPS:-100000}
BATCH_BINS=${BATCH_BINS:-4000000}
WORKERS=${WORKERS:-8}
USE_TF32=${USE_TF32:-false}
ITERS_PER_EPOCH=${ITERS_PER_EPOCH:-1000}
if (( STEPS < 1 || ITERS_PER_EPOCH < 1 || STEPS % ITERS_PER_EPOCH != 0 )); then
  echo 'STEPS must be a positive multiple of ITERS_PER_EPOCH.' >&2
  exit 2
fi
if [[ -e "$TTS_EXP/checkpoint.pth" && "${RESUME:-false}" != true ]]; then
  echo 'Existing experiment: explicitly set RESUME=true to resume this same run.' >&2
  exit 2
fi
mkdir -p "$TTS_EXP"
git rev-parse HEAD > "$TTS_EXP/training_git_commit.txt"
printf '%s\n' "$STEPS" > "$TTS_EXP/target_iterations.txt"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
# Keep CPU math limits consistent between the calibration and training launchers.
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
# Scratch training is the default. An initializer must be explicitly supplied
# and have matching tokens/architecture; optimizer and scheduler start fresh.
INIT_ARGS=""
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  [[ -f "$INIT_CHECKPOINT" ]] || exit 2
  INIT_ARGS="--init_param ${INIT_CHECKPOINT}:::normalize,pitch_normalize,energy_normalize --ignore_init_mismatch false"
fi
METRICS_RUNNER=()
if [[ "$TRAIN_SET" == quality_v11_train ]]; then
  WANDB_METRICS=${WANDB_METRICS:-true}
fi
if [[ "${WANDB_METRICS:-false}" == true ]]; then
  METRICS_RUNNER=(python -m training.quality.wandb_metrics
    --events "$TTS_EXP/tensorboard" --state "$TTS_EXP/wandb_metrics/state.json"
    --project "${WANDB_PROJECT:-ukrainian-tts}" --name "${WANDB_NAME:-quality-v11-50k}" --)
fi
"${METRICS_RUNNER[@]}" "${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" \
  --stage 7 --stop_stage 7 --tts_exp "$TTS_EXP" \
  --train_args "--max_epoch $((STEPS / ITERS_PER_EPOCH)) --num_iters_per_epoch ${ITERS_PER_EPOCH} --batch_bins ${BATCH_BINS} --num_workers ${WORKERS} --accum_grad 1 --use_amp false --use_tf32 ${USE_TF32} --cudnn_benchmark false --keep_nbest_models 3 --num_att_plot 0 --use_tensorboard true --use_wandb false --wandb_model_log_interval -1 --resume ${RESUME:-false} ${INIT_ARGS}" &
TRAIN_PID=$!
WATCH_PID=""
if (( ITERS_PER_EPOCH == 1000 )); then
  bash "${ROOT}/scripts/preserve_expanded_v9_milestones.sh" "$TRAIN_PID" "$STEPS" "$TTS_EXP" &
  WATCH_PID=$!
fi
trap 'kill "$TRAIN_PID" ${WATCH_PID:+"$WATCH_PID"} 2>/dev/null || true' TERM INT EXIT
set +e
wait "$TRAIN_PID"
STATUS=$?
set -e
if [[ -n "$WATCH_PID" ]]; then wait "$WATCH_PID"; fi
trap - TERM INT EXIT
(( STATUS == 0 )) || exit "$STATUS"
if (( ITERS_PER_EPOCH == 1000 )); then
  python -m training.scripts.audit_finetune_checkpoint --checkpoint "$TTS_EXP/checkpoint.pth" \
    --expected-steps "$STEPS" --model-artifact "$TTS_EXP/$((STEPS / ITERS_PER_EPOCH))epoch.pth" \
    --output "${QUALITY_RUN_DIR:-${ROOT}/quality_runs/v10}/checkpoint_${STEPS}.json"
  if [[ "${RUN_CHECKPOINT_EVAL:-true}" == true ]] && \
     { (( STEPS == 50000 || STEPS == 100000 )) || [[ -n "${MILESTONES:-}" ]]; }; then
    if (( STEPS == 50000 )); then export MILESTONES=${MILESTONES:-"25k 50k"}; fi
    TTS_EXP="$TTS_EXP" bash "${ROOT}/scripts/evaluate_quality_v10_checkpoints.sh"
  fi
fi

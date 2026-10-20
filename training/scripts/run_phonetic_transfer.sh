#!/usr/bin/env bash
# Full-model phonetic transfer, with a fresh optimizer and numeric-only W&B.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a SLURM GPU allocation}"
# sbatch executes a spooled copy, so BASH_SOURCE is not the repository path.
ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}/training"
cd "$ROOT/.."
source "$ROOT/activate.sh"
TRANSFER_ROOT=${TRANSFER_ROOT:-training/quality_runs/phonetic_transfer}
TRANSFER_BUNDLE=${TRANSFER_BUNDLE:-$TRANSFER_ROOT/converted}
STEPS=${STEPS:-20000}
BATCH_SIZE=${BATCH_SIZE:-8}
SPEAKER=${SPEAKER:-}
RUN_TAG=${RUN_TAG:-${SPEAKER:+${SPEAKER}_}${STEPS}}
TTS_EXP=${TTS_EXP:-training/exp/phonetic_transfer_${RUN_TAG}}
export UKTTS_ESPEAK_DATA_HASH_FILE="$TRANSFER_ROOT/frontend/ESPEAK_NG_DATA_HASH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export UKTTS_CUDA_CACHE_INTERVAL=10
for split in train dev; do
    if [[ ! -f "$TRANSFER_ROOT/$split/metadata.json" ]]; then
        python -m training.transfer.prepare --bundle "$TRANSFER_BUNDLE" \
            --pairs "training/data/quality_v12/manifests/quality_v12_$split.parquet" \
            --output "$TRANSFER_ROOT/$split"
    fi
done
DATA_ROOT="$TRANSFER_ROOT"
VOICE_ARGS=()
BACKEND_ARGS=()
[[ ${FLASH_ATTENTION:-0} == 1 ]] && BACKEND_ARGS+=(--flash-attention)
[[ ${RESUME:-0} == 1 ]] && BACKEND_ARGS+=(--resume)
if [[ -n "$SPEAKER" ]]; then
    : "${SOURCE_ID:?A single-voice run requires a source-qualified speaker}"
    DATA_ROOT="$TRANSFER_ROOT/voices/$SPEAKER"
    if [[ ! -f "$DATA_ROOT/selection.json" ]]; then
        python -m training.transfer.select_voice \
            --manifests training/data/quality_v12/manifests \
            --prepared "$TRANSFER_ROOT" --output "$DATA_ROOT" \
            --speaker "$SPEAKER" --source "$SOURCE_ID"
    fi
    VOICE_ARGS=(--speaker "$SPEAKER")
fi
python -m training.quality.supervise \
    --output "$TRANSFER_ROOT/run_${RUN_TAG}" --interval 60 --minimum-available-gib 24 \
    -- python -m training.quality.wandb_metrics \
    --events "$TTS_EXP/tensorboard" --state "$TRANSFER_ROOT/wandb_${RUN_TAG}/state.json" \
    --project ukrainian-tts --name "jets-english-ukrainian-phonetic-${RUN_TAG}" \
    -- python -m training.transfer.train --bundle "$TRANSFER_BUNDLE" \
    --train "$DATA_ROOT/train" --valid "$DATA_ROOT/dev" \
    --output "$TTS_EXP" --steps "$STEPS" --batch-size "$BATCH_SIZE" \
    --external-metrics "${VOICE_ARGS[@]}" "${BACKEND_ARGS[@]}" &
SUPERVISOR_PID=$!
python -m training.transfer.monitor --pid "$SUPERVISOR_PID" \
    --log "$TRANSFER_ROOT/run_${RUN_TAG}/command.log" --steps "$STEPS" \
    --output "$TRANSFER_ROOT/run_${RUN_TAG}/progress.jsonl" &
MONITOR_PID=$!
trap 'kill "$SUPERVISOR_PID" "$MONITOR_PID" 2>/dev/null || true' TERM INT EXIT
STATUS=0
wait "$SUPERVISOR_PID" || STATUS=$?
wait "$MONITOR_PID" || true
trap - TERM INT EXIT
exit "$STATUS"

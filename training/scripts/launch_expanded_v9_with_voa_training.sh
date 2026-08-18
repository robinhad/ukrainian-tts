#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=expanded_v9_cascade_with_voa
TARGET=${1:-500000}
TARGET_LABEL="$((TARGET / 1000))k"
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_${NAME}/tts_jets_uk_24k_${NAME}_scratch_500k"}
LOG="${ROOT}/reports/${NAME}_${TARGET_LABEL}_launcher.log"
STATUS="${ROOT}/reports/training_status_${NAME}_${TARGET_LABEL}.jsonl"
PYTHON="${ROOT}/.venv/bin/python"

setsid env TTS_EXP="$TTS_EXP" ALLOW_TRAINING_RESUME="${ALLOW_TRAINING_RESUME:-0}" \
    "${ROOT}/scripts/run_expanded_v9_with_voa_training.sh" "$TARGET" \
    >"$LOG" 2>&1 &
train_pid=$!
STATUS="$STATUS" INTERVAL_SECONDS=900 FREE_DISK_STOP_GIB=30 \
    STOP_ON_CRITICAL=0 TRAIN_PGID="$train_pid" \
    "${ROOT}/scripts/monitor_expanded_v3_training.sh" \
    "$train_pid" "$TARGET" "$TTS_EXP" &
monitor=$!
"${ROOT}/scripts/preserve_expanded_v9_milestones.sh" \
    "$train_pid" "$TARGET" "$TTS_EXP" &
milestones=$!
"$PYTHON" "${ROOT}/scripts/monitor_expanded_v5_disk.py" \
    --watch-pid "$train_pid" --workspace "$ROOT" \
    --status "${ROOT}/reports/${NAME}_disk.jsonl" \
    --trigger-gib 30 --interval-seconds 60 --stop-pgid "$train_pid" --report-only &
disk=$!
(
    while kill -0 "$train_pid" 2>/dev/null && [[ ! -s "${TTS_EXP}/train.log" ]]; do
        sleep 5
    done
    if [[ -s "${TTS_EXP}/train.log" ]]; then
        "$PYTHON" "${ROOT}/scripts/preserve_validation_best_checkpoints.py" \
            --log "${TTS_EXP}/train.log" --exp-dir "$TTS_EXP" --keep 3 \
            --target-epoch "$((TARGET / 1000))" --poll-seconds 30
    fi
) &
best=$!
set +e
wait "$train_pid"; train_status=$?
wait "$monitor"; monitor_status=$?
wait "$milestones"; milestone_status=$?
wait "$disk"; disk_status=$?
wait "$best"; best_status=$?
set -e
if (( train_status || monitor_status || milestone_status || disk_status || best_status )); then
    echo "Training chain failed: ${train_status} ${monitor_status} ${milestone_status} ${disk_status} ${best_status}" >&2
    exit 1
fi
TTS_EXP="$TTS_EXP" STATUS="$STATUS" \
    "${ROOT}/scripts/finalize_expanded_v9_with_voa_500k.sh" "$TARGET"
echo "The v9 scratch ${TARGET_LABEL} training and evaluation chain is complete."

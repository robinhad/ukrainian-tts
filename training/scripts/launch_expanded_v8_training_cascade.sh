#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa
TARGET=${1:-100000}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_${NAME}/tts_jets_uk_24k_${NAME}_from_v7e93_100k"}
LOG="${ROOT}/reports/${NAME}_100k_launcher.log"
STATUS="${ROOT}/reports/training_status_${NAME}_100k.jsonl"
PYTHON="${ROOT}/.venv/bin/python"

setsid env TTS_EXP="$TTS_EXP" ALLOW_TRAINING_RESUME=1 \
    "${ROOT}/scripts/run_expanded_v8_training_cascade_training.sh" "$TARGET" \
    >"$LOG" 2>&1 &
train_pid=$!
STATUS="$STATUS" INTERVAL_SECONDS=900 FREE_DISK_STOP_GIB=30 \
    STOP_ON_CRITICAL=0 TRAIN_PGID="$train_pid" \
    "${ROOT}/scripts/monitor_expanded_v3_training.sh" \
    "$train_pid" "$TARGET" "$TTS_EXP" &
monitor=$!
"${ROOT}/scripts/preserve_expanded_v5_milestones.sh" \
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
            --log "${TTS_EXP}/train.log" --exp-dir "$TTS_EXP" --keep 1 \
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
    "${ROOT}/scripts/finalize_expanded_v8_training_cascade_100k.sh" "$TARGET"
echo "The v8 100K training and milestone evaluation chain is complete."

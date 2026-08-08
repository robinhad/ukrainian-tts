#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_ITERATIONS=${1:-100000}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v5_enhanced_novoa/tts_jets_uk_24k_expanded_v5_enhanced_novoa_ft_v4e28_100k"}
LOG="${ROOT}/reports/expanded_v5_enhanced_novoa_100k_launcher.log"
STATUS="${ROOT}/reports/training_status_expanded_v5_enhanced_novoa_100k.jsonl"
PYTHON="${ROOT}/.venv/bin/python"

setsid env TTS_EXP="$TTS_EXP" ALLOW_TRAINING_RESUME=1 \
    "${ROOT}/scripts/run_expanded_v5_enhanced_novoa_training.sh" \
    "$TARGET_ITERATIONS" >"$LOG" 2>&1 &
train_pid=$!
train_pgid=$train_pid
STATUS="$STATUS" INTERVAL_SECONDS=900 FREE_DISK_STOP_GIB=30 \
    STOP_ON_CRITICAL=0 TRAIN_PGID="$train_pgid" \
    "${ROOT}/scripts/monitor_expanded_v3_training.sh" \
    "$train_pid" "$TARGET_ITERATIONS" "$TTS_EXP" &
monitor_pid=$!
"${ROOT}/scripts/preserve_expanded_v5_milestones.sh" \
    "$train_pid" "$TARGET_ITERATIONS" "$TTS_EXP" &
milestone_pid=$!
"$PYTHON" "${ROOT}/scripts/monitor_expanded_v5_disk.py" \
    --watch-pid "$train_pid" --workspace "$ROOT" \
    --status "${ROOT}/reports/expanded_v5_enhanced_novoa_disk.jsonl" \
    --trigger-gib 30 --interval-seconds 60 --stop-pgid "$train_pgid" \
    --report-only &
disk_pid=$!
(
    while kill -0 "$train_pid" 2>/dev/null && [[ ! -s "${TTS_EXP}/train.log" ]]; do
        sleep 5
    done
    if [[ -s "${TTS_EXP}/train.log" ]]; then
        "$PYTHON" "${ROOT}/scripts/preserve_validation_best_checkpoints.py" \
            --log "${TTS_EXP}/train.log" --exp-dir "$TTS_EXP" --keep 1 \
            --target-epoch "$((TARGET_ITERATIONS / 1000))" --poll-seconds 30
    fi
) &
best_pid=$!

set +e
wait "$train_pid"; train_status=$?
wait "$monitor_pid"; monitor_status=$?
wait "$milestone_pid"; milestone_status=$?
wait "$disk_pid"; disk_status=$?
wait "$best_pid"; best_status=$?
set -e
if (( train_status || monitor_status || milestone_status || disk_status || best_status )); then
    echo "The enhanced non-VOA training chain failed." >&2
    echo "train=${train_status} monitor=${monitor_status} milestone=${milestone_status} disk=${disk_status} best=${best_status}" >&2
    exit 1
fi

TTS_EXP="$TTS_EXP" STATUS="$STATUS" \
    "${ROOT}/scripts/finalize_expanded_v5_enhanced_novoa_100k.sh" \
    "$TARGET_ITERATIONS"
echo "The enhanced non-VOA 100K training chain is complete."

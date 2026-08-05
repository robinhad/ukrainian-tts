#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_ITERATIONS=${1:-100000}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v4_trim_only/tts_jets_uk_24k_expanded_v4_trim_only_ft367_100k"}
LOG="${ROOT}/reports/expanded_v4_trim_only_100k_launcher.log"
STATUS="${ROOT}/reports/training_status_expanded_v4_trim_only_100k.jsonl"
MARKER="${ROOT}/reports/expanded_v4_trim_only_cleanup_eligible.json"
PYTHON="${ROOT}/.venv/bin/python"

TTS_EXP="$TTS_EXP" "${ROOT}/scripts/run_expanded_v4_trim_only_training.sh" "$TARGET_ITERATIONS" >"$LOG" 2>&1 &
train_pid=$!
STATUS="$STATUS" INTERVAL_SECONDS=900 \
    "${ROOT}/scripts/monitor_expanded_v3_training.sh" "$train_pid" "$TARGET_ITERATIONS" "$TTS_EXP" &
monitor_pid=$!
"${ROOT}/scripts/preserve_expanded_v3_milestones.sh" "$train_pid" "$TARGET_ITERATIONS" "$TTS_EXP" &
milestone_pid=$!
"$PYTHON" "${ROOT}/scripts/monitor_trim_only_disk.py" \
    --watch-pid "$train_pid" --workspace "$ROOT" --marker "$MARKER" \
    --status "${ROOT}/reports/expanded_v4_trim_only_disk.jsonl" &
disk_pid=$!
(
    while kill -0 "$train_pid" 2>/dev/null && [[ ! -s "${TTS_EXP}/train.log" ]]; do sleep 5; done
    if [[ -s "${TTS_EXP}/train.log" ]]; then
        "$PYTHON" "${ROOT}/scripts/preserve_validation_best_checkpoints.py" \
            --log "${TTS_EXP}/train.log" --exp-dir "$TTS_EXP" \
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
    echo "The trim-only training chain failed." >&2
    echo "train=${train_status} monitor=${monitor_status} milestone=${milestone_status} disk=${disk_status} best=${best_status}" >&2
    exit 1
fi
"$PYTHON" "${ROOT}/scripts/summarize_training_status.py" \
    --input "$STATUS" \
    --output "${ROOT}/reports/expanded_v4_trim_only_training_monitor_summary.json" \
    --target-iterations "$TARGET_ITERATIONS" --maximum-gap-minutes 30
"$PYTHON" "${ROOT}/scripts/summarize_tensorboard.py" \
    --logdir "${TTS_EXP}/tensorboard" \
    --output "${ROOT}/reports/expanded_v4_trim_only_tensorboard_metrics.json"
audit_args=(
    --checkpoint "${TTS_EXP}/checkpoint.pth"
    --expected-steps "$TARGET_ITERATIONS"
    --output "${ROOT}/reports/expanded_v4_trim_only_100k_checkpoint.json"
)
if [[ -s "${TTS_EXP}/milestones/100k.pth" ]]; then
    audit_args+=(--model-artifact "${TTS_EXP}/milestones/100k.pth")
fi
"$PYTHON" "${ROOT}/scripts/audit_finetune_checkpoint.py" "${audit_args[@]}"
for label in 25k 50k 75k 100k; do
    if [[ -s "${TTS_EXP}/milestones/${label}.pth" ]]; then
        TTS_EXP="$TTS_EXP" \
            "${ROOT}/scripts/evaluate_expanded_v4_trim_only_milestone.sh" "$label"
    fi
done
if [[ -s "${TTS_EXP}/milestones/100k.pth" ]]; then
    TTS_EXP="$TTS_EXP" \
        "${ROOT}/scripts/generate_five_voice_expanded_v4_trim_only_eval.sh" 100k
fi
echo "The trim-only 100K training chain is complete."

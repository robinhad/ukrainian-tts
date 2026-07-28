#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_ITERATIONS=${1:-25000}
TARGET_LABEL="$((TARGET_ITERATIONS / 1000))k"
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v3/tts_jets_uk_24k_expanded_v3_${TARGET_LABEL}"}
LOG="${ROOT}/reports/expanded_v3_${TARGET_LABEL}_launcher.log"

TTS_EXP="$TTS_EXP" \
    "${ROOT}/scripts/run_expanded_v3_training.sh" "$TARGET_ITERATIONS" \
    >"$LOG" 2>&1 &
train_pid=$!
"${ROOT}/scripts/monitor_expanded_v3_training.sh" \
    "$train_pid" "$TARGET_ITERATIONS" "$TTS_EXP" &
monitor_pid=$!
"${ROOT}/scripts/preserve_expanded_v3_milestones.sh" \
    "$train_pid" "$TARGET_ITERATIONS" "$TTS_EXP" &
milestone_pid=$!

set +e
wait "$train_pid"
train_status=$?
wait "$monitor_pid"
monitor_status=$?
wait "$milestone_pid"
milestone_status=$?
set -e

if (( train_status != 0 || monitor_status != 0 || milestone_status != 0 )); then
    echo "The expanded-v3 training run failed." >&2
    echo "Training exit: ${train_status}; monitor exit: ${monitor_status}; milestone exit: ${milestone_status}." >&2
    exit 1
fi

if (( TARGET_ITERATIONS == 25000 )); then
    MODEL_FILE=milestones/25k.pth \
        "${ROOT}/scripts/generate_five_voice_expanded_v3_eval.sh"
fi

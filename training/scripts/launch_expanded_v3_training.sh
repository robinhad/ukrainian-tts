#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG="${ROOT}/reports/expanded_v3_25k_launcher.log"

"${ROOT}/scripts/run_expanded_v3_training.sh" 25000 >"$LOG" 2>&1 &
train_pid=$!
"${ROOT}/scripts/monitor_expanded_v3_training.sh" "$train_pid" &
monitor_pid=$!
"${ROOT}/scripts/preserve_expanded_v3_milestones.sh" "$train_pid" &
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

MODEL_FILE=milestones/25k.pth "${ROOT}/scripts/generate_five_voice_expanded_v3_eval.sh"

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_PID=${1:?Usage: monitor_expanded_v3_training.sh TRAIN_PID}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-300}
LOG="${ROOT}/exp_expanded_v3/tts_jets_uk_24k_expanded_v3_25k/train.log"
STATUS="${ROOT}/reports/training_status_expanded_v3.jsonl"
PYTHON="${ROOT}/.venv/bin/python"

if (( INTERVAL_SECONDS < 1 || INTERVAL_SECONDS > 300 )); then
    echo "INTERVAL_SECONDS must be in the range 1 to 300." >&2
    exit 2
fi
probe() {
    set +e
    "$PYTHON" "${ROOT}/scripts/training_status.py" \
        --log "$LOG" --target-iterations 25000 \
        --workspace "$ROOT" --free-disk-stop-gib 60 \
        --maximum-temperature-c 90 --output "$STATUS"
    result=$?
    set -e
    if (( result != 0 )); then
        echo "A critical training condition exists. Stop PID ${TRAIN_PID}." >&2
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        return 1
    fi
}

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    if [[ -s "$LOG" ]]; then
        probe
    else
        TZ=Europe/Kyiv date '+%Y-%m-%dT%H:%M:%S%:z INITIALIZING; ETA not available'
    fi
    sleep "$INTERVAL_SECONDS"
done
if [[ -s "$LOG" ]]; then
    probe
fi

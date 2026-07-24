#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_PID=${1:?Usage: monitor_multispeaker_training.sh TRAIN_PID [TARGET_ITERATIONS] [INTERVAL_SECONDS]}
TARGET_ITERATIONS=${2:-25000}
INTERVAL_SECONDS=${3:-300}
LOG=${TRAIN_LOG:-"$ROOT/exp_multispeaker_full/tts_jets_uk_24k_multispeaker/train.log"}
STATUS_LOG=${STATUS_LOG:-"$ROOT/reports/training_status_multispeaker.jsonl"}
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}

if ! [[ "$TRAIN_PID" =~ ^[0-9]+$ ]]; then
    echo "TRAIN_PID must be a positive integer." >&2
    exit 2
fi
if (( TARGET_ITERATIONS < 1 )); then
    echo "TARGET_ITERATIONS must be positive." >&2
    exit 2
fi
if (( INTERVAL_SECONDS < 1 || INTERVAL_SECONDS > 300 )); then
    echo "INTERVAL_SECONDS must be in the range 1--300." >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Python is not executable: $PYTHON" >&2
    exit 2
fi

record_status() {
    if ! "$PYTHON" "$ROOT/scripts/training_status.py" \
        --log "$LOG" \
        --target-iterations "$TARGET_ITERATIONS" \
        --output "$STATUS_LOG"; then
        echo "The status probe found an error marker. Monitoring continues." >&2
    fi
}

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    record_status
    sleep "$INTERVAL_SECONDS"
done

record_status

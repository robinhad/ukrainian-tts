#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_PID=${1:?Usage: monitor_expanded_v3_training.sh TRAIN_PID TARGET_ITERATIONS TTS_EXP}
TARGET_ITERATIONS=${2:-25000}
TARGET_LABEL="$((TARGET_ITERATIONS / 1000))k"
TTS_EXP=${3:-"${ROOT}/exp_expanded_v3/tts_jets_uk_24k_expanded_v3_${TARGET_LABEL}"}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-900}
LOG="${TTS_EXP}/train.log"
STATUS=${STATUS:-"${ROOT}/reports/training_status_expanded_v3_${TARGET_LABEL}.jsonl"}
PYTHON="${ROOT}/.venv/bin/python"

if (( INTERVAL_SECONDS < 1 || INTERVAL_SECONDS > 900 )); then
    echo "INTERVAL_SECONDS must be in the range 1 to 900." >&2
    exit 2
fi
probe() {
    set +e
    "$PYTHON" "${ROOT}/scripts/training_status.py" \
        --log "$LOG" --target-iterations "$TARGET_ITERATIONS" \
        --workspace "$ROOT" --free-disk-stop-gib "${FREE_DISK_STOP_GIB:-30}" \
        --maximum-temperature-c 90 --output "$STATUS"
    result=$?
    set -e
    if (( result != 0 )); then
        if [[ -n "${TRAIN_PGID:-}" ]]; then
            if ! [[ "$TRAIN_PGID" =~ ^[0-9]+$ ]] || (( TRAIN_PGID <= 1 )); then
                echo "TRAIN_PGID is not safe: ${TRAIN_PGID}" >&2
                return 1
            fi
            echo "A critical training condition exists. Stop process group ${TRAIN_PGID}." >&2
            kill -TERM -- "-${TRAIN_PGID}" 2>/dev/null || true
        else
            echo "A critical training condition exists. Stop PID ${TRAIN_PID}." >&2
            kill -TERM "$TRAIN_PID" 2>/dev/null || true
        fi
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

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT="${ROOT}/data/expanded_v3"
VERSION=v4-c050-d20-g050-defer-long
PSEUDO_ROOT="${DATA_ROOT}/sources/pseudo_uk_${VERSION}"
MARKER_ROOT="${DATA_ROOT}/sources/voa_ukr_user_grant/markers/${VERSION}"
STAGE_FILE="${ROOT}/reports/expanded_v3_500k_stage.txt"
COMMAND_LOG="${ROOT}/reports/expanded_v3_500k_commands.jsonl"
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
export GPU_UUIDS
export ALLOW_USER_AUTHORIZED_UNDER_MINIMUM_HOURS=1
export ALLOW_TRAINING_RESUME=1
export VOA_PIPELINE_VERSION="$VERSION"
export VOA_GPU_IDS=0,1
export INCLUDE_RECOVERED_LONG=${INCLUDE_RECOVERED_LONG:-1}
if [[ "$INCLUDE_RECOVERED_LONG" != 0 && "$INCLUDE_RECOVERED_LONG" != 1 ]]; then
    echo "INCLUDE_RECOVERED_LONG must be 0 or 1." >&2
    exit 2
fi

kyiv_time() {
    TZ=Europe/Kyiv date '+%Y-%m-%d %H:%M:%S %Z'
}

set_stage() {
    printf '%s | %s\n' "$(kyiv_time)" "$1" >"$STAGE_FILE"
    echo "[chain] $(cat "$STAGE_FILE")"
}

monitor_chain() {
    while kill -0 "$$" 2>/dev/null; do
        echo "[chain-status] $(cat "$STAGE_FILE" 2>/dev/null || true)"
        nvidia-smi \
            --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu \
            --format=csv,noheader,nounits
        PYTHONPATH="${ROOT}/.." "${ROOT}/.venv/bin/python" - \
            "$ROOT" "$MARKER_ROOT" <<'PY'
import shutil
import sys
from pathlib import Path
from training.scripts.cleanup_unlabeled_cache import sweep_deferred_batches

workspace = sys.argv[1]
usage = shutil.disk_usage(workspace)
free_gib = usage.free / 1024**3
print(f"free_disk_gib={free_gib:.2f}")
if free_gib <= 30.0:
    cleaned = sweep_deferred_batches(
        Path(sys.argv[2]),
        Path(workspace),
        30.0,
    )
    usage = shutil.disk_usage(workspace)
    print(f"cleanup_batches={len(cleaned)}")
    print(f"free_disk_gib_after_cleanup={usage.free / 1024**3:.2f}")
PY
        sleep 900
    done
}

mkdir -p "${ROOT}/reports"
set_stage "WAIT_FOR_VOA_V4"
monitor_chain &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

while pgrep -f "process_voa_streaming.sh" >/dev/null; do
    sleep 60
done

set_stage "VERIFY_AND_RESUME_VOA_V4"
DATA_ROOT="$DATA_ROOT" \
    "${ROOT}/.venv/bin/python" "${ROOT}/scripts/run_logged.py" \
    --log "$COMMAND_LOG" --eta-minutes 600 -- \
    "${ROOT}/scripts/process_voa_streaming.sh"

marker_count=$(find "$MARKER_ROOT" -maxdepth 1 -name '*.json' -type f | wc -l)
if (( marker_count != 20 )); then
    echo "VOA v4 has ${marker_count} markers. It requires 20." >&2
    exit 1
fi
if [[ ! -s "${PSEUDO_ROOT}/records.jsonl" ]]; then
    echo "The merged VOA pseudo-label manifest does not exist." >&2
    exit 1
fi
if [[ "$INCLUDE_RECOVERED_LONG" == 1 ]]; then
    if [[ ! -s "${PSEUDO_ROOT}/deferred_too_long/records.jsonl" ]]; then
        echo "The deferred-long manifest does not exist." >&2
        exit 1
    fi
    set_stage "RECOVER_DEFERRED_LONG"
    "${ROOT}/.venv/bin/python" "${ROOT}/scripts/run_logged.py" \
        --log "$COMMAND_LOG" --eta-minutes 600 -- \
        "${ROOT}/scripts/process_deferred_long_parallel.sh"
    if [[ ! -s "${PSEUDO_ROOT}/records_with_recovered_long.jsonl" ]]; then
        echo "The recovered VOA manifest does not exist." >&2
        exit 1
    fi
else
    set_stage "SKIP_DEFERRED_LONG"
    echo "[chain] Use the base VOA manifest without recovered long segments."
fi

set_stage "PREPARE_CLEAN_EXPANDED_V3"
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/run_logged.py" \
    --log "$COMMAND_LOG" --eta-minutes 2400 -- \
    "${ROOT}/scripts/prepare_expanded_v3.sh" full

set_stage "TRAIN_JETS_500K"
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/run_logged.py" \
    --log "$COMMAND_LOG" --eta-minutes 6000 -- \
    "${ROOT}/scripts/launch_expanded_v3_training.sh" 500000

set_stage "COMPLETE"

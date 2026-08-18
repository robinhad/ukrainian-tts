#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=expanded_v9_cascade_with_voa
STATUS="${ROOT}/reports/${NAME}_pipeline_status_30m.jsonl"
MONITOR_LOG="${ROOT}/reports/${NAME}_pipeline_monitor.log"

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/monitor_expanded_v9_pipeline.py" \
    --watch-pid "$$" --root "$ROOT" --status "$STATUS" \
    --interval-seconds 1800 >>"$MONITOR_LOG" 2>&1 &
"${ROOT}/scripts/run_expanded_v9_with_voa_preprocessing.sh"
"${ROOT}/scripts/prepare_expanded_v9_with_voa.sh"
"${ROOT}/scripts/run_expanded_v9_with_voa_smoke.sh"
"${ROOT}/scripts/launch_expanded_v9_with_voa_training.sh" 500000

echo "The v9 VOA-inclusive scratch pipeline is complete. Status: ${STATUS}"

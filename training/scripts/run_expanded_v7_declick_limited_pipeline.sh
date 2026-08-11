#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=expanded_v7_sidon_deess_declick_limit_novoa
STATUS="${ROOT}/reports/${NAME}_pipeline_status_30m.jsonl"
MONITOR_LOG="${ROOT}/reports/${NAME}_pipeline_monitor.log"

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/monitor_expanded_v7_pipeline.py" \
    --watch-pid "$$" --root "$ROOT" --status "$STATUS" \
    --interval-seconds 1800 >>"$MONITOR_LOG" 2>&1 &
"${ROOT}/scripts/run_expanded_v7_audio_postprocessing.sh"
"${ROOT}/scripts/prepare_expanded_v7_declick_limited.sh"
"${ROOT}/scripts/run_expanded_v7_declick_limited_smoke.sh"
"${ROOT}/scripts/launch_expanded_v7_declick_limited_training.sh" 100000
echo "The v7 processing and 100K training pipeline is complete."

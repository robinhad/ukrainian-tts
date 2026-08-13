#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa
STATUS="${ROOT}/reports/${NAME}_pipeline_status_30m.jsonl"
MONITOR_LOG="${ROOT}/reports/${NAME}_pipeline_monitor.log"

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/monitor_expanded_v8_pipeline.py" \
    --watch-pid "$$" --root "$ROOT" --status "$STATUS" \
    --interval-seconds 1800 >>"$MONITOR_LOG" 2>&1 &
"${ROOT}/scripts/run_expanded_v8_training_cascade_preprocessing.sh"
"${ROOT}/scripts/prepare_expanded_v8_training_cascade.sh"
"${ROOT}/scripts/run_expanded_v8_training_cascade_smoke.sh"
"${ROOT}/scripts/launch_expanded_v8_training_cascade.sh" 100000
echo "The v8 processing and 100K training pipeline is complete."

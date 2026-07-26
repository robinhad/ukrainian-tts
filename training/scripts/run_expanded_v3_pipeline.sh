#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COMMAND_LOG="${ROOT}/reports/expanded_v3_full_commands.jsonl"

kyiv_time() {
    TZ=Europe/Kyiv date '+%Y-%m-%d %H:%M:%S %Z'
}

echo "[pipeline] $(kyiv_time); full data preparation starts."
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/run_logged.py" \
    --log "$COMMAND_LOG" --eta-minutes 1200 -- \
    "${ROOT}/scripts/prepare_expanded_v3.sh" full

echo "[pipeline] $(kyiv_time); full data preparation is PASS."
echo "[pipeline] $(kyiv_time); dual-GPU JETS training to 25K starts."
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/run_logged.py" \
    --log "$COMMAND_LOG" --eta-minutes 4320 -- \
    "${ROOT}/scripts/launch_expanded_v3_training.sh"

echo "[pipeline] $(kyiv_time); training and listening evaluation are complete."

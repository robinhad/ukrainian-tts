#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT="${ROOT}/data/expanded_v3"
STAGE_FILE="${ROOT}/reports/expanded_v3_500k_stage.txt"
COMMAND_LOG="${ROOT}/reports/expanded_v3_500k_commands.jsonl"
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}

export GPU_UUIDS
export ALLOW_USER_AUTHORIZED_UNDER_MINIMUM_HOURS=1
export ALLOW_TRAINING_RESUME=1
export INCLUDE_RECOVERED_LONG=0
export TRAINING_INCLUDE_RECOVERED_LONG=0
export MANIFEST_DIR="${DATA_ROOT}/manifests"
export TRAIN_SET=expanded_v3_train
export VALID_SET=expanded_v3_dev
export TEST_SETS=expanded_v3_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_expanded_v3"
export EXP_DIR="${ROOT}/exp_expanded_v3"
IFS=, read -r -a GPUS <<<"$GPU_UUIDS"
export GPU_COUNT=${#GPUS[@]}

kyiv_time() {
    TZ=Europe/Kyiv date '+%Y-%m-%d %H:%M:%S %Z'
}

set_stage() {
    printf '%s | %s\n' "$(kyiv_time)" "$1" >"$STAGE_FILE"
    echo "[chain] $(cat "$STAGE_FILE")"
}

verify_filtered_counts() {
    PYTHONPATH="${ROOT}/.." "${ROOT}/.venv/bin/python" - \
        "${DATA_ROOT}/manifests" "${DUMP_DIR}/raw" <<'PY'
import sys
from pathlib import Path

import pandas as pd

manifest_root = Path(sys.argv[1])
dump_root = Path(sys.argv[2])
for split in ("train", "dev", "eval"):
    frame = pd.read_parquet(manifest_root / f"{split}.parquet")
    expected = len(frame)
    text_path = dump_root / f"expanded_v3_{split}" / "text"
    actual = sum(1 for line in text_path.open(encoding="utf-8") if line.strip())
    if actual != expected:
        raise SystemExit(
            f"{split} has {actual} ESPnet records. It requires {expected}."
        )
    print(f"{split}_records={actual}")
PY
}

source "${ROOT}/activate.sh"
mkdir -p "${ROOT}/reports"

set_stage "PREPARE_ESPNET_TOKENS_AND_STATISTICS"
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/run_logged.py" \
    --log "$COMMAND_LOG" --eta-minutes 360 -- \
    "${ROOT}/espnet_recipe/run_expanded_v3.sh" --stage 4 --stop_stage 6

set_stage "VERIFY_NO_LONG_SEGMENTS"
verify_filtered_counts

set_stage "AUDIT_TRAINING_INPUT"
set +e
PYTHONPATH="${ROOT}/.." "${ROOT}/.venv/bin/python" \
    "${ROOT}/scripts/audit_expanded_v3_readiness.py" \
    --registry "${ROOT}/conf/expanded_v3_sources.yaml" \
    --workspace "$ROOT" \
    --data-root "$DATA_ROOT" \
    --reports-root "${ROOT}/reports" \
    --mode full \
    --output "${ROOT}/reports/expanded_v3_full_scale_readiness.json"
readiness_status=$?
set -e
if (( readiness_status != 0 )); then
    echo "The readiness report keeps its audit failure." >&2
    echo "The user authorized training on the available data." >&2
fi

set_stage "TRAIN_JETS_500K"
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/run_logged.py" \
    --log "$COMMAND_LOG" --eta-minutes 6000 -- \
    "${ROOT}/scripts/launch_expanded_v3_training.sh" 500000

set_stage "COMPLETE"

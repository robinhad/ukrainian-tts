#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT="${ROOT}/data/expanded_v3"
STAGE_FILE="${ROOT}/reports/expanded_v3_500k_stage.txt"
COMMAND_LOG="${ROOT}/reports/expanded_v3_500k_commands.jsonl"
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
ESPNET_STATS_NJ=${ESPNET_STATS_NJ:-10}

if (( ESPNET_STATS_NJ < 1 || ESPNET_STATS_NJ > 24 )); then
    echo "ESPNET_STATS_NJ must be from 1 to 24." >&2
    exit 2
fi

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
    frame = pd.read_parquet(manifest_root / f"expanded_v3_{split}.parquet")
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

restore_embedding_indexes() {
    local split
    local output
    local restored
    local expected
    local actual
    local -a shards
    for split in train dev eval; do
        output="${DUMP_DIR}/xvector/expanded_v3_${split}/xvector.scp"
        restored="${output}.restored"
        mapfile -t shards < <(
            find "$(dirname "$output")" -mindepth 2 -maxdepth 2 \
                -path '*/part-*/xvector.scp' -type f -print | sort
        )
        if (( ${#shards[@]} == 0 )); then
            echo "No embedding shards exist for ${split}." >&2
            exit 1
        fi
        LC_ALL=C sort -k1,1 -u "${shards[@]}" >"$restored"
        expected=$(
            "${ROOT}/.venv/bin/python" -c \
                "import pandas as pd; print(len(pd.read_parquet('${DATA_ROOT}/manifests/expanded_v3_${split}.parquet')))"
        )
        actual=$(wc -l <"$restored")
        if (( actual != expected )); then
            echo "${split} has ${actual} restored embeddings. It requires ${expected}." >&2
            exit 1
        fi
        mv "$restored" "$output"
        echo "${split}_restored_embeddings=${actual}"
    done
}

source "${ROOT}/activate.sh"
mkdir -p "${ROOT}/reports"

set_stage "RESTORE_COMPLETE_EMBEDDING_INDEXES"
restore_embedding_indexes

set_stage "PREPARE_ESPNET_TOKENS_AND_STATISTICS"
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/run_logged.py" \
    --log "$COMMAND_LOG" --eta-minutes 360 -- \
    "${ROOT}/espnet_recipe/run_expanded_v3.sh" \
    --stage 4 --stop_stage 6 --nj "$ESPNET_STATS_NJ"

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

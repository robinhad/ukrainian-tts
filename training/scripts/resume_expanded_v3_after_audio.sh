#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
COMMAND_LOG="${ROOT}/reports/expanded_v3_full_commands.jsonl"

export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
IFS=, read -r -a GPUS <<< "$GPU_UUIDS"
export GPU_COUNT=${#GPUS[@]}
export MANIFEST_DIR="${ROOT}/data/expanded_v3/manifests"
export TRAIN_SET=expanded_v3_train
export VALID_SET=expanded_v3_dev
export TEST_SETS=expanded_v3_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_expanded_v3"
export EXP_DIR="${ROOT}/exp_expanded_v3"

source "${ROOT}/activate.sh"
python "${ROOT}/scripts/check_resources.py" \
    --mode full --require-torch --gpu-uuids "$GPU_UUIDS" \
    --workspace "$ROOT" \
    --output "${ROOT}/reports/resource_usage_expanded_v3.jsonl"

python "${ROOT}/scripts/run_logged.py" \
    --log "$COMMAND_LOG" --eta-minutes 5 -- \
    "${ROOT}/espnet_recipe/run_expanded_v3.sh" --stage 1 --stop_stage 2

MANIFEST="${ROOT}/data/expanded_v3/manifests/all.parquet" \
RAW_MANIFEST="${ROOT}/data/expanded_v3/source_manifests/all.parquet" \
OUTPUT_MANIFEST="${ROOT}/data/expanded_v3/hybrid_manifest/all.parquet" \
KALDI_ROOT="${ROOT}/data/expanded_v3/hybrid_embedding_data" \
DUMP_DIR="$DUMP_DIR" \
REPORT="${ROOT}/reports/expanded_v3_full_hybrid_embeddings.json" \
GPU_UUIDS="$GPU_UUIDS" \
    python "${ROOT}/scripts/run_logged.py" \
        --log "$COMMAND_LOG" --eta-minutes 75 -- \
        "${ROOT}/scripts/extract_hybrid_embeddings.sh"

python "${ROOT}/scripts/run_logged.py" \
    --log "$COMMAND_LOG" --eta-minutes 45 -- \
    "${ROOT}/espnet_recipe/run_expanded_v3.sh" --stage 4 --stop_stage 6

python "${ROOT}/scripts/audit_expanded_v3_readiness.py" \
    --registry "${ROOT}/conf/expanded_v3_sources.yaml" \
    --workspace "$ROOT" \
    --data-root "${ROOT}/data/expanded_v3" \
    --reports-root "${ROOT}/reports" \
    --mode full \
    --output "${ROOT}/reports/expanded_v3_full_scale_readiness.json"

python "${ROOT}/scripts/run_logged.py" \
    --log "$COMMAND_LOG" --eta-minutes 480 -- \
    "${ROOT}/scripts/launch_expanded_v3_training.sh"

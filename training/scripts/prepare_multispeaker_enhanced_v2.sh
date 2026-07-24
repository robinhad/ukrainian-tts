#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_MANIFEST=${SOURCE_MANIFEST:-"${ROOT}/data/multispeaker_full/manifests/all.parquet"}
DATA_ROOT=${DATA_ROOT:-"${ROOT}/data/multispeaker_enhanced_v2"}
GPU_UUIDS=${GPU_UUIDS:-"GPU-be591530-39fd-0b1c-50af-8c75548cb6b8,GPU-de1be084-ce05-942c-cb74-78e80652f184"}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-8}
IFS=, read -r -a GPUS <<< "$GPU_UUIDS"
if (( ${#GPUS[@]} != 2 )); then
    echo "This iteration requires exactly two GPU UUIDs." >&2
    exit 2
fi
if (( WORKERS_PER_GPU < 1 )); then
    echo "WORKERS_PER_GPU must be positive." >&2
    exit 2
fi
NUM_SHARDS=$((WORKERS_PER_GPU * ${#GPUS[@]}))

source "${ROOT}/activate.sh"
python "${ROOT}/scripts/check_resources.py" \
    --mode full --require-torch --gpu-uuids "$GPU_UUIDS" \
    --workspace "$ROOT" --output "${ROOT}/reports/resource_usage.jsonl"

mkdir -p "${DATA_ROOT}/records" "${DATA_ROOT}/logs" "${DATA_ROOT}/processed_24k"
pids=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    gpu=${GPUS[$((shard % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 \
        python "${ROOT}/scripts/preprocess_enhanced_audio.py" \
        --manifest "$SOURCE_MANIFEST" \
        --output-root "${DATA_ROOT}/processed_24k" \
        --output-records "${DATA_ROOT}/records/records-${shard}.jsonl" \
        --model-cache "${ROOT}/vendor/deepfilternet-cache" \
        --shard-index "$shard" --num-shards "$NUM_SHARDS" --resume \
        > "${DATA_ROOT}/logs/preprocess-${shard}.log" 2>&1 &
    pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "${DATA_ROOT}/preprocess.pids"

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
if (( failed )); then
    echo "At least one preprocessing shard failed. See ${DATA_ROOT}/logs." >&2
    exit 1
fi

python "${ROOT}/scripts/merge_enhanced_manifests.py" \
    --source-manifest "$SOURCE_MANIFEST" \
    --records-dir "${DATA_ROOT}/records" \
    --output-dir "${DATA_ROOT}/manifests" \
    --report "${ROOT}/reports/multispeaker_enhanced_v2_dataset.json"
python "${ROOT}/scripts/validate_dataset.py" \
    --manifest "${DATA_ROOT}/manifests/all.parquet" \
    --report "${ROOT}/reports/multispeaker_enhanced_v2_validation.json"
python "${ROOT}/scripts/analyze_dataset.py" \
    --manifest "${DATA_ROOT}/manifests/all.parquet" \
    --output "${ROOT}/reports/multispeaker_enhanced_v2_data_analysis.json"

export MANIFEST_DIR="${DATA_ROOT}/manifests"
export TRAIN_SET=multispeaker_train
export VALID_SET=multispeaker_dev
export TEST_SETS=multispeaker_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_multispeaker_enhanced_v2"
export EXP_DIR="${ROOT}/exp_multispeaker_enhanced_v2"
export GPU_COUNT=2

"${ROOT}/espnet_recipe/run_multispeaker_enhanced_v2.sh" \
    --stage 1 --stop_stage 6


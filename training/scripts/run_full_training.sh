#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_ITERATIONS=${1:-1000}
if (( TARGET_ITERATIONS < 1000 || TARGET_ITERATIONS % 1000 != 0 )); then
    echo "target iterations must be a positive multiple of 1000" >&2
    exit 2
fi
TARGET_EPOCHS=$((TARGET_ITERATIONS / 1000))
# Use a measured value for each dataset and batch configuration.
ETA_MINUTES_PER_1000=${ETA_MINUTES_PER_1000:-18}
ETA_MINUTES=$(((TARGET_ITERATIONS * ETA_MINUTES_PER_1000 + 999) / 1000 + 10))
DATASET_NAME=${DATASET_NAME:-full}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-tts_jets_uk_24k_full}
DUMP_NAME=${DUMP_NAME:-dump_full}
EXP_NAME=${EXP_NAME:-exp_full}
BATCH_BINS=${BATCH_BINS:-3000000}
NUM_WORKERS=${NUM_WORKERS:-8}
CUDNN_BENCHMARK=${CUDNN_BENCHMARK:-false}

source "${ROOT}/activate.sh"
GPU_UUIDS=${GPU_UUIDS:-"GPU-be591530-39fd-0b1c-50af-8c75548cb6b8,GPU-de1be084-ce05-942c-cb74-78e80652f184"}
export CUDA_VISIBLE_DEVICES="${GPU_UUIDS}"
export GPU_COUNT=${GPU_COUNT:-2}
python "${ROOT}/scripts/check_resources.py" \
    --mode full --require-torch --gpu-uuids "${GPU_UUIDS}" --workspace "${ROOT}" \
    --output "${ROOT}/reports/resource_usage.jsonl"

export MANIFEST_DIR="${ROOT}/data/${DATASET_NAME}/manifests"
export TRAIN_SET=train
export VALID_SET=dev
export TEST_SETS=eval
export DATA_SETS="train dev eval"
export DUMP_DIR="${ROOT}/${DUMP_NAME}"
export EXP_DIR="${ROOT}/${EXP_NAME}"

python "${ROOT}/scripts/run_logged.py" \
    --log "${ROOT}/reports/commands.jsonl" \
    --eta-minutes "${ETA_MINUTES}" -- \
    "${ROOT}/espnet_recipe/run.sh" \
    --stage 7 --stop_stage 7 \
    --tts_exp "${EXP_DIR}/${EXPERIMENT_NAME}" \
    --train_args "--max_epoch ${TARGET_EPOCHS} --num_iters_per_epoch 1000 --batch_bins ${BATCH_BINS} --num_workers ${NUM_WORKERS} --cudnn_benchmark ${CUDNN_BENCHMARK} --use_amp false"

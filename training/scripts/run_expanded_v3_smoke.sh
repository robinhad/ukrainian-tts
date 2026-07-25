#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
DATA_ROOT="${ROOT}/data/expanded_v3_smoke"
TTS_EXP="${ROOT}/exp_expanded_v3_smoke/tts_jets_uk_24k_expanded_v3_smoke"

source "${ROOT}/activate.sh"
export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
IFS=, read -r -a GPUS <<< "$GPU_UUIDS"
export GPU_COUNT=${#GPUS[@]}
export MANIFEST_DIR="${DATA_ROOT}/manifests"
export TRAIN_SET=expanded_v3_smoke_train
export VALID_SET=expanded_v3_smoke_dev
export TEST_SETS=expanded_v3_smoke_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_expanded_v3_smoke"
export EXP_DIR="${ROOT}/exp_expanded_v3_smoke"

python "${ROOT}/scripts/check_resources.py" \
    --mode smoke --require-torch --gpu-uuids "$GPU_UUIDS" \
    --workspace "$ROOT" \
    --output "${ROOT}/reports/resource_usage_expanded_v3.jsonl"
"${ROOT}/espnet_recipe/run_expanded_v3.sh" \
    --stage 7 --stop_stage 7 --tts_exp "$TTS_EXP" \
    --train_args \
        "--max_epoch 1 --num_iters_per_epoch 100 --batch_bins 1000000 --num_workers 8 --cudnn_benchmark false --use_amp false"
"${ROOT}/espnet_recipe/run_expanded_v3.sh" \
    --stage 8 --stop_stage 8 --tts_exp "$TTS_EXP" \
    --inference_model train.total_count.ave.pth

DECODE_DIR="${TTS_EXP}/decode_jets_train.total_count.ave/${TEST_SETS}"
python "${ROOT}/scripts/synthesize_eval.py" \
    --wav-dir "${DECODE_DIR}/wav" \
    --manifest "${DATA_ROOT}/manifests/${TEST_SETS}.parquet" \
    --checkpoint "${TTS_EXP}/train.total_count.ave.pth" \
    --config "${TTS_EXP}/config.yaml" \
    --inference-log "${DECODE_DIR}/log/tts_inference.1.log" \
    --output "${ROOT}/reports/expanded_v3_smoke_inference.json"

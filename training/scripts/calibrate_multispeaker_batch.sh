#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BATCH_BINS=${1:-2500000}
ITERATIONS=${2:-200}
if (( BATCH_BINS < 1000000 || ITERATIONS < 50 || ITERATIONS > 500 )); then
    echo "Use batch_bins >= 1000000 and 50--500 iterations." >&2
    exit 2
fi

source "${ROOT}/activate.sh"
GPU_UUIDS=${GPU_UUIDS:-"GPU-be591530-39fd-0b1c-50af-8c75548cb6b8,GPU-de1be084-ce05-942c-cb74-78e80652f184"}
export CUDA_VISIBLE_DEVICES="${GPU_UUIDS}"
export GPU_COUNT=2
python "${ROOT}/scripts/check_resources.py" \
    --mode full --require-torch --gpu-uuids "${GPU_UUIDS}" \
    --workspace "${ROOT}" --output "${ROOT}/reports/resource_usage.jsonl"

export MANIFEST_DIR="${ROOT}/data/multispeaker_full/manifests"
export TRAIN_SET=multispeaker_train
export VALID_SET=multispeaker_dev
export TEST_SETS=multispeaker_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_multispeaker_full"
export EXP_DIR="${ROOT}/exp_multispeaker_full"
CALIBRATION_EXP="${EXP_DIR}/calibration_${BATCH_BINS}"

python "${ROOT}/scripts/run_logged.py" \
    --log "${ROOT}/reports/commands.jsonl" --eta-minutes 15 -- \
    "${ROOT}/espnet_recipe/run_multispeaker.sh" \
    --stage 7 --stop_stage 7 \
    --tts_exp "${CALIBRATION_EXP}" \
    --train_args \
        "--max_epoch 1 --num_iters_per_epoch ${ITERATIONS} --batch_bins ${BATCH_BINS} --num_workers 8 --cudnn_benchmark false --use_amp false"

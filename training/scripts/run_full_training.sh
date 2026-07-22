#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_ITERATIONS=${1:-1000}
if (( TARGET_ITERATIONS < 1000 || TARGET_ITERATIONS % 1000 != 0 )); then
    echo "target iterations must be a positive multiple of 1000" >&2
    exit 2
fi
TARGET_EPOCHS=$((TARGET_ITERATIONS / 1000))
# Two-GPU ETA is deliberately conservative until the first DDP epoch is measured.
ETA_MINUTES=$(((TARGET_ITERATIONS * 15 + 999) / 1000 + 10))

source "${ROOT}/activate.sh"
GPU_UUIDS=${GPU_UUIDS:-"GPU-be591530-39fd-0b1c-50af-8c75548cb6b8,GPU-de1be084-ce05-942c-cb74-78e80652f184"}
export CUDA_VISIBLE_DEVICES="${GPU_UUIDS}"
export GPU_COUNT=${GPU_COUNT:-2}
python "${ROOT}/scripts/check_resources.py" \
    --mode full --require-torch --gpu-uuids "${GPU_UUIDS}" --workspace "${ROOT}" \
    --output "${ROOT}/reports/resource_usage.jsonl"

export MANIFEST_DIR="${ROOT}/data/full/manifests"
export TRAIN_SET=train
export VALID_SET=dev
export TEST_SETS=eval
export DATA_SETS="train dev eval"
export DUMP_DIR="${ROOT}/dump_full"
export EXP_DIR="${ROOT}/exp_full"

python "${ROOT}/scripts/run_logged.py" \
    --log "${ROOT}/reports/commands.jsonl" \
    --eta-minutes "${ETA_MINUTES}" -- \
    "${ROOT}/espnet_recipe/run.sh" \
    --stage 7 --stop_stage 7 \
    --tts_exp "${EXP_DIR}/tts_jets_uk_24k_full" \
    --train_args "--max_epoch ${TARGET_EPOCHS} --num_iters_per_epoch 1000 --batch_bins 3000000 --use_amp false"

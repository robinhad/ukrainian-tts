#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_ITERATIONS=${1:-1000}
if (( TARGET_ITERATIONS < 1000 || TARGET_ITERATIONS % 1000 != 0 )); then
    echo "target iterations must be a positive multiple of 1000" >&2
    exit 2
fi
TARGET_EPOCHS=$((TARGET_ITERATIONS / 1000))
# FP32 calibration measured about 1.54 seconds/iteration plus validation.
ETA_MINUTES=$(((TARGET_ITERATIONS * 26 + 999) / 1000 + 10))

source "${ROOT}/activate.sh"
python "${ROOT}/scripts/check_resources.py" \
    --mode full --require-torch --workspace "${ROOT}" \
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

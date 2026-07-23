#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${1:-latest.pth}
source "${ROOT}/activate.sh"
export MANIFEST_DIR="${ROOT}/data/multispeaker_full/manifests"
export TRAIN_SET=multispeaker_train
export VALID_SET=multispeaker_dev
export TEST_SETS=multispeaker_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_multispeaker_full"
export EXP_DIR="${ROOT}/exp_multispeaker_full"
export GPU_COUNT=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-GPU-be591530-39fd-0b1c-50af-8c75548cb6b8}

python "${ROOT}/scripts/run_logged.py" \
    --log "${ROOT}/reports/commands.jsonl" --eta-minutes 30 -- \
    "${ROOT}/espnet_recipe/run_multispeaker.sh" \
    --stage 8 --stop_stage 8 \
    --tts_exp "${EXP_DIR}/tts_jets_uk_24k_multispeaker" \
    --inference_model "${MODEL}"

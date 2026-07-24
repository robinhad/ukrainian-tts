#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_ITERATIONS=${1:-25000}
if (( TARGET_ITERATIONS < 1000 || TARGET_ITERATIONS % 1000 != 0 )); then
    echo "The target must be a positive multiple of 1000 iterations." >&2
    exit 2
fi
TARGET_EPOCHS=$((TARGET_ITERATIONS / 1000))
BATCH_BINS=${BATCH_BINS:-2000000}
NUM_WORKERS=${NUM_WORKERS:-8}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

source "${ROOT}/activate.sh"
GPU_UUIDS=${GPU_UUIDS:-"GPU-be591530-39fd-0b1c-50af-8c75548cb6b8,GPU-de1be084-ce05-942c-cb74-78e80652f184"}
export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
export GPU_COUNT=2
python "${ROOT}/scripts/check_resources.py" \
    --mode full --require-torch --gpu-uuids "$GPU_UUIDS" \
    --workspace "$ROOT" --output "${ROOT}/reports/resource_usage.jsonl"

export MANIFEST_DIR="${ROOT}/data/multispeaker_enhanced_v2/manifests"
export TRAIN_SET=multispeaker_train
export VALID_SET=multispeaker_dev
export TEST_SETS=multispeaker_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_multispeaker_enhanced_v2"
export EXP_DIR="${ROOT}/exp_multispeaker_enhanced_v2"
TTS_EXP="${EXP_DIR}/tts_jets_uk_24k_multispeaker_enhanced_v2"
mkdir -p "$TTS_EXP"
git -C "${ROOT}/.." rev-parse HEAD > "${TTS_EXP}/training_git_commit.txt"

"${ROOT}/espnet_recipe/run_multispeaker_enhanced_v2.sh" \
    --stage 7 --stop_stage 7 \
    --tts_exp "$TTS_EXP" \
    --train_args \
        "--max_epoch ${TARGET_EPOCHS} --num_iters_per_epoch 1000 --batch_bins ${BATCH_BINS} --num_workers ${NUM_WORKERS} --cudnn_benchmark false --use_amp false"


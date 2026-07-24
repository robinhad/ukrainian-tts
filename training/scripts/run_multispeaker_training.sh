#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_ITERATIONS=${1:-25000}
if (( TARGET_ITERATIONS < 1000 || TARGET_ITERATIONS % 1000 != 0 )); then
    echo "The target must be a positive multiple of 1000 iterations." >&2
    exit 2
fi
TARGET_EPOCHS=$((TARGET_ITERATIONS / 1000))
BATCH_BINS=${BATCH_BINS:-3800000}
NUM_WORKERS=${NUM_WORKERS:-8}
ETA_MINUTES_PER_1000=${ETA_MINUTES_PER_1000:-32}
ETA_MINUTES=$(((TARGET_ITERATIONS * ETA_MINUTES_PER_1000 + 999) / 1000 + 15))

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
TTS_EXP="${EXP_DIR}/tts_jets_uk_24k_multispeaker"
mkdir -p "$TTS_EXP"
if [[ ! -e "${TTS_EXP}/training_git_commit.txt" ]]; then
    git -C "${ROOT}/.." rev-parse HEAD > "${TTS_EXP}/training_git_commit.txt"
fi

python "${ROOT}/scripts/run_logged.py" \
    --log "${ROOT}/reports/commands.jsonl" --eta-minutes "${ETA_MINUTES}" -- \
    "${ROOT}/espnet_recipe/run_multispeaker.sh" \
    --stage 7 --stop_stage 7 \
    --tts_exp "${TTS_EXP}" \
    --train_args \
        "--max_epoch ${TARGET_EPOCHS} --num_iters_per_epoch 1000 --batch_bins ${BATCH_BINS} --num_workers ${NUM_WORKERS} --cudnn_benchmark false --use_amp false"

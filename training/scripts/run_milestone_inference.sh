#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${1:-latest.pth}
DATASET_NAME=${DATASET_NAME:-full}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-tts_jets_uk_24k_full}
DUMP_NAME=${DUMP_NAME:-dump_full}
EXP_NAME=${EXP_NAME:-exp_full}
source "${ROOT}/activate.sh"
export MANIFEST_DIR="${ROOT}/data/${DATASET_NAME}/manifests"
export TRAIN_SET=train
export VALID_SET=dev
export TEST_SETS=eval
export DATA_SETS="train dev eval"
export DUMP_DIR="${ROOT}/${DUMP_NAME}"
export EXP_DIR="${ROOT}/${EXP_NAME}"

python "${ROOT}/scripts/run_logged.py" \
    --log "${ROOT}/reports/commands.jsonl" --eta-minutes 15 -- \
    "${ROOT}/espnet_recipe/run.sh" \
    --stage 8 --stop_stage 8 \
    --tts_exp "${EXP_DIR}/${EXPERIMENT_NAME}" \
    --inference_model "${MODEL}"

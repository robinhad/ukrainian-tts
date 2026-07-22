#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${1:-latest.pth}
source "${ROOT}/activate.sh"
export MANIFEST_DIR="${ROOT}/data/full/manifests"
export TRAIN_SET=train
export VALID_SET=dev
export TEST_SETS=eval
export DATA_SETS="train dev eval"
export DUMP_DIR="${ROOT}/dump_full"
export EXP_DIR="${ROOT}/exp_full"

python "${ROOT}/scripts/run_logged.py" \
    --log "${ROOT}/reports/commands.jsonl" --eta-minutes 15 -- \
    "${ROOT}/espnet_recipe/run.sh" \
    --stage 8 --stop_stage 8 \
    --tts_exp "${EXP_DIR}/tts_jets_uk_24k_full" \
    --inference_model "${MODEL}"

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=expanded_v7_sidon_deess_declick_limit_novoa
TARGET_ITERATIONS=${1:-100000}
if (( TARGET_ITERATIONS < 1000 || TARGET_ITERATIONS % 1000 != 0 )); then
    echo "TARGET_ITERATIONS must be a positive multiple of 1,000." >&2
    exit 2
fi
MAX_EPOCH=$((TARGET_ITERATIONS / 1000))
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
BATCH_BINS=${BATCH_BINS:-2000000}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_${NAME}/tts_jets_uk_24k_${NAME}_from_v6e94_100k"}
INIT_CHECKPOINT="${ROOT}/exp_expanded_v6_sidon_deess_novoa/tts_jets_uk_24k_expanded_v6_sidon_deess_novoa_from_v5e81_100k/best_checkpoints/94epoch.pth"
INIT_SHA256=3a8f3f40d00abfcaadc3f457a7485ddcc816cbb8a26d0cec193eeae39c7d7d48
READINESS="${ROOT}/reports/${NAME}_full_scale_readiness.json"
SMOKE_REPORT="${ROOT}/reports/${NAME}_smoke_inference.json"

source "${ROOT}/activate.sh"
echo "${INIT_SHA256}  ${INIT_CHECKPOINT}" | sha256sum -c -
python - "$READINESS" "$SMOKE_REPORT" <<'PY'
import json
import sys
from pathlib import Path

for path in map(Path, sys.argv[1:]):
    if json.loads(path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise SystemExit(f"A required gate is not PASS: {path}")
PY
if [[ -e "${TTS_EXP}/checkpoint.pth" && "${ALLOW_TRAINING_RESUME:-0}" != 1 ]]; then
    echo "The v7 experiment already has a checkpoint." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_UUIDS" GPU_COUNT=2
export MANIFEST_DIR="${ROOT}/data/${NAME}/manifests"
export TRAIN_SET=${NAME}_train VALID_SET=${NAME}_dev TEST_SETS=${NAME}_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_${NAME}" EXP_DIR="${ROOT}/exp_${NAME}"
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
python "${ROOT}/scripts/check_resources.py" --mode full --require-torch \
    --gpu-uuids "$GPU_UUIDS" --workspace "$ROOT" \
    --output "${ROOT}/reports/resource_usage_${NAME}.jsonl"
mkdir -p "$TTS_EXP"
git -C "${ROOT}/.." rev-parse HEAD > "${TTS_EXP}/training_git_commit.txt"
printf '%s\n' "$INIT_CHECKPOINT" > "${TTS_EXP}/initial_checkpoint.txt"
printf '%s\n' "$INIT_SHA256" > "${TTS_EXP}/initial_checkpoint.sha256"
"${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" \
    --stage 7 --stop_stage 7 --tts_exp "$TTS_EXP" \
    --train_args "--max_epoch ${MAX_EPOCH} --num_iters_per_epoch 1000 --batch_bins ${BATCH_BINS} --num_workers 8 --accum_grad 1 --cudnn_benchmark false --use_amp false --keep_nbest_models 3 --num_att_plot 0 --init_param ${INIT_CHECKPOINT}:::normalize,pitch_normalize,energy_normalize --ignore_init_mismatch false"

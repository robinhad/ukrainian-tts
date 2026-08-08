#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_ITERATIONS=${1:-100000}
if (( TARGET_ITERATIONS < 1000 || TARGET_ITERATIONS % 1000 != 0 )); then
    echo "TARGET_ITERATIONS must be a positive multiple of 1000." >&2
    exit 2
fi
MAX_EPOCH=$((TARGET_ITERATIONS / 1000))
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
BATCH_BINS=${BATCH_BINS:-2000000}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v5_enhanced_novoa/tts_jets_uk_24k_expanded_v5_enhanced_novoa_ft_v4e28_100k"}
INIT_CHECKPOINT="${ROOT}/exp_expanded_v4_trim_only/tts_jets_uk_24k_expanded_v4_trim_only_ft367_100k/best_checkpoints/28epoch.pth"
INIT_SHA256=01ae21912295c1e100b0f30325761420f19161463dfdb048b50d2c82396591fa
READINESS="${ROOT}/reports/expanded_v5_enhanced_novoa_full_scale_readiness.json"
SMOKE_REPORT="${ROOT}/reports/expanded_v5_enhanced_novoa_smoke_inference.json"

source "${ROOT}/activate.sh"
echo "${INIT_SHA256}  ${INIT_CHECKPOINT}" | sha256sum -c -
python - "$READINESS" "$SMOKE_REPORT" <<'PY'
import json
import sys
from pathlib import Path

for path in map(Path, sys.argv[1:]):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise SystemExit(f"A required gate is not PASS: {path}")
PY
if [[ -e "${TTS_EXP}/checkpoint.pth" || -e "${TTS_EXP}/1epoch.pth" ]]; then
    if [[ "${ALLOW_TRAINING_RESUME:-0}" != 1 ]]; then
        echo "The v5 experiment already has a checkpoint." >&2
        exit 2
    fi
fi

export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
export GPU_COUNT=2
export MANIFEST_DIR="${ROOT}/data/expanded_v5_enhanced_novoa/manifests"
export TRAIN_SET=expanded_v5_enhanced_novoa_train
export VALID_SET=expanded_v5_enhanced_novoa_dev
export TEST_SETS=expanded_v5_enhanced_novoa_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_expanded_v5_enhanced_novoa"
export EXP_DIR="${ROOT}/exp_expanded_v5_enhanced_novoa"
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

python "${ROOT}/scripts/check_resources.py" \
    --mode full --require-torch --gpu-uuids "$GPU_UUIDS" \
    --workspace "$ROOT" --output "${ROOT}/reports/resource_usage_expanded_v5_enhanced_novoa.jsonl"
mkdir -p "$TTS_EXP"
git -C "${ROOT}/.." rev-parse HEAD > "${TTS_EXP}/training_git_commit.txt"
printf '%s\n' "$INIT_CHECKPOINT" > "${TTS_EXP}/initial_checkpoint.txt"
printf '%s\n' "$INIT_SHA256" > "${TTS_EXP}/initial_checkpoint.sha256"
"${ROOT}/espnet_recipe/run_expanded_v5_enhanced_novoa.sh" \
    --stage 7 --stop_stage 7 --tts_exp "$TTS_EXP" \
    --train_args \
        "--max_epoch ${MAX_EPOCH} --num_iters_per_epoch 1000 --batch_bins ${BATCH_BINS} --num_workers 8 --accum_grad 1 --cudnn_benchmark false --use_amp false --keep_nbest_models 3 --num_att_plot 0 --init_param ${INIT_CHECKPOINT}:::normalize,pitch_normalize,energy_normalize --ignore_init_mismatch false"

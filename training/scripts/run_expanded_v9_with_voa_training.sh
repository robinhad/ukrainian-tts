#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=expanded_v9_cascade_with_voa
TARGET_ITERATIONS=${1:-500000}
if (( TARGET_ITERATIONS < 1000 || TARGET_ITERATIONS % 1000 != 0 )); then
    echo "TARGET_ITERATIONS must be a positive multiple of 1,000." >&2
    exit 2
fi
MAX_EPOCH=$((TARGET_ITERATIONS / 1000))
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
BATCH_BINS=${BATCH_BINS:-2000000}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_${NAME}/tts_jets_uk_24k_${NAME}_scratch_500k"}
READINESS="${ROOT}/reports/${NAME}_full_scale_readiness.json"
SMOKE_REPORT="${ROOT}/reports/${NAME}_smoke_inference.json"
SMOKE_CHECKPOINT="${ROOT}/reports/${NAME}_smoke_checkpoint.json"

source "${ROOT}/activate.sh"
python - "$READINESS" "$SMOKE_REPORT" "$SMOKE_CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path

for path in map(Path, sys.argv[1:]):
    if json.loads(path.read_text()).get("status") != "PASS":
        raise SystemExit(f"A required v9 gate is not PASS: {path}")
PY
if [[ -e "${TTS_EXP}/checkpoint.pth" && "${ALLOW_TRAINING_RESUME:-0}" != 1 ]]; then
    echo "The scratch experiment already has a checkpoint. Set ALLOW_TRAINING_RESUME=1 only to resume this same v9 run." >&2
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
printf '%s\n' "scratch_random_initialization" > "${TTS_EXP}/initialization.txt"
printf '%s\n' "$TARGET_ITERATIONS" > "${TTS_EXP}/target_iterations.txt"
"${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" \
    --stage 7 --stop_stage 7 --tts_exp "$TTS_EXP" \
    --train_args "--max_epoch ${MAX_EPOCH} --num_iters_per_epoch 1000 --batch_bins ${BATCH_BINS} --num_workers 8 --accum_grad 1 --cudnn_benchmark false --use_amp false --keep_nbest_models 3 --num_att_plot 0"

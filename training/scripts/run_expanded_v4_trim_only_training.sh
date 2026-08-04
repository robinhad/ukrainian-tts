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
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v4_trim_only/tts_jets_uk_24k_expanded_v4_trim_only_ft367_100k"}
INIT_CHECKPOINT="${ROOT}/exp_expanded_v3/tts_jets_uk_24k_expanded_v3_500k/best_checkpoints/367epoch.pth"
INIT_SHA256=b7f093cd5c437f9b0122a13c48018c33b154698bb8149d26544401e99126d617
READINESS="${ROOT}/reports/expanded_v4_trim_only_full_scale_readiness.json"

source "${ROOT}/activate.sh"
echo "${INIT_SHA256}  ${INIT_CHECKPOINT}" | sha256sum -c -
python - "$READINESS" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
report = json.loads(path.read_text())
if report.get("status") != "PASS":
    raise SystemExit("The trim-only readiness report is not PASS.")
PY
if [[ -e "${TTS_EXP}/checkpoint.pth" || -e "${TTS_EXP}/1epoch.pth" ]]; then
    if [[ "${ALLOW_TRAINING_RESUME:-0}" != 1 ]]; then
        echo "The trim-only experiment already has a checkpoint." >&2
        exit 2
    fi
fi

export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
export GPU_COUNT=2
export MANIFEST_DIR="${ROOT}/data/expanded_v4_trim_only/manifests"
export TRAIN_SET=expanded_v4_trim_only_train
export VALID_SET=expanded_v4_trim_only_dev
export TEST_SETS=expanded_v4_trim_only_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_expanded_v4_trim_only"
export EXP_DIR="${ROOT}/exp_expanded_v4_trim_only"
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

python "${ROOT}/scripts/check_resources.py" \
    --mode full --require-torch --gpu-uuids "$GPU_UUIDS" \
    --workspace "$ROOT" --output "${ROOT}/reports/resource_usage_expanded_v4_trim_only.jsonl"
mkdir -p "$TTS_EXP"
git -C "${ROOT}/.." rev-parse HEAD > "${TTS_EXP}/training_git_commit.txt"
"${ROOT}/espnet_recipe/run_expanded_v4_trim_only.sh" \
    --stage 7 --stop_stage 7 --tts_exp "$TTS_EXP" \
    --train_args \
        "--max_epoch ${MAX_EPOCH} --num_iters_per_epoch 1000 --batch_bins ${BATCH_BINS} --num_workers 8 --accum_grad 1 --cudnn_benchmark false --use_amp false --init_param ${INIT_CHECKPOINT}:::normalize,pitch_normalize,energy_normalize --ignore_init_mismatch false"

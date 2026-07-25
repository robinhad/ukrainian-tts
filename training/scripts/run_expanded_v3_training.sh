#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_ITERATIONS=${1:-25000}
if (( TARGET_ITERATIONS != 25000 )); then
    echo "The review gate for expanded-v3 is fixed at 25000 iterations." >&2
    exit 2
fi
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
BATCH_BINS=${BATCH_BINS:-2000000}
TTS_EXP="${ROOT}/exp_expanded_v3/tts_jets_uk_24k_expanded_v3_25k"
READINESS="${ROOT}/reports/expanded_v3_full_scale_readiness.json"

source "${ROOT}/activate.sh"
python - "$READINESS" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"The scale readiness report does not exist: {path}")
report = json.loads(path.read_text())
if report.get("status") != "PASS":
    raise SystemExit("The scale readiness report does not have PASS status.")
PY
if [[ -e "${TTS_EXP}/checkpoint.pth" || -e "${TTS_EXP}/1epoch.pth" ]]; then
    echo "The fresh expanded-v3 experiment already has a checkpoint." >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
IFS=, read -r -a GPUS <<< "$GPU_UUIDS"
export GPU_COUNT=${#GPUS[@]}
export MANIFEST_DIR="${ROOT}/data/expanded_v3/manifests"
export TRAIN_SET=expanded_v3_train
export VALID_SET=expanded_v3_dev
export TEST_SETS=expanded_v3_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_expanded_v3"
export EXP_DIR="${ROOT}/exp_expanded_v3"
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

python "${ROOT}/scripts/check_resources.py" \
    --mode full --require-torch --gpu-uuids "$GPU_UUIDS" \
    --workspace "$ROOT" \
    --output "${ROOT}/reports/resource_usage_expanded_v3.jsonl"
mkdir -p "$TTS_EXP"
git -C "${ROOT}/.." rev-parse HEAD > "${TTS_EXP}/training_git_commit.txt"
"${ROOT}/espnet_recipe/run_expanded_v3.sh" \
    --stage 7 --stop_stage 7 --tts_exp "$TTS_EXP" \
    --train_args \
        "--max_epoch 25 --num_iters_per_epoch 1000 --batch_bins ${BATCH_BINS} --num_workers 8 --accum_grad 1 --cudnn_benchmark false --use_amp false"

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v4_trim_only_smoke/tts_jets_uk_24k_expanded_v4_trim_only_ft367_smoke"}
INIT_CHECKPOINT="${ROOT}/exp_expanded_v3/tts_jets_uk_24k_expanded_v3_500k/best_checkpoints/367epoch.pth"
INIT_SHA256=b7f093cd5c437f9b0122a13c48018c33b154698bb8149d26544401e99126d617
READINESS="${ROOT}/reports/expanded_v4_trim_only_smoke_scale_readiness.json"

source "${ROOT}/activate.sh"
echo "${INIT_SHA256}  ${INIT_CHECKPOINT}" | sha256sum -c -
python - "$READINESS" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("status") != "PASS":
    raise SystemExit("The trim-only smoke readiness report is not PASS.")
PY

export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
export GPU_COUNT=2
export MANIFEST_DIR="${ROOT}/data/expanded_v4_trim_only_smoke/manifests"
export TRAIN_SET=expanded_v4_trim_only_train
export VALID_SET=expanded_v4_trim_only_dev
export TEST_SETS=expanded_v4_trim_only_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_expanded_v4_trim_only_smoke"
export EXP_DIR="${ROOT}/exp_expanded_v4_trim_only_smoke"
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

python "${ROOT}/scripts/check_resources.py" \
    --mode smoke --require-torch --gpu-uuids "$GPU_UUIDS" \
    --workspace "$ROOT" --output "${ROOT}/reports/resource_usage_expanded_v4_trim_only.jsonl"
"${ROOT}/espnet_recipe/run_expanded_v4_trim_only.sh" \
    --stage 7 --stop_stage 7 --tts_exp "$TTS_EXP" \
    --train_args \
        "--max_epoch 1 --num_iters_per_epoch 100 --batch_bins 1000000 --num_workers 8 --accum_grad 1 --cudnn_benchmark false --use_amp false --init_param ${INIT_CHECKPOINT}:::normalize,pitch_normalize,energy_normalize --ignore_init_mismatch false"
"${ROOT}/espnet_recipe/run_expanded_v4_trim_only.sh" \
    --stage 8 --stop_stage 8 --tts_exp "$TTS_EXP" \
    --inference_model train.total_count.ave.pth

DECODE_DIR="${TTS_EXP}/decode_jets_train.total_count.ave/${TEST_SETS}"
python "${ROOT}/scripts/synthesize_eval.py" \
    --wav-dir "${DECODE_DIR}/wav" \
    --manifest "${MANIFEST_DIR}/${TEST_SETS}.parquet" \
    --checkpoint "${TTS_EXP}/train.total_count.ave.pth" \
    --config "${TTS_EXP}/config.yaml" \
    --inference-log "${DECODE_DIR}/log/tts_inference.1.log" \
    --output "${ROOT}/reports/expanded_v4_trim_only_smoke_inference.json"

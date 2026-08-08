#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v5_enhanced_novoa_smoke/tts_jets_uk_24k_expanded_v5_enhanced_novoa_ft_v4e28_smoke"}
INIT_CHECKPOINT="${ROOT}/exp_expanded_v4_trim_only/tts_jets_uk_24k_expanded_v4_trim_only_ft367_100k/best_checkpoints/28epoch.pth"
INIT_SHA256=01ae21912295c1e100b0f30325761420f19161463dfdb048b50d2c82396591fa
READINESS="${ROOT}/reports/expanded_v5_enhanced_novoa_full_scale_readiness.json"

source "${ROOT}/activate.sh"
echo "${INIT_SHA256}  ${INIT_CHECKPOINT}" | sha256sum -c -
python - "$READINESS" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("status") != "PASS":
    raise SystemExit("The enhanced non-VOA readiness report is not PASS.")
PY

export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
export GPU_COUNT=2
export MANIFEST_DIR="${ROOT}/data/expanded_v5_enhanced_novoa/manifests"
export TRAIN_SET=expanded_v5_enhanced_novoa_train
export VALID_SET=expanded_v5_enhanced_novoa_dev
export TEST_SETS=expanded_v5_enhanced_novoa_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_expanded_v5_enhanced_novoa"
# Keep the smoke model separate, but use the full filtered-corpus statistics.
export EXP_DIR="${ROOT}/exp_expanded_v5_enhanced_novoa"
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

python "${ROOT}/scripts/check_resources.py" \
    --mode smoke --require-torch --gpu-uuids "$GPU_UUIDS" \
    --workspace "$ROOT" --output "${ROOT}/reports/resource_usage_expanded_v5_enhanced_novoa.jsonl"
"${ROOT}/espnet_recipe/run_expanded_v5_enhanced_novoa.sh" \
    --stage 7 --stop_stage 7 --tts_exp "$TTS_EXP" \
    --train_args \
        "--max_epoch 1 --num_iters_per_epoch 100 --batch_bins 1000000 --num_workers 8 --accum_grad 1 --cudnn_benchmark false --use_amp false --keep_nbest_models 1 --num_att_plot 0 --init_param ${INIT_CHECKPOINT}:::normalize,pitch_normalize,energy_normalize --ignore_init_mismatch false"
"${ROOT}/espnet_recipe/run_expanded_v5_enhanced_novoa.sh" \
    --stage 8 --stop_stage 8 --tts_exp "$TTS_EXP" \
    --inference_model train.total_count.ave.pth

DECODE_DIR="${TTS_EXP}/decode_jets_train.total_count.ave/${TEST_SETS}"
python "${ROOT}/scripts/synthesize_eval.py" \
    --wav-dir "${DECODE_DIR}/wav" \
    --manifest "${MANIFEST_DIR}/${TEST_SETS}.parquet" \
    --checkpoint "${TTS_EXP}/train.total_count.ave.pth" \
    --config "${TTS_EXP}/config.yaml" \
    --inference-log "${DECODE_DIR}/log/tts_inference.1.log" \
    --output "${ROOT}/reports/expanded_v5_enhanced_novoa_smoke_inference.json"
python "${ROOT}/scripts/audit_finetune_checkpoint.py" \
    --checkpoint "${TTS_EXP}/checkpoint.pth" --expected-steps 100 \
    --output "${ROOT}/reports/expanded_v5_enhanced_novoa_smoke_checkpoint.json"
echo "The enhanced non-VOA smoke gate is complete."

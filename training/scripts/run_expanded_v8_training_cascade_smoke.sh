#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_${NAME}_smoke/tts_jets_uk_24k_${NAME}_from_v7e93_smoke"}
INIT_CHECKPOINT="${ROOT}/exp_expanded_v7_sidon_deess_declick_limit_novoa/tts_jets_uk_24k_expanded_v7_sidon_deess_declick_limit_novoa_from_v6e94_100k/best_checkpoints/93epoch.pth"
INIT_SHA256=bec6bb4587a16250e39ce554cb6f2a6563f4d67fc825fc711a2c07436560129a
READINESS="${ROOT}/reports/${NAME}_full_scale_readiness.json"

source "${ROOT}/activate.sh"
echo "${INIT_SHA256}  ${INIT_CHECKPOINT}" | sha256sum -c -
python - "$READINESS" <<'PY'
import json, sys
from pathlib import Path
path=Path(sys.argv[1])
if json.loads(path.read_text()).get('status') != 'PASS':
    raise SystemExit('The v8 readiness report is not PASS.')
PY
export CUDA_VISIBLE_DEVICES="$GPU_UUIDS" GPU_COUNT=2
export MANIFEST_DIR="${ROOT}/data/${NAME}/manifests"
export TRAIN_SET=${NAME}_train VALID_SET=${NAME}_dev TEST_SETS=${NAME}_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_${NAME}" EXP_DIR="${ROOT}/exp_${NAME}"
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
python "${ROOT}/scripts/check_resources.py" --mode smoke --require-torch \
    --gpu-uuids "$GPU_UUIDS" --workspace "$ROOT" \
    --output "${ROOT}/reports/resource_usage_${NAME}.jsonl"
"${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" \
    --stage 7 --stop_stage 7 --tts_exp "$TTS_EXP" \
    --train_args "--max_epoch 1 --num_iters_per_epoch 100 --batch_bins 1000000 --num_workers 8 --accum_grad 1 --cudnn_benchmark false --use_amp false --keep_nbest_models 1 --num_att_plot 0 --init_param ${INIT_CHECKPOINT}:::normalize,pitch_normalize,energy_normalize --ignore_init_mismatch false"
"${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" \
    --stage 8 --stop_stage 8 --tts_exp "$TTS_EXP" \
    --inference_model train.total_count.ave.pth
DECODE_DIR="${TTS_EXP}/decode_jets_train.total_count.ave/${TEST_SETS}"
python "${ROOT}/scripts/synthesize_eval.py" --wav-dir "${DECODE_DIR}/wav" \
    --manifest "${MANIFEST_DIR}/${TEST_SETS}.parquet" \
    --checkpoint "${TTS_EXP}/train.total_count.ave.pth" \
    --config "${TTS_EXP}/config.yaml" \
    --inference-log "${DECODE_DIR}/log/tts_inference.1.log" \
    --output "${ROOT}/reports/${NAME}_smoke_inference.json"
python "${ROOT}/scripts/audit_finetune_checkpoint.py" \
    --checkpoint "${TTS_EXP}/checkpoint.pth" --expected-steps 100 \
    --output "${ROOT}/reports/${NAME}_smoke_checkpoint.json"
echo "The v8 smoke gate is complete."

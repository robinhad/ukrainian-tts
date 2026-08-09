#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v6_sidon_deess_novoa_smoke/tts_jets_uk_24k_expanded_v6_sidon_deess_novoa_from_v5e81_smoke"}
INIT_CHECKPOINT="${ROOT}/exp_expanded_v5_enhanced_novoa/tts_jets_uk_24k_expanded_v5_enhanced_novoa_ft_v4e28_100k/best_checkpoints/81epoch.pth"
INIT_SHA256=ec8f6ec4e6ac330b796d91cb1c55239d81b1f4d8bb8aefd399b6c44b4206ff90
READINESS="${ROOT}/reports/expanded_v6_sidon_deess_novoa_full_scale_readiness.json"
source "${ROOT}/activate.sh"
echo "${INIT_SHA256}  ${INIT_CHECKPOINT}" | sha256sum -c -
python - "$READINESS" <<'PY'
import json, sys
from pathlib import Path
if json.loads(Path(sys.argv[1]).read_text()).get('status') != 'PASS':
    raise SystemExit('The v6 readiness report is not PASS.')
PY
export CUDA_VISIBLE_DEVICES="$GPU_UUIDS" GPU_COUNT=2
export MANIFEST_DIR="${ROOT}/data/expanded_v6_sidon_deess_novoa/manifests"
export TRAIN_SET=expanded_v6_sidon_deess_novoa_train VALID_SET=expanded_v6_sidon_deess_novoa_dev TEST_SETS=expanded_v6_sidon_deess_novoa_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}" DUMP_DIR="${ROOT}/dump_expanded_v6_sidon_deess_novoa" EXP_DIR="${ROOT}/exp_expanded_v6_sidon_deess_novoa"
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
python "${ROOT}/scripts/check_resources.py" --mode smoke --require-torch --gpu-uuids "$GPU_UUIDS" --workspace "$ROOT" --output "${ROOT}/reports/resource_usage_expanded_v6_sidon_deess_novoa.jsonl"
"${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" --stage 7 --stop_stage 7 --tts_exp "$TTS_EXP" \
    --train_args "--max_epoch 1 --num_iters_per_epoch 100 --batch_bins 1000000 --num_workers 8 --accum_grad 1 --cudnn_benchmark false --use_amp false --keep_nbest_models 1 --num_att_plot 0 --init_param ${INIT_CHECKPOINT}:::normalize,pitch_normalize,energy_normalize --ignore_init_mismatch false"
"${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" --stage 8 --stop_stage 8 --tts_exp "$TTS_EXP" --inference_model train.total_count.ave.pth
DECODE_DIR="${TTS_EXP}/decode_jets_train.total_count.ave/${TEST_SETS}"
python "${ROOT}/scripts/synthesize_eval.py" --wav-dir "${DECODE_DIR}/wav" \
    --manifest "${MANIFEST_DIR}/${TEST_SETS}.parquet" --checkpoint "${TTS_EXP}/train.total_count.ave.pth" \
    --config "${TTS_EXP}/config.yaml" --inference-log "${DECODE_DIR}/log/tts_inference.1.log" \
    --output "${ROOT}/reports/expanded_v6_sidon_deess_novoa_smoke_inference.json"
python "${ROOT}/scripts/audit_finetune_checkpoint.py" --checkpoint "${TTS_EXP}/checkpoint.pth" --expected-steps 100 \
    --output "${ROOT}/reports/expanded_v6_sidon_deess_novoa_smoke_checkpoint.json"
echo "The v6 smoke gate is complete."

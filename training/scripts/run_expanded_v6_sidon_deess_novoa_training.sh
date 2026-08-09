#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_ITERATIONS=${1:-100000}
if (( TARGET_ITERATIONS < 1000 || TARGET_ITERATIONS % 1000 != 0 )); then
    echo "TARGET_ITERATIONS must be a positive multiple of 1000." >&2; exit 2
fi
MAX_EPOCH=$((TARGET_ITERATIONS / 1000))
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
BATCH_BINS=${BATCH_BINS:-2000000}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v6_sidon_deess_novoa/tts_jets_uk_24k_expanded_v6_sidon_deess_novoa_from_v5e81_100k"}
INIT_CHECKPOINT="${ROOT}/exp_expanded_v5_enhanced_novoa/tts_jets_uk_24k_expanded_v5_enhanced_novoa_ft_v4e28_100k/best_checkpoints/81epoch.pth"
INIT_SHA256=ec8f6ec4e6ac330b796d91cb1c55239d81b1f4d8bb8aefd399b6c44b4206ff90
READINESS="${ROOT}/reports/expanded_v6_sidon_deess_novoa_full_scale_readiness.json"
SMOKE_REPORT="${ROOT}/reports/expanded_v6_sidon_deess_novoa_smoke_inference.json"
source "${ROOT}/activate.sh"
echo "${INIT_SHA256}  ${INIT_CHECKPOINT}" | sha256sum -c -
python - "$READINESS" "$SMOKE_REPORT" <<'PY'
import json, sys
from pathlib import Path
for path in map(Path, sys.argv[1:]):
    if json.loads(path.read_text()).get('status') != 'PASS':
        raise SystemExit(f'A required gate is not PASS: {path}')
PY
if [[ -e "${TTS_EXP}/checkpoint.pth" && "${ALLOW_TRAINING_RESUME:-0}" != 1 ]]; then
    echo "The v6 experiment already has a checkpoint." >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_UUIDS" GPU_COUNT=2
export MANIFEST_DIR="${ROOT}/data/expanded_v6_sidon_deess_novoa/manifests"
export TRAIN_SET=expanded_v6_sidon_deess_novoa_train VALID_SET=expanded_v6_sidon_deess_novoa_dev
export TEST_SETS=expanded_v6_sidon_deess_novoa_eval DATA_SETS="expanded_v6_sidon_deess_novoa_train expanded_v6_sidon_deess_novoa_dev expanded_v6_sidon_deess_novoa_eval"
export DUMP_DIR="${ROOT}/dump_expanded_v6_sidon_deess_novoa" EXP_DIR="${ROOT}/exp_expanded_v6_sidon_deess_novoa"
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
python "${ROOT}/scripts/check_resources.py" --mode full --require-torch --gpu-uuids "$GPU_UUIDS" --workspace "$ROOT" --output "${ROOT}/reports/resource_usage_expanded_v6_sidon_deess_novoa.jsonl"
mkdir -p "$TTS_EXP"
git -C "${ROOT}/.." rev-parse HEAD > "${TTS_EXP}/training_git_commit.txt"
printf '%s\n' "$INIT_CHECKPOINT" > "${TTS_EXP}/initial_checkpoint.txt"
printf '%s\n' "$INIT_SHA256" > "${TTS_EXP}/initial_checkpoint.sha256"
"${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" --stage 7 --stop_stage 7 --tts_exp "$TTS_EXP" \
    --train_args "--max_epoch ${MAX_EPOCH} --num_iters_per_epoch 1000 --batch_bins ${BATCH_BINS} --num_workers 8 --accum_grad 1 --cudnn_benchmark false --use_amp false --keep_nbest_models 3 --num_att_plot 0 --init_param ${INIT_CHECKPOINT}:::normalize,pitch_normalize,energy_normalize --ignore_init_mismatch false"

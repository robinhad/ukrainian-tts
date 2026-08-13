#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa
TARGET_ITERATIONS=${1:-100000}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_${NAME}/tts_jets_uk_24k_${NAME}_from_v7e93_100k"}
STATUS=${STATUS:-"${ROOT}/reports/training_status_${NAME}_100k.jsonl"}
PYTHON="${ROOT}/.venv/bin/python"

source "${ROOT}/activate.sh"
"$PYTHON" "${ROOT}/scripts/summarize_training_status.py" \
    --input "$STATUS" \
    --output "${ROOT}/reports/${NAME}_training_monitor_summary.json" \
    --target-iterations "$TARGET_ITERATIONS" --maximum-gap-minutes 30
"$PYTHON" "${ROOT}/scripts/summarize_tensorboard.py" \
    --logdir "${TTS_EXP}/tensorboard" \
    --output "${ROOT}/reports/${NAME}_tensorboard_metrics.json"
"$PYTHON" "${ROOT}/scripts/audit_finetune_checkpoint.py" \
    --checkpoint "${TTS_EXP}/checkpoint.pth" \
    --expected-steps "$TARGET_ITERATIONS" \
    --model-artifact "${TTS_EXP}/milestones/100k.pth" \
    --output "${ROOT}/reports/${NAME}_100k_checkpoint.json"

for label in 25k 50k 75k 100k; do
    TTS_EXP="$TTS_EXP" \
        "${ROOT}/scripts/evaluate_expanded_v8_training_cascade_milestone.sh" "$label"
done
TTS_EXP="$TTS_EXP" \
    "${ROOT}/scripts/generate_five_voice_expanded_v8_eval.sh" 100k
echo "The v8 training-cascade 100K finalization is complete."

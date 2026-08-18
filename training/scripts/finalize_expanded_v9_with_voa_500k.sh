#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=expanded_v9_cascade_with_voa
TARGET_ITERATIONS=${1:-500000}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_${NAME}/tts_jets_uk_24k_${NAME}_scratch_500k"}
STATUS=${STATUS:-"${ROOT}/reports/training_status_${NAME}_500k.jsonl"}
PYTHON="${ROOT}/.venv/bin/python"
FINAL_LABEL="$((TARGET_ITERATIONS / 1000))k"

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
    --model-artifact "${TTS_EXP}/milestones/${FINAL_LABEL}.pth" \
    --output "${ROOT}/reports/${NAME}_${FINAL_LABEL}_checkpoint.json"

MILESTONES=(25k 50k 75k 100k 200k 300k 400k)
if [[ "$FINAL_LABEL" != 400k ]]; then
    MILESTONES+=("$FINAL_LABEL")
fi
for label in "${MILESTONES[@]}"; do
    TTS_EXP="$TTS_EXP" \
        "${ROOT}/scripts/evaluate_expanded_v9_with_voa_milestone.sh" "$label"
done
TTS_EXP="$TTS_EXP" \
    "${ROOT}/scripts/generate_five_voice_expanded_v9_eval.sh" "$FINAL_LABEL"
echo "The VOA-inclusive v9 scratch ${FINAL_LABEL} finalization is complete."

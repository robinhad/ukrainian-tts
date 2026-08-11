#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_ITERATIONS=${1:-100000}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v6_sidon_deess_novoa/tts_jets_uk_24k_expanded_v6_sidon_deess_novoa_from_v5e81_100k"}
STATUS=${STATUS:-"${ROOT}/reports/training_status_expanded_v6_sidon_deess_novoa_100k.jsonl"}
PYTHON="${ROOT}/.venv/bin/python"

source "${ROOT}/activate.sh"
"$PYTHON" "${ROOT}/scripts/summarize_training_status.py" \
    --input "$STATUS" \
    --output "${ROOT}/reports/expanded_v6_sidon_deess_novoa_training_monitor_summary.json" \
    --target-iterations "$TARGET_ITERATIONS" --maximum-gap-minutes 30
"$PYTHON" "${ROOT}/scripts/summarize_tensorboard.py" \
    --logdir "${TTS_EXP}/tensorboard" \
    --output "${ROOT}/reports/expanded_v6_sidon_deess_novoa_tensorboard_metrics.json"
"$PYTHON" "${ROOT}/scripts/audit_finetune_checkpoint.py" \
    --checkpoint "${TTS_EXP}/checkpoint.pth" \
    --expected-steps "$TARGET_ITERATIONS" \
    --model-artifact "${TTS_EXP}/milestones/100k.pth" \
    --output "${ROOT}/reports/expanded_v6_sidon_deess_novoa_100k_checkpoint.json"

for label in 25k 50k 75k 100k; do
    TTS_EXP="$TTS_EXP" \
        "${ROOT}/scripts/evaluate_expanded_v6_sidon_deess_novoa_milestone.sh" "$label"
done
TTS_EXP="$TTS_EXP" \
    "${ROOT}/scripts/generate_five_voice_expanded_v6_eval.sh" 100k
echo "The Sidon and de-essing 100K finalization is complete."

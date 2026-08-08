#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_ITERATIONS=${1:-100000}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v5_enhanced_novoa/tts_jets_uk_24k_expanded_v5_enhanced_novoa_ft_v4e28_100k"}
STATUS=${STATUS:-"${ROOT}/reports/training_status_expanded_v5_enhanced_novoa_100k.jsonl"}
PYTHON="${ROOT}/.venv/bin/python"

source "${ROOT}/activate.sh"
set +e
"$PYTHON" "${ROOT}/scripts/summarize_training_status.py" \
    --input "$STATUS" \
    --output "${ROOT}/reports/expanded_v5_enhanced_novoa_training_monitor_summary.json" \
    --target-iterations "$TARGET_ITERATIONS" --maximum-gap-minutes 30
monitor_summary_status=$?
set -e
if (( monitor_summary_status != 0 )); then
    echo "The monitoring summary records a real monitoring failure. Continue artifact validation." >&2
fi
"$PYTHON" "${ROOT}/scripts/summarize_tensorboard.py" \
    --logdir "${TTS_EXP}/tensorboard" \
    --output "${ROOT}/reports/expanded_v5_enhanced_novoa_tensorboard_metrics.json"

audit_args=(
    --checkpoint "${TTS_EXP}/checkpoint.pth"
    --expected-steps "$TARGET_ITERATIONS"
    --output "${ROOT}/reports/expanded_v5_enhanced_novoa_100k_checkpoint.json"
)
if [[ -s "${TTS_EXP}/milestones/100k.pth" ]]; then
    audit_args+=(--model-artifact "${TTS_EXP}/milestones/100k.pth")
fi
"$PYTHON" "${ROOT}/scripts/audit_finetune_checkpoint.py" "${audit_args[@]}"

for label in 25k 50k 75k 100k; do
    if [[ -s "${TTS_EXP}/milestones/${label}.pth" ]]; then
        TTS_EXP="$TTS_EXP" \
            "${ROOT}/scripts/evaluate_expanded_v5_enhanced_novoa_milestone.sh" "$label"
        if [[ "$label" != 100k && "${KEEP_INTERMEDIATE_DECODE:-0}" != 1 ]]; then
            decode="${TTS_EXP}/decode_jets_milestone_${label}"
            if [[ -d "$decode" && "$decode" == "${TTS_EXP}"/decode_jets_milestone_* ]]; then
                rm -r -- "$decode"
            fi
        fi
    fi
done
if [[ -s "${TTS_EXP}/milestones/100k.pth" ]]; then
    TTS_EXP="$TTS_EXP" \
        "${ROOT}/scripts/generate_five_voice_expanded_v5_eval.sh" 100k
fi
echo "The enhanced non-VOA 100K finalization is complete."

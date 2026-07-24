#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MILESTONE_LABEL=${1:?Usage: evaluate_multispeaker_milestone.sh LABEL}
if ! [[ "$MILESTONE_LABEL" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "The milestone label has invalid characters." >&2
    exit 2
fi

TTS_EXP="${ROOT}/exp_multispeaker_full/tts_jets_uk_24k_multispeaker"
MILESTONE_CHECKPOINT="${ROOT}/exp_multispeaker_full/milestones/${MILESTONE_LABEL}.pth"
INFERENCE_MODEL="milestone_${MILESTONE_LABEL}.pth"
DECODE_DIR="${TTS_EXP}/decode_jets_milestone_${MILESTONE_LABEL}/multispeaker_eval"
MANIFEST="${ROOT}/data/multispeaker_full/manifests/multispeaker_eval.parquet"
REPORT="${ROOT}/reports/multispeaker_full_inference_${MILESTONE_LABEL}.json"

if [[ ! -s "$MILESTONE_CHECKPOINT" ]]; then
    echo "The milestone checkpoint does not exist: $MILESTONE_CHECKPOINT" >&2
    exit 2
fi

source "${ROOT}/activate.sh"
ln -sfn "../milestones/${MILESTONE_LABEL}.pth" \
    "${TTS_EXP}/${INFERENCE_MODEL}"
"${ROOT}/scripts/run_multispeaker_milestone_inference.sh" "$INFERENCE_MODEL"
python "${ROOT}/scripts/synthesize_eval.py" \
    --wav-dir "${DECODE_DIR}/wav" \
    --manifest "$MANIFEST" \
    --checkpoint "$MILESTONE_CHECKPOINT" \
    --config "${TTS_EXP}/config.yaml" \
    --inference-log "${DECODE_DIR}/log/tts_inference.1.log" \
    --output "$REPORT"

echo "Milestone: $MILESTONE_LABEL"
echo "Evaluation report: $REPORT"

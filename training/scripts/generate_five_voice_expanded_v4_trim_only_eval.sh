#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPOSITORY_ROOT=$(cd "${ROOT}/.." && pwd)
MILESTONE_LABEL=${1:-100k}
if ! [[ "$MILESTONE_LABEL" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "The milestone label has invalid characters." >&2
    exit 2
fi
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v4_trim_only/tts_jets_uk_24k_expanded_v4_trim_only_ft367_100k"}
MODEL_FILE=${MODEL_FILE:-"milestones/${MILESTONE_LABEL}.pth"}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/eval/generated/five_voice_expanded_v4_trim_only_${MILESTONE_LABEL}"}
REPORT=${REPORT:-"${ROOT}/reports/five_voice_expanded_v4_trim_only_${MILESTONE_LABEL}.json"}
VOICE_ARK="${ROOT}/eval/generated/five_voice_expanded_v3_500k/voices.ark"
REFERENCE_REPORT="${ROOT}/reports/five_voice_expanded_v3_500k.json"
TEXT=${LISTENING_TEXT:-"Кам'янець-Подільський - місто в Хмельницькій області України, центр Кам'янець-Подільської міської об'єднаної територіальної громади і Кам'янець-Подільського району."}

if [[ ! -s "${TTS_EXP}/${MODEL_FILE}" || ! -s "$VOICE_ARK" || ! -s "$REFERENCE_REPORT" ]]; then
    echo "A checkpoint or fixed voice artifact does not exist." >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"
cp "$REFERENCE_REPORT" "$REPORT"

source "${ROOT}/activate.sh"
for voice in voice_01 voice_02 voice_03 voice_04 voice_05; do
    (
        cd "$REPOSITORY_ROOT"
        python -m training.inference.synthesize \
            --text "$TEXT" --output "${OUTPUT_DIR}/${voice}.wav" \
            --config "${TTS_EXP}/config.yaml" \
            --checkpoint "${TTS_EXP}/${MODEL_FILE}" \
            --speaker-embedding-ark "$VOICE_ARK" --speaker "$voice"
    )
done
python "${ROOT}/scripts/validate_listening_voice_eval.py" \
    --report "$REPORT" --wav-dir "$OUTPUT_DIR" --text "$TEXT" \
    --checkpoint "${TTS_EXP}/${MODEL_FILE}" --config "${TTS_EXP}/config.yaml"

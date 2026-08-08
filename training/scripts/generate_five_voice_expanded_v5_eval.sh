#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPOSITORY_ROOT=$(cd "${ROOT}/.." && pwd)
MILESTONE_LABEL=${1:-100k}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v5_enhanced_novoa/tts_jets_uk_24k_expanded_v5_enhanced_novoa_ft_v4e28_100k"}
MODEL_FILE=${MODEL_FILE:-"milestones/${MILESTONE_LABEL}.pth"}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/eval/generated/five_voice_expanded_v5_enhanced_novoa_${MILESTONE_LABEL}"}
REPORT=${REPORT:-"${ROOT}/reports/five_voice_expanded_v5_enhanced_novoa_${MILESTONE_LABEL}.json"}
VOICE_ARK="${OUTPUT_DIR}/voices.ark"
SELECTION_REPORT="${OUTPUT_DIR}/voice_selection.json"
MANIFEST="${ROOT}/data/expanded_v5_enhanced_novoa/hybrid_manifest/all.parquet"
EVAL_SCP="${ROOT}/dump_expanded_v5_enhanced_novoa/xvector/expanded_v5_enhanced_novoa_eval/xvector.scp"
TEXT=${LISTENING_TEXT:-"Кам'янець-Подільський - місто в Хмельницькій області України, центр Кам'янець-Подільської міської об'єднаної територіальної громади і Кам'янець-Подільського району."}

if [[ ! -s "${TTS_EXP}/${MODEL_FILE}" || ! -s "$MANIFEST" || ! -s "$EVAL_SCP" ]]; then
    echo "A checkpoint, manifest, or embedding archive does not exist." >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"
source "${ROOT}/activate.sh"
python "${ROOT}/scripts/select_five_expanded_voices.py" \
    --manifest "$MANIFEST" --corpus-ark "$EVAL_SCP" \
    --output-ark "$VOICE_ARK" --output-report "$SELECTION_REPORT"
cp "$SELECTION_REPORT" "$REPORT"

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

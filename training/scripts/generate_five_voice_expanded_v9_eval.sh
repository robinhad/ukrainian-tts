#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPOSITORY_ROOT=$(cd "${ROOT}/.." && pwd)
NAME=expanded_v9_cascade_with_voa
LABEL=${1:-500k}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_${NAME}/tts_jets_uk_24k_${NAME}_scratch_500k"}
MODEL_FILE=${MODEL_FILE:-"milestones/${LABEL}.pth"}
OUTPUT=${OUTPUT_DIR:-"${ROOT}/eval/generated/five_voice_${NAME}_${LABEL}"}
REPORT=${REPORT:-"${ROOT}/reports/five_voice_${NAME}_${LABEL}.json"}
MANIFEST="${ROOT}/data/${NAME}/hybrid_manifest/all.parquet"
EVAL_SCP="${ROOT}/dump_${NAME}/xvector/${NAME}_eval/xvector.scp"
TEXT=${LISTENING_TEXT:-"Кам'янець-Подільський - місто в Хмельницькій області України, центр Кам'янець-Подільської міської об'єднаної територіальної громади і Кам'янець-Подільського району."}

if [[ ! -s "${TTS_EXP}/${MODEL_FILE}" || ! -s "$MANIFEST" || ! -s "$EVAL_SCP" ]]; then
    echo "A checkpoint, manifest, or embedding archive does not exist." >&2
    exit 2
fi
mkdir -p "$OUTPUT"
source "${ROOT}/activate.sh"
python "${ROOT}/scripts/select_five_expanded_voices.py" \
    --manifest "$MANIFEST" --corpus-ark "$EVAL_SCP" \
    --output-ark "${OUTPUT}/voices.ark" \
    --output-report "${OUTPUT}/voice_selection.json"
cp "${OUTPUT}/voice_selection.json" "$REPORT"
for voice in voice_01 voice_02 voice_03 voice_04 voice_05; do
    (
        cd "$REPOSITORY_ROOT"
        python -m training.inference.synthesize \
            --text "$TEXT" --output "${OUTPUT}/${voice}.wav" \
            --config "${TTS_EXP}/config.yaml" \
            --checkpoint "${TTS_EXP}/${MODEL_FILE}" \
            --speaker-embedding-ark "${OUTPUT}/voices.ark" \
            --speaker "$voice"
    )
done
python "${ROOT}/scripts/validate_listening_voice_eval.py" \
    --report "$REPORT" --wav-dir "$OUTPUT" --text "$TEXT" \
    --checkpoint "${TTS_EXP}/${MODEL_FILE}" --config "${TTS_EXP}/config.yaml"

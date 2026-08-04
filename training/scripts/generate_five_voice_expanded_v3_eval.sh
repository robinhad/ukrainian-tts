#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPOSITORY_ROOT=$(cd "${ROOT}/.." && pwd)
TARGET_ITERATIONS=${TARGET_ITERATIONS:-25000}
TARGET_LABEL=${TARGET_LABEL:-"$((TARGET_ITERATIONS / 1000))k"}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v3/tts_jets_uk_24k_expanded_v3_${TARGET_LABEL}"}
MODEL_FILE=${MODEL_FILE:-"milestones/${TARGET_LABEL}.pth"}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/eval/generated/five_voice_expanded_v3_${TARGET_LABEL}"}
REPORT=${REPORT:-"${ROOT}/reports/five_voice_expanded_v3_${TARGET_LABEL}.json"}
VOICE_ARK="${OUTPUT_DIR}/voices.ark"
TEXT=${LISTENING_TEXT:-"Кам'янець-Подільський - місто в Хмельницькій області України, центр Кам'янець-Подільської міської об'єднаної територіальної громади і Кам'янець-Подільського району."}

source "${ROOT}/activate.sh"
python "${ROOT}/scripts/select_five_expanded_voices.py" \
    --manifest "${ROOT}/data/expanded_v3/manifests/expanded_v3_train.parquet" \
    --corpus-ark "${ROOT}/dump_expanded_v3/xvector/expanded_v3_train/xvector.scp" \
    --output-ark "$VOICE_ARK" \
    --output-report "$REPORT"

for voice in voice_01 voice_02 voice_03 voice_04 voice_05; do
    (
        cd "$REPOSITORY_ROOT"
        python -m training.inference.synthesize \
            --text "$TEXT" \
            --output "${OUTPUT_DIR}/${voice}.wav" \
            --config "${TTS_EXP}/config.yaml" \
            --checkpoint "${TTS_EXP}/${MODEL_FILE}" \
            --speaker-embedding-ark "$VOICE_ARK" \
            --speaker "$voice"
    )
done

python "${ROOT}/scripts/validate_listening_voice_eval.py" \
    --report "$REPORT" --wav-dir "$OUTPUT_DIR" --text "$TEXT"

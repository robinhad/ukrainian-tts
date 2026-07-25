#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPOSITORY_ROOT=$(cd "${ROOT}/.." && pwd)
TTS_EXP="${ROOT}/exp_expanded_v3/tts_jets_uk_24k_expanded_v3_25k"
MODEL_FILE=${MODEL_FILE:-train.total_count.ave.pth}
OUTPUT_DIR="${ROOT}/eval/generated/five_voice_expanded_v3_25k"
REPORT="${ROOT}/reports/five_voice_expanded_v3_25k.json"
VOICE_ARK="${OUTPUT_DIR}/voices.ark"
TEXT=${LISTENING_TEXT:-"Кам'янець-Подільський - місто в Хмельницькій області України, центр Кам'янець-Подільської міської об'єднаної територіальної громади і Кам'янець-Подільського району."}

source "${ROOT}/activate.sh"
python "${ROOT}/scripts/select_five_expanded_voices.py" \
    --manifest "${ROOT}/data/expanded_v3/manifests/expanded_v3_train.parquet" \
    --corpus-ark "${ROOT}/dump_expanded_v3/xvector/expanded_v3_train/spk_xvector.ark" \
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

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPOSITORY_ROOT=$(cd "${ROOT}/.." && pwd)
OUTPUT_DIR="${ROOT}/eval/generated/five_voice_metallic_review"
REPORT="${ROOT}/reports/five_voice_listening_eval.json"
VOICE_ARK="${OUTPUT_DIR}/voices.ark"
CHECKPOINT="${ROOT}/exp_multispeaker_full/milestones/25k.pth"
CONFIG="${ROOT}/exp_multispeaker_full/tts_jets_uk_24k_multispeaker/config.yaml"
TEXT=${LISTENING_TEXT:-"Кам'янець-Подільський - місто в Хмельницькій області України, центр Кам'янець-Подільської міської об'єднаної територіальної громади і Кам'янець-Подільського району."}

source "${ROOT}/activate.sh"
python "${ROOT}/scripts/select_diverse_listening_voices.py" \
    --manifest "${ROOT}/data/multispeaker_full/manifests/multispeaker_train.parquet" \
    --corpus-ark "${ROOT}/dump_multispeaker_full/xvector/multispeaker_train/spk_xvector.ark" \
    --named-ark "${ROOT}/data/multispeaker/source_metadata/spk_xvector_v6.ark" \
    --output-ark "$VOICE_ARK" \
    --output-report "$REPORT"

for voice in \
    voice_01_lada \
    voice_02_dmytro_zero_shot \
    voice_03_common_voice \
    voice_04_common_voice \
    voice_05_common_voice
do
    (
        cd "$REPOSITORY_ROOT"
        python -m training.inference.synthesize \
            --text "$TEXT" \
            --output "${OUTPUT_DIR}/${voice}.wav" \
            --config "$CONFIG" \
            --checkpoint "$CHECKPOINT" \
            --speaker-embedding-ark "$VOICE_ARK" \
            --speaker "$voice"
    )
done

python "${ROOT}/scripts/validate_listening_voice_eval.py" \
    --report "$REPORT" \
    --wav-dir "$OUTPUT_DIR" \
    --text "$TEXT"

echo "Listening directory: $OUTPUT_DIR"
echo "Report: $REPORT"

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECKPOINT_NAME=${1:-25epoch.pth}
MILESTONE_LABEL=${2:-25k}
TTS_EXP="${ROOT}/exp_multispeaker_full/tts_jets_uk_24k_multispeaker"
SOURCE_CHECKPOINT="${TTS_EXP}/${CHECKPOINT_NAME}"
MILESTONE_DIR="${ROOT}/exp_multispeaker_full/milestones"
MILESTONE_CHECKPOINT="${MILESTONE_DIR}/${MILESTONE_LABEL}.pth"
DECODE_DIR="${TTS_EXP}/decode_jets_milestone_${MILESTONE_LABEL}/multispeaker_eval"
MANIFEST="${ROOT}/data/multispeaker_full/manifests/multispeaker_eval.parquet"
REPORT="${ROOT}/reports/multispeaker_full_inference_${MILESTONE_LABEL}.json"
SPEAKER_ARK="${ROOT}/data/multispeaker/source_metadata/spk_xvector_v6.ark"
EXAMPLE_TEXT="Український синтез мовлення працює офлайн."
RELEASE_DIR="${ROOT}/releases/uk-tts-jets-multispeaker-${MILESTONE_LABEL}-rc"

if [[ ! -s "$SOURCE_CHECKPOINT" ]]; then
    echo "The checkpoint does not exist: $SOURCE_CHECKPOINT" >&2
    exit 2
fi
if [[ ! -s "$SPEAKER_ARK" ]]; then
    echo "The speaker embedding file does not exist: $SPEAKER_ARK" >&2
    exit 2
fi

source "${ROOT}/activate.sh"

python "${ROOT}/scripts/summarize_training_status.py" \
    --input "${ROOT}/reports/training_status_multispeaker.jsonl" \
    --output "${ROOT}/reports/power_summary_multispeaker.json"
python "${ROOT}/scripts/summarize_tensorboard.py" \
    --logdir "${TTS_EXP}/tensorboard" \
    --output "${ROOT}/reports/tensorboard_metrics_multispeaker.json"

mkdir -p "$MILESTONE_DIR"
cp --reflink=auto "$SOURCE_CHECKPOINT" "$MILESTONE_CHECKPOINT"
cmp --silent "$SOURCE_CHECKPOINT" "$MILESTONE_CHECKPOINT"

for label in 1k 5k "$MILESTONE_LABEL"; do
    "${ROOT}/scripts/evaluate_multispeaker_milestone.sh" "$label"
done

(
    cd "${ROOT}/.."
    python -m training.inference.synthesize \
        --text "$EXAMPLE_TEXT" \
        --output "${ROOT}/eval/generated/multispeaker_${MILESTONE_LABEL}_lada.wav" \
        --config "${TTS_EXP}/config.yaml" \
        --checkpoint "$MILESTONE_CHECKPOINT" \
        --speaker-embedding-ark "$SPEAKER_ARK" \
        --speaker lada
    python -m training.inference.synthesize \
        --text "$EXAMPLE_TEXT" \
        --output "${ROOT}/eval/generated/multispeaker_${MILESTONE_LABEL}_dmytro_zero_shot.wav" \
        --config "${TTS_EXP}/config.yaml" \
        --checkpoint "$MILESTONE_CHECKPOINT" \
        --speaker-embedding-ark "$SPEAKER_ARK" \
        --speaker dmytro
)

python "${ROOT}/scripts/build_listening_set.py" \
    --manifest "$MANIFEST" \
    --candidate \
        "jets_multispeaker_1k=${TTS_EXP}/decode_jets_milestone_1k/multispeaker_eval/wav" \
    --candidate \
        "jets_multispeaker_5k=${TTS_EXP}/decode_jets_milestone_5k/multispeaker_eval/wav" \
    --candidate "jets_multispeaker_${MILESTONE_LABEL}=${DECODE_DIR}/wav" \
    --balance-column source \
    --count 20 \
    --output \
        "${ROOT}/eval/generated/listening_multispeaker_${MILESTONE_LABEL}"

python "${ROOT}/scripts/package_multispeaker_release.py" \
    --checkpoint "$MILESTONE_CHECKPOINT" \
    --evaluation-report "$REPORT" \
    --milestone "$MILESTONE_LABEL" \
    --output "$RELEASE_DIR"

echo "Checkpoint: $MILESTONE_CHECKPOINT"
echo "Evaluation report: $REPORT"
echo "Listening set: ${ROOT}/eval/generated/listening_multispeaker_${MILESTONE_LABEL}"
echo "Release candidate: $RELEASE_DIR"
echo "The Dmytro example is zero-shot. Dmytro data was not in the train set."

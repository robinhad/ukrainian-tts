#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECKPOINT_NAME=${1:-25epoch.pth}
MILESTONE_LABEL=${2:-25k}
TTS_EXP="${ROOT}/exp_full_trimmed/tts_jets_uk_24k_trimmed"
SOURCE_CHECKPOINT="${TTS_EXP}/${CHECKPOINT_NAME}"
MILESTONE_DIR="${ROOT}/exp_full_trimmed/milestones"
MILESTONE_CHECKPOINT="${MILESTONE_DIR}/${MILESTONE_LABEL}.pth"
INFERENCE_MODEL="milestone_${MILESTONE_LABEL}.pth"
DECODE_DIR="${TTS_EXP}/decode_jets_milestone_${MILESTONE_LABEL}"
REPORT="${ROOT}/reports/full_trimmed_inference_${MILESTONE_LABEL}.json"

if [[ ! -s "${SOURCE_CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${SOURCE_CHECKPOINT}" >&2
    exit 2
fi

mkdir -p "${MILESTONE_DIR}"
cp --reflink=auto "${SOURCE_CHECKPOINT}" "${MILESTONE_CHECKPOINT}"
cmp --silent "${SOURCE_CHECKPOINT}" "${MILESTONE_CHECKPOINT}"
ln -sfn "../milestones/${MILESTONE_LABEL}.pth" "${TTS_EXP}/${INFERENCE_MODEL}"

DATASET_NAME=full_trimmed \
EXPERIMENT_NAME=tts_jets_uk_24k_trimmed \
DUMP_NAME=dump_full_trimmed \
EXP_NAME=exp_full_trimmed \
    "${ROOT}/scripts/run_milestone_inference.sh" "${INFERENCE_MODEL}"

source "${ROOT}/activate.sh"
python "${ROOT}/scripts/synthesize_eval.py" \
    --wav-dir "${DECODE_DIR}/eval/wav" \
    --manifest "${ROOT}/data/full_trimmed/manifests/eval.parquet" \
    --checkpoint "${MILESTONE_CHECKPOINT}" \
    --config "${TTS_EXP}/config.yaml" \
    --inference-log "${DECODE_DIR}/eval/log/tts_inference.1.log" \
    --output "${REPORT}"

(
    cd "${ROOT}/.."
    python -m training.inference.synthesize \
        --text "Український синтез мовлення працює офлайн." \
        --output "${ROOT}/eval/generated/example_trimmed_${MILESTONE_LABEL}.wav" \
        --config "${TTS_EXP}/config.yaml" \
        --checkpoint "${MILESTONE_CHECKPOINT}"
)

python "${ROOT}/scripts/build_listening_set.py" \
    --manifest "${ROOT}/data/full_trimmed/manifests/eval.parquet" \
    --raw-dir "${ROOT}/data/full/raw" \
    --candidate "jets_trimmed_${MILESTONE_LABEL}=${DECODE_DIR}/eval/wav" \
    --count 20 \
    --output "${ROOT}/eval/generated/listening_trimmed_${MILESTONE_LABEL}"

echo "Checkpoint: ${MILESTONE_CHECKPOINT}"
echo "Evaluation report: ${REPORT}"
echo "Listening set: ${ROOT}/eval/generated/listening_trimmed_${MILESTONE_LABEL}"

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MILESTONE_LABEL=${1:?Usage: evaluate_expanded_v3_milestone.sh LABEL}
if ! [[ "$MILESTONE_LABEL" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "The milestone label has invalid characters." >&2
    exit 2
fi

TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v3/tts_jets_uk_24k_expanded_v3_500k"}
MILESTONE_CHECKPOINT=${MILESTONE_CHECKPOINT:-"${TTS_EXP}/milestones/${MILESTONE_LABEL}.pth"}
INFERENCE_MODEL="milestone_${MILESTONE_LABEL}.pth"
DECODE_DIR="${TTS_EXP}/decode_jets_milestone_${MILESTONE_LABEL}/expanded_v3_eval"
MANIFEST="${ROOT}/data/expanded_v3/manifests/expanded_v3_eval.parquet"
REPORT=${REPORT:-"${ROOT}/reports/expanded_v3_inference_${MILESTONE_LABEL}.json"}
COMMAND_LOG="${ROOT}/reports/expanded_v3_${MILESTONE_LABEL}_evaluation_commands.jsonl"

if [[ ! -s "$MILESTONE_CHECKPOINT" ]]; then
    echo "The milestone checkpoint does not exist: $MILESTONE_CHECKPOINT" >&2
    exit 2
fi
if [[ ! -s "$MANIFEST" ]]; then
    echo "The evaluation manifest does not exist: $MANIFEST" >&2
    exit 2
fi

source "${ROOT}/activate.sh"
export GPU_COUNT=1
export MANIFEST_DIR="${ROOT}/data/expanded_v3/manifests"
export TRAIN_SET=expanded_v3_train
export VALID_SET=expanded_v3_dev
export TEST_SETS=expanded_v3_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_expanded_v3"
export EXP_DIR="${ROOT}/exp_expanded_v3"

ln -sfn "milestones/${MILESTONE_LABEL}.pth" "${TTS_EXP}/${INFERENCE_MODEL}"
python "${ROOT}/scripts/run_logged.py" \
    --log "$COMMAND_LOG" --eta-minutes 60 -- \
    "${ROOT}/espnet_recipe/run_expanded_v3.sh" \
    --stage 8 --stop_stage 8 \
    --tts_exp "$TTS_EXP" \
    --inference_model "$INFERENCE_MODEL"

python "${ROOT}/scripts/synthesize_eval.py" \
    --wav-dir "${DECODE_DIR}/wav" \
    --manifest "$MANIFEST" \
    --checkpoint "$MILESTONE_CHECKPOINT" \
    --config "${TTS_EXP}/config.yaml" \
    --inference-log "${DECODE_DIR}/log/tts_inference.1.log" \
    --output "$REPORT"

echo "Milestone: $MILESTONE_LABEL"
echo "Evaluation report: $REPORT"

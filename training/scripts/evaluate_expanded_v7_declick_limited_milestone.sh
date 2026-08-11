#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=expanded_v7_sidon_deess_declick_limit_novoa
LABEL=${1:?Set a milestone label.}
[[ "$LABEL" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_${NAME}/tts_jets_uk_24k_${NAME}_from_v6e94_100k"}
CHECKPOINT=${MILESTONE_CHECKPOINT:-"${TTS_EXP}/milestones/${LABEL}.pth"}
MODEL="milestone_${LABEL}.pth"
TEST_SET=${NAME}_eval
DECODE="${TTS_EXP}/decode_jets_milestone_${LABEL}/${TEST_SET}"
MANIFEST="${ROOT}/data/${NAME}/manifests/${TEST_SET}.parquet"
REPORT=${REPORT:-"${ROOT}/reports/${NAME}_inference_${LABEL}.json"}
[[ -s "$CHECKPOINT" && -s "$MANIFEST" ]] || {
    echo "The checkpoint or manifest is missing." >&2
    exit 2
}
source "${ROOT}/activate.sh"
export GPU_COUNT=1 MANIFEST_DIR="${ROOT}/data/${NAME}/manifests"
export TRAIN_SET=${NAME}_train VALID_SET=${NAME}_dev TEST_SETS="$TEST_SET"
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_${NAME}" EXP_DIR="${ROOT}/exp_${NAME}"
ln -sfn "milestones/${LABEL}.pth" "${TTS_EXP}/${MODEL}"
"${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" \
    --stage 8 --stop_stage 8 --tts_exp "$TTS_EXP" --inference_model "$MODEL"
python "${ROOT}/scripts/synthesize_eval.py" --wav-dir "${DECODE}/wav" \
    --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" \
    --config "${TTS_EXP}/config.yaml" \
    --inference-log "${DECODE}/log/tts_inference.1.log" --output "$REPORT"

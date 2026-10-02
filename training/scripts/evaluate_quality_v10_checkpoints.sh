#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}/.."
source "${ROOT}/activate.sh"
export UKTTS_ESPEAK_DATA_HASH_FILE="${ROOT}/quality_runs/v10/frontend/ESPEAK_NG_DATA_HASH"
export GPU_COUNT=1
export TRAIN_SET=${TRAIN_SET:-quality_v10_train} VALID_SET=${VALID_SET:-quality_v10_dev} TEST_SETS=${TEST_SETS:-quality_v10_eval}
export DATA_SETS="$TRAIN_SET $VALID_SET $TEST_SETS"
export MANIFEST_DIR=${MANIFEST_DIR:-"${ROOT}/data/quality_v10/manifests"}
export DUMP_DIR=${DUMP_DIR:-"${ROOT}/dump_quality_v10"} EXP_DIR=${EXP_DIR:-"${ROOT}/exp_quality_v10"}
TTS_EXP=${TTS_EXP:-"${EXP_DIR}/tts_jets_quality_v10_100k"}
QUALITY_ROOT=${QUALITY_ROOT:-"${ROOT}/quality_runs/v10/checkpoints"}
export UKTTS_QUALITY_PANEL=${UKTTS_QUALITY_PANEL:-"${ROOT}/quality_runs/v10/panels/heldout.jsonl"}
export UKTTS_QUALITY_CONFIG=${UKTTS_QUALITY_CONFIG:-"${ROOT}/conf/quality.yaml"}
export UKTTS_QUALITY_ORIGINAL="$QUALITY_ROOT/original"
python -m training.quality evaluate --panel "$UKTTS_QUALITY_PANEL" --label original \
  --output "$UKTTS_QUALITY_ORIGINAL" --config "$UKTTS_QUALITY_CONFIG" --resume
for LABEL in ${MILESTONES:-25k 50k 75k 100k}; do
  [[ "$LABEL" =~ ^[0-9]+k$ ]] || exit 2
  MODEL="milestone_${LABEL}.pth"
  CHECKPOINT="$TTS_EXP/milestones/${LABEL}.pth"
  [[ -s "$CHECKPOINT" ]] || exit 2
  ln -sfn "milestones/${LABEL}.pth" "$TTS_EXP/$MODEL"
  "${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" \
    --stage 8 --stop_stage 8 --tts_exp "$TTS_EXP" --inference_model "$MODEL"
  DECODE="$TTS_EXP/decode_jets_milestone_${LABEL}/${TEST_SETS}"
  python -m training.scripts.synthesize_eval --wav-dir "$DECODE/wav" \
    --manifest "$MANIFEST_DIR/${TEST_SETS}.parquet" --checkpoint "$CHECKPOINT" \
    --config "$TTS_EXP/config.yaml" --inference-log "$DECODE/log/tts_inference.1.log" \
    --output "$QUALITY_ROOT/${LABEL}.json"
  export UKTTS_QUALITY_PREVIOUS="$QUALITY_ROOT/${LABEL}_quality"
done

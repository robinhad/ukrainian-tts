#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}/.."
source "${ROOT}/activate.sh"
export OMP_NUM_THREADS=${PROCESS_CPU_THREADS:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
python -m training.scripts.verify_local_frontend
export UKTTS_ESPEAK_DATA_HASH_FILE="${ROOT}/quality_runs/v10/frontend/ESPEAK_NG_DATA_HASH"
python -m training.scripts.prepare_ecapa_for_training
export GPU_COUNT=1
export TRAIN_SET=quality_v10_train VALID_SET=quality_v10_dev TEST_SETS=quality_v10_eval
export DATA_SETS="$TRAIN_SET $VALID_SET $TEST_SETS"
export MANIFEST_DIR="${ROOT}/data/quality_v10/manifests"
export DUMP_DIR="${ROOT}/dump_quality_v10" EXP_DIR="${ROOT}/exp_quality_v10"
PROFILE=${PROFILE:?Set PROFILE to the reviewed processing profile YAML}
RAW_MANIFEST=${RAW_MANIFEST:-"${ROOT}/data/quality_v10_raw/all.parquet"}
python -m training.quality process --panel "$RAW_MANIFEST" --profile "$PROFILE" \
  --output "${ROOT}/data/quality_v10/audio_24k" --resume \
  --model-workers "${PROCESS_MODEL_WORKERS:-1}" --cpu-workers "${PROCESS_CPU_WORKERS:-2}"
python -m training.scripts.build_quality_v10_manifest --manifest "$RAW_MANIFEST" \
  --processed "${ROOT}/data/quality_v10/audio_24k" --profile "$PROFILE" \
  --output "${ROOT}/data/quality_v10/records.jsonl"
python -m training.scripts.build_manifest --records "${ROOT}/data/quality_v10/records.jsonl" \
  --output-dir "$MANIFEST_DIR" --cache "${ROOT}/data/quality_v10/frontend.sqlite" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
python -m training.scripts.validate_dataset --manifest "$MANIFEST_DIR/all.parquet" \
  --report "${ROOT}/quality_runs/v10/dataset_validation.json"
RECIPE="${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh"
"$RECIPE" --stage 1 --stop_stage 1
python -m training.scripts.prepare_direct_pcm24_raw_data --manifest-dir "$MANIFEST_DIR" \
  --kaldi-data-root "${ROOT}/espnet_recipe/data" --dump-dir "$DUMP_DIR" \
  --train-set "$TRAIN_SET" --valid-set "$VALID_SET" --test-set "$TEST_SETS"
MANIFEST="$MANIFEST_DIR/all.parquet" RAW_MANIFEST="$RAW_MANIFEST" \
  KALDI_ROOT="${ROOT}/data/quality_v10/hybrid" DUMP_DIR="$DUMP_DIR" \
  GPU_UUIDS="${CUDA_VISIBLE_DEVICES:-0}" \
  REPORT="${ROOT}/quality_runs/v10/embeddings.json" \
  OUTPUT_MANIFEST="${ROOT}/data/quality_v10/hybrid_manifest/all.parquet" \
  "${ROOT}/scripts/extract_hybrid_embeddings.sh"
# The recipe automatically consumes cached pitch/energy in stage 7. Avoid
# repeating CPU pitch extraction in every optimizer iteration.
"$RECIPE" --stage 4 --stop_stage 6 --nj "${SLURM_CPUS_PER_TASK:-8}" \
  --write_collected_feats true

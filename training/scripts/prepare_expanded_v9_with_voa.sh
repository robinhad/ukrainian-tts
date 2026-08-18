#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=expanded_v9_cascade_with_voa
DATA_ROOT="${ROOT}/data/${NAME}"
DUMP_DIR="${ROOT}/dump_${NAME}"
EXP_DIR="${ROOT}/exp_${NAME}"
MANIFEST_DIR="${DATA_ROOT}/manifests"
RAW_MANIFEST="${DATA_ROOT}/pre_enhancement_embedding_manifest.parquet"
KALDI_ROOT="${DATA_ROOT}/hybrid_embedding_data"
HYBRID_MANIFEST="${DATA_ROOT}/hybrid_manifest/all.parquet"
RESULT_ROOT="${ROOT}/reports/${NAME}_processing"
REPORTS="${ROOT}/reports"
PREFIX=${NAME}_full
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}

mapfile -t RESULTS < <(find "$RESULT_ROOT" -maxdepth 1 -name 'shard-*.jsonl' -type f | sort -V)
if (( ${#RESULTS[@]} < 2 )); then
    echo "The v9 processing result shards are missing." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
source "${ROOT}/activate.sh"
python "${ROOT}/scripts/check_resources.py" --mode full --require-torch \
    --gpu-uuids "$GPU_UUIDS" --workspace "$ROOT" \
    --output "${REPORTS}/resource_usage_${NAME}.jsonl"
python "${ROOT}/scripts/build_expanded_v9_with_voa.py" \
    --base-manifest "${ROOT}/data/expanded_v3/manifests/all.parquet" \
    --raw-manifest "${ROOT}/data/expanded_v3/source_manifests/all.parquet" \
    --processing-results "${RESULTS[@]}" --output-dir "$MANIFEST_DIR" \
    --output-raw-manifest "$RAW_MANIFEST" \
    --report "${REPORTS}/${PREFIX}_dataset.json"
python "${ROOT}/scripts/validate_dataset.py" --manifest "${MANIFEST_DIR}/all.parquet" \
    --report "${REPORTS}/${PREFIX}_validation.json"
python "${ROOT}/scripts/analyze_dataset.py" --manifest "${MANIFEST_DIR}/all.parquet" \
    --output "${REPORTS}/${PREFIX}_analysis.json"

export MANIFEST_DIR DUMP_DIR EXP_DIR
export TRAIN_SET=${NAME}_train
export VALID_SET=${NAME}_dev
export TEST_SETS=${NAME}_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export GPU_COUNT=2
"${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" --stage 1 --stop_stage 1
python "${ROOT}/scripts/prepare_direct_pcm24_raw_data.py" \
    --manifest-dir "$MANIFEST_DIR" \
    --kaldi-data-root "${ROOT}/espnet_recipe/data" --dump-dir "$DUMP_DIR" \
    --train-set "$TRAIN_SET" --valid-set "$VALID_SET" --test-set "$TEST_SETS"

MANIFEST="${MANIFEST_DIR}/all.parquet" RAW_MANIFEST="$RAW_MANIFEST" \
    KALDI_ROOT="$KALDI_ROOT" DUMP_DIR="$DUMP_DIR" \
    REPORT="${REPORTS}/${PREFIX}_hybrid_embeddings.json" \
    OUTPUT_MANIFEST="$HYBRID_MANIFEST" GPU_UUIDS="$GPU_UUIDS" \
    "${ROOT}/scripts/extract_hybrid_embeddings.sh"

"${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" \
    --stage 4 --stop_stage 4 --nj 10
"${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" \
    --stage 5 --stop_stage 5
"${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" \
    --stage 6 --stop_stage 6 --nj "${STATS_NJ:-20}"
python "${ROOT}/scripts/audit_expanded_v9_with_voa_readiness.py" \
    --workspace "$ROOT" --data-root "$DATA_ROOT" --dump-dir "$DUMP_DIR" \
    --exp-dir "$EXP_DIR" --reports-root "$REPORTS" \
    --output "${REPORTS}/${PREFIX}_scale_readiness.json"
echo "The VOA-inclusive v9 preparation is complete."

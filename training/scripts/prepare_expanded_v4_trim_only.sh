#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-smoke}
if [[ "$MODE" != smoke && "$MODE" != full ]]; then
    echo "Usage: prepare_expanded_v4_trim_only.sh [smoke|full]" >&2
    exit 2
fi
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
DATA_ROOT="${ROOT}/data/expanded_v4_trim_only"
DUMP_DIR="${ROOT}/dump_expanded_v4_trim_only"
EXP_DIR="${ROOT}/exp_expanded_v4_trim_only"
if [[ "$MODE" == smoke ]]; then
    DATA_ROOT="${ROOT}/data/expanded_v4_trim_only_smoke"
    DUMP_DIR="${ROOT}/dump_expanded_v4_trim_only_smoke"
    EXP_DIR="${ROOT}/exp_expanded_v4_trim_only_smoke"
fi
PREFIX="expanded_v4_trim_only_${MODE}"
MANIFEST_DIR="${DATA_ROOT}/manifests"
RAW_MANIFEST="${DATA_ROOT}/raw_embedding_manifest.parquet"
KALDI_ROOT="${DATA_ROOT}/hybrid_embedding_data"
REPORTS="${ROOT}/reports"
TOKEN_DIR="${DUMP_DIR}/token_list/phn_espeak_ng_ukrainian"
REFERENCE_TOKENS="${ROOT}/dump_expanded_v3/token_list/phn_espeak_ng_ukrainian/tokens.txt"

export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
source "${ROOT}/activate.sh"
python "${ROOT}/scripts/check_resources.py" \
    --mode "$MODE" --require-torch --gpu-uuids "$GPU_UUIDS" \
    --workspace "$ROOT" --output "${REPORTS}/resource_usage_expanded_v4_trim_only.jsonl"
python "${ROOT}/scripts/build_expanded_v4_trim_only.py" \
    --active-manifest "${ROOT}/data/expanded_v3/manifests/all.parquet" \
    --canonical-manifest "${ROOT}/data/expanded_v3/source_manifests/all.parquet" \
    --output-dir "$MANIFEST_DIR" --raw-manifest "$RAW_MANIFEST" \
    --report "${REPORTS}/${PREFIX}_dataset.json" --mode "$MODE"
python "${ROOT}/scripts/validate_dataset.py" \
    --manifest "${MANIFEST_DIR}/all.parquet" \
    --report "${REPORTS}/${PREFIX}_validation.json"
python "${ROOT}/scripts/analyze_dataset.py" \
    --manifest "${MANIFEST_DIR}/all.parquet" \
    --output "${REPORTS}/${PREFIX}_analysis.json"

export MANIFEST_DIR DUMP_DIR EXP_DIR
export TRAIN_SET=expanded_v4_trim_only_train
export VALID_SET=expanded_v4_trim_only_dev
export TEST_SETS=expanded_v4_trim_only_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export GPU_COUNT=2
"${ROOT}/espnet_recipe/run_expanded_v4_trim_only.sh" --stage 1 --stop_stage 2

MANIFEST="${MANIFEST_DIR}/all.parquet" \
RAW_MANIFEST="$RAW_MANIFEST" \
OUTPUT_MANIFEST="${DATA_ROOT}/hybrid_manifest/all.parquet" \
KALDI_ROOT="$KALDI_ROOT" DUMP_DIR="$DUMP_DIR" \
REPORT="${REPORTS}/${PREFIX}_hybrid_embeddings.json" \
GPU_UUIDS="$GPU_UUIDS" \
    "${ROOT}/scripts/extract_hybrid_embeddings.sh"

"${ROOT}/espnet_recipe/run_expanded_v4_trim_only.sh" --stage 4 --stop_stage 4 --nj 10
mkdir -p "$TOKEN_DIR"
cp "$REFERENCE_TOKENS" "${TOKEN_DIR}/tokens.txt"
"${ROOT}/espnet_recipe/run_expanded_v4_trim_only.sh" --stage 6 --stop_stage 6 --nj 10
python "${ROOT}/scripts/audit_expanded_v4_trim_only_readiness.py" \
    --workspace "$ROOT" --data-root "$DATA_ROOT" --dump-dir "$DUMP_DIR" \
    --exp-dir "$EXP_DIR" --reports-root "$REPORTS" --mode "$MODE" \
    --reference-token-list "$REFERENCE_TOKENS" \
    --output "${REPORTS}/${PREFIX}_scale_readiness.json"
if [[ "$MODE" == full ]]; then
    python "${ROOT}/scripts/create_trim_only_cleanup_marker.py" \
        --hybrid-report "${REPORTS}/${PREFIX}_hybrid_embeddings.json" \
        --source-root "${ROOT}/data/expanded_v3/sources" \
        --output "${REPORTS}/expanded_v4_trim_only_cleanup_eligible.json"
fi
echo "The expanded-v4 trim-only ${MODE} preparation is complete."

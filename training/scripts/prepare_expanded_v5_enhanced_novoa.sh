#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
DATA_ROOT="${ROOT}/data/expanded_v5_enhanced_novoa"
DUMP_DIR="${ROOT}/dump_expanded_v5_enhanced_novoa"
EXP_DIR="${ROOT}/exp_expanded_v5_enhanced_novoa"
MANIFEST_DIR="${DATA_ROOT}/manifests"
RAW_MANIFEST="${DATA_ROOT}/pre_enhancement_embedding_manifest.parquet"
KALDI_ROOT="${DATA_ROOT}/hybrid_embedding_data"
HYBRID_MANIFEST="${DATA_ROOT}/hybrid_manifest/all.parquet"
REPORTS="${ROOT}/reports"
TOKEN_DIR="${DUMP_DIR}/token_list/phn_espeak_ng_ukrainian"
REFERENCE_TOKENS="${ROOT}/dump_expanded_v3/token_list/phn_espeak_ng_ukrainian/tokens.txt"
PREFIX=expanded_v5_enhanced_novoa_full

export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
source "${ROOT}/activate.sh"
python "${ROOT}/scripts/check_resources.py" \
    --mode full --require-torch --gpu-uuids "$GPU_UUIDS" \
    --workspace "$ROOT" --output "${REPORTS}/resource_usage_expanded_v5_enhanced_novoa.jsonl"
python "${ROOT}/scripts/build_expanded_v5_enhanced_novoa.py" \
    --enhanced-manifest "${ROOT}/data/expanded_v3/manifests/all.parquet" \
    --pre-enhancement-manifest "${ROOT}/data/expanded_v3/source_manifests/all.parquet" \
    --output-dir "$MANIFEST_DIR" --raw-manifest "$RAW_MANIFEST" \
    --report "${REPORTS}/${PREFIX}_dataset.json" --mode full
python "${ROOT}/scripts/validate_dataset.py" \
    --manifest "${MANIFEST_DIR}/all.parquet" \
    --report "${REPORTS}/${PREFIX}_validation.json"
python "${ROOT}/scripts/analyze_dataset.py" \
    --manifest "${MANIFEST_DIR}/all.parquet" \
    --output "${REPORTS}/${PREFIX}_analysis.json"

export MANIFEST_DIR DUMP_DIR EXP_DIR
export TRAIN_SET=expanded_v5_enhanced_novoa_train
export VALID_SET=expanded_v5_enhanced_novoa_dev
export TEST_SETS=expanded_v5_enhanced_novoa_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export GPU_COUNT=2
"${ROOT}/espnet_recipe/run_expanded_v5_enhanced_novoa.sh" --stage 1 --stop_stage 2

python "${ROOT}/scripts/prepare_hybrid_embeddings.py" \
    --manifest "${MANIFEST_DIR}/all.parquet" \
    --raw-manifest "$RAW_MANIFEST" \
    --output-manifest "$HYBRID_MANIFEST" \
    --kaldi-root "$KALDI_ROOT" \
    --report "${REPORTS}/${PREFIX}_hybrid_assignment.json"
python "${ROOT}/scripts/reuse_expanded_v5_embeddings.py" \
    --hybrid-manifest "$HYBRID_MANIFEST" \
    --v3-hybrid-manifest "${ROOT}/data/expanded_v3/hybrid_manifest/all.parquet" \
    --v3-xvector-root "${ROOT}/dump_expanded_v3/xvector" \
    --v4-hybrid-manifest "${ROOT}/data/expanded_v4_trim_only/hybrid_manifest/all.parquet" \
    --v4-xvector-root "${ROOT}/dump_expanded_v4_trim_only/xvector" \
    --output-root "${DUMP_DIR}/xvector" \
    --report "${REPORTS}/${PREFIX}_hybrid_embeddings.json"

"${ROOT}/espnet_recipe/run_expanded_v5_enhanced_novoa.sh" --stage 4 --stop_stage 4 --nj 10
mkdir -p "$TOKEN_DIR"
cp "$REFERENCE_TOKENS" "${TOKEN_DIR}/tokens.txt"
"${ROOT}/espnet_recipe/run_expanded_v5_enhanced_novoa.sh" \
    --stage 6 --stop_stage 6 --nj "${STATS_NJ:-20}"
python "${ROOT}/scripts/audit_expanded_v5_enhanced_novoa_readiness.py" \
    --workspace "$ROOT" --data-root "$DATA_ROOT" --dump-dir "$DUMP_DIR" \
    --exp-dir "$EXP_DIR" --reports-root "$REPORTS" \
    --output "${REPORTS}/${PREFIX}_scale_readiness.json"
echo "The enhanced non-VOA full preparation is complete."

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPOSITORY_ROOT=$(cd "${ROOT}/.." && pwd)
OUTPUT=${OUTPUT:-"${ROOT}/eval/generated/random_training_enhancement_comparison_v1"}
REPORT=${REPORT:-"${ROOT}/reports/random_training_enhancement_comparison_v1.json"}
MANIFEST=${MANIFEST:-"${ROOT}/data/expanded_v4_trim_only/manifests/all.parquet"}
CACHE=${CACHE:-"${ROOT}/vendor/audio-enhancement-review-models"}
RUNNER="${ROOT}/scripts/run_fair_enhancement_comparison.py"

cd "$REPOSITORY_ROOT"
if [[ -e "$OUTPUT" ]]; then
    echo "The comparison output already exists: ${OUTPUT}" >&2
    exit 2
fi
"${ROOT}/.venv/bin/python" "$RUNNER" select \
    --manifest "$MANIFEST" --output "$OUTPUT" --count 10

"${ROOT}/.venv/bin/python" "$RUNNER" process \
    --backend rnnoise85 --selection "$OUTPUT/selection.jsonl" \
    --output "$OUTPUT" --model-cache "$CACHE/rnnoise"
"${ROOT}/.venv/bin/python" "$RUNNER" process \
    --backend deepfilternet3 --selection "$OUTPUT/selection.jsonl" \
    --output "$OUTPUT" --model-cache "${ROOT}/vendor/deepfilternet-cache"

CUDA_VISIBLE_DEVICES=0 "${ROOT}/.venv-audio-review/bin/python" "$RUNNER" process \
    --backend resemble_enhance --selection "$OUTPUT/selection.jsonl" \
    --output "$OUTPUT" --model-cache "$CACHE/resemble" --device cuda:0 &
resemble_pid=$!
CUDA_VISIBLE_DEVICES=1 "${ROOT}/.venv-audio-review/bin/python" "$RUNNER" process \
    --backend clearervoice --selection "$OUTPUT/selection.jsonl" \
    --output "$OUTPUT" --model-cache "$CACHE/mossformer2" --device cuda:0 &
clearvoice_pid=$!
wait "$resemble_pid"
wait "$clearvoice_pid"

"${ROOT}/.venv/bin/python" "$RUNNER" finalize \
    --selection "$OUTPUT/selection.jsonl" --output "$OUTPUT" --report "$REPORT"

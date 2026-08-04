#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXP="${ROOT}/exp_expanded_v3/tts_jets_uk_24k_expanded_v3_500k"
CHECKPOINT="${EXP}/milestones/500k.pth"
STAGE_FILE="${ROOT}/reports/expanded_v3_500k_stage.txt"
FINAL_LOG="${ROOT}/reports/expanded_v3_500k_finalization.log"
CHECKSUMS="${ROOT}/reports/expanded_v3_500k_artifacts.sha256"

if [[ ! -s "$STAGE_FILE" ]] || ! grep -qE '\| COMPLETE$' "$STAGE_FILE"; then
    echo "The training chain is not complete." >&2
    exit 2
fi
if [[ ! -s "$CHECKPOINT" ]]; then
    echo "The 500K checkpoint does not exist: $CHECKPOINT" >&2
    exit 2
fi
if tail -n 50000 "${EXP}/train.log" | grep -Eiq \
    '(^|[^[:alpha:]])nan([^[:alpha:]]|$)|out of memory|runtimeerror|traceback|nccl.*error'; then
    echo "The training log contains a critical error." >&2
    exit 1
fi

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
if (( ${#GPU_UUIDS[@]} < 2 )); then
    echo "The final evaluation requires two GPUs." >&2
    exit 2
fi

echo "Start the fixed evaluation on GPU 0." | tee "$FINAL_LOG"
CUDA_VISIBLE_DEVICES="${GPU_UUIDS[0]}" \
    "${ROOT}/scripts/evaluate_expanded_v3_milestone.sh" 500k \
    >>"$FINAL_LOG" 2>&1 &
fixed_pid=$!

echo "Start the five-voice evaluation on GPU 1." | tee -a "$FINAL_LOG"
CUDA_VISIBLE_DEVICES="${GPU_UUIDS[1]}" TARGET_ITERATIONS=500000 \
    "${ROOT}/scripts/generate_five_voice_expanded_v3_eval.sh" \
    >>"$FINAL_LOG" 2>&1 &
voices_pid=$!

set +e
wait "$fixed_pid"
fixed_status=$?
wait "$voices_pid"
voices_status=$?
set -e
if (( fixed_status != 0 || voices_status != 0 )); then
    echo "Final evaluation failed. Fixed: ${fixed_status}; voices: ${voices_status}." \
        | tee -a "$FINAL_LOG" >&2
    exit 1
fi

artifacts=(
    "$CHECKPOINT"
    "${EXP}/config.yaml"
    "${ROOT}/reports/expanded_v3_inference_500k.json"
    "${ROOT}/reports/five_voice_expanded_v3_500k.json"
)
while IFS= read -r wav; do
    artifacts+=("$wav")
done < <(find "${ROOT}/eval/generated/five_voice_expanded_v3_500k" \
    -maxdepth 1 -name '*.wav' -type f | sort)

sha256sum "${artifacts[@]}" >"$CHECKSUMS"
echo "Final evaluation is complete." | tee -a "$FINAL_LOG"
echo "Checksums: $CHECKSUMS" | tee -a "$FINAL_LOG"

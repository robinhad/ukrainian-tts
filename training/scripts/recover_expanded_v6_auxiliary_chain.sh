#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_PID=${1:?Set the active v6 training PID.}
TARGET=${2:-100000}
TTS_EXP=${3:-"${ROOT}/exp_expanded_v6_sidon_deess_novoa/tts_jets_uk_24k_expanded_v6_sidon_deess_novoa_from_v5e81_100k"}
PYTHON="${ROOT}/.venv/bin/python"
if ! [[ "$TRAIN_PID" =~ ^[0-9]+$ ]] || (( TRAIN_PID <= 1 )); then
    echo "The training PID is not safe." >&2
    exit 2
fi

"$PYTHON" "${ROOT}/scripts/monitor_expanded_v5_disk.py" \
    --watch-pid "$TRAIN_PID" --workspace "$ROOT" \
    --status "${ROOT}/reports/expanded_v6_sidon_deess_novoa_disk.jsonl" \
    --trigger-gib 30 --interval-seconds 60 --stop-pgid "$TRAIN_PID" --report-only &
disk_pid=$!
"$PYTHON" "${ROOT}/scripts/preserve_validation_best_checkpoints.py" \
    --log "${TTS_EXP}/train.log" --exp-dir "$TTS_EXP" --keep 1 \
    --target-epoch "$((TARGET / 1000))" --poll-seconds 30 &
best_pid=$!

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
done
wait "$disk_pid"
wait "$best_pid"

deadline=$((SECONDS + 600))
while [[ ! -s "${TTS_EXP}/milestones/100k.pth" ]] && (( SECONDS < deadline )); do
    sleep 10
done
if [[ ! -s "${TTS_EXP}/milestones/100k.pth" ]]; then
    echo "The 100K milestone does not exist after training." >&2
    exit 1
fi
if grep -Eiq 'Traceback|out of memory|NaN|RuntimeError' "${TTS_EXP}/train.log"; then
    echo "The training log has a critical error pattern." >&2
    exit 1
fi
for label in 25k 50k 75k 100k; do
    TTS_EXP="$TTS_EXP" "${ROOT}/scripts/evaluate_expanded_v6_sidon_deess_novoa_milestone.sh" "$label"
done
TTS_EXP="$TTS_EXP" "${ROOT}/scripts/generate_five_voice_expanded_v6_eval.sh" 100k
echo "The recovered v6 evaluation chain is complete."

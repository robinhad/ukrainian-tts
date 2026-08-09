#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET=${1:-100000}
TTS_EXP=${TTS_EXP:-"${ROOT}/exp_expanded_v6_sidon_deess_novoa/tts_jets_uk_24k_expanded_v6_sidon_deess_novoa_from_v5e81_100k"}
LOG="${ROOT}/reports/expanded_v6_sidon_deess_novoa_100k_launcher.log"
STATUS="${ROOT}/reports/training_status_expanded_v6_sidon_deess_novoa_100k.jsonl"
setsid env TTS_EXP="$TTS_EXP" ALLOW_TRAINING_RESUME=1 "${ROOT}/scripts/run_expanded_v6_sidon_deess_novoa_training.sh" "$TARGET" >"$LOG" 2>&1 &
train_pid=$!
STATUS="$STATUS" INTERVAL_SECONDS=900 FREE_DISK_STOP_GIB=30 STOP_ON_CRITICAL=0 TRAIN_PGID="$train_pid" \
    "${ROOT}/scripts/monitor_expanded_v3_training.sh" "$train_pid" "$TARGET" "$TTS_EXP" & monitor=$!
"${ROOT}/scripts/preserve_expanded_v5_milestones.sh" "$train_pid" "$TARGET" "$TTS_EXP" & milestones=$!
python "${ROOT}/scripts/monitor_expanded_v5_disk.py" --watch-pid "$train_pid" --workspace "$ROOT" \
    --status "${ROOT}/reports/expanded_v6_sidon_deess_novoa_disk.jsonl" --trigger-gib 30 --interval-seconds 60 --stop-pgid "$train_pid" --report-only & disk=$!
(
  while kill -0 "$train_pid" 2>/dev/null && [[ ! -s "${TTS_EXP}/train.log" ]]; do sleep 5; done
  [[ -s "${TTS_EXP}/train.log" ]] && python "${ROOT}/scripts/preserve_validation_best_checkpoints.py" --log "${TTS_EXP}/train.log" --exp-dir "$TTS_EXP" --keep 1 --target-epoch "$((TARGET / 1000))" --poll-seconds 30
) & best=$!
set +e; wait "$train_pid"; a=$?; wait "$monitor"; b=$?; wait "$milestones"; c=$?; wait "$disk"; d=$?; wait "$best"; e=$?; set -e
(( a || b || c || d || e )) && { echo "Training chain failed: $a $b $c $d $e" >&2; exit 1; }
for label in 25k 50k 75k 100k; do
    [[ -s "${TTS_EXP}/milestones/${label}.pth" ]] && TTS_EXP="$TTS_EXP" "${ROOT}/scripts/evaluate_expanded_v6_sidon_deess_novoa_milestone.sh" "$label"
done
[[ -s "${TTS_EXP}/milestones/100k.pth" ]] && TTS_EXP="$TTS_EXP" "${ROOT}/scripts/generate_five_voice_expanded_v6_eval.sh" 100k
echo "The v6 100K training chain is complete."

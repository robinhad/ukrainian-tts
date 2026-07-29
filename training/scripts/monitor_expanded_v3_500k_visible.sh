#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WATCH_PID=${1:?Usage: monitor_expanded_v3_500k_visible.sh WATCH_PID}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-900}
STAGE_FILE="${ROOT}/reports/expanded_v3_500k_stage.txt"
CHAIN_LOG="${ROOT}/reports/expanded_v3_500k_chain.log"
VOA_LOG="${ROOT}/reports/expanded_v3_voa_v4_reprocess.log"
TRAIN_LOG="${ROOT}/exp_expanded_v3/tts_jets_uk_24k_expanded_v3_500k/train.log"

if (( INTERVAL_SECONDS < 60 || INTERVAL_SECONDS > 1800 )); then
    echo "INTERVAL_SECONDS must be from 60 to 1800." >&2
    exit 2
fi

print_status() {
    echo "============================================================"
    TZ=Europe/Kyiv date '+status_time_kyiv=%Y-%m-%d %H:%M:%S %Z'
    echo "stage=$(cut -d'|' -f2- "$STAGE_FILE" 2>/dev/null | xargs || echo UNKNOWN)"
    if [[ -s "$VOA_LOG" ]]; then
        echo "voa_$(grep 'ETA' "$VOA_LOG" | tail -n 1 || true)"
    fi
    if [[ -s "$CHAIN_LOG" ]]; then
        echo "chain_$(grep 'ETA' "$CHAIN_LOG" | tail -n 1 || true)"
    fi
    "${ROOT}/.venv/bin/python" - "$ROOT" <<'PY'
import collections
import glob
import json
import os
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
progress = {}
pattern = root / (
    "data/expanded_v3/sources/voa_ukr_user_grant/batches/"
    "process-v4-c050-d20-g050-defer-long-*.progress.jsonl"
)
for name in glob.glob(str(pattern)):
    for line in Path(name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            progress[item["source_key"]] = item
rejected = collections.Counter()
accepted = 0
deferred = 0
for item in progress.values():
    accepted += int(item.get("accepted_segments", 0))
    deferred += int(item.get("deferred_too_long_segments", 0))
    rejected.update(item.get("rejected", {}))

def record_metrics(pattern: Path) -> tuple[int, float]:
    records = {}
    for name in glob.glob(str(pattern)):
        for line in Path(name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                records[item["utterance_id"]] = item
    hours = sum(float(item.get("duration", 0)) for item in records.values()) / 3600
    return len(records), hours

pseudo = root / "data/expanded_v3/sources/pseudo_uk_v4-c050-d20-g050-defer-long"
accepted_records, accepted_hours = record_metrics(
    pseudo / "records-gpu*.jsonl"
)
deferred_records, deferred_hours = record_metrics(
    pseudo / "deferred_too_long/records-gpu*.jsonl"
)
recovered_records, recovered_hours = record_metrics(
    pseudo / "recovered_long/records-gpu*.jsonl"
)
enhanced = {}
for name in glob.glob(
    str(root / "data/expanded_v3/enhanced_records/records-*.jsonl")
):
    for line in Path(name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            enhanced[item["utterance_id"]] = item
enhancement_status = collections.Counter(
    item.get("enhancement_status", "unknown")
    for item in enhanced.values()
)
enhancement_attempts = collections.Counter(
    int(item.get("enhancement_attempts", 0))
    for item in enhanced.values()
)
enhancement_errors = collections.Counter(
    str(item.get("enhancement_error", "unknown")).split(":", 1)[0]
    for item in enhanced.values()
    if item.get("enhancement_status") == "failed"
)
enhancement_total = 218980
usage = shutil.disk_usage(root)
print(json.dumps({
    "preparation": {
        "enhancement_attempts": dict(sorted(enhancement_attempts.items())),
        "enhancement_error_types": dict(sorted(enhancement_errors.items())),
        "enhancement_progress_percent": round(
            100 * len(enhanced) / enhancement_total,
            2,
        ),
        "enhancement_records": len(enhanced),
        "enhancement_status": dict(sorted(enhancement_status.items())),
        "enhancement_total": enhancement_total,
        "includes_recovered_long": (
            os.environ.get("TRAINING_INCLUDE_RECOVERED_LONG", "1") == "1"
        ),
    },
    "voa": {
        "accepted_hours": round(accepted_hours, 3),
        "accepted_records": accepted_records,
        "deferred_long_hours": round(deferred_hours, 3),
        "deferred_long_records": deferred_records,
        "processed_source_files": len(progress),
        "progress_percent": round(100 * len(progress) / 20505, 2),
        "recovered_long_hours": round(recovered_hours, 3),
        "recovered_long_records": recovered_records,
        "rejected": dict(sorted(rejected.items())),
    },
    "free_disk_gib": round(usage.free / 1024**3, 2),
}, sort_keys=True))
PY
    nvidia-smi \
        --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu \
        --format=csv,noheader,nounits
    if [[ -s "$TRAIN_LOG" ]]; then
        "${ROOT}/.venv/bin/python" "${ROOT}/scripts/training_status.py" \
            --log "$TRAIN_LOG" \
            --target-iterations 500000 \
            --workspace "$ROOT" \
            --free-disk-stop-gib 0 \
            --maximum-temperature-c 100 || true
    fi
}

while kill -0 "$WATCH_PID" 2>/dev/null; do
    print_status
    sleep "$INTERVAL_SECONDS"
done
print_status
echo "monitor_state=COMPLETE"

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
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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

stats_root = (
    root
    / "exp_expanded_v3/tts_stats_raw_phn_espeak_ng_ukrainian/logdir"
)
stats_jobs = []
for log_path in sorted(stats_root.glob("stats.[0-9]*.log")):
    match = re.fullmatch(r"stats\.(\d+)\.log", log_path.name)
    if match is None:
        continue
    job = int(match.group(1))
    shape_path = stats_root / f"train.{job}.scp"
    total = (
        sum(1 for line in shape_path.open(encoding="utf-8") if line.strip())
        if shape_path.is_file()
        else 0
    )
    points = []
    text_value = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text_value.splitlines():
        point = re.search(
            r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*Niter: (\d+)",
            line,
        )
        if point is not None:
            points.append(
                (
                    datetime.strptime(point.group(1), "%Y-%m-%d %H:%M:%S"),
                    int(point.group(2)),
                )
            )
    current = points[-1][1] if points else 0
    rate = None
    eta_seconds = None
    if len(points) >= 2 and points[-1][0] > points[0][0]:
        rate = (points[-1][1] - points[0][1]) / (
            points[-1][0] - points[0][0]
        ).total_seconds()
        if rate > 0 and total >= current:
            eta_seconds = (total - current) / rate
    stats_jobs.append(
        {
            "current": current,
            "error": (
                "Traceback" in text_value
                or "Ended (code 1)" in text_value
                or "Keys are mismatched" in text_value
            ),
            "eta_seconds": eta_seconds,
            "job": job,
            "rate_iterations_per_second": (
                round(rate, 4) if rate is not None else None
            ),
            "total": total,
        }
    )
known_etas = [
    item["eta_seconds"]
    for item in stats_jobs
    if item["eta_seconds"] is not None
]
stats_eta_seconds = max(known_etas) if known_etas else None
stats_eta_kyiv = None
if stats_eta_seconds is not None:
    stats_eta_kyiv = (
        datetime.now(ZoneInfo("Europe/Kyiv"))
        + timedelta(seconds=stats_eta_seconds)
    ).isoformat(timespec="seconds")
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
    "statistics": {
        "completed_iterations": sum(item["current"] for item in stats_jobs),
        "error_jobs": [
            item["job"] for item in stats_jobs if item["error"]
        ],
        "estimated_finish_kyiv": stats_eta_kyiv,
        "jobs": stats_jobs,
        "progress_percent": (
            round(
                100
                * sum(item["current"] for item in stats_jobs)
                / sum(item["total"] for item in stats_jobs),
                2,
            )
            if stats_jobs and sum(item["total"] for item in stats_jobs)
            else 0.0
        ),
        "total_iterations": sum(item["total"] for item in stats_jobs),
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

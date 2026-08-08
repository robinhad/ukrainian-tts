#!/usr/bin/env python3
"""Report v5 statistics progress and a Kyiv ETA."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


NITER = re.compile(r"INFO: Niter: (\d+)")
LOG_TIMESTAMP = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")
KYIV = ZoneInfo("Europe/Kyiv")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def completed_iterations(text: str) -> int:
    """Count logged iterations, including a counter reset between splits."""
    values = [int(value) for value in NITER.findall(text)]
    if not values:
        return 0
    completed = 0
    previous = values[0]
    for value in values[1:]:
        if value < previous:
            completed += previous
        previous = value
    return completed + previous


def first_log_time(text: str) -> float | None:
    """Return the first ESPnet timestamp as a Unix value in Kyiv time."""
    match = LOG_TIMESTAMP.search(text)
    if match is None:
        return None
    value = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    return value.replace(tzinfo=KYIV).timestamp()


def append(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def sample(logdir: Path, workspace: Path, total: int) -> dict[str, object]:
    logs = sorted(logdir.glob("stats.*.log"))
    texts = [path.read_text(encoding="utf-8", errors="replace") for path in logs]
    progress = sum(completed_iterations(text) for text in texts)
    now_unix = time.time()
    start_times = [value for text in texts if (value := first_log_time(text))]
    start_unix = min(start_times, default=now_unix)
    elapsed = max(now_unix - start_unix, 1.0)
    rate = progress / elapsed
    remaining = max(total - progress, 0)
    now = datetime.now(KYIV)
    eta = now + timedelta(seconds=remaining / rate) if rate > 0 else None
    return {
        "timestamp_kyiv": now.isoformat(),
        "stage": "speech_pitch_energy_statistics",
        "worker_logs": len(logs),
        "logged_iterations": progress,
        "target_iterations": total,
        "progress_percent": round(100 * progress / total, 2) if total else None,
        "iterations_per_second": round(rate, 4),
        "eta_kyiv": eta.isoformat() if eta else None,
        "free_disk_gib": round(shutil.disk_usage(workspace).free / 1024**3, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-iterations", type=int, required=True)
    parser.add_argument("--interval-seconds", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.interval_seconds <= 1800:
        parser.error("--interval-seconds must be in the range 1 to 1800")

    while alive(args.watch_pid):
        row = sample(args.logdir, args.workspace, args.total_iterations)
        append(args.output, row)
        print(json.dumps(row, sort_keys=True), flush=True)
        time.sleep(args.interval_seconds)
    row = sample(args.logdir, args.workspace, args.total_iterations)
    row["stage"] = "preparation_process_finished"
    append(args.output, row)
    print(json.dumps(row, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

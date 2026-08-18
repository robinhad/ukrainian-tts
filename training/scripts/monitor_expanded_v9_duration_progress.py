#!/usr/bin/env python3
"""Monitor v9 preprocessing by input-audio duration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


def process_is_active(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def load_results(paths: list[Path]) -> dict[str, dict]:
    """Read complete JSONL records and ignore an incomplete final write."""
    records: dict[str, dict] = {}
    for path in paths:
        if not path.is_file():
            continue
        for raw in path.read_bytes().splitlines():
            if not raw.strip(b"\0 \t\r\n"):
                continue
            try:
                row = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            identifier = str(row.get("utterance_id", ""))
            if identifier:
                records[identifier] = row
    return records


def load_start(path: Path, fallback: datetime) -> datetime:
    if not path.is_file():
        return fallback
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            return datetime.fromisoformat(json.loads(line)["timestamp_kyiv"])
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return fallback


def gpu_status() -> list[dict]:
    query = "index,utilization.gpu,memory.used,power.draw,power.limit,temperature.gpu"
    output = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    rows = []
    for line in output.splitlines():
        index, utilization, memory, power, limit, temperature = [
            item.strip() for item in line.split(",")
        ]
        rows.append(
            {
                "index": int(index),
                "utilization_percent": float(utilization),
                "memory_used_mib": float(memory),
                "power_w": float(power),
                "power_limit_w": float(limit),
                "power_utilization_percent": round(100 * float(power) / float(limit), 1),
                "temperature_c": float(temperature),
            }
        )
    return rows


def make_record(
    durations: dict[str, float],
    results: dict[str, dict],
    started: datetime,
    workspace: Path,
) -> dict:
    now = datetime.now(ZoneInfo("Europe/Kyiv"))
    unknown = sorted(set(results) - set(durations))
    completed_ids = set(results) & set(durations)
    completed_seconds = sum(durations[identifier] for identifier in completed_ids)
    total_seconds = sum(durations.values())
    elapsed = max((now - started).total_seconds(), 1.0)
    audio_seconds_per_wall_second = completed_seconds / elapsed
    remaining_wall_seconds = (
        max(total_seconds - completed_seconds, 0.0) / audio_seconds_per_wall_second
        if audio_seconds_per_wall_second > 0
        else None
    )
    failed = sum(
        row.get("processing_status") != "ok" for row in results.values()
    )
    return {
        "timestamp_kyiv": now.isoformat(),
        "phase": "duration_aware_audio_preprocessing",
        "processed_records": len(completed_ids),
        "total_records": len(durations),
        "record_progress_percent": round(100 * len(completed_ids) / len(durations), 4),
        "processed_audio_hours": round(completed_seconds / 3600, 6),
        "total_audio_hours": round(total_seconds / 3600, 6),
        "duration_progress_percent": round(100 * completed_seconds / total_seconds, 4),
        "audio_seconds_per_wall_second": round(audio_seconds_per_wall_second, 5),
        "failed": int(failed),
        "unknown_result_ids": unknown[:20],
        "unknown_result_id_count": len(unknown),
        "free_disk_gib": round(shutil.disk_usage(workspace).free / 1024**3, 2),
        "gpu": gpu_status(),
        "eta_kyiv": (
            (now + timedelta(seconds=remaining_wall_seconds)).isoformat()
            if remaining_wall_seconds is not None
            else None
        ),
        "eta_basis": "cumulative_processed_audio_duration",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--worker-pids", type=int, nargs="+", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--start-status", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.interval_seconds <= 1800:
        raise SystemExit("The monitor interval must be from 1 to 1,800 seconds.")
    frame = pd.read_parquet(args.manifest, columns=["utterance_id", "duration"])
    if frame["utterance_id"].duplicated().any() or len(frame) == 0:
        raise SystemExit("The duration manifest has invalid utterance IDs.")
    durations = dict(
        zip(frame["utterance_id"].astype(str), frame["duration"].astype(float))
    )
    started = load_start(
        args.start_status, datetime.now(ZoneInfo("Europe/Kyiv"))
    )
    args.status.parent.mkdir(parents=True, exist_ok=True)
    while True:
        results = load_results(args.results)
        record = make_record(durations, results, started, args.workspace)
        with args.status.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        if not any(process_is_active(pid) for pid in args.worker_pids):
            return 0 if not record["failed"] else 1
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

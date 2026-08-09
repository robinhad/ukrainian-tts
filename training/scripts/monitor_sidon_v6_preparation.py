#!/usr/bin/env python3
"""Monitor two Sidon preprocessing workers and report a Kyiv ETA."""

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


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            result.append(json.loads(line))
        except (json.JSONDecodeError, TypeError):
            pass
    return result


def gpu_metrics() -> list[dict]:
    command = ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw,power.limit,temperature.gpu", "--format=csv,noheader,nounits"]
    output = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    result = []
    for line in output.splitlines():
        index, uuid, util, memory, power, limit, temperature = [x.strip() for x in line.split(",")]
        result.append({"index": int(index), "uuid": uuid, "utilization_percent": float(util), "memory_used_mib": float(memory), "power_w": float(power), "power_limit_w": float(limit), "temperature_c": float(temperature)})
    return result


def active(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--worker-pids", type=int, nargs="+", required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--minimum-free-gib", type=float, default=30)
    args = parser.parse_args()
    started = time.monotonic()
    args.status.parent.mkdir(parents=True, exist_ok=True)
    while True:
        all_rows = [row for path in args.results for row in rows(path)]
        ok = sum(row.get("processing_status") == "ok" for row in all_rows)
        failed = sum(row.get("processing_status") == "failed" for row in all_rows)
        processed = ok + failed
        elapsed = max(time.monotonic() - started, 1e-6)
        rate = processed / elapsed
        eta_seconds = (args.total - processed) / rate if rate > 0 else None
        now = datetime.now(ZoneInfo("Europe/Kyiv"))
        free = shutil.disk_usage(args.workspace).free / 1024**3
        record = {
            "timestamp_kyiv": now.isoformat(), "phase": "sidon_deess_preprocessing",
            "processed": processed, "retained": ok, "failed": failed, "total": args.total,
            "progress_percent": round(100 * processed / args.total, 4),
            "rate_files_per_second": round(rate, 5), "elapsed_seconds": round(elapsed, 1),
            "eta_kyiv": (now + timedelta(seconds=eta_seconds)).isoformat() if eta_seconds is not None else None,
            "free_disk_gib": round(free, 2), "gpu": gpu_metrics(),
        }
        with args.status.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        if free < args.minimum_free_gib:
            for pid in args.worker_pids:
                if active(pid):
                    os.kill(pid, 15)
            return 2
        if not any(active(pid) for pid in args.worker_pids):
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

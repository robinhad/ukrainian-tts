#!/usr/bin/env python3
"""Monitor free disk space and run deferred cleanup at the fixed threshold."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


GIB = 1024**3
KYIV = ZoneInfo("Europe/Kyiv")


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def append_status(path: Path, item: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--free-disk-trigger-gib", type=float, default=30.0)
    args = parser.parse_args()

    if args.watch_pid <= 0:
        parser.error("--watch-pid must be positive")
    if not 1 <= args.interval_seconds <= 300:
        parser.error("--interval-seconds must be in the range 1 to 300")
    if args.free_disk_trigger_gib < 0:
        parser.error("--free-disk-trigger-gib must not be negative")

    cleanup = Path(__file__).with_name("cleanup_unlabeled_cache.py")
    while process_exists(args.watch_pid):
        free_gib = shutil.disk_usage(args.workspace).free / GIB
        item: dict[str, object] = {
            "action": "NONE",
            "free_disk_gib": round(free_gib, 2),
            "threshold_gib": args.free_disk_trigger_gib,
            "timestamp_kyiv": datetime.now(KYIV).isoformat(),
            "watch_pid": args.watch_pid,
        }
        if free_gib <= args.free_disk_trigger_gib:
            marker = json.loads(args.marker.read_text(encoding="utf-8"))
            command = [
                sys.executable,
                str(cleanup),
                "--records",
                marker["records_path"],
                "--collect-report",
                marker["collect_report_path"],
                "--marker",
                str(args.marker),
                "--free-disk-trigger-gib",
                str(args.free_disk_trigger_gib),
            ]
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            item.update(
                {
                    "action": "CLEANUP",
                    "cleanup_exit_status": result.returncode,
                    "cleanup_result": result.stdout.strip(),
                    "cleanup_stderr": result.stderr.strip(),
                    "free_disk_gib_after": round(
                        shutil.disk_usage(args.workspace).free / GIB,
                        2,
                    ),
                }
            )
        append_status(args.status, item)
        print(json.dumps(item, sort_keys=True), flush=True)
        time.sleep(args.interval_seconds)

    append_status(
        args.status,
        {
            "action": "STOP",
            "timestamp_kyiv": datetime.now(KYIV).isoformat(),
            "watch_pid": args.watch_pid,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

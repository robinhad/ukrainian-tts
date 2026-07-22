#!/usr/bin/env python3
"""Run a command, persist its result and emit a five-minute heartbeat."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    started_wall = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    process = subprocess.Popen(command)
    while True:
        try:
            status = process.wait(timeout=300)
            break
        except subprocess.TimeoutExpired:
            elapsed = round(time.monotonic() - started)
            print(f"[monitor] still running after {elapsed}s: {command[0]}", flush=True)
    record = {
        "command": command, "started_utc": started_wall,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3), "exit_status": status,
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

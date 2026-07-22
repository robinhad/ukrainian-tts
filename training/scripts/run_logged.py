#!/usr/bin/env python3
"""Run a command, persist its result and emit a five-minute heartbeat."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--eta-minutes", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    started_wall = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    kyiv = ZoneInfo("Europe/Kyiv")
    estimated_finish = datetime.now(kyiv) + timedelta(minutes=args.eta_minutes)
    process = subprocess.Popen(command)
    while True:
        try:
            status = process.wait(timeout=300)
            break
        except subprocess.TimeoutExpired:
            elapsed = round(time.monotonic() - started)
            now = datetime.now(kyiv).strftime("%Y-%m-%d %H:%M:%S %Z")
            eta = estimated_finish.strftime("%Y-%m-%d %H:%M:%S %Z")
            print(
                f"[monitor] {now}; running {command[0]} for {elapsed}s; ETA {eta}",
                flush=True,
            )
    record = {
        "command": command, "started_utc": started_wall,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3), "exit_status": status,
        "estimated_finish_kyiv": estimated_finish.isoformat(),
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

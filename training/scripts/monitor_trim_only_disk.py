#!/usr/bin/env python3
"""Delete only authorized source-cache directories at the 30 GiB trigger."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


KYIV = ZoneInfo("Europe/Kyiv")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def append(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--trigger-gib", type=float, default=30.0)
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()
    marker = json.loads(args.marker.read_text(encoding="utf-8"))
    if marker.get("status") != "PASS" or not marker.get("raw_embeddings_complete"):
        raise SystemExit("The source-cache cleanup marker is not valid.")
    root = Path(marker["source_root"]).resolve()
    targets = [Path(path).resolve() for path in marker.get("eligible_paths", [])]
    if any(path.parent != root or path.is_symlink() for path in targets):
        raise SystemExit("The cleanup marker has an unsafe path.")

    while alive(args.watch_pid):
        free = shutil.disk_usage(args.workspace).free / 1024**3
        row: dict[str, object] = {
            "timestamp_kyiv": datetime.now(KYIV).isoformat(),
            "free_disk_gib": round(free, 2),
            "trigger_gib": args.trigger_gib,
            "action": "NONE",
        }
        if free <= args.trigger_gib:
            target = next((path for path in targets if path.is_dir()), None)
            if target is None:
                row["action"] = "NO_ELIGIBLE_CACHE"
                append(args.status, row)
                print(json.dumps(row, sort_keys=True), flush=True)
                return 1
            shutil.rmtree(target)
            row.update(
                {
                    "action": "DELETE_SOURCE_CACHE",
                    "deleted_path": str(target),
                    "recoverable": False,
                    "free_disk_gib_after": round(
                        shutil.disk_usage(args.workspace).free / 1024**3, 2
                    ),
                }
            )
        append(args.status, row)
        print(json.dumps(row, sort_keys=True), flush=True)
        time.sleep(args.interval_seconds)
    append(args.status, {"timestamp_kyiv": datetime.now(KYIV).isoformat(), "action": "STOP"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

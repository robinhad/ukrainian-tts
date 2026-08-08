#!/usr/bin/env python3
"""Reclaim only approved derived v4 files at the 30 GiB trigger."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
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


def approved_targets(workspace: Path) -> list[Path]:
    experiment = (
        workspace
        / "exp_expanded_v4_trim_only"
        / "tts_jets_uk_24k_expanded_v4_trim_only_ft367_100k"
    )
    return [
        *(experiment / f"{epoch}epoch.pth" for epoch in range(96, 100)),
        experiment / "decode_jets_milestone_25k",
        experiment / "decode_jets_milestone_50k",
        experiment / "decode_jets_milestone_75k",
        experiment / "att_ws",
    ]


def remove_target(target: Path, workspace: Path) -> None:
    resolved_workspace = workspace.resolve()
    resolved = target.resolve()
    if resolved_workspace not in resolved.parents:
        raise RuntimeError(f"The cleanup target is outside the workspace: {target}")
    if target.is_symlink():
        raise RuntimeError(f"The cleanup target is a symbolic link: {target}")
    if target.is_dir():
        shutil.rmtree(target)
    elif target.is_file():
        target.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--stop-pgid", type=int)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Report exhausted cleanup capacity, but do not stop training.",
    )
    parser.add_argument("--trigger-gib", type=float, default=30.0)
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.stop_pgid is not None and args.stop_pgid <= 1:
        parser.error("--stop-pgid must be greater than 1")
    targets = approved_targets(args.workspace)

    while alive(args.watch_pid):
        free = shutil.disk_usage(args.workspace).free / 1024**3
        row: dict[str, object] = {
            "timestamp_kyiv": datetime.now(KYIV).isoformat(),
            "free_disk_gib": round(free, 2),
            "trigger_gib": args.trigger_gib,
            "action": "NONE",
        }
        removed = []
        if free <= args.trigger_gib:
            for target in targets:
                if free > args.trigger_gib:
                    break
                if not target.exists():
                    continue
                size = target.stat().st_size if target.is_file() else None
                remove_target(target, args.workspace)
                removed.append({"path": str(target), "file_size": size})
                free = shutil.disk_usage(args.workspace).free / 1024**3
            if removed:
                row.update(
                    {
                        "action": "DELETE_APPROVED_DERIVED_ARTIFACTS",
                        "deleted": removed,
                        "recoverable": False,
                        "free_disk_gib_after": round(free, 2),
                    }
                )
            if free <= args.trigger_gib and not any(path.exists() for path in targets):
                row.update(
                    {
                        "action": (
                            "ALERT_NO_APPROVED_CLEANUP_REMAINS"
                            if args.report_only
                            else "STOP_NO_APPROVED_CLEANUP_REMAINS"
                        ),
                        "free_disk_gib_after": round(free, 2),
                    }
                )
                append(args.status, row)
                print(json.dumps(row, sort_keys=True), flush=True)
                if args.report_only:
                    time.sleep(args.interval_seconds)
                    continue
                if args.stop_pgid is not None:
                    os.killpg(args.stop_pgid, signal.SIGTERM)
                else:
                    os.kill(args.watch_pid, signal.SIGTERM)
                return 1
        append(args.status, row)
        print(json.dumps(row, sort_keys=True), flush=True)
        time.sleep(args.interval_seconds)
    append(
        args.status,
        {"timestamp_kyiv": datetime.now(KYIV).isoformat(), "action": "STOP"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

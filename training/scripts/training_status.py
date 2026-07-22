#!/usr/bin/env python3
"""Print the current ESPnet training status as JSON."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


PROGRESS_RE = re.compile(r"(?P<epoch>\d+)epoch:train:\d+-(?P<batch>\d+)batch")
ETA_RE = re.compile(
    r"Estimated time to finish: "
    r"(?:(?P<hours>\d+) hours?, )?"
    r"(?:(?P<minutes>\d+) minutes? and )?"
    r"(?P<seconds>[\d.]+) seconds"
)
ERROR_RE = re.compile(r"\b(?:nan|runtimeerror|traceback)\b|out of memory", re.IGNORECASE)


def parse_log(text: str, iterations_per_epoch: int) -> dict:
    progress = list(PROGRESS_RE.finditer(text))
    estimates = list(ETA_RE.finditer(text))
    result: dict[str, object] = {
        "epoch": None,
        "batch": None,
        "total_iterations": None,
        "estimated_seconds_remaining": None,
        "error_matches": len(ERROR_RE.findall(text)),
    }
    if progress:
        match = progress[-1]
        epoch = int(match.group("epoch"))
        batch = int(match.group("batch"))
        result.update({
            "epoch": epoch,
            "batch": batch,
            "total_iterations": (epoch - 1) * iterations_per_epoch + batch,
        })
    if estimates:
        match = estimates[-1]
        result["estimated_seconds_remaining"] = (
            int(match.group("hours") or 0) * 3600
            + int(match.group("minutes") or 0) * 60
            + float(match.group("seconds"))
        )
    return result


def gpu_status() -> list[dict[str, object]]:
    fields = [
        "index",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
        "power.limit",
    ]
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = csv.reader(io.StringIO(output), skipinitialspace=True)
    result = []
    for row in rows:
        power_draw = float(row[5])
        power_limit = float(row[6])
        result.append({
            "index": int(row[0]),
            "utilization_percent": int(row[1]),
            "memory_used_mib": int(row[2]),
            "memory_total_mib": int(row[3]),
            "temperature_c": int(row[4]),
            "power_draw_w": round(power_draw, 2),
            "power_limit_w": round(power_limit, 2),
            "power_utilization_percent": round(100 * power_draw / power_limit, 1),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--target-iterations", type=int, default=25000)
    parser.add_argument("--iterations-per-epoch", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.log.is_file():
        parser.error(f"log does not exist: {args.log}")
    now = datetime.now(ZoneInfo("Europe/Kyiv"))
    status = parse_log(args.log.read_text(encoding="utf-8", errors="replace"), args.iterations_per_epoch)
    seconds = status.pop("estimated_seconds_remaining")
    total = status.get("total_iterations")
    status.update({
        "status": "RUNNING" if isinstance(total, int) and total < args.target_iterations else "COMPLETE",
        "target_iterations": args.target_iterations,
        "timestamp_kyiv": now.isoformat(),
        "eta_kyiv": (now + timedelta(seconds=seconds)).isoformat() if seconds is not None else None,
        "gpus": gpu_status(),
        "checkpoints": sorted(path.name for path in args.log.parent.glob("[0-9]*epoch.pth")),
    })
    rendered = json.dumps(status, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(status, sort_keys=True) + "\n")
    return 0 if status["error_matches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

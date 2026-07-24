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
TIMED_PROGRESS_RE = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)"
    r".*?(?P<epoch>\d+)epoch:train:\d+-(?P<batch>\d+)batch"
)
ETA_RE = re.compile(
    r"Estimated time to finish: "
    r"(?:(?P<hours>\d+) hours?, )?"
    r"(?:(?P<minutes>\d+) minutes? and )?"
    r"(?P<seconds>[\d.]+) seconds"
)
ERROR_RE = re.compile(r"\b(?:nan|runtimeerror|traceback)\b|out of memory", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+")
TRAIN_METRIC_NAMES = (
    "generator_loss",
    "generator_g_mel_loss",
    "generator_align_loss",
    "discriminator_loss",
    "train_time",
)
NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"


def parse_log(text: str, iterations_per_epoch: int) -> dict:
    progress = list(PROGRESS_RE.finditer(text))
    estimates = list(ETA_RE.finditer(text))
    result: dict[str, object] = {
        "epoch": None,
        "batch": None,
        "total_iterations": None,
        "estimated_seconds_remaining": None,
        "estimate_timestamp": None,
        "observed_seconds_per_iteration": None,
        "observed_iterations_per_minute": None,
        "latest_train_metrics": {},
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
        line_end = text.find("\n", match.end())
        line = text[match.start():] if line_end == -1 else text[match.start():line_end]
        result["latest_train_metrics"] = {
            name: float(metric.group(1))
            for name in TRAIN_METRIC_NAMES
            if (
                metric := re.search(
                    rf"\b{re.escape(name)}=({NUMBER_PATTERN})\b",
                    line,
                    re.IGNORECASE,
                )
            )
        }
    timed_progress = list(TIMED_PROGRESS_RE.finditer(text))
    if len(timed_progress) >= 2:
        samples = timed_progress[-20:]
        first = samples[0]
        last = samples[-1]
        first_time = datetime.strptime(
            first.group("timestamp"), "%Y-%m-%d %H:%M:%S,%f"
        )
        last_time = datetime.strptime(
            last.group("timestamp"), "%Y-%m-%d %H:%M:%S,%f"
        )
        first_iteration = (
            (int(first.group("epoch")) - 1) * iterations_per_epoch
            + int(first.group("batch"))
        )
        last_iteration = (
            (int(last.group("epoch")) - 1) * iterations_per_epoch
            + int(last.group("batch"))
        )
        iteration_delta = last_iteration - first_iteration
        elapsed = (last_time - first_time).total_seconds()
        if iteration_delta > 0 and elapsed > 0:
            seconds_per_iteration = elapsed / iteration_delta
            result.update({
                "observed_seconds_per_iteration": round(seconds_per_iteration, 4),
                "observed_iterations_per_minute": round(
                    60.0 / seconds_per_iteration, 2
                ),
            })
    if estimates:
        match = estimates[-1]
        result["estimated_seconds_remaining"] = (
            int(match.group("hours") or 0) * 3600
            + int(match.group("minutes") or 0) * 60
            + float(match.group("seconds"))
        )
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        line = text[line_start:] if line_end == -1 else text[line_start:line_end]
        timestamp = TIMESTAMP_RE.search(line)
        if timestamp:
            result["estimate_timestamp"] = timestamp.group()
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
    estimate_timestamp = status.pop("estimate_timestamp")
    total = status.get("total_iterations")
    eta_source = "espnet" if seconds is not None else None
    eta = None
    if seconds is not None:
        if isinstance(estimate_timestamp, str):
            estimated_at = datetime.strptime(
                estimate_timestamp, "%Y-%m-%d %H:%M:%S,%f"
            ).replace(tzinfo=ZoneInfo("Europe/Kyiv"))
            eta = estimated_at + timedelta(seconds=seconds)
        else:
            eta = now + timedelta(seconds=seconds)
    observed_seconds = status.get("observed_seconds_per_iteration")
    if (
        seconds is None
        and isinstance(total, int)
        and isinstance(observed_seconds, float)
        and total < args.target_iterations
    ):
        seconds = (args.target_iterations - total) * observed_seconds
        eta_source = "observed_progress"
        eta = now + timedelta(seconds=seconds)
    status.update({
        "status": "RUNNING" if isinstance(total, int) and total < args.target_iterations else "COMPLETE",
        "target_iterations": args.target_iterations,
        "timestamp_kyiv": now.isoformat(),
        "eta_kyiv": eta.isoformat() if eta is not None else None,
        "eta_source": eta_source,
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

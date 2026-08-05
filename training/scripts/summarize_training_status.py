#!/usr/bin/env python3
"""Summarize the monitored state of one training run."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path


def summarize(path: Path, target_iterations: int, maximum_gap_minutes: float) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("target_iterations") == target_iterations]
    if not rows:
        raise RuntimeError("The status file has no samples for the target run.")

    timestamps = [datetime.fromisoformat(row["timestamp_kyiv"]) for row in rows]
    gaps = [
        (current - previous).total_seconds() / 60.0
        for previous, current in zip(timestamps, timestamps[1:])
    ]
    gpu_indices = sorted(
        {gpu["index"] for row in rows for gpu in row.get("gpus", [])}
    )
    gpu_summary = []
    for index in gpu_indices:
        samples = [
            gpu
            for row in rows
            for gpu in row.get("gpus", [])
            if gpu["index"] == index
        ]
        gpu_summary.append(
            {
                "index": index,
                "sample_count": len(samples),
                "average_power_w": statistics.mean(
                    sample["power_draw_w"] for sample in samples
                ),
                "maximum_power_w": max(sample["power_draw_w"] for sample in samples),
                "average_power_limit_percent": statistics.mean(
                    sample["power_utilization_percent"] for sample in samples
                ),
                "maximum_temperature_c": max(
                    sample["temperature_c"] for sample in samples
                ),
                "maximum_memory_used_mib": max(
                    sample["memory_used_mib"] for sample in samples
                ),
                "maximum_utilization_percent": max(
                    sample["utilization_percent"] for sample in samples
                ),
            }
        )

    critical_sample_count = sum(bool(row.get("critical_conditions")) for row in rows)
    maximum_error_matches = max(row.get("error_matches", 0) for row in rows)
    maximum_gap = max(gaps, default=0.0)
    status = (
        "PASS"
        if maximum_gap <= maximum_gap_minutes
        and critical_sample_count == 0
        and maximum_error_matches == 0
        and gpu_indices == [0, 1]
        else "FAIL"
    )
    return {
        "status": status,
        "target_iterations": target_iterations,
        "sample_count": len(rows),
        "first_timestamp_kyiv": rows[0]["timestamp_kyiv"],
        "last_timestamp_kyiv": rows[-1]["timestamp_kyiv"],
        "latest_total_iterations": rows[-1].get("total_iterations"),
        "latest_eta_kyiv": rows[-1].get("eta_kyiv"),
        "maximum_allowed_gap_minutes": maximum_gap_minutes,
        "maximum_observed_gap_minutes": maximum_gap,
        "gaps_over_limit": sum(gap > maximum_gap_minutes for gap in gaps),
        "critical_sample_count": critical_sample_count,
        "maximum_error_matches": maximum_error_matches,
        "minimum_free_disk_gib": min(row["free_disk_gib"] for row in rows),
        "gpus": gpu_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-iterations", type=int, required=True)
    parser.add_argument("--maximum-gap-minutes", type=float, default=30.0)
    args = parser.parse_args()

    report = summarize(args.input, args.target_iterations, args.maximum_gap_minutes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

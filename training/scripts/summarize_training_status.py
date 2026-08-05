#!/usr/bin/env python3
"""Summarize the monitored state of one training run."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def summarize(
    rows: list[dict],
    target_iterations: int | None = None,
    maximum_gap_minutes: float = 30.0,
) -> dict:
    """Return a compatible GPU summary and monitor-continuity checks."""
    if target_iterations is not None:
        rows = [
            row for row in rows if row.get("target_iterations") == target_iterations
        ]
    if not rows:
        return {
            "status": "FAIL",
            "target_iterations": target_iterations,
            "sample_count": 0,
            "first_sample_kyiv": None,
            "last_sample_kyiv": None,
            "first_timestamp_kyiv": None,
            "last_timestamp_kyiv": None,
            "gpus": [],
        }

    timestamps = [datetime.fromisoformat(row["timestamp_kyiv"]) for row in rows]
    gaps = [
        (current - previous).total_seconds() / 60.0
        for previous, current in zip(timestamps, timestamps[1:])
    ]
    samples_by_gpu: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        for gpu in row.get("gpus", []):
            samples_by_gpu[int(gpu["index"])].append(gpu)
    gpu_indices = sorted(samples_by_gpu)
    gpu_summary = []
    for index in gpu_indices:
        samples = samples_by_gpu[index]
        powers = [float(sample["power_draw_w"]) for sample in samples]
        utilizations = [int(sample["utilization_percent"]) for sample in samples]
        temperatures = [int(sample["temperature_c"]) for sample in samples]
        memory = [int(sample["memory_used_mib"]) for sample in samples]
        average_power = statistics.mean(powers)
        maximum_power = max(powers)
        gpu_summary.append(
            {
                "index": index,
                "sample_count": len(samples),
                "power_limit_w": float(samples[-1].get("power_limit_w", 0.0)),
                "mean_sampled_power_w": round(average_power, 2),
                "maximum_sampled_power_w": round(maximum_power, 2),
                "mean_sampled_utilization_percent": round(
                    statistics.mean(utilizations), 2
                ),
                "maximum_sampled_utilization_percent": max(utilizations),
                "maximum_sampled_temperature_c": max(temperatures),
                "maximum_sampled_memory_mib": max(memory),
                "average_power_w": average_power,
                "maximum_power_w": maximum_power,
                "average_power_limit_percent": statistics.mean(
                    sample.get("power_utilization_percent", 0.0)
                    for sample in samples
                ),
                "maximum_temperature_c": max(temperatures),
                "maximum_memory_used_mib": max(memory),
                "maximum_utilization_percent": max(utilizations),
            }
        )

    critical_sample_count = sum(bool(row.get("critical_conditions")) for row in rows)
    maximum_error_matches = max(row.get("error_matches", 0) for row in rows)
    maximum_gap = max(gaps, default=0.0)
    monitor_checks_pass = (
        maximum_gap <= maximum_gap_minutes
        and critical_sample_count == 0
        and maximum_error_matches == 0
    )
    if target_iterations is not None:
        monitor_checks_pass = monitor_checks_pass and gpu_indices == [0, 1]
    status = "PASS" if rows and gpu_summary and monitor_checks_pass else "FAIL"
    return {
        "status": status,
        "target_iterations": target_iterations,
        "sample_count": len(rows),
        "first_sample_kyiv": rows[0]["timestamp_kyiv"],
        "last_sample_kyiv": rows[-1]["timestamp_kyiv"],
        "first_timestamp_kyiv": rows[0]["timestamp_kyiv"],
        "last_timestamp_kyiv": rows[-1]["timestamp_kyiv"],
        "latest_total_iterations": rows[-1].get("total_iterations"),
        "latest_eta_kyiv": rows[-1].get("eta_kyiv"),
        "maximum_allowed_gap_minutes": maximum_gap_minutes,
        "maximum_observed_gap_minutes": maximum_gap,
        "gaps_over_limit": sum(gap > maximum_gap_minutes for gap in gaps),
        "critical_sample_count": critical_sample_count,
        "maximum_error_matches": maximum_error_matches,
        "minimum_free_disk_gib": min(
            (row["free_disk_gib"] for row in rows if "free_disk_gib" in row),
            default=None,
        ),
        "gpus": gpu_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-iterations", type=int)
    parser.add_argument("--maximum-gap-minutes", type=float, default=30.0)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = summarize(rows, args.target_iterations, args.maximum_gap_minutes)
    report["source"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

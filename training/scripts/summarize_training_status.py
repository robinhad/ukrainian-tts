#!/usr/bin/env python3
"""Summarize GPU power and utilization samples from the training monitor."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(rows: list[dict]) -> dict:
    """Return one summary for each GPU index."""
    samples: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        for gpu in row.get("gpus", []):
            samples[int(gpu["index"])].append(gpu)

    gpu_summaries = []
    for index, gpu_rows in sorted(samples.items()):
        powers = [float(row["power_draw_w"]) for row in gpu_rows]
        utilization = [int(row["utilization_percent"]) for row in gpu_rows]
        temperatures = [int(row["temperature_c"]) for row in gpu_rows]
        memory = [int(row["memory_used_mib"]) for row in gpu_rows]
        gpu_summaries.append({
            "index": index,
            "sample_count": len(gpu_rows),
            "power_limit_w": float(gpu_rows[-1]["power_limit_w"]),
            "mean_sampled_power_w": round(sum(powers) / len(powers), 2),
            "maximum_sampled_power_w": round(max(powers), 2),
            "mean_sampled_utilization_percent": round(
                sum(utilization) / len(utilization), 2
            ),
            "maximum_sampled_utilization_percent": max(utilization),
            "maximum_sampled_temperature_c": max(temperatures),
            "maximum_sampled_memory_mib": max(memory),
        })

    return {
        "sample_count": len(rows),
        "first_sample_kyiv": rows[0].get("timestamp_kyiv") if rows else None,
        "last_sample_kyiv": rows[-1].get("timestamp_kyiv") if rows else None,
        "gpus": gpu_summaries,
        "status": "PASS" if rows and gpu_summaries else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = summarize(rows)
    report["source"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

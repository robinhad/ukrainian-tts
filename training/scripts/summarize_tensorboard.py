#!/usr/bin/env python3
"""Export TensorBoard scalar metrics to a machine-readable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol

import numpy as np
from tensorboard.backend.event_processing.event_multiplexer import (
    EventMultiplexer,
)


class ScalarEvent(Protocol):
    wall_time: float
    step: int
    value: float


def summarize_events(events: list[ScalarEvent]) -> dict[str, float | int]:
    """Summarize one scalar series after a deterministic step sort."""
    if not events:
        raise ValueError("a scalar series must not be empty")
    ordered = sorted(events, key=lambda event: (event.step, event.wall_time))
    latest_by_step = {event.step: event for event in ordered}
    values = [latest_by_step[step] for step in sorted(latest_by_step)]
    first = values[0]
    last = values[-1]
    numeric = np.asarray([event.value for event in values], dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("a scalar series contains a non-finite value")
    return {
        "count": len(values),
        "first_step": int(first.step),
        "first_value": float(first.value),
        "last_step": int(last.step),
        "last_value": float(last.value),
        "minimum_value": float(numeric.min()),
        "maximum_value": float(numeric.max()),
        "absolute_change": float(last.value - first.value),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.logdir.is_dir():
        parser.error(f"TensorBoard log directory does not exist: {args.logdir}")
    multiplexer = EventMultiplexer(size_guidance={"scalars": 0})
    multiplexer.AddRunsFromDirectory(str(args.logdir))
    multiplexer.Reload()
    runs = {}
    errors = []
    for run, tags in sorted(multiplexer.Runs().items()):
        scalar_tags = tags.get("scalars", [])
        summaries = {}
        for tag in sorted(scalar_tags):
            try:
                summaries[tag] = summarize_events(
                    multiplexer.Scalars(run, tag)
                )
            except ValueError as error:
                errors.append(f"{run}/{tag}: {error}")
        runs[run] = {
            "scalar_tag_count": len(summaries),
            "scalars": summaries,
        }

    report = {
        "status": "PASS" if runs and not errors else "FAIL",
        "source": str(args.logdir.resolve()),
        "run_count": len(runs),
        "runs": runs,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_count": report["run_count"],
                "scalar_tag_counts": {
                    run: data["scalar_tag_count"]
                    for run, data in runs.items()
                },
                "errors": errors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Assign pending enhancement shards to balanced worker slots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def plan_assignments(
    total_records: int,
    completed: dict[int, int],
    num_shards: int,
    worker_count: int,
) -> tuple[list[list[int]], list[int]]:
    if total_records < 0 or num_shards < 1 or worker_count < 1:
        raise ValueError("The record and worker values are invalid.")
    tasks = []
    for shard in range(num_shards):
        expected = (total_records + num_shards - 1 - shard) // num_shards
        remaining = expected - completed.get(shard, 0)
        if remaining < 0:
            raise ValueError(f"Shard {shard} has more records than expected.")
        if remaining:
            tasks.append((remaining, shard))
    assignments = [[] for _ in range(min(worker_count, len(tasks)))]
    loads = [0 for _ in assignments]
    for remaining, shard in sorted(tasks, reverse=True):
        slot = min(range(len(loads)), key=lambda index: (loads[index], index))
        assignments[slot].append(shard)
        loads[slot] += remaining
    return assignments, loads


def count_completed(records_dir: Path) -> dict[int, int]:
    completed = {}
    for path in records_dir.glob("records-*.jsonl"):
        shard = int(path.stem.rsplit("-", 1)[1])
        completed[shard] = sum(
            1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    total_records = len(pd.read_parquet(args.manifest, columns=["utterance_id"]))
    assignments, loads = plan_assignments(
        total_records,
        count_completed(args.records_dir),
        args.num_shards,
        args.workers,
    )
    print(
        json.dumps(
            {
                "assignments": assignments,
                "pending_shards": sum(map(len, assignments)),
                "planned_loads": loads,
                "remaining_records": sum(loads),
                "status": "PASS",
                "worker_slots": len(assignments),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Delete one processed unlabeled batch and its cached source shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def remove_file(path: Path) -> int:
    if not path.is_file() and not path.is_symlink():
        return 0
    size = path.stat().st_size
    path.unlink()
    return size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--collect-report", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()

    removed_bytes = 0
    if args.records.is_file():
        for line in args.records.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                removed_bytes += remove_file(Path(record["audio_path"]))
    report = json.loads(args.collect_report.read_text(encoding="utf-8"))
    for item in report.get("cache_artifacts", []):
        snapshot = Path(item["snapshot_path"])
        blob = Path(item["blob_path"])
        removed_bytes += remove_file(snapshot)
        if blob != snapshot:
            removed_bytes += remove_file(blob)
    remove_file(args.records)
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    args.marker.write_text(
        json.dumps(
            {
                "status": "PASS",
                "shard_start": report["shard_start"],
                "shard_count": report["shard_count"],
                "removed_bytes": removed_bytes,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.marker.read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

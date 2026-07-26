#!/usr/bin/env python3
"""Merge JSONL records and remove duplicate utterance identifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records: dict[str, dict] = {}
    for path in args.inputs:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row["utterance_id"])
            previous = records.get(identifier)
            if previous is not None and previous != row:
                raise ValueError(f"Conflicting records for {identifier}")
            records[identifier] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(records[key], ensure_ascii=False, sort_keys=True) + "\n"
            for key in sorted(records)
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "records": len(records),
                "status": "PASS",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

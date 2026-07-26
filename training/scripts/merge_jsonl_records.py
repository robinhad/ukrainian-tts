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
    records_by_id: dict[str, dict] = {}
    content_to_id: dict[tuple[str, str], str] = {}
    for path in args.inputs:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row["utterance_id"])
            content_key = (
                str(row["audio_sha256_source"]),
                str(row["text_sha256_source"]),
            )
            previous = records_by_id.get(identifier)
            if previous is not None:
                previous_key = (
                    str(previous["audio_sha256_source"]),
                    str(previous["text_sha256_source"]),
                )
                if previous_key != content_key:
                    raise ValueError(f"Conflicting content for {identifier}")
                continue
            prior_identifier = content_to_id.get(content_key)
            if prior_identifier is not None:
                if identifier < prior_identifier:
                    del records_by_id[prior_identifier]
                    records_by_id[identifier] = row
                    content_to_id[content_key] = identifier
                continue
            records_by_id[identifier] = row
            content_to_id[content_key] = identifier
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(records_by_id[key], ensure_ascii=False, sort_keys=True)
            + "\n"
            for key in sorted(records_by_id)
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "records": len(records_by_id),
                "status": "PASS",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

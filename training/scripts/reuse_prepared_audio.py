#!/usr/bin/env python3
"""Reuse prepared WAV metadata after a source-record filter changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PREPARATION_FIELDS = (
    "audio_path",
    "duration",
    "sample_rate",
    "channels",
    "format",
    "qc_flags",
    "audio_sha256",
    "trim_applied",
    "duration_before_trim",
    "trim_start_seconds",
    "trim_end_seconds",
    "trim_removed_seconds",
    "trim_threshold_rms",
    "trim_config_hash",
)
MATCH_FIELDS = (
    "utterance_id",
    "text_raw",
    "split",
    "audio_sha256_source",
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reconcile(source_rows: list[dict], prepared_rows: list[dict]) -> list[dict]:
    """Select prepared rows for the new source set and verify source identity."""
    prepared_by_id = {
        str(row["utterance_id"]): row for row in prepared_rows
    }
    if len(prepared_by_id) != len(prepared_rows):
        raise RuntimeError("prepared records contain duplicate utterance IDs")
    output = []
    for source in source_rows:
        utterance_id = str(source["utterance_id"])
        if utterance_id not in prepared_by_id:
            raise RuntimeError(f"prepared record is missing: {utterance_id}")
        prepared = prepared_by_id[utterance_id]
        for field in MATCH_FIELDS:
            if source.get(field) != prepared.get(field):
                raise RuntimeError(
                    f"{utterance_id}: source identity changed in {field}"
                )
        for field in PREPARATION_FIELDS:
            if field not in prepared:
                raise RuntimeError(
                    f"{utterance_id}: prepared field is missing: {field}"
                )
        if not Path(prepared["audio_path"]).is_file():
            raise RuntimeError(
                f"{utterance_id}: prepared audio does not exist: "
                f"{prepared['audio_path']}"
            )
        merged = source.copy()
        merged.update({
            field: prepared[field]
            for field in PREPARATION_FIELDS
        })
        output.append(merged)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--prepared-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_rows = load_jsonl(args.source_records)
    prepared_rows = load_jsonl(args.prepared_records)
    output = reconcile(source_rows, prepared_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in output:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(args.output)
    print(json.dumps({
        "status": "PASS",
        "source_records": len(source_rows),
        "prepared_records_before": len(prepared_rows),
        "prepared_records_after": len(output),
        "removed_records": len(prepared_rows) - len(output),
        "artifact": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
